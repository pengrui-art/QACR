#!/usr/bin/env python3
"""Phase 3.15 - End-to-end QACR training on real VQA data.

Freezes the VLM backbone, trains only Router + Executor using the LLM's
next-token-prediction loss as the task signal plus budget regularization.

Usage (smoke test with streaming - no download needed):
    conda run -n qacr python scripts/train_qacr_e2e.py \
        --model Model/Qwen35-08B --dataset vqav2 --streaming \
        --max-samples 50 --epochs 1 --budget 0.45

Usage (full training with local data):
    conda run -n qacr python scripts/train_qacr_e2e.py \
        --model Model/Qwen35-08B --dataset vqav2 \
        --max-samples 20000 --epochs 3 --budget 0.45 \
        --save-dir checkpoints/qacr_vqav2_b045

Usage (4 GPUs, with progress bar):
    conda run -n qacr torchrun --standalone --nproc_per_node=4 \
        scripts/train_qacr_e2e.py \
        --model Model/Qwen35-08B --dataset vqav2 \
        --local-data-dir data/VQAv2 \
        --max-samples 20000 --epochs 3 --budget 0.45
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from qacr.data.vqa_dataset import VQADataset, vqa_collate_fn
from qacr.qacr_model import QACRRoutingHook, build_qacr_components
from qacr.routing import linear_temperature

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline end-to-end training")
    p.add_argument("--baseline", default="token_pruning", choices=["token_pruning", "low_res", "image_only"])
    p.add_argument("--keep-ratio", type=float, default=0.45, help="Tokens to keep. Used for token_pruning.")
    p.add_argument("--low-res-grid", type=int, default=9, help="Grid size for low_res. Natively resizes PIL image to grid*28.")
    
    # Model
    p.add_argument("--model", default="Model/Qwen35-08B", help="Path to Qwen3.5-VL")
    # Data
    p.add_argument("--dataset", default="vqav2", choices=["vqav2", "textvqa", "pope"])
    p.add_argument("--max-samples", type=int, default=20000)
    p.add_argument(
        "--streaming", action="store_true", help="Use HF streaming (no full download)"
    )
    p.add_argument("--local-data-dir", type=str, default=None)
    # Training
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4) # baseline learning rate
    
    # ImageOnly Router / Executor (matches QACR budget and architecture exactly)
    p.add_argument("--budget", type=float, default=0.45, help="Compute budget for ImageOnly")
    p.add_argument("--lambda-compute", type=float, default=0.8)
    p.add_argument("--lambda-entropy", type=float, default=0.02)
    p.add_argument("--temp-start", type=float, default=1.5)
    p.add_argument("--temp-end", type=float, default=0.3)
    p.add_argument("--router-hidden", type=int, default=128)
    p.add_argument("--executor-hidden", type=int, default=256)
    p.add_argument("--deep-layers", type=int, default=3)
    p.add_argument(
        "--executor-output-alpha",
        type=float,
        default=1.0,
        help="Blend factor for routed visual tokens (1.0=legacy full rewrite, lower=more conservative)",
    )
    
    # Output
    p.add_argument("--save-dir", type=str, default=None)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every-epoch", action="store_true", default=True)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--profile-every", type=int, default=0)
    p.add_argument("--profile-sync-cuda", action="store_true")
    p.add_argument("--disable-tqdm", action="store_true")
    return p.parse_args()

class BaselineRoutingHook:
    """Consolidated hook for TokenPruning, LowRes, and ImageOnly baselines."""
    def __init__(self, baseline: str, executor: torch.nn.Module, router: torch.nn.Module | None = None, keep_ratio: float = 0.45, lambda_compute: float = 0.8, lambda_entropy: float = 0.02):
        self.baseline = baseline
        self.executor = executor
        self.router = router
        self.keep_ratio = keep_ratio
        
        # Budget loss variables for ImageOnly
        self.lambda_compute = lambda_compute
        self.lambda_entropy = lambda_entropy
        self.budget = keep_ratio
        self.temperature = 1.0
        
        self.routing_loss = torch.tensor(0.0)
        self.stats = type("Stats", (), {"expected_compute": 0.0, "mean_skip": 0.0, "mean_shallow": 0.0, "mean_deep": 0.0, "budget_loss": None})()

    def __call__(self, module, inputs, output):
        # inputs is tuple(hidden_states)
        hidden_states = inputs[0] if isinstance(inputs, tuple) else inputs
        pooler_output = output.pooler_output if hasattr(output, "pooler_output") else output
        
        x = pooler_output  
        orig_ndim = x.ndim
        if orig_ndim == 2:
            x = x.unsqueeze(0)
        x_orig_dtype = x.dtype
        x = x.float()
        B, N, D = x.shape
        self.routing_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        
        if self.baseline == "low_res":
            # LowRes doesn't route (it just processes fewer tokens in the grid).
            # We send all components to the deep path of the executor so parameter capacity matches QACR.
            route_probs = torch.zeros(B, N, 3, device=x.device, dtype=x.dtype)
            route_probs[..., 2] = 1.0 # All deep
            self.stats.expected_compute = 1.0
            self.stats.mean_skip = 0.0
            self.stats.mean_shallow = 0.0
            self.stats.mean_deep = 1.0
            
        elif self.baseline == "token_pruning":
            # Heuristic pruning by L2 norm
            keep_k = max(1, int(round(N * self.keep_ratio)))
            scores = torch.norm(x, p=2, dim=-1) # [B, N]
            topk_idx = torch.topk(scores, k=keep_k, dim=-1, largest=True).indices
            
            route_probs = torch.zeros(B, N, 3, device=x.device, dtype=x.dtype)
            route_probs[..., 0] = 1.0 # Default skip
            batch_idx = torch.arange(B).unsqueeze(1).expand_as(topk_idx)
            route_probs[batch_idx, topk_idx, 0] = 0.0
            route_probs[batch_idx, topk_idx, 2] = 1.0 # Kept tokens Deep
            
            self.stats.expected_compute = self.keep_ratio
            self.stats.mean_skip = float(route_probs[..., 0].mean())
            self.stats.mean_shallow = 0.0
            self.stats.mean_deep = float(route_probs[..., 2].mean())

        elif self.baseline == "image_only":
            assert self.router is not None
            # Predict routes using image tokens only
            logits = self.router(image_tokens=x).logits # [B, N, 3]
            
            from qacr.routing.soft_hard import soft_routing_probs, compute_regularization_loss
            route_probs = soft_routing_probs(logits, temperature=self.temperature, use_gumbel=self.router.training)
            
            costs = torch.tensor([0.0, 0.35, 1.0], device=x.device, dtype=torch.float32)
            loss_budget, expected_compute = compute_regularization_loss(
                route_probs, route_costs=costs, budget_ratio=self.budget
            )
            entropy = - (route_probs * route_probs.clamp_min(1e-8).log()).sum(-1).mean()
            self.routing_loss = self.lambda_compute * loss_budget - self.lambda_entropy * entropy
            self.stats.budget_loss = loss_budget
            self.stats.expected_compute = expected_compute.item()
            self.stats.mean_skip = float(route_probs[..., 0].mean())
            self.stats.mean_shallow = float(route_probs[..., 1].mean())
            self.stats.mean_deep = float(route_probs[..., 2].mean())

        out, _ = self.executor(image_tokens=x, route_probs=route_probs, mode="soft")
        out = out.to(x_orig_dtype)
        
        if orig_ndim == 2:
            out = out.squeeze(0)

        if hasattr(output, "pooler_output"):
            from transformers.modeling_outputs import BaseModelOutputWithPooling
            return BaseModelOutputWithPooling(
                last_hidden_state=output.last_hidden_state,
                pooler_output=out,
                hidden_states=output.hidden_states,
                attentions=output.attentions,
            )
        return out


def _init_distributed() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "DDP training requires CUDA GPUs, but CUDA is not available."
            )
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")

    return distributed, rank, local_rank, world_size


def _is_main_process(rank: int) -> bool:
    return rank == 0


def _unwrap_module(m: torch.nn.Module) -> torch.nn.Module:
    return m.module if isinstance(m, DDP) else m


def _reduce_mean(
    value: float, device: torch.device, distributed: bool, world_size: int
) -> float:
    if not distributed:
        return float(value)
    t = torch.tensor(float(value), device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t = t / world_size
    return float(t.item())


def _maybe_sync_cuda(device: torch.device, enable: bool) -> None:
    if enable and device.type == "cuda":
        torch.cuda.synchronize(device)


def _save_checkpoint(
    router: torch.nn.Module,
    executor: torch.nn.Module,
    path: Path,
    args: argparse.Namespace,
    step: int,
    loss: float,
) -> None:
    router_to_save = _unwrap_module(router)
    executor_to_save = _unwrap_module(executor)
    torch.save(
        {
            "router": _unwrap_module(router).state_dict() if router is not executor else {},
            "executor": executor_to_save.state_dict(),
            "step": step,
            "loss": loss,
            "baseline": getattr(args, "baseline", "unknown"),
            "router_hidden": getattr(args, "router_hidden", 128),
            "executor_hidden": getattr(args, "executor_hidden", 256),
            "deep_layers": getattr(args, "deep_layers", 3),
            "executor_output_alpha": getattr(args, "executor_output_alpha", 1.0),
        },
        path,
    )
    log.info("Saved checkpoint -> %s (loss=%.4f)", path.name, loss)


def main() -> None:
    args = parse_args()
    distributed, rank, local_rank, world_size = _init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    main_process = _is_main_process(rank)

    if main_process:
        log.info(
            "Runtime: distributed=%s world_size=%d rank=%d local_rank=%d device=%s",
            distributed,
            world_size,
            rank,
            local_rank,
            device,
        )
        if (
            torch.cuda.is_available()
            and torch.cuda.device_count() > 1
            and not distributed
        ):
            log.warning(
                "Detected %d CUDA devices but running single-process. "
                "Use torchrun --nproc_per_node=%d for multi-GPU training.",
                torch.cuda.device_count(),
                min(4, torch.cuda.device_count()),
            )

    # 1. Load frozen VLM
    if main_process:
        log.info("Loading VLM from %s ...", args.model)

    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
    )

    load_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=load_dtype,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = model.to(device)

    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    hidden_dim = model.model.visual.merger.linear_fc2.out_features

    # 2. Build trainable Router + Executor
    from qacr.vision import DepthMultiPathExecutor
    from qacr.routing.image_only_router import ImageOnlyRouter
    
    executor = DepthMultiPathExecutor(
        token_dim=hidden_dim,
        hidden_dim=args.executor_hidden,
        deep_layers=args.deep_layers,
        output_alpha=args.executor_output_alpha,
    ).to(device)

    router = None
    if args.baseline == "image_only":
        router = ImageOnlyRouter(
            image_dim=hidden_dim, hidden_dim=args.router_hidden
        ).to(device)

    if distributed:
        if router is not None:
            router = DDP(router, device_ids=[local_rank], output_device=local_rank)
        executor = DDP(executor, device_ids=[local_rank], output_device=local_rank)

    if router is not None:
        router.train()
    executor.train()

    n_params = sum(p.numel() for p in _unwrap_module(executor).parameters())
    if router is not None:
        n_params += sum(p.numel() for p in _unwrap_module(router).parameters())
    if main_process:
        log.info("Trainable params: Router + Executor = %s", f"{n_params:,}")

    # 3. Build dataset & dataloader
    if main_process:
        log.info(
            "Loading dataset: %s (max_samples=%s, streaming=%s)",
            args.dataset,
            args.max_samples,
            args.streaming,
        )

    if args.baseline == "low_res":
        from PIL import Image
        def low_res_transform(img):
            size = args.low_res_grid * 28
            return img.resize((size, size))
        
        # We subclass VQADataset locally to inject an image transform
        class TransformedVQADataset(VQADataset):
            def __getitem__(self, idx):
                item = super().__getitem__(idx)
                item["image"] = low_res_transform(item["image"])
                return item
        ds = TransformedVQADataset(
            dataset_name=args.dataset,
            split="train",
            max_samples=args.max_samples,
            streaming=args.streaming,
            local_dir=args.local_data_dir,
        )
    else:
        ds = VQADataset(
            dataset_name=args.dataset,
            split="train",
            max_samples=args.max_samples,
            streaming=args.streaming,
            local_dir=args.local_data_dir,
        )

    collate = partial(vqa_collate_fn, processor=processor)
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )

    loader_kwargs = {
        "dataset": ds,
        "batch_size": args.batch_size,
        "shuffle": (sampler is None),
        "sampler": sampler,
        "num_workers": args.num_workers,
        "collate_fn": collate,
        "drop_last": True,
        "pin_memory": torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    loader = DataLoader(**loader_kwargs)

    # 4. Optimizer
    params_to_opt = list(executor.parameters())
    if router is not None:
        params_to_opt += list(router.parameters())
    
    optimizer = torch.optim.AdamW(
        params_to_opt,
        lr=args.lr,
        weight_decay=0.01,
    )

    # 5. Routing hook
    hook = BaselineRoutingHook(
        baseline=args.baseline,
        executor=executor,
        router=router,
        keep_ratio=args.keep_ratio if args.baseline == "token_pruning" else args.budget,
        lambda_compute=args.lambda_compute,
        lambda_entropy=args.lambda_entropy,
    )

    # 6. Training
    if not args.save_dir:
        label = args.baseline
        if args.baseline == "token_pruning":
            label += f"_kr{args.keep_ratio:.2f}"
        elif args.baseline == "image_only":
            label += f"_b{args.budget:.2f}"
        elif args.baseline == "low_res":
            label += f"_g{args.low_res_grid}"
        
        save_dir = REPO / "checkpoints" / f"{label}_{args.dataset}"
    else:
        save_dir = Path(args.save_dir)
    if main_process:
        save_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    global_step = 0
    best_loss = float("inf")
    train_log: list[dict] = []
    profile_every = args.profile_every if args.profile_every > 0 else args.log_every

    if main_process:
        log.info(
            "Starting training: %d epochs, %d batches/epoch (per-rank), grad_accum=%d, world_size=%d",
            args.epochs,
            len(loader),
            args.grad_accum,
            world_size,
        )
        log.info(
            "Budget=%.2f, lr=%.1e, temp=%.2f->%.2f",
            args.budget,
            args.lr,
            args.temp_start,
            args.temp_end,
        )
        log.info(
            "Timing profiler: every=%d sync_cuda=%s",
            profile_every,
            args.profile_sync_cuda,
        )

    try:
        for epoch in range(args.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)

            epoch_loss = 0.0
            epoch_ntp = 0.0
            epoch_steps = 0
            optimizer.zero_grad(set_to_none=True)
            timing_window = {
                "data": 0.0,
                "forward": 0.0,
                "backward": 0.0,
                "opt": 0.0,
                "iter": 0.0,
                "count": 0,
            }
            last_iter_end = time.perf_counter()

            pbar = None
            if main_process and not args.disable_tqdm and tqdm is not None:
                pbar = tqdm(
                    total=len(loader),
                    desc=f"Epoch {epoch + 1}/{args.epochs}",
                    dynamic_ncols=True,
                )

            for batch_idx, batch in enumerate(loader):
                iter_start = time.perf_counter()
                data_time = iter_start - last_iter_end

                step_in_epoch = epoch * len(loader) + batch_idx
                temp = linear_temperature(
                    step_in_epoch,
                    len(loader) * args.epochs,
                    args.temp_start,
                    args.temp_end,
                )

                # For baselines, we no longer need query embeddings in the hook (ImageOnly uses image, Pruning uses norm).
                # hook.query_embeds = ... is bypassed.
                
                # Set hook state (for ImageOnly entropy temp)
                hook.temperature = temp

                # Register hook & forward
                handle = model.model.visual.register_forward_hook(hook)
                try:
                    _maybe_sync_cuda(device, args.profile_sync_cuda)
                    t_forward_start = time.perf_counter()
                    model_kwargs = {}
                    for k, v in batch.items():
                        if k.startswith("_"):
                            continue
                        if isinstance(v, torch.Tensor):
                            if v.dtype in (torch.float32, torch.float16):
                                model_kwargs[k] = v.to(
                                    device=device, dtype=torch.bfloat16
                                )
                            else:
                                model_kwargs[k] = v.to(device=device)
                        else:
                            model_kwargs[k] = v
                    outputs = model(**model_kwargs)
                    _maybe_sync_cuda(device, args.profile_sync_cuda)
                    forward_time = time.perf_counter() - t_forward_start
                finally:
                    handle.remove()

                ntp_loss = outputs.loss
                routing_loss = hook.routing_loss.to(ntp_loss.device)
                total_loss = ntp_loss + routing_loss

                _maybe_sync_cuda(device, args.profile_sync_cuda)
                t_backward_start = time.perf_counter()
                (total_loss / args.grad_accum).backward()
                _maybe_sync_cuda(device, args.profile_sync_cuda)
                backward_time = time.perf_counter() - t_backward_start

                step_loss = _reduce_mean(
                    float(total_loss.detach()), device, distributed, world_size
                )
                step_ntp = _reduce_mean(
                    float(ntp_loss.detach()), device, distributed, world_size
                )
                epoch_loss += step_loss
                epoch_ntp += step_ntp
                epoch_steps += 1

                opt_time = 0.0

                if (batch_idx + 1) % args.grad_accum == 0 or (batch_idx + 1) == len(
                    loader
                ):
                    _maybe_sync_cuda(device, args.profile_sync_cuda)
                    t_opt_start = time.perf_counter()
                    torch.nn.utils.clip_grad_norm_(
                        params_to_opt,
                        1.0,
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    _maybe_sync_cuda(device, args.profile_sync_cuda)
                    opt_time = time.perf_counter() - t_opt_start
                    global_step += 1

                    iter_time = time.perf_counter() - iter_start
                    timing_window["data"] += data_time
                    timing_window["forward"] += forward_time
                    timing_window["backward"] += backward_time
                    timing_window["opt"] += opt_time
                    timing_window["iter"] += iter_time
                    timing_window["count"] += 1

                    if global_step % args.log_every == 0 or global_step == 1:
                        avg_loss = epoch_loss / max(epoch_steps, 1)
                        avg_ntp = epoch_ntp / max(epoch_steps, 1)
                        s = hook.stats

                        expected_compute = _reduce_mean(
                            float(s.expected_compute), device, distributed, world_size
                        )
                        mean_skip = _reduce_mean(
                            float(s.mean_skip), device, distributed, world_size
                        )
                        mean_shallow = _reduce_mean(
                            float(s.mean_shallow), device, distributed, world_size
                        )
                        mean_deep = _reduce_mean(
                            float(s.mean_deep), device, distributed, world_size
                        )
                        budget_loss = _reduce_mean(
                            (
                                float(s.budget_loss.detach())
                                if s.budget_loss is not None
                                else 0.0
                            ),
                            device,
                            distributed,
                            world_size,
                        )

                        denom = max(timing_window["count"], 1)
                        data_ms = 1000.0 * timing_window["data"] / denom
                        forward_ms = 1000.0 * timing_window["forward"] / denom
                        backward_ms = 1000.0 * timing_window["backward"] / denom
                        opt_ms = 1000.0 * timing_window["opt"] / denom
                        iter_ms = 1000.0 * timing_window["iter"] / denom

                        entry = {
                            "epoch": epoch,
                            "global_step": global_step,
                            "total_loss": round(avg_loss, 6),
                            "ntp_loss": round(avg_ntp, 6),
                            "budget_loss": round(budget_loss, 6),
                            "expected_compute": round(expected_compute, 4),
                            "skip": round(mean_skip, 4),
                            "shallow": round(mean_shallow, 4),
                            "deep": round(mean_deep, 4),
                            "data_ms": round(data_ms, 3),
                            "forward_ms": round(forward_ms, 3),
                            "backward_ms": round(backward_ms, 3),
                            "opt_ms": round(opt_ms, 3),
                            "iter_ms": round(iter_ms, 3),
                            "temp": round(temp, 4),
                        }

                        if main_process:
                            train_log.append(entry)
                            log.info(
                                "step=%d  loss=%.4f  ntp=%.4f  compute=%.3f  "
                                "skip/sh/deep=%.2f/%.2f/%.2f  temp=%.3f",
                                global_step,
                                avg_loss,
                                avg_ntp,
                                expected_compute,
                                mean_skip,
                                mean_shallow,
                                mean_deep,
                                temp,
                            )
                            if global_step % profile_every == 0 or global_step == 1:
                                log.info(
                                    "timing(ms): data=%.2f forward=%.2f backward=%.2f opt=%.2f iter=%.2f (window=%d)",
                                    data_ms,
                                    forward_ms,
                                    backward_ms,
                                    opt_ms,
                                    iter_ms,
                                    timing_window["count"],
                                )
                        if pbar is not None:
                            pbar.set_postfix(
                                step=global_step,
                                loss=f"{avg_loss:.4f}",
                                ntp=f"{avg_ntp:.4f}",
                                comp=f"{expected_compute:.3f}",
                                deep=f"{mean_deep:.2f}",
                                itms=f"{iter_ms:.1f}",
                                temp=f"{temp:.2f}",
                            )

                        if global_step % profile_every == 0 or global_step == 1:
                            timing_window = {
                                "data": 0.0,
                                "forward": 0.0,
                                "backward": 0.0,
                                "opt": 0.0,
                                "iter": 0.0,
                                "count": 0,
                            }

                    curr_loss = epoch_loss / max(epoch_steps, 1)
                    if curr_loss < best_loss:
                        best_loss = curr_loss
                        if main_process:
                            _save_checkpoint(
                                router if router is not None else executor,
                                executor,
                                save_dir / "best.pt",
                                args,
                                global_step,
                                best_loss,
                            )

                if pbar is not None:
                    pbar.update(1)

                last_iter_end = time.perf_counter()

            if pbar is not None:
                pbar.close()

            avg = epoch_loss / max(epoch_steps, 1)
            if main_process:
                log.info("Epoch %d done - avg_loss=%.4f", epoch, avg)

            if args.save_every_epoch and main_process:
                _save_checkpoint(
                    router if router is not None else executor,
                    executor,
                    save_dir / f"epoch{epoch}.pt",
                    args,
                    global_step,
                    avg,
                )

        if main_process:
            _save_checkpoint(
                router if router is not None else executor, # fallback save logic for no router
                executor,
                save_dir / "last.pt",
                args,
                global_step,
                epoch_loss / max(epoch_steps, 1),
            )
            log_path = save_dir / "train_log.json"
            log_path.write_text(json.dumps(train_log, indent=2))
            log.info("Training complete. Checkpoints & log saved to %s", save_dir)
    finally:
        if distributed:
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
