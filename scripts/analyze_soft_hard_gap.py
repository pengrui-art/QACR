#!/usr/bin/env python3
"""Phase 1.4: analyze train-soft vs infer-hard gap and convergence behavior."""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageTextToText, AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qacr.routing import (
    DepthOnlyRouter,
    compute_regularization_loss,
    hard_routing_from_logits,
    linear_temperature,
    soft_routing_probs,
)
from qacr.vision import DepthMultiPathExecutor


QUERIES = [
    "请重点看图中目标主体，背景可节省计算。",
    "请优先处理可能影响答案的关键区域。",
    "保证基础语义准确同时尽量控制预算。",
    "对于不重要区域允许浅层或跳过。",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Model/Qwen35-08B")
    parser.add_argument("--image", default="outputs/demo_phase01.png")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--eval-interval", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-compute", type=float, default=0.8)
    parser.add_argument("--lambda-entropy", type=float, default=0.02)
    parser.add_argument("--budget", type=float, default=0.45)
    parser.add_argument("--temp-start", type=float, default=1.5)
    parser.add_argument("--temp-end", type=float, default=0.4)
    parser.add_argument("--router-hidden", type=int, default=96)
    parser.add_argument("--executor-hidden", type=int, default=128)
    parser.add_argument("--deep-layers", type=int, default=3)
    parser.add_argument("--coarse-grid", type=int, default=14)
    parser.add_argument("--convergence-ratio", type=float, default=0.7)
    return parser.parse_args()


def load_query_tokens(
    processor: AutoProcessor, model: AutoModelForImageTextToText, query: str
) -> torch.Tensor:
    tokenized = processor.tokenizer([query], return_tensors="pt", padding=True)
    input_ids = tokenized["input_ids"].to(next(model.parameters()).device)
    with torch.no_grad():
        query_tokens = model.get_input_embeddings()(input_ids)
    return query_tokens.detach().float().cpu()


def load_coarse_image_tokens(image_path: Path, grid: int) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    tfm = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor()])
    pixel = tfm(image).unsqueeze(0)
    pooled = F.adaptive_avg_pool2d(pixel, output_size=(grid, grid))
    return pooled.flatten(2).transpose(1, 2).contiguous()


def is_finite_grads(modules: list[torch.nn.Module]) -> bool:
    for module in modules:
        for p in module.parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                return False
    return True


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    image_path = Path(args.image)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image path not found: {image_path}")
    if args.eval_interval < 1:
        raise ValueError("eval-interval must be >= 1")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()

    image_tokens = load_coarse_image_tokens(image_path, grid=args.coarse_grid)
    query_cache = {q: load_query_tokens(processor, model, q) for q in QUERIES}
    eval_query = QUERIES[0]
    route_costs = torch.tensor([0.0, 0.35, 1.0], dtype=torch.float32)

    router = DepthOnlyRouter(
        query_dim=query_cache[eval_query].size(-1),
        image_dim=image_tokens.size(-1),
        hidden_dim=args.router_hidden,
    )
    executor = DepthMultiPathExecutor(
        token_dim=image_tokens.size(-1),
        hidden_dim=args.executor_hidden,
        deep_layers=args.deep_layers,
    )
    optimizer = torch.optim.AdamW(
        list(router.parameters()) + list(executor.parameters()),
        lr=args.lr,
    )

    soft_loss_history = []
    hard_loss_history = []
    gap_history = []
    train_total_loss_history = []
    finite_gradients_all_steps = True
    convergence_step = -1
    convergence_threshold = None

    print("===== Soft-Train vs Hard-Infer Gap Analysis (Task 1.4) =====")
    print(f"model: {model_path}")
    print(f"steps: {args.steps}")
    print(f"budget_target: {args.budget:.4f}")
    print("eval_step | soft_eval_loss | hard_eval_loss | hard-soft_gap | expected_compute")

    for step in range(args.steps):
        query = QUERIES[step % len(QUERIES)]
        query_tokens = query_cache[query]
        temp = linear_temperature(step, args.steps, args.temp_start, args.temp_end)

        router_out = router(query_tokens=query_tokens, image_tokens=image_tokens)
        soft_probs = soft_routing_probs(
            router_out.logits, temperature=temp, use_gumbel=True
        )
        routed_tokens, _ = executor(
            image_tokens=image_tokens, route_probs=soft_probs, mode="soft"
        )
        task_loss = F.mse_loss(routed_tokens, image_tokens)
        compute_loss, expected_compute = compute_regularization_loss(
            route_probs=soft_probs,
            route_costs=route_costs,
            budget_ratio=args.budget,
        )
        entropy = -(soft_probs * soft_probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
        train_total_loss = (
            task_loss
            + args.lambda_compute * compute_loss
            - args.lambda_entropy * entropy
        )

        optimizer.zero_grad(set_to_none=True)
        train_total_loss.backward()
        finite_gradients_all_steps = (
            finite_gradients_all_steps
            and is_finite_grads([router, executor])
            and math.isfinite(float(train_total_loss.detach()))
        )
        optimizer.step()

        train_total_loss_history.append(float(train_total_loss.detach()))
        if convergence_threshold is None:
            convergence_threshold = float(train_total_loss.detach()) * args.convergence_ratio
        if convergence_step < 0 and float(train_total_loss.detach()) <= convergence_threshold:
            convergence_step = step

        if (step + 1) % args.eval_interval == 0 or step == args.steps - 1:
            eval_query_tokens = query_cache[eval_query]
            eval_out = router(
                query_tokens=eval_query_tokens,
                image_tokens=image_tokens,
            )
            soft_eval_probs = torch.softmax(eval_out.logits, dim=-1)
            soft_eval_tokens, _ = executor(
                image_tokens=image_tokens, route_probs=soft_eval_probs, mode="soft"
            )
            soft_eval_loss = F.mse_loss(soft_eval_tokens, image_tokens)

            hard_idx = hard_routing_from_logits(eval_out.logits.detach())
            hard_one_hot = F.one_hot(hard_idx, num_classes=3).to(soft_eval_probs.dtype)
            hard_eval_tokens, _ = executor(
                image_tokens=image_tokens,
                route_probs=hard_one_hot,
                route_indices=hard_idx,
                mode="hard",
            )
            hard_eval_loss = F.mse_loss(hard_eval_tokens, image_tokens)
            gap = float((hard_eval_loss - soft_eval_loss).detach())

            soft_loss_history.append(float(soft_eval_loss.detach()))
            hard_loss_history.append(float(hard_eval_loss.detach()))
            gap_history.append(gap)
            print(
                f"{step + 1:>8} | {float(soft_eval_loss.detach()):.6f} | "
                f"{float(hard_eval_loss.detach()):.6f} | {gap:.6f} | "
                f"{float(expected_compute.detach()):.6f}"
            )

    final_eval_out = router(
        query_tokens=query_cache[eval_query],
        image_tokens=image_tokens,
    )
    final_soft_probs = torch.softmax(final_eval_out.logits, dim=-1).detach()
    final_hard_idx = hard_routing_from_logits(final_eval_out.logits.detach())
    final_hard_one_hot = F.one_hot(final_hard_idx, num_classes=3).to(final_soft_probs.dtype)
    _, hard_stats = executor(
        image_tokens=image_tokens,
        route_probs=final_hard_one_hot,
        route_indices=final_hard_idx,
        mode="hard",
    )

    mean_gap = statistics.mean(gap_history) if gap_history else float("nan")
    max_gap = max(gap_history) if gap_history else float("nan")
    final_gap = gap_history[-1] if gap_history else float("nan")
    train_loss_std = statistics.pstdev(train_total_loss_history)

    print(f"finite_gradients_all_steps: {finite_gradients_all_steps}")
    print(f"convergence_threshold: {convergence_threshold:.6f}")
    print(f"convergence_step: {convergence_step}")
    print(f"train_total_loss_std: {train_loss_std:.6f}")
    print(f"final_soft_eval_loss: {soft_loss_history[-1]:.6f}")
    print(f"final_hard_eval_loss: {hard_loss_history[-1]:.6f}")
    print(f"final_hard_minus_soft_gap: {final_gap:.6f}")
    print(f"mean_hard_minus_soft_gap: {mean_gap:.6f}")
    print(f"max_hard_minus_soft_gap: {max_gap:.6f}")
    print(f"final_soft_ratio_skip: {float(final_soft_probs[..., 0].mean()):.6f}")
    print(f"final_soft_ratio_shallow: {float(final_soft_probs[..., 1].mean()):.6f}")
    print(f"final_soft_ratio_deep: {float(final_soft_probs[..., 2].mean()):.6f}")
    print(f"final_hard_ratio_skip: {hard_stats['skip_ratio']:.6f}")
    print(f"final_hard_ratio_shallow: {hard_stats['shallow_ratio']:.6f}")
    print(f"final_hard_ratio_deep: {hard_stats['deep_ratio']:.6f}")
    print("status: train-soft vs infer-hard gap analysis succeeded")


if __name__ == "__main__":
    main()
