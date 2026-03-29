#!/usr/bin/env python3
"""Phase 1.3: Query-adaptive routing with multi-budget sweep."""

from __future__ import annotations

import argparse
import math
import sys
import time
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
    AttentionLevelRouter,
    DepthOnlyRouter,
    compute_regularization_loss,
    hard_routing_from_logits,
    linear_temperature,
    soft_routing_probs,
)
from qacr.vision import DepthMultiPathExecutor


QUERIES = [
    "请优先给文字区域更深计算，背景尽量浅层。",
    "只需粗略描述全图时，尽量控制计算预算。",
    "重点识别显著目标边缘与语义关键区域。",
    "请平衡答案质量与延迟，避免全部走deep。",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Model/Qwen35-08B")
    parser.add_argument("--image", default="outputs/demo_phase01.png")
    parser.add_argument("--budgets", default="0.35,0.45,0.60")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-compute", type=float, default=0.8)
    parser.add_argument("--lambda-entropy", type=float, default=0.02)
    parser.add_argument("--temp-start", type=float, default=1.5)
    parser.add_argument("--temp-end", type=float, default=0.4)
    parser.add_argument("--router-hidden", type=int, default=96)
    parser.add_argument("--executor-hidden", type=int, default=128)
    parser.add_argument("--deep-layers", type=int, default=3)
    parser.add_argument("--coarse-grid", type=int, default=14)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--benchmark-runs", type=int, default=20)
    parser.add_argument(
        "--router-type", choices=["depth", "attention"], default="depth"
    )
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


def sync_if_needed() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_latency_ms(
    router: torch.nn.Module,
    executor: DepthMultiPathExecutor,
    query_tokens: torch.Tensor,
    image_tokens: torch.Tensor,
    warmup_runs: int,
    benchmark_runs: int,
) -> float:
    router.eval()
    executor.eval()
    with torch.no_grad():
        for _ in range(warmup_runs):
            out = router(query_tokens=query_tokens, image_tokens=image_tokens)
            probs = torch.softmax(out.logits, dim=-1)
            executor(image_tokens=image_tokens, route_probs=probs, mode="soft")
        sync_if_needed()

        start = time.perf_counter()
        for _ in range(benchmark_runs):
            out = router(query_tokens=query_tokens, image_tokens=image_tokens)
            probs = torch.softmax(out.logits, dim=-1)
            executor(image_tokens=image_tokens, route_probs=probs, mode="soft")
        sync_if_needed()

    elapsed = time.perf_counter() - start
    return elapsed * 1000.0 / max(benchmark_runs, 1)


def parse_budgets(s: str) -> list[float]:
    budgets = [float(x.strip()) for x in s.split(",") if x.strip()]
    if not budgets:
        raise ValueError("At least one budget is required")
    for b in budgets:
        if not (0.0 <= b <= 1.0):
            raise ValueError(f"budget must be in [0, 1], got {b}")
    return budgets


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    image_path = Path(args.image)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image path not found: {image_path}")

    budgets = parse_budgets(args.budgets)

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

    print("===== Query-Adaptive Budget Sweep (Task 1.3) =====")
    print(f"model: {model_path}")
    print(f"image_token_shape: {tuple(image_tokens.shape)}")
    print(f"budgets: {budgets}")
    print(
        "budget | expected_compute | task_loss | total_loss | "
        "soft_deep | hard_deep | latency_ms"
    )

    for budget in budgets:
        if args.router_type == "attention":
            router = AttentionLevelRouter(
                query_dim=query_cache[eval_query].size(-1),
                image_dim=image_tokens.size(-1),
                hidden_dim=args.router_hidden,
            )
        else:
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

        finite_gradients_all_steps = True
        final_expected_compute = None
        final_task_loss = None
        final_total_loss = None
        final_soft_probs = None
        final_logits = None

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
                budget_ratio=budget,
            )
            entropy = (
                -(soft_probs * soft_probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
            )
            total_loss = (
                task_loss
                + args.lambda_compute * compute_loss
                - args.lambda_entropy * entropy
            )

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            grad_ok = True
            for module in (router, executor):
                for p in module.parameters():
                    if p.grad is None:
                        continue
                    if not torch.isfinite(p.grad).all():
                        grad_ok = False
                        break
                if not grad_ok:
                    break
            finite_gradients_all_steps = (
                finite_gradients_all_steps
                and grad_ok
                and math.isfinite(float(total_loss.detach()))
            )
            optimizer.step()

            final_expected_compute = expected_compute.detach()
            final_task_loss = task_loss.detach()
            final_total_loss = total_loss.detach()
            final_soft_probs = soft_probs.detach()
            final_logits = router_out.logits.detach()

        hard_idx = hard_routing_from_logits(final_logits)
        hard_one_hot = F.one_hot(hard_idx, num_classes=3).to(final_soft_probs.dtype)
        _, hard_stats = executor(
            image_tokens=image_tokens,
            route_probs=hard_one_hot,
            route_indices=hard_idx,
            mode="hard",
        )
        soft_deep = float(final_soft_probs[..., 2].mean())
        hard_deep = float(hard_stats["deep_ratio"])

        latency_ms = benchmark_latency_ms(
            router=router,
            executor=executor,
            query_tokens=query_cache[eval_query],
            image_tokens=image_tokens,
            warmup_runs=args.warmup_runs,
            benchmark_runs=args.benchmark_runs,
        )

        print(
            f"{budget:.2f} | {float(final_expected_compute):.6f} | "
            f"{float(final_task_loss):.6f} | {float(final_total_loss):.6f} | "
            f"{soft_deep:.6f} | "
            f"{hard_deep:.6f} | {latency_ms:.6f}"
        )
        print(
            f"finite_gradients_all_steps_budget_{budget:.2f}: {finite_gradients_all_steps}"
        )

    print("status: query-adaptive multi-budget sweep succeeded")


if __name__ == "__main__":
    main()
