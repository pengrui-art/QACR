#!/usr/bin/env python3
"""Phase 1.2 baseline: image-only routing (without query conditioning)."""

from __future__ import annotations

import argparse
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
    ImageOnlyRouter,
    compute_regularization_loss,
    hard_routing_from_logits,
    linear_temperature,
    soft_routing_probs,
)
from qacr.vision import DepthMultiPathExecutor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Model/Qwen35-08B")
    parser.add_argument("--image", default="outputs/demo_phase01.png")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lambda-compute", type=float, default=0.7)
    parser.add_argument("--budget", type=float, default=0.45)
    parser.add_argument("--temp-start", type=float, default=1.4)
    parser.add_argument("--temp-end", type=float, default=0.35)
    parser.add_argument("--router-hidden", type=int, default=96)
    parser.add_argument("--executor-hidden", type=int, default=128)
    parser.add_argument("--deep-layers", type=int, default=3)
    parser.add_argument("--coarse-grid", type=int, default=14)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--benchmark-runs", type=int, default=20)
    return parser.parse_args()


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
    router: ImageOnlyRouter,
    executor: DepthMultiPathExecutor,
    image_tokens: torch.Tensor,
    warmup_runs: int,
    benchmark_runs: int,
) -> float:
    router.eval()
    executor.eval()
    with torch.no_grad():
        for _ in range(warmup_runs):
            out = router(image_tokens=image_tokens)
            route_probs = torch.softmax(out.logits, dim=-1)
            executor(image_tokens=image_tokens, route_probs=route_probs, mode="soft")
        sync_if_needed()

        start = time.perf_counter()
        for _ in range(benchmark_runs):
            out = router(image_tokens=image_tokens)
            route_probs = torch.softmax(out.logits, dim=-1)
            executor(image_tokens=image_tokens, route_probs=route_probs, mode="soft")
        sync_if_needed()

    elapsed = time.perf_counter() - start
    return elapsed * 1000.0 / max(benchmark_runs, 1)


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    image_path = Path(args.image)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image path not found: {image_path}")

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
    router = ImageOnlyRouter(
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
    route_costs = torch.tensor([0.0, 0.35, 1.0], dtype=torch.float32)

    last_soft_probs = None
    final_task_loss = None
    final_compute_loss = None
    final_total_loss = None
    final_expected_compute = None

    for step in range(args.steps):
        temp = linear_temperature(step, args.steps, args.temp_start, args.temp_end)
        router_out = router(image_tokens=image_tokens)
        soft_probs = soft_routing_probs(
            router_out.logits,
            temperature=temp,
            use_gumbel=True,
        )
        routed_tokens, _ = executor(
            image_tokens=image_tokens,
            route_probs=soft_probs,
            mode="soft",
        )

        task_loss = F.mse_loss(routed_tokens, image_tokens)
        compute_loss, expected_compute = compute_regularization_loss(
            route_probs=soft_probs,
            route_costs=route_costs,
            budget_ratio=args.budget,
        )
        total_loss = task_loss + args.lambda_compute * compute_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        last_soft_probs = soft_probs.detach()
        final_task_loss = task_loss.detach()
        final_compute_loss = compute_loss.detach()
        final_total_loss = total_loss.detach()
        final_expected_compute = expected_compute.detach()

    hard_idx = hard_routing_from_logits(router_out.logits.detach())
    hard_one_hot = F.one_hot(hard_idx, num_classes=3).to(last_soft_probs.dtype)
    _, hard_stats = executor(
        image_tokens=image_tokens,
        route_probs=hard_one_hot,
        route_indices=hard_idx,
        mode="hard",
    )
    soft_stats = {
        "skip": float(last_soft_probs[..., 0].mean()),
        "shallow": float(last_soft_probs[..., 1].mean()),
        "deep": float(last_soft_probs[..., 2].mean()),
    }

    avg_latency_ms = benchmark_latency_ms(
        router=router,
        executor=executor,
        image_tokens=image_tokens,
        warmup_runs=args.warmup_runs,
        benchmark_runs=args.benchmark_runs,
    )

    print("===== Image-only Routing Baseline (Task 1.2) =====")
    print(f"model: {model_path}")
    print(f"processor_loaded: {processor.__class__.__name__}")
    print(f"steps: {args.steps}")
    print(f"temperature_start_end: {args.temp_start} -> {args.temp_end}")
    print(f"budget_target: {args.budget:.4f}")
    print(f"image_token_shape: {tuple(image_tokens.shape)}")
    print(f"expected_compute: {float(final_expected_compute):.6f}")
    print(f"task_loss: {float(final_task_loss):.6f}")
    print(f"compute_loss: {float(final_compute_loss):.6f}")
    print(f"total_loss: {float(final_total_loss):.6f}")
    print(f"soft_ratio_skip: {soft_stats['skip']:.6f}")
    print(f"soft_ratio_shallow: {soft_stats['shallow']:.6f}")
    print(f"soft_ratio_deep: {soft_stats['deep']:.6f}")
    print(f"hard_ratio_skip: {hard_stats['skip_ratio']:.6f}")
    print(f"hard_ratio_shallow: {hard_stats['shallow_ratio']:.6f}")
    print(f"hard_ratio_deep: {hard_stats['deep_ratio']:.6f}")
    print(f"avg_forward_latency_ms: {avg_latency_ms:.6f}")
    print("status: image-only routing train-soft + infer-hard baseline succeeded")


if __name__ == "__main__":
    main()

