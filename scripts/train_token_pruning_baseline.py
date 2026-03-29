#!/usr/bin/env python3
"""Phase 2.2 baseline: heuristic token pruning (keep/drop) under compute budget."""

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

from qacr.vision import DepthMultiPathExecutor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Model/Qwen35-08B")
    parser.add_argument("--image", default="outputs/demo_phase01.png")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--keep-ratio", type=float, default=0.45)
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


def build_pruning_routes(image_tokens: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    if not (0.0 < keep_ratio <= 1.0):
        raise ValueError("keep-ratio must be in (0, 1]")
    batch_size, num_tokens, _ = image_tokens.shape
    keep_k = max(1, int(round(num_tokens * keep_ratio)))

    # Heuristic importance: larger L2-norm tokens are kept for deep computation.
    scores = torch.norm(image_tokens, p=2, dim=-1)  # [B, Ti]
    topk_idx = torch.topk(scores, k=keep_k, dim=-1, largest=True).indices

    routes = torch.zeros(batch_size, num_tokens, 3, dtype=image_tokens.dtype)
    routes[..., 0] = 1.0  # default: skip
    batch_index = torch.arange(batch_size).unsqueeze(1).expand_as(topk_idx)
    routes[batch_index, topk_idx, 0] = 0.0
    routes[batch_index, topk_idx, 2] = 1.0  # kept tokens: deep
    return routes


def sync_if_needed() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_latency_ms(
    executor: DepthMultiPathExecutor,
    image_tokens: torch.Tensor,
    route_probs: torch.Tensor,
    warmup_runs: int,
    benchmark_runs: int,
) -> float:
    executor.eval()
    with torch.no_grad():
        for _ in range(warmup_runs):
            executor(image_tokens=image_tokens, route_probs=route_probs, mode="hard")
        sync_if_needed()

        start = time.perf_counter()
        for _ in range(benchmark_runs):
            executor(image_tokens=image_tokens, route_probs=route_probs, mode="hard")
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
    route_probs = build_pruning_routes(image_tokens=image_tokens, keep_ratio=args.keep_ratio)
    executor = DepthMultiPathExecutor(
        token_dim=image_tokens.size(-1),
        hidden_dim=args.executor_hidden,
        deep_layers=args.deep_layers,
    )
    optimizer = torch.optim.AdamW(executor.parameters(), lr=args.lr)

    final_loss = None
    for _ in range(args.steps):
        routed_tokens, _ = executor(
            image_tokens=image_tokens, route_probs=route_probs, mode="hard"
        )
        loss = F.mse_loss(routed_tokens, image_tokens)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = loss.detach()

    avg_latency_ms = benchmark_latency_ms(
        executor=executor,
        image_tokens=image_tokens,
        route_probs=route_probs,
        warmup_runs=args.warmup_runs,
        benchmark_runs=args.benchmark_runs,
    )

    skip_ratio = float(route_probs[..., 0].mean())
    deep_ratio = float(route_probs[..., 2].mean())

    print("===== Token Pruning Baseline (Task 2.2) =====")
    print(f"model: {model_path}")
    print(f"processor_loaded: {processor.__class__.__name__}")
    print(f"steps: {args.steps}")
    print(f"keep_ratio_target: {args.keep_ratio:.6f}")
    print(f"image_token_shape: {tuple(image_tokens.shape)}")
    print(f"pruning_ratio_skip: {skip_ratio:.6f}")
    print(f"pruning_ratio_keep_deep: {deep_ratio:.6f}")
    print(f"expected_compute_ratio: {deep_ratio:.6f}")
    print(f"final_loss: {float(final_loss):.6f}")
    print(f"avg_forward_latency_ms: {avg_latency_ms:.6f}")
    print("status: heuristic token pruning baseline forward/backward succeeded")


if __name__ == "__main__":
    main()

