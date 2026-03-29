#!/usr/bin/env python3
"""Phase 3.5: hardware-level latency and memory profiling for routing."""

from __future__ import annotations

import argparse
import json
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

from qacr.routing import DepthOnlyRouter
from qacr.vision import DepthMultiPathExecutor


PROFILE_QUERIES = [
    "请关注左侧图形。",
    "请关注右侧图形。",
    "请关注底部文字。",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Model/Qwen35-08B")
    parser.add_argument("--image", default="outputs/demo_phase01.png")
    parser.add_argument("--coarse-grid", type=int, default=14)
    parser.add_argument("--router-hidden", type=int, default=96)
    parser.add_argument("--executor-hidden", type=int, default=128)
    parser.add_argument("--deep-layers", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=10)
    parser.add_argument("--benchmark-runs", type=int, default=120)
    parser.add_argument("--out-json", default="outputs/phase35_hardware_profile.json")
    return parser.parse_args()


def load_query_tokens(
    processor: AutoProcessor, model: AutoModelForImageTextToText, query: str
) -> torch.Tensor:
    tokenized = processor.tokenizer([query], return_tensors="pt", padding=True)
    input_ids = tokenized["input_ids"].to(next(model.parameters()).device)
    with torch.no_grad():
        query_tokens = model.get_input_embeddings()(input_ids)
    return query_tokens.detach().float()


def load_coarse_image_tokens(image_path: Path, grid: int) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    tfm = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor()])
    pixel = tfm(image).unsqueeze(0)
    pooled = F.adaptive_avg_pool2d(pixel, output_size=(grid, grid))
    return pooled.flatten(2).transpose(1, 2).contiguous()


def sync_if_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def make_route_from_ratios(num_tokens: int, skip: float, shallow: float) -> torch.Tensor:
    deep = max(0.0, 1.0 - skip - shallow)
    n_skip = int(round(num_tokens * skip))
    n_shallow = int(round(num_tokens * shallow))
    n_deep = max(0, num_tokens - n_skip - n_shallow)
    route = torch.zeros(num_tokens, dtype=torch.long)
    route[:n_skip] = 0
    route[n_skip : n_skip + n_shallow] = 1
    route[n_skip + n_shallow : n_skip + n_shallow + n_deep] = 2
    return route


def dense_forward(
    executor: DepthMultiPathExecutor,
    image_tokens: torch.Tensor,
    route_idx: torch.Tensor,
) -> torch.Tensor:
    one_hot = F.one_hot(route_idx, num_classes=3).to(image_tokens.dtype).unsqueeze(0)
    out, _ = executor(
        image_tokens=image_tokens,
        route_probs=one_hot,
        route_indices=route_idx.unsqueeze(0),
        mode="hard",
    )
    return out


def conditional_forward(
    executor: DepthMultiPathExecutor,
    image_tokens: torch.Tensor,
    route_idx: torch.Tensor,
) -> torch.Tensor:
    one_hot = F.one_hot(route_idx, num_classes=3).to(image_tokens.dtype).unsqueeze(0)
    out, _ = executor(
        image_tokens=image_tokens,
        route_probs=one_hot,
        route_indices=route_idx.unsqueeze(0),
        mode="hard_conditional",
    )
    return out


def benchmark_fn(
    fn,
    warmup_runs: int,
    benchmark_runs: int,
) -> float:
    for _ in range(warmup_runs):
        fn()
    sync_if_cuda()
    start = time.perf_counter()
    for _ in range(benchmark_runs):
        fn()
    sync_if_cuda()
    elapsed = time.perf_counter() - start
    return elapsed * 1000.0 / max(benchmark_runs, 1)


def profile_peak_memory(fn) -> float:
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.reset_peak_memory_stats()
    fn()
    sync_if_cuda()
    return float(torch.cuda.max_memory_allocated() / (1024**2))


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    image_path = Path(args.image)
    out_json = Path(args.out_json)
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

    profile_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tokens = load_coarse_image_tokens(image_path, grid=args.coarse_grid).to(profile_device)
    query_tokens = [
        load_query_tokens(processor, model, q).to(profile_device) for q in PROFILE_QUERIES
    ]
    router = DepthOnlyRouter(
        query_dim=query_tokens[0].size(-1),
        image_dim=image_tokens.size(-1),
        hidden_dim=args.router_hidden,
    ).to(profile_device)
    executor = DepthMultiPathExecutor(
        token_dim=image_tokens.size(-1),
        hidden_dim=args.executor_hidden,
        deep_layers=args.deep_layers,
    ).to(profile_device)

    num_tokens = image_tokens.size(1)
    routes = {
        "all_deep": make_route_from_ratios(num_tokens, skip=0.0, shallow=0.0).to(profile_device),
        "balanced": make_route_from_ratios(num_tokens, skip=0.33, shallow=0.33).to(profile_device),
        "mostly_skip": make_route_from_ratios(num_tokens, skip=0.75, shallow=0.20).to(profile_device),
    }
    # Add a query-driven hard route profile.
    router_out = router(query_tokens=query_tokens[0], image_tokens=image_tokens)
    routes["query_hard"] = router_out.hard_routes[0].detach().to(profile_device)

    results = {}
    for name, route_idx in routes.items():
        route_idx = route_idx.long()
        dense_fn = lambda: dense_forward(executor, image_tokens, route_idx)
        conditional_fn = lambda: conditional_forward(executor, image_tokens, route_idx)

        dense_ms = benchmark_fn(
            dense_fn, warmup_runs=args.warmup_runs, benchmark_runs=args.benchmark_runs
        )
        conditional_ms = benchmark_fn(
            conditional_fn,
            warmup_runs=args.warmup_runs,
            benchmark_runs=args.benchmark_runs,
        )
        dense_mem = profile_peak_memory(dense_fn)
        conditional_mem = profile_peak_memory(conditional_fn)

        one_hot = F.one_hot(route_idx, num_classes=3).float()
        latency_delta_ms = conditional_ms - dense_ms
        latency_ratio = conditional_ms / max(dense_ms, 1e-8)
        deep_ratio = float(one_hot[:, 2].mean())
        if deep_ratio < 0.5 and conditional_ms < dense_ms:
            bottleneck = "conditional_execution_shows_real_speedup"
        elif deep_ratio < 0.5:
            bottleneck = "index_dispatch_and_kernel_launch_overhead_dominate"
        else:
            bottleneck = "deep_path_dominates_compute_so_sparse_gain_is_limited"
        results[name] = {
            "skip_ratio": float(one_hot[:, 0].mean()),
            "shallow_ratio": float(one_hot[:, 1].mean()),
            "deep_ratio": deep_ratio,
            "dense_latency_ms": dense_ms,
            "conditional_latency_ms": conditional_ms,
            "latency_delta_ms": latency_delta_ms,
            "conditional_over_dense_ratio": latency_ratio,
            "dense_peak_mem_mb": dense_mem,
            "conditional_peak_mem_mb": conditional_mem,
            "memory_delta_mb": conditional_mem - dense_mem,
            "bottleneck_interpretation": bottleneck,
        }

    flash_or_fla_installed = True
    try:
        import fla  # type: ignore  # noqa: F401
    except Exception:
        flash_or_fla_installed = False

    summary = {
        "model": str(model_path),
        "image": str(image_path),
        "device": str(profile_device),
        "dtype": str(dtype),
        "batch_size": 1,
        "coarse_grid": args.coarse_grid,
        "num_tokens": int(num_tokens),
        "benchmark_runs": args.benchmark_runs,
        "warmup_runs": args.warmup_runs,
        "flash_or_fla_installed": flash_or_fla_installed,
        "profiles": results,
        "note": (
            "dense_latency基于旧式三分支全算再融合；"
            "conditional_latency基于当前执行器的真实 hard conditional execution。"
        ),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("===== Hardware Latency & Memory Profiling (Task 3.5) =====")
    print(f"device: {summary['device']}")
    print(f"dtype: {summary['dtype']}")
    print(f"num_tokens: {summary['num_tokens']}")
    print(f"flash_or_fla_installed: {flash_or_fla_installed}")
    print(
        "profile | skip | shallow | deep | dense_ms | conditional_ms | "
        "ratio | dense_mem_mb | conditional_mem_mb | bottleneck"
    )
    for name, p in results.items():
        print(
            f"{name} | {p['skip_ratio']:.6f} | {p['shallow_ratio']:.6f} | {p['deep_ratio']:.6f} | "
            f"{p['dense_latency_ms']:.6f} | {p['conditional_latency_ms']:.6f} | "
            f"{p['conditional_over_dense_ratio']:.6f} | {p['dense_peak_mem_mb']:.3f} | "
            f"{p['conditional_peak_mem_mb']:.3f} | {p['bottleneck_interpretation']}"
        )
    print(f"summary_json: {out_json}")
    print("status: hardware profiling succeeded")


if __name__ == "__main__":
    main()
