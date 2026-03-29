#!/usr/bin/env python3
"""Task 3.7: attention-level routing pilot for Phase 4 go/no-go."""

from __future__ import annotations

import argparse
import json
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
    hard_routing_from_logits,
    linear_temperature,
    soft_routing_probs,
)
from qacr.vision import DepthMultiPathExecutor


TRAIN_SPECS = [
    ("left_focus", "请优先关注图像左侧对象。", "left_focus"),
    ("right_focus", "请优先关注图像右侧对象。", "right_focus"),
    ("center_focus", "请优先关注图像中心区域。", "center_focus"),
    ("bottom_text", "请重点关注底部文字区域。", "bottom_text"),
    ("shape_region", "请忽略底部文字，只关注图形区域。", "shape_region"),
    ("left_right_dual", "请同时看左右两个主体。", "left_right_dual"),
]

CORNER_CASES = [
    ("left_easy", "请只关注左边蓝色区域，右边可以忽略。", "left_focus"),
    ("right_easy", "请只关注右边黄色区域，左边可以忽略。", "right_focus"),
    ("text_hard", "请重点识别底部文字，其他几何区域降算力。", "bottom_text"),
    ("ignore_text", "请忽略底部文字，只关注图形区域。", "shape_region"),
    ("dual_object", "左右两个主体都要看，中心和下方可以减少计算。", "left_right_dual"),
    ("center_conflict", "只看中心区域，不要关注左右边缘。", "center_focus"),
]

EVAL_SPECS = [
    ("left_eval", "回答左侧主体问题时，请把注意力集中到左边。", "left_focus"),
    ("text_eval", "文字阅读任务需要更深地处理底部文本。", "bottom_text"),
    ("center_eval", "空间定位问题只依赖中心区域。", "center_focus"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Model/Qwen35-08B")
    parser.add_argument("--image", default="outputs/demo_phase01.png")
    parser.add_argument("--coarse-grid", type=int, default=14)
    parser.add_argument("--steps", type=int, default=140)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--router-hidden", type=int, default=96)
    parser.add_argument("--temp-start", type=float, default=1.4)
    parser.add_argument("--temp-end", type=float, default=0.35)
    parser.add_argument("--deep-threshold", type=float, default=0.33)
    parser.add_argument("--skip-threshold", type=float, default=0.50)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--benchmark-runs", type=int, default=60)
    parser.add_argument(
        "--reference-corner-json",
        default="outputs/phase33_corner_case_summary.json",
    )
    parser.add_argument(
        "--summary-json",
        default="outputs/phase37_attention_routing_summary.json",
    )
    return parser.parse_args()


def load_query_tokens(
    processor: AutoProcessor, model: AutoModelForImageTextToText, query: str
) -> torch.Tensor:
    tokenized = processor.tokenizer([query], return_tensors="pt", padding=True)
    input_ids = tokenized["input_ids"].to(next(model.parameters()).device)
    with torch.no_grad():
        query_tokens = model.get_input_embeddings()(input_ids)
    return query_tokens.detach().float()


def load_coarse_image_tokens(image_path: Path, grid: int, device: torch.device) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    tfm = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor()])
    pixel = tfm(image).unsqueeze(0).to(device)
    pooled = F.adaptive_avg_pool2d(pixel, output_size=(grid, grid))
    return pooled.flatten(2).transpose(1, 2).contiguous()


def make_focus_mask(grid: int, mode: str, device: torch.device) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, grid, device=device),
        torch.linspace(0, 1, grid, device=device),
        indexing="ij",
    )
    if mode == "left_focus":
        mask = (xx < 0.45).float()
    elif mode == "right_focus":
        mask = (xx > 0.55).float()
    elif mode == "center_focus":
        dist = torch.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2)
        mask = (dist < 0.28).float()
    elif mode == "bottom_text":
        mask = ((yy > 0.62) & (xx > 0.05) & (xx < 0.6)).float()
    elif mode == "shape_region":
        mask = (yy < 0.62).float()
    elif mode == "left_right_dual":
        mask = ((xx < 0.30) | (xx > 0.70)).float()
    else:
        raise ValueError(f"Unknown mask mode: {mode}")
    if float(mask.mean()) < 0.05:
        mask[grid // 2, grid // 2] = 1.0
    return mask.reshape(1, grid * grid, 1)


def build_target_route_probs(mask: torch.Tensor) -> torch.Tensor:
    focus_probs = torch.tensor([0.05, 0.15, 0.80], dtype=mask.dtype, device=mask.device).view(1, 1, 3)
    nonfocus_probs = torch.tensor([0.45, 0.45, 0.10], dtype=mask.dtype, device=mask.device).view(1, 1, 3)
    return mask * focus_probs + (1.0 - mask) * nonfocus_probs


def sync_if_needed() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_latency_ms(fn, warmup_runs: int, benchmark_runs: int) -> float:
    for _ in range(warmup_runs):
        fn()
    sync_if_needed()
    start = time.perf_counter()
    for _ in range(benchmark_runs):
        fn()
    sync_if_needed()
    return (time.perf_counter() - start) * 1000.0 / max(benchmark_runs, 1)


def load_reference_corner(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def train_router(
    router: torch.nn.Module,
    train_queries: dict[str, torch.Tensor],
    image_tokens: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.nn.Module, bool]:
    optimizer = torch.optim.AdamW(router.parameters(), lr=args.lr)
    finite_gradients_all_steps = True
    for step in range(args.steps):
        name, _, mask_mode = TRAIN_SPECS[step % len(TRAIN_SPECS)]
        target = build_target_route_probs(make_focus_mask(args.coarse_grid, mask_mode, image_tokens.device))
        temp = linear_temperature(step, args.steps, args.temp_start, args.temp_end)
        out = router(query_tokens=train_queries[name], image_tokens=image_tokens)
        soft_probs = soft_routing_probs(out.logits, temperature=temp, use_gumbel=True)
        entropy = -(soft_probs * soft_probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
        loss = F.mse_loss(soft_probs, target) - 0.01 * entropy
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_ok = True
        for p in router.parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                grad_ok = False
                break
        finite_gradients_all_steps = finite_gradients_all_steps and grad_ok and math.isfinite(float(loss.detach()))
        optimizer.step()
    return router, finite_gradients_all_steps


def evaluate_router(
    router: torch.nn.Module,
    query_cache: dict[str, torch.Tensor],
    image_tokens: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, float]:
    case_results = []
    gaps = []
    eval_losses = []

    for name, query, mask_mode in CORNER_CASES:
        target_mask = make_focus_mask(args.coarse_grid, mask_mode, image_tokens.device)
        target = build_target_route_probs(target_mask)
        out = router(query_tokens=query_cache[name], image_tokens=image_tokens)
        soft_probs = soft_routing_probs(
            out.logits, temperature=args.temp_end, use_gumbel=False
        )
        hard_idx = hard_routing_from_logits(out.logits)
        hard_one_hot = F.one_hot(hard_idx, num_classes=3).to(soft_probs.dtype)

        soft_loss = F.mse_loss(soft_probs, target)
        hard_loss = F.mse_loss(hard_one_hot, target)
        gaps.append(float(hard_loss.detach() - soft_loss.detach()))

        pred_skip = soft_probs[0, :, 0]
        pred_deep = soft_probs[0, :, 2]
        key = target_mask[0, :, 0] > 0.5
        nonkey = ~key
        miss_rate = float((pred_deep[key] < args.deep_threshold).float().mean()) if key.any() else 0.0
        early_skip = float((pred_skip[key] > args.skip_threshold).float().mean()) if key.any() else 0.0
        key_deep = float(pred_deep[key].mean().detach()) if key.any() else 0.0
        nonkey_deep = float(pred_deep[nonkey].mean().detach()) if nonkey.any() else 0.0
        separation = key_deep - nonkey_deep
        score = float(miss_rate + early_skip + max(0.0, -separation))
        flagged = bool((miss_rate > 0.40) or (early_skip > 0.25) or (separation <= 0.0))
        case_results.append(
            {
                "name": name,
                "miss_rate_key_tokens": miss_rate,
                "early_skip_rate_key_tokens": early_skip,
                "separation_key_minus_nonkey": separation,
                "corner_score": score,
                "flagged_as_error": flagged,
            }
        )

    for name, _, mask_mode in EVAL_SPECS:
        target = build_target_route_probs(make_focus_mask(args.coarse_grid, mask_mode, image_tokens.device))
        out = router(query_tokens=query_cache[name], image_tokens=image_tokens)
        soft_probs = soft_routing_probs(
            out.logits, temperature=args.temp_end, use_gumbel=False
        )
        eval_losses.append(float(F.mse_loss(soft_probs, target).detach()))

    num_flagged = sum(int(x["flagged_as_error"]) for x in case_results)
    mean_miss = sum(x["miss_rate_key_tokens"] for x in case_results) / len(case_results)
    mean_score = sum(x["corner_score"] for x in case_results) / len(case_results)

    return {
        "num_cases": len(case_results),
        "num_flagged_errors": num_flagged,
        "mean_miss_rate_key_tokens": float(mean_miss),
        "mean_corner_score": float(mean_score),
        "mean_hard_minus_soft_gap": float(sum(gaps) / len(gaps)),
        "eval_proxy_route_loss": float(sum(eval_losses) / len(eval_losses)),
        "details": case_results,
    }


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    image_path = Path(args.image)
    summary_json = Path(args.summary_json)
    reference_corner_json = Path(args.reference_corner_json)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image path not found: {image_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    all_specs = TRAIN_SPECS + CORNER_CASES + EVAL_SPECS
    query_cache = {
        name: load_query_tokens(processor, model, query).to(device)
        for name, query, _ in all_specs
    }
    image_tokens = load_coarse_image_tokens(image_path=image_path, grid=args.coarse_grid, device=device)
    executor = DepthMultiPathExecutor(token_dim=image_tokens.size(-1)).to(device)

    depth_router = DepthOnlyRouter(
        query_dim=next(iter(query_cache.values())).size(-1),
        image_dim=image_tokens.size(-1),
        hidden_dim=args.router_hidden,
    ).to(device)
    attn_router = AttentionLevelRouter(
        query_dim=next(iter(query_cache.values())).size(-1),
        image_dim=image_tokens.size(-1),
        hidden_dim=args.router_hidden,
    ).to(device)

    depth_router, depth_grad_ok = train_router(depth_router, query_cache, image_tokens, args)
    attn_router, attn_grad_ok = train_router(attn_router, query_cache, image_tokens, args)

    depth_metrics = evaluate_router(depth_router, query_cache, image_tokens, args)
    attn_metrics = evaluate_router(attn_router, query_cache, image_tokens, args)
    reference_corner = load_reference_corner(reference_corner_json)

    latency_depth = benchmark_latency_ms(
        lambda: executor(
            image_tokens=image_tokens,
            route_probs=depth_router(
                query_tokens=query_cache[EVAL_SPECS[0][0]],
                image_tokens=image_tokens,
            ).route_probs,
            mode="soft",
        ),
        warmup_runs=args.warmup_runs,
        benchmark_runs=args.benchmark_runs,
    )
    latency_attn = benchmark_latency_ms(
        lambda: executor(
            image_tokens=image_tokens,
            route_probs=attn_router(
                query_tokens=query_cache[EVAL_SPECS[0][0]],
                image_tokens=image_tokens,
            ).route_probs,
            mode="soft",
        ),
        warmup_runs=args.warmup_runs,
        benchmark_runs=args.benchmark_runs,
    )

    if reference_corner is not None:
        reference_miss = sum(
            float(case["miss_rate_key_tokens"]) for case in reference_corner["cases"]
        ) / max(len(reference_corner["cases"]), 1)
        reference_flagged = int(reference_corner["num_flagged_errors"])
    else:
        reference_miss = depth_metrics["mean_miss_rate_key_tokens"]
        reference_flagged = depth_metrics["num_flagged_errors"]

    miss_reduction = 1.0 - (
        attn_metrics["mean_miss_rate_key_tokens"] / max(reference_miss, 1e-8)
    )
    pass_gate = bool(
        (
            attn_metrics["num_flagged_errors"] <= 2
            or miss_reduction >= 0.40
        )
        and attn_grad_ok
        and attn_metrics["mean_hard_minus_soft_gap"] <= 0.02
    )

    summary = {
        "task": "3.7_attention_level_routing_gate",
        "model": str(model_path),
        "image": str(image_path),
        "device": str(device),
        "depth_only": {
            **depth_metrics,
            "finite_gradients_all_steps": depth_grad_ok,
            "latency_ms": float(latency_depth),
        },
        "attention_level": {
            **attn_metrics,
            "finite_gradients_all_steps": attn_grad_ok,
            "latency_ms": float(latency_attn),
        },
        "reference_corner_case": {
            "path": str(reference_corner_json),
            "num_flagged_errors": int(reference_flagged),
            "mean_miss_rate_key_tokens": float(reference_miss),
        },
        "miss_rate_reduction": float(miss_reduction),
        "pass_gate": pass_gate,
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("===== Attention-level Routing Pilot (Task 3.7) =====")
    print("method | eval_proxy_route_loss | num_flagged_errors | mean_miss_rate | mean_gap | latency_ms")
    print(
        "DepthOnly | "
        f"{depth_metrics['eval_proxy_route_loss']:.6f} | "
        f"{depth_metrics['num_flagged_errors']} | "
        f"{depth_metrics['mean_miss_rate_key_tokens']:.6f} | "
        f"{depth_metrics['mean_hard_minus_soft_gap']:.6f} | "
        f"{latency_depth:.6f}"
    )
    print(
        "AttentionLevel | "
        f"{attn_metrics['eval_proxy_route_loss']:.6f} | "
        f"{attn_metrics['num_flagged_errors']} | "
        f"{attn_metrics['mean_miss_rate_key_tokens']:.6f} | "
        f"{attn_metrics['mean_hard_minus_soft_gap']:.6f} | "
        f"{latency_attn:.6f}"
    )
    print(f"miss_rate_reduction: {miss_reduction:.6f}")
    print(f"depth_gradients_finite: {depth_grad_ok}")
    print(f"attention_gradients_finite: {attn_grad_ok}")
    print(f"pass_gate: {pass_gate}")
    print(f"summary_json: {summary_json}")


if __name__ == "__main__":
    main()
