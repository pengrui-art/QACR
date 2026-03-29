#!/usr/bin/env python3
"""Task 3.6: high-resolution re-encoding pilot for Phase 4 go/no-go."""

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
    DepthOnlyRouter,
    compute_regularization_loss,
    linear_temperature,
    soft_routing_probs,
)
from qacr.vision import DepthMultiPathExecutor, HighResReEncoder


OCR_TRAIN_SPECS = [
    ("text_read", "请重点识别底部文字与细粒度笔画。", "bottom_text"),
    ("doc_word", "请读取页面中唯一明显的词。", "bottom_text"),
    ("fine_strokes", "请优先保留细粒度文本细节。", "bottom_text"),
    ("shape_global", "如果只需看图形，请优先保持整体上下文。", "shape_region"),
]

OCR_EVAL_SPECS = [
    ("textvqa_proxy", "请读取底部文字内容。", "bottom_text"),
    ("docvqa_proxy", "文档中最关键的词是什么？", "bottom_text"),
    ("ocr_detail_proxy", "请保留底部文字的细粒度局部细节。", "bottom_text"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Model/Qwen35-08B")
    parser.add_argument("--image", default="outputs/demo_phase01.png")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--budget", type=float, default=0.45)
    parser.add_argument("--lambda-compute", type=float, default=0.8)
    parser.add_argument("--lambda-entropy", type=float, default=0.02)
    parser.add_argument("--router-hidden", type=int, default=96)
    parser.add_argument("--executor-hidden", type=int, default=128)
    parser.add_argument("--deep-layers", type=int, default=3)
    parser.add_argument("--coarse-grid", type=int, default=14)
    parser.add_argument("--highres-grid", type=int, default=28)
    parser.add_argument("--low-grid", type=int, default=9)
    parser.add_argument("--select-ratio", type=float, default=0.15)
    parser.add_argument("--extra-compute-factor", type=float, default=0.4)
    parser.add_argument("--focus-weight", type=float, default=5.0)
    parser.add_argument("--temp-start", type=float, default=1.4)
    parser.add_argument("--temp-end", type=float, default=0.45)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--benchmark-runs", type=int, default=20)
    parser.add_argument(
        "--summary-json",
        default="outputs/phase36_highres_reencoding_summary.json",
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


def load_image_tokens(image_path: Path, grid: int, device: torch.device) -> torch.Tensor:
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
    if mode == "bottom_text":
        mask = ((yy > 0.62) & (xx > 0.05) & (xx < 0.6)).float()
    elif mode == "shape_region":
        mask = (yy < 0.62).float()
    else:
        raise ValueError(f"Unknown mask mode: {mode}")
    if float(mask.mean()) < 0.05:
        mask[grid - 2, grid // 3] = 1.0
    return mask.reshape(1, grid * grid, 1)


def build_detail_features(
    highres_tokens: torch.Tensor, coarse_grid: int, highres_grid: int
) -> torch.Tensor:
    patch_scale = highres_grid // coarse_grid
    if highres_grid != coarse_grid * patch_scale:
        raise ValueError("highres grid must be an integer multiple of coarse grid")
    batch_size, _, token_dim = highres_tokens.shape
    local = highres_tokens.view(
        batch_size,
        coarse_grid,
        patch_scale,
        coarse_grid,
        patch_scale,
        token_dim,
    )
    local = local.permute(0, 1, 3, 2, 4, 5).contiguous()
    local = local.view(batch_size, coarse_grid * coarse_grid, patch_scale * patch_scale, token_dim)
    local_std = local.std(dim=2, unbiased=False)
    local_range = local.max(dim=2).values - local.min(dim=2).values
    return local_std + 0.5 * local_range


def build_target_tokens(
    coarse_tokens: torch.Tensor,
    highres_tokens: torch.Tensor,
    mask: torch.Tensor,
    coarse_grid: int,
    highres_grid: int,
) -> torch.Tensor:
    detail = build_detail_features(highres_tokens, coarse_grid=coarse_grid, highres_grid=highres_grid)
    return coarse_tokens * (1.0 - mask) + (0.20 * coarse_tokens + 0.80 * detail) * mask


def weighted_focus_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, focus_weight: float
) -> torch.Tensor:
    weights = 1.0 + focus_weight * mask
    return ((pred - target) ** 2 * weights).mean()


def detail_proxy_accuracy(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    focus = mask[0, :, 0] > 0.5
    if int(focus.sum()) <= 1:
        return 1.0
    pred_score = pred[0].norm(dim=-1)[focus]
    target_score = target[0].norm(dim=-1)[focus]
    k = max(1, int(math.ceil(pred_score.numel() * 0.35)))
    pred_top = torch.topk(pred_score, k=k).indices
    target_top = torch.topk(target_score, k=k).indices
    overlap = len(set(pred_top.tolist()) & set(target_top.tolist()))
    return float(overlap / k)


def upsample_tokens(tokens: torch.Tensor, src_grid: int, dst_grid: int) -> torch.Tensor:
    token_dim = tokens.size(-1)
    feat = tokens.transpose(1, 2).reshape(tokens.size(0), token_dim, src_grid, src_grid)
    up = F.interpolate(feat, size=(dst_grid, dst_grid), mode="bilinear", align_corners=False)
    return up.flatten(2).transpose(1, 2).contiguous()


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


def train_depth_only(
    query_cache: dict[str, torch.Tensor],
    coarse_tokens: torch.Tensor,
    highres_tokens: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    router = DepthOnlyRouter(
        query_dim=next(iter(query_cache.values())).size(-1),
        image_dim=coarse_tokens.size(-1),
        hidden_dim=args.router_hidden,
    ).to(device)
    executor = DepthMultiPathExecutor(
        token_dim=coarse_tokens.size(-1),
        hidden_dim=args.executor_hidden,
        deep_layers=args.deep_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(router.parameters()) + list(executor.parameters()),
        lr=args.lr,
    )
    route_costs = torch.tensor([0.0, 0.35, 1.0], dtype=coarse_tokens.dtype, device=device)

    final_expected_compute = torch.tensor(0.0, device=device)
    finite_gradients_all_steps = True

    for step in range(args.steps):
        name, _, mask_mode = OCR_TRAIN_SPECS[step % len(OCR_TRAIN_SPECS)]
        mask = make_focus_mask(args.coarse_grid, mask_mode, device=device)
        target = build_target_tokens(
            coarse_tokens=coarse_tokens,
            highres_tokens=highres_tokens,
            mask=mask,
            coarse_grid=args.coarse_grid,
            highres_grid=args.highres_grid,
        )
        temp = linear_temperature(step, args.steps, args.temp_start, args.temp_end)

        out = router(query_tokens=query_cache[name], image_tokens=coarse_tokens)
        soft_probs = soft_routing_probs(out.logits, temperature=temp, use_gumbel=True)
        routed_tokens, _ = executor(
            image_tokens=coarse_tokens, route_probs=soft_probs, mode="soft"
        )
        task_loss = weighted_focus_loss(
            pred=routed_tokens,
            target=target,
            mask=mask,
            focus_weight=args.focus_weight,
        )
        compute_loss, expected_compute = compute_regularization_loss(
            route_probs=soft_probs, route_costs=route_costs, budget_ratio=args.budget
        )
        entropy = -(soft_probs * soft_probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
        total_loss = task_loss + args.lambda_compute * compute_loss - args.lambda_entropy * entropy

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
        finite_gradients_all_steps = finite_gradients_all_steps and grad_ok
        optimizer.step()
        final_expected_compute = expected_compute.detach()

    losses = []
    accuracies = []
    for name, _, mask_mode in OCR_EVAL_SPECS:
        mask = make_focus_mask(args.coarse_grid, mask_mode, device=device)
        target = build_target_tokens(
            coarse_tokens=coarse_tokens,
            highres_tokens=highres_tokens,
            mask=mask,
            coarse_grid=args.coarse_grid,
            highres_grid=args.highres_grid,
        )
        out = router(query_tokens=query_cache[name], image_tokens=coarse_tokens)
        routed_tokens, _ = executor(
            image_tokens=coarse_tokens, route_probs=out.route_probs, mode="soft"
        )
        losses.append(
            float(
                weighted_focus_loss(
                    pred=routed_tokens,
                    target=target,
                    mask=mask,
                    focus_weight=args.focus_weight,
                ).detach()
            )
        )
        accuracies.append(detail_proxy_accuracy(routed_tokens.detach(), target.detach(), mask.detach()))

    latency_ms = benchmark_latency_ms(
        lambda: executor(
            image_tokens=coarse_tokens,
            route_probs=router(
                query_tokens=query_cache[OCR_EVAL_SPECS[0][0]],
                image_tokens=coarse_tokens,
            ).route_probs,
            mode="soft",
        ),
        warmup_runs=args.warmup_runs,
        benchmark_runs=args.benchmark_runs,
    )
    return {
        "proxy_task_loss": float(sum(losses) / len(losses)),
        "proxy_accuracy": float(sum(accuracies) / len(accuracies)),
        "expected_compute": float(final_expected_compute),
        "latency_ms": float(latency_ms),
        "finite_gradients_all_steps": bool(finite_gradients_all_steps),
    }


def train_highres(
    query_cache: dict[str, torch.Tensor],
    coarse_tokens: torch.Tensor,
    highres_tokens: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    router = DepthOnlyRouter(
        query_dim=next(iter(query_cache.values())).size(-1),
        image_dim=coarse_tokens.size(-1),
        hidden_dim=args.router_hidden,
    ).to(device)
    executor = DepthMultiPathExecutor(
        token_dim=coarse_tokens.size(-1),
        hidden_dim=args.executor_hidden,
        deep_layers=args.deep_layers,
    ).to(device)
    reencoder = HighResReEncoder(
        token_dim=coarse_tokens.size(-1),
        hidden_dim=args.router_hidden,
        patch_scale=args.highres_grid // args.coarse_grid,
        extra_compute_factor=args.extra_compute_factor,
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(router.parameters()) + list(executor.parameters()) + list(reencoder.parameters()),
        lr=args.lr,
    )
    route_costs = torch.tensor([0.0, 0.35, 1.0], dtype=coarse_tokens.dtype, device=device)

    final_expected_compute = torch.tensor(0.0, device=device)
    final_extra_compute = 0.0
    finite_gradients_all_steps = True

    for step in range(args.steps):
        name, _, mask_mode = OCR_TRAIN_SPECS[step % len(OCR_TRAIN_SPECS)]
        mask = make_focus_mask(args.coarse_grid, mask_mode, device=device)
        target = build_target_tokens(
            coarse_tokens=coarse_tokens,
            highres_tokens=highres_tokens,
            mask=mask,
            coarse_grid=args.coarse_grid,
            highres_grid=args.highres_grid,
        )
        temp = linear_temperature(step, args.steps, args.temp_start, args.temp_end)

        out = router(query_tokens=query_cache[name], image_tokens=coarse_tokens)
        soft_probs = soft_routing_probs(out.logits, temperature=temp, use_gumbel=True)
        routed_tokens, _ = executor(
            image_tokens=coarse_tokens, route_probs=soft_probs, mode="soft"
        )
        refined_tokens, reencode_stats = reencoder(
            base_tokens=routed_tokens,
            highres_tokens=highres_tokens,
            selection_scores=soft_probs[..., 2],
            selected_ratio=args.select_ratio,
        )
        task_loss = weighted_focus_loss(
            pred=refined_tokens,
            target=target,
            mask=mask,
            focus_weight=args.focus_weight,
        )
        compute_loss, expected_compute = compute_regularization_loss(
            route_probs=soft_probs, route_costs=route_costs, budget_ratio=args.budget
        )
        entropy = -(soft_probs * soft_probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
        total_loss = task_loss + args.lambda_compute * compute_loss - args.lambda_entropy * entropy

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_ok = True
        for module in (router, executor, reencoder):
            for p in module.parameters():
                if p.grad is None:
                    continue
                if not torch.isfinite(p.grad).all():
                    grad_ok = False
                    break
            if not grad_ok:
                break
        finite_gradients_all_steps = finite_gradients_all_steps and grad_ok
        optimizer.step()
        final_expected_compute = expected_compute.detach()
        final_extra_compute = reencode_stats["extra_compute_ratio"]

    losses = []
    accuracies = []
    for name, _, mask_mode in OCR_EVAL_SPECS:
        mask = make_focus_mask(args.coarse_grid, mask_mode, device=device)
        target = build_target_tokens(
            coarse_tokens=coarse_tokens,
            highres_tokens=highres_tokens,
            mask=mask,
            coarse_grid=args.coarse_grid,
            highres_grid=args.highres_grid,
        )
        out = router(query_tokens=query_cache[name], image_tokens=coarse_tokens)
        routed_tokens, _ = executor(
            image_tokens=coarse_tokens, route_probs=out.route_probs, mode="soft"
        )
        refined_tokens, reencode_stats = reencoder(
            base_tokens=routed_tokens,
            highres_tokens=highres_tokens,
            selection_scores=out.route_probs[..., 2],
            selected_ratio=args.select_ratio,
        )
        losses.append(
            float(
                weighted_focus_loss(
                    pred=refined_tokens,
                    target=target,
                    mask=mask,
                    focus_weight=args.focus_weight,
                ).detach()
            )
        )
        accuracies.append(detail_proxy_accuracy(refined_tokens.detach(), target.detach(), mask.detach()))
        final_extra_compute = reencode_stats["extra_compute_ratio"]

    latency_ms = benchmark_latency_ms(
        lambda: reencoder(
            base_tokens=executor(
                image_tokens=coarse_tokens,
                route_probs=router(
                    query_tokens=query_cache[OCR_EVAL_SPECS[0][0]],
                    image_tokens=coarse_tokens,
                ).route_probs,
                mode="soft",
            )[0],
            highres_tokens=highres_tokens,
            selection_scores=router(
                query_tokens=query_cache[OCR_EVAL_SPECS[0][0]],
                image_tokens=coarse_tokens,
            ).route_probs[..., 2],
            selected_ratio=args.select_ratio,
        ),
        warmup_runs=args.warmup_runs,
        benchmark_runs=args.benchmark_runs,
    )
    return {
        "proxy_task_loss": float(sum(losses) / len(losses)),
        "proxy_accuracy": float(sum(accuracies) / len(accuracies)),
        "expected_compute": float(final_expected_compute),
        "extra_compute_ratio": float(final_extra_compute),
        "total_compute_ratio": float(final_expected_compute + final_extra_compute),
        "latency_ms": float(latency_ms),
        "finite_gradients_all_steps": bool(finite_gradients_all_steps),
    }


def train_lowres(
    coarse_tokens: torch.Tensor,
    highres_tokens: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    image_path: Path,
) -> dict[str, float]:
    lowres_tokens = load_image_tokens(image_path=image_path, grid=args.low_grid, device=device)
    executor = DepthMultiPathExecutor(
        token_dim=lowres_tokens.size(-1),
        hidden_dim=args.executor_hidden,
        deep_layers=args.deep_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(executor.parameters(), lr=args.lr)
    route_probs = torch.zeros(
        lowres_tokens.size(0), lowres_tokens.size(1), 3, dtype=lowres_tokens.dtype, device=device
    )
    route_probs[..., 2] = 1.0

    final_loss = torch.tensor(0.0, device=device)
    for step in range(args.steps):
        _, _, mask_mode = OCR_TRAIN_SPECS[step % len(OCR_TRAIN_SPECS)]
        mask = make_focus_mask(args.coarse_grid, mask_mode, device=device)
        target = build_target_tokens(
            coarse_tokens=coarse_tokens,
            highres_tokens=highres_tokens,
            mask=mask,
            coarse_grid=args.coarse_grid,
            highres_grid=args.highres_grid,
        )
        pred, _ = executor(image_tokens=lowres_tokens, route_probs=route_probs, mode="soft")
        pred_up = upsample_tokens(pred, src_grid=args.low_grid, dst_grid=args.coarse_grid)
        final_loss = weighted_focus_loss(
            pred=pred_up, target=target, mask=mask, focus_weight=args.focus_weight
        )
        optimizer.zero_grad(set_to_none=True)
        final_loss.backward()
        optimizer.step()

    losses = []
    accuracies = []
    for _, _, mask_mode in OCR_EVAL_SPECS:
        mask = make_focus_mask(args.coarse_grid, mask_mode, device=device)
        target = build_target_tokens(
            coarse_tokens=coarse_tokens,
            highres_tokens=highres_tokens,
            mask=mask,
            coarse_grid=args.coarse_grid,
            highres_grid=args.highres_grid,
        )
        pred, _ = executor(image_tokens=lowres_tokens, route_probs=route_probs, mode="soft")
        pred_up = upsample_tokens(pred, src_grid=args.low_grid, dst_grid=args.coarse_grid)
        losses.append(
            float(
                weighted_focus_loss(
                    pred=pred_up, target=target, mask=mask, focus_weight=args.focus_weight
                ).detach()
            )
        )
        accuracies.append(detail_proxy_accuracy(pred_up.detach(), target.detach(), mask.detach()))

    latency_ms = benchmark_latency_ms(
        lambda: upsample_tokens(
            executor(image_tokens=lowres_tokens, route_probs=route_probs, mode="soft")[0],
            src_grid=args.low_grid,
            dst_grid=args.coarse_grid,
        ),
        warmup_runs=args.warmup_runs,
        benchmark_runs=args.benchmark_runs,
    )
    return {
        "proxy_task_loss": float(sum(losses) / len(losses)),
        "proxy_accuracy": float(sum(accuracies) / len(accuracies)),
        "expected_compute": float((args.low_grid * args.low_grid) / (args.coarse_grid * args.coarse_grid)),
        "latency_ms": float(latency_ms),
        "finite_gradients_all_steps": True,
    }


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    image_path = Path(args.image)
    summary_json = Path(args.summary_json)
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

    all_specs = OCR_TRAIN_SPECS + OCR_EVAL_SPECS
    query_cache = {
        name: load_query_tokens(processor, model, query).to(device)
        for name, query, _ in all_specs
    }
    coarse_tokens = load_image_tokens(image_path=image_path, grid=args.coarse_grid, device=device)
    highres_tokens = load_image_tokens(image_path=image_path, grid=args.highres_grid, device=device)

    depth_metrics = train_depth_only(
        query_cache=query_cache,
        coarse_tokens=coarse_tokens,
        highres_tokens=highres_tokens,
        args=args,
        device=device,
    )
    highres_metrics = train_highres(
        query_cache=query_cache,
        coarse_tokens=coarse_tokens,
        highres_tokens=highres_tokens,
        args=args,
        device=device,
    )
    lowres_metrics = train_lowres(
        coarse_tokens=coarse_tokens,
        highres_tokens=highres_tokens,
        args=args,
        device=device,
        image_path=image_path,
    )

    loss_drop = (
        (depth_metrics["proxy_task_loss"] - highres_metrics["proxy_task_loss"])
        / max(depth_metrics["proxy_task_loss"], 1e-8)
    )
    accuracy_gain = highres_metrics["proxy_accuracy"] - depth_metrics["proxy_accuracy"]
    extra_compute_ok = highres_metrics["extra_compute_ratio"] <= depth_metrics["expected_compute"] * 0.20
    pass_gate = bool(
        extra_compute_ok
        and (loss_drop >= 0.10 or accuracy_gain >= 0.05)
    )

    summary = {
        "task": "3.6_highres_reencoding_gate",
        "model": str(model_path),
        "image": str(image_path),
        "device": str(device),
        "budget": args.budget,
        "coarse_grid": args.coarse_grid,
        "highres_grid": args.highres_grid,
        "low_grid": args.low_grid,
        "depth_only": depth_metrics,
        "highres_reencode": highres_metrics,
        "lowres_baseline": lowres_metrics,
        "relative_loss_drop_vs_depth_only": float(loss_drop),
        "accuracy_gain_vs_depth_only": float(accuracy_gain),
        "extra_compute_within_20_percent": bool(extra_compute_ok),
        "pass_gate": pass_gate,
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("===== High-Resolution Re-encoding Pilot (Task 3.6) =====")
    print(f"device: {device}")
    print(f"budget: {args.budget:.2f}")
    print("method | proxy_task_loss | proxy_accuracy | expected_compute | extra_compute | total_compute | latency_ms")
    print(
        "DepthOnly-QACR | "
        f"{depth_metrics['proxy_task_loss']:.6f} | {depth_metrics['proxy_accuracy']:.6f} | "
        f"{depth_metrics['expected_compute']:.6f} | 0.000000 | "
        f"{depth_metrics['expected_compute']:.6f} | {depth_metrics['latency_ms']:.6f}"
    )
    print(
        "HighRes-Reencode | "
        f"{highres_metrics['proxy_task_loss']:.6f} | {highres_metrics['proxy_accuracy']:.6f} | "
        f"{highres_metrics['expected_compute']:.6f} | {highres_metrics['extra_compute_ratio']:.6f} | "
        f"{highres_metrics['total_compute_ratio']:.6f} | {highres_metrics['latency_ms']:.6f}"
    )
    print(
        "LowRes-9x9 | "
        f"{lowres_metrics['proxy_task_loss']:.6f} | {lowres_metrics['proxy_accuracy']:.6f} | "
        f"{lowres_metrics['expected_compute']:.6f} | 0.000000 | "
        f"{lowres_metrics['expected_compute']:.6f} | {lowres_metrics['latency_ms']:.6f}"
    )
    print(f"relative_loss_drop_vs_depth_only: {loss_drop:.6f}")
    print(f"accuracy_gain_vs_depth_only: {accuracy_gain:.6f}")
    print(f"extra_compute_within_20_percent: {extra_compute_ok}")
    print(f"pass_gate: {pass_gate}")
    print(f"summary_json: {summary_json}")


if __name__ == "__main__":
    main()
