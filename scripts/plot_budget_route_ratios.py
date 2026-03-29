#!/usr/bin/env python3
"""Phase 3.2: budget vs route-ratio statistics and plotting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
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
from qacr.vision import DepthMultiPathExecutor


QUERIES = [
    "请重点理解左侧目标，其他区域可适当降算力。",
    "请优先关注右侧主要目标，背景尽量节省计算。",
    "请识别文字区域并保证语义准确。",
    "总体描述时平衡预算和质量，避免全部走deep。",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Model/Qwen35-08B")
    parser.add_argument("--image", default="outputs/demo_phase01.png")
    parser.add_argument("--budgets", default="0.30,0.40,0.50,0.60,0.70")
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
    parser.add_argument("--out-plot", default="outputs/phase32_budget_route_ratio_plot.png")
    parser.add_argument(
        "--out-json", default="outputs/phase32_budget_route_ratio_stats.json"
    )
    return parser.parse_args()


def parse_budgets(text: str) -> list[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("At least one budget is required")
    for b in values:
        if not (0.0 <= b <= 1.0):
            raise ValueError(f"budget out of range: {b}")
    return values


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


def draw_budget_plot(
    budgets: list[float],
    skip: list[float],
    shallow: list[float],
    deep: list[float],
    out_path: Path,
) -> None:
    w, h = 920, 560
    margin_l, margin_r, margin_t, margin_b = 90, 40, 60, 80
    plot_w = w - margin_l - margin_r
    plot_h = h - margin_t - margin_b

    img = Image.new("RGB", (w, h), color=(247, 249, 253))
    draw = ImageDraw.Draw(img)
    draw.rectangle(
        [margin_l, margin_t, margin_l + plot_w, margin_t + plot_h],
        outline=(120, 130, 145),
        width=2,
    )

    for i in range(6):
        yv = i / 5.0
        y = margin_t + int(plot_h * (1 - yv))
        draw.line([(margin_l, y), (margin_l + plot_w, y)], fill=(220, 225, 236), width=1)
        draw.text((20, y - 8), f"{yv:.1f}", fill=(70, 75, 88))

    bmin, bmax = min(budgets), max(budgets)
    if abs(bmax - bmin) < 1e-8:
        bmax = bmin + 1e-3

    def to_xy(b: float, r: float) -> tuple[int, int]:
        x = margin_l + int((b - bmin) / (bmax - bmin) * plot_w)
        y = margin_t + int((1.0 - r) * plot_h)
        return x, y

    for b in budgets:
        x, _ = to_xy(b, 0.0)
        draw.line([(x, margin_t), (x, margin_t + plot_h)], fill=(233, 236, 244), width=1)
        draw.text((x - 14, margin_t + plot_h + 10), f"{b:.2f}", fill=(60, 65, 78))

    series = [
        ("skip", skip, (70, 129, 215)),
        ("shallow", shallow, (67, 170, 139)),
        ("deep", deep, (219, 102, 74)),
    ]
    for _, vals, color in series:
        pts = [to_xy(b, r) for b, r in zip(budgets, vals)]
        draw.line(pts, fill=color, width=4)
        for p in pts:
            draw.ellipse([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], fill=color)

    draw.text((margin_l, 20), "Budget vs Route Ratio (Task 3.2)", fill=(35, 40, 55))
    draw.text((w // 2 - 30, h - 30), "Budget", fill=(35, 40, 55))
    draw.text((8, margin_t - 16), "Ratio", fill=(35, 40, 55))

    lx, ly = w - 230, 24
    for idx, (name, _, color) in enumerate(series):
        y = ly + idx * 24
        draw.rectangle([lx, y + 4, lx + 18, y + 16], fill=color)
        draw.text((lx + 26, y + 2), name, fill=(35, 40, 55))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    image_path = Path(args.image)
    out_plot = Path(args.out_plot)
    out_json = Path(args.out_json)
    budgets = parse_budgets(args.budgets)

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
    query_cache = {q: load_query_tokens(processor, model, q) for q in QUERIES}
    route_costs = torch.tensor([0.0, 0.35, 1.0], dtype=torch.float32)

    skip_list, shallow_list, deep_list, expected_list = [], [], [], []

    for budget in budgets:
        router = DepthOnlyRouter(
            query_dim=next(iter(query_cache.values())).size(-1),
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

        last_soft = None
        last_expected = None
        for step in range(args.steps):
            query = QUERIES[step % len(QUERIES)]
            query_tokens = query_cache[query]
            temp = linear_temperature(step, args.steps, args.temp_start, args.temp_end)
            out = router(query_tokens=query_tokens, image_tokens=image_tokens)
            soft_probs = soft_routing_probs(out.logits, temperature=temp, use_gumbel=True)
            routed_tokens, _ = executor(
                image_tokens=image_tokens, route_probs=soft_probs, mode="soft"
            )
            task_loss = F.mse_loss(routed_tokens, image_tokens)
            comp_loss, expected = compute_regularization_loss(
                route_probs=soft_probs,
                route_costs=route_costs,
                budget_ratio=budget,
            )
            entropy = -(soft_probs * soft_probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
            loss = task_loss + args.lambda_compute * comp_loss - args.lambda_entropy * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last_soft = soft_probs.detach()
            last_expected = expected.detach()

        skip_list.append(float(last_soft[..., 0].mean()))
        shallow_list.append(float(last_soft[..., 1].mean()))
        deep_list.append(float(last_soft[..., 2].mean()))
        expected_list.append(float(last_expected))

    draw_budget_plot(
        budgets=budgets,
        skip=skip_list,
        shallow=shallow_list,
        deep=deep_list,
        out_path=out_plot,
    )

    payload = {
        "model": str(model_path),
        "image": str(image_path),
        "budgets": budgets,
        "skip_ratio": skip_list,
        "shallow_ratio": shallow_list,
        "deep_ratio": deep_list,
        "expected_compute": expected_list,
        "plot": str(out_plot),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("===== Budget vs Route Ratio Stats (Task 3.2) =====")
    print(f"model: {model_path}")
    print(f"image: {image_path}")
    print("budget | expected_compute | skip_ratio | shallow_ratio | deep_ratio")
    for b, e, s, sh, d in zip(budgets, expected_list, skip_list, shallow_list, deep_list):
        print(f"{b:.2f} | {e:.6f} | {s:.6f} | {sh:.6f} | {d:.6f}")
    print(f"plot: {out_plot}")
    print(f"stats_json: {out_json}")
    print("status: budget-route ratio plotting succeeded")


if __name__ == "__main__":
    main()

