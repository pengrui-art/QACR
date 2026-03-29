#!/usr/bin/env python3
"""Phase 3.3: analyze routing errors and corner cases."""

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

from qacr.routing import DepthOnlyRouter


TRAIN_SPECS = [
    ("left_focus", "请优先关注图像左侧对象。"),
    ("right_focus", "请优先关注图像右侧对象。"),
    ("center_focus", "请优先关注图像中心区域。"),
    ("bottom_text", "请重点关注底部文字区域。"),
]

CORNER_CASES = [
    {
        "name": "left_easy",
        "query": "请只关注左边蓝色区域，右边可以忽略。",
        "mask_mode": "left_focus",
    },
    {
        "name": "right_easy",
        "query": "请只关注右边黄色区域，左边可以忽略。",
        "mask_mode": "right_focus",
    },
    {
        "name": "text_hard",
        "query": "请重点识别底部文字，其他几何区域降算力。",
        "mask_mode": "bottom_text",
    },
    {
        "name": "ignore_text",
        "query": "请忽略底部文字，只关注图形区域。",
        "mask_mode": "shape_region",
    },
    {
        "name": "dual_object",
        "query": "左右两个主体都要看，中心和下方可以减少计算。",
        "mask_mode": "left_right_dual",
    },
    {
        "name": "center_conflict",
        "query": "只看中心区域，不要关注左右边缘。",
        "mask_mode": "center_focus",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Model/Qwen35-08B")
    parser.add_argument("--image", default="outputs/demo_phase01.png")
    parser.add_argument("--coarse-grid", type=int, default=14)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--router-hidden", type=int, default=96)
    parser.add_argument("--deep-threshold", type=float, default=0.33)
    parser.add_argument("--skip-threshold", type=float, default=0.50)
    parser.add_argument("--out-dir", default="outputs/phase33_corner_cases")
    parser.add_argument(
        "--summary-json", default="outputs/phase33_corner_case_summary.json"
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


def make_focus_mask(grid: int, mode: str) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, grid), torch.linspace(0, 1, grid), indexing="ij"
    )
    if mode == "left_focus":
        mask = (xx < 0.45).float()
    elif mode == "right_focus":
        mask = (xx > 0.55).float()
    elif mode == "center_focus":
        cx, cy = 0.5, 0.5
        dist = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
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
    return mask


def build_target_route_probs(mask: torch.Tensor) -> torch.Tensor:
    focus = mask.reshape(-1, 1)
    focus_probs = torch.tensor([0.05, 0.15, 0.80], dtype=torch.float32).view(1, 3)
    nonfocus_probs = torch.tensor([0.45, 0.45, 0.10], dtype=torch.float32).view(1, 3)
    return focus * focus_probs + (1.0 - focus) * nonfocus_probs


def colorize(map_2d: np.ndarray) -> np.ndarray:
    x = np.clip(map_2d, 0.0, 1.0)
    r = np.clip(2.0 * x - 0.2, 0.0, 1.0)
    g = np.clip(1.5 - np.abs(2.0 * x - 1.0), 0.0, 1.0)
    b = np.clip(1.2 - 2.0 * x, 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def save_case_panel(
    base_img: Image.Image,
    pred_deep: np.ndarray,
    target_mask: np.ndarray,
    out_path: Path,
    case_name: str,
    metrics_text: str,
) -> None:
    w, h = base_img.size
    pred_resized = (
        np.array(
            Image.fromarray((pred_deep * 255).astype(np.uint8), mode="L").resize((w, h)),
            dtype=np.float32,
        )
        / 255.0
    )
    tgt_resized = (
        np.array(
            Image.fromarray((target_mask * 255).astype(np.uint8), mode="L").resize((w, h)),
            dtype=np.float32,
        )
        / 255.0
    )
    pred_img = Image.fromarray(
        colorize(pred_resized),
        mode="RGB",
    )
    tgt_img = Image.fromarray(
        colorize(tgt_resized),
        mode="RGB",
    )
    left = Image.blend(base_img, pred_img, 0.45)
    right = Image.blend(base_img, tgt_img, 0.45)

    canvas = Image.new("RGB", (w * 2, h + 48), color=(245, 247, 252))
    draw = ImageDraw.Draw(canvas)
    canvas.paste(left, (0, 0))
    canvas.paste(right, (w, 0))
    draw.rectangle([0, h, w * 2, h + 48], fill=(230, 235, 245))
    draw.text((10, h + 6), f"{case_name} | Left: Pred Deep  Right: Target Focus", fill=(30, 35, 50))
    draw.text((10, h + 24), metrics_text, fill=(30, 35, 50))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    image_path = Path(args.image)
    out_dir = Path(args.out_dir)
    summary_json = Path(args.summary_json)
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

    image_tokens = load_coarse_image_tokens(image_path, args.coarse_grid)
    train_q = {name: load_query_tokens(processor, model, q) for name, q in TRAIN_SPECS}
    router = DepthOnlyRouter(
        query_dim=next(iter(train_q.values())).size(-1),
        image_dim=image_tokens.size(-1),
        hidden_dim=args.router_hidden,
    )
    optimizer = torch.optim.AdamW(router.parameters(), lr=args.lr)

    train_targets = {}
    for name, _ in TRAIN_SPECS:
        m = make_focus_mask(args.coarse_grid, mode=name)
        train_targets[name] = build_target_route_probs(m).unsqueeze(0)

    for _ in range(args.steps):
        total = 0.0
        for name, _ in TRAIN_SPECS:
            out = router(query_tokens=train_q[name], image_tokens=image_tokens)
            total = total + F.mse_loss(out.route_probs, train_targets[name])
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()

    base_img = Image.open(image_path).convert("RGB").resize((448, 448))
    out_dir.mkdir(parents=True, exist_ok=True)
    case_results = []

    for case in CORNER_CASES:
        q_tok = load_query_tokens(processor, model, case["query"])
        out = router(query_tokens=q_tok, image_tokens=image_tokens)
        pred = out.route_probs[0]
        pred_skip = pred[:, 0].reshape(args.coarse_grid, args.coarse_grid).detach().cpu().numpy()
        pred_deep = pred[:, 2].reshape(args.coarse_grid, args.coarse_grid).detach().cpu().numpy()
        target_mask = make_focus_mask(args.coarse_grid, case["mask_mode"]).numpy()

        key = target_mask > 0.5
        nonkey = target_mask <= 0.5
        key_deep_mean = float(pred_deep[key].mean()) if key.any() else 0.0
        nonkey_deep_mean = float(pred_deep[nonkey].mean()) if nonkey.any() else 0.0
        key_skip_mean = float(pred_skip[key].mean()) if key.any() else 0.0
        miss_rate = float((pred_deep[key] < args.deep_threshold).mean()) if key.any() else 0.0
        early_skip_rate = float((pred_skip[key] > args.skip_threshold).mean()) if key.any() else 0.0
        separation = key_deep_mean - nonkey_deep_mean
        score = float(miss_rate + early_skip_rate + max(0.0, -separation))
        flagged = bool((miss_rate > 0.40) or (early_skip_rate > 0.25) or (separation <= 0.0))

        metrics_text = (
            f"key_deep={key_deep_mean:.3f}, nonkey_deep={nonkey_deep_mean:.3f}, "
            f"miss={miss_rate:.3f}, early_skip={early_skip_rate:.3f}"
        )
        panel_file = out_dir / f"corner_{case['name']}.png"
        save_case_panel(
            base_img=base_img,
            pred_deep=pred_deep,
            target_mask=target_mask,
            out_path=panel_file,
            case_name=case["name"],
            metrics_text=metrics_text,
        )
        case_results.append(
            {
                "name": case["name"],
                "query": case["query"],
                "mask_mode": case["mask_mode"],
                "key_deep_mean": key_deep_mean,
                "nonkey_deep_mean": nonkey_deep_mean,
                "key_skip_mean": key_skip_mean,
                "miss_rate_key_tokens": miss_rate,
                "early_skip_rate_key_tokens": early_skip_rate,
                "separation_key_minus_nonkey": separation,
                "corner_score": score,
                "flagged_as_error": flagged,
                "panel": str(panel_file),
            }
        )

    case_results.sort(key=lambda x: x["corner_score"], reverse=True)
    flagged = [c for c in case_results if c["flagged_as_error"]]
    summary = {
        "model": str(model_path),
        "image": str(image_path),
        "coarse_grid": args.coarse_grid,
        "steps": args.steps,
        "deep_threshold": args.deep_threshold,
        "skip_threshold": args.skip_threshold,
        "num_cases": len(case_results),
        "num_flagged_errors": len(flagged),
        "worst_case": case_results[0] if case_results else None,
        "cases": case_results,
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("===== Routing Corner Case Analysis (Task 3.3) =====")
    print(f"model: {model_path}")
    print(f"image: {image_path}")
    print(f"num_cases: {len(case_results)}")
    print(f"num_flagged_errors: {len(flagged)}")
    print("case | miss_rate | early_skip_rate | separation | corner_score | flagged")
    for c in case_results:
        print(
            f"{c['name']} | {c['miss_rate_key_tokens']:.6f} | "
            f"{c['early_skip_rate_key_tokens']:.6f} | {c['separation_key_minus_nonkey']:.6f} | "
            f"{c['corner_score']:.6f} | {c['flagged_as_error']}"
        )
    print(f"summary_json: {summary_json}")
    print(f"case_panels_dir: {out_dir}")
    print("status: corner case analysis succeeded")


if __name__ == "__main__":
    main()
