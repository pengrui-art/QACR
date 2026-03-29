#!/usr/bin/env python3
"""Phase 3.1: visualize token-level routing heatmaps under different queries."""

from __future__ import annotations

import argparse
import itertools
import json
import math
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


QUERY_SPECS = [
    ("left_focus", "请优先关注图像左侧对象。"),
    ("right_focus", "请优先关注图像右侧对象。"),
    ("center_focus", "请优先关注图像中心区域。"),
    ("bottom_text", "请重点关注底部文字区域。"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Model/Qwen35-08B")
    parser.add_argument("--image", default="outputs/demo_phase01.png")
    parser.add_argument("--coarse-grid", type=int, default=14)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--router-hidden", type=int, default=96)
    parser.add_argument("--output-dir", default="outputs/phase31_heatmaps")
    parser.add_argument(
        "--summary-json", default="outputs/phase31_query_heatmap_summary.json"
    )
    return parser.parse_args()


def load_query_tokens(
    processor: AutoProcessor,
    model: AutoModelForImageTextToText,
    query: str,
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
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if float(mask.mean()) < 0.05:
        mask[grid // 2, grid // 2] = 1.0
    return mask


def build_target_route_probs(mask: torch.Tensor) -> torch.Tensor:
    # focus token -> deep; non-focus token -> skip/shallow.
    focus = mask.reshape(-1, 1)
    focus_probs = torch.tensor([0.05, 0.15, 0.80], dtype=torch.float32).view(1, 3)
    nonfocus_probs = torch.tensor([0.45, 0.45, 0.10], dtype=torch.float32).view(1, 3)
    return focus * focus_probs + (1.0 - focus) * nonfocus_probs


def colorize_heatmap(map_2d: np.ndarray) -> np.ndarray:
    # blue -> cyan -> yellow -> red
    x = np.clip(map_2d, 0.0, 1.0)
    r = np.clip(2.0 * x - 0.2, 0.0, 1.0)
    g = np.clip(1.5 - np.abs(2.0 * x - 1.0), 0.0, 1.0)
    b = np.clip(1.2 - 2.0 * x, 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def overlay_heatmap_on_image(
    base_img: Image.Image, heatmap_2d: np.ndarray, out_path: Path
) -> None:
    h, w = base_img.height, base_img.width
    heat_uint8 = (np.clip(heatmap_2d, 0.0, 1.0) * 255).astype(np.uint8)
    heat_img = Image.fromarray(heat_uint8, mode="L").resize((w, h), Image.BILINEAR)
    heat_rgb = Image.fromarray(colorize_heatmap(np.array(heat_img) / 255.0), mode="RGB")
    blended = Image.blend(base_img.convert("RGB"), heat_rgb, alpha=0.45)
    blended.save(out_path)


def draw_panel(
    image_paths: list[Path],
    labels: list[str],
    out_path: Path,
    tile_w: int = 448,
    tile_h: int = 448,
) -> None:
    cols, rows = 2, 2
    canvas = Image.new("RGB", (cols * tile_w, rows * (tile_h + 30)), color=(245, 247, 252))
    draw = ImageDraw.Draw(canvas)
    for idx, (p, label) in enumerate(zip(image_paths, labels)):
        img = Image.open(p).convert("RGB").resize((tile_w, tile_h))
        c = idx % cols
        r = idx // cols
        x0 = c * tile_w
        y0 = r * (tile_h + 30)
        canvas.paste(img, (x0, y0))
        draw.rectangle([x0, y0 + tile_h, x0 + tile_w, y0 + tile_h + 30], fill=(230, 235, 245))
        draw.text((x0 + 8, y0 + tile_h + 8), label, fill=(30, 35, 50))
    canvas.save(out_path)


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    image_path = Path(args.image)
    out_dir = Path(args.output_dir)
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

    image_tokens = load_coarse_image_tokens(image_path, grid=args.coarse_grid)
    query_tokens_map = {
        name: load_query_tokens(processor, model, query) for name, query in QUERY_SPECS
    }
    router = DepthOnlyRouter(
        query_dim=next(iter(query_tokens_map.values())).size(-1),
        image_dim=image_tokens.size(-1),
        hidden_dim=args.router_hidden,
    )
    optimizer = torch.optim.AdamW(router.parameters(), lr=args.lr)

    targets = {}
    for name, _ in QUERY_SPECS:
        mask = make_focus_mask(args.coarse_grid, mode=name)
        targets[name] = build_target_route_probs(mask).unsqueeze(0)  # [1, Ti, 3]

    for step in range(args.steps):
        total_loss = 0.0
        for name, _ in QUERY_SPECS:
            out = router(query_tokens=query_tokens_map[name], image_tokens=image_tokens)
            loss = F.mse_loss(out.route_probs, targets[name])
            total_loss = total_loss + loss
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

    out_dir.mkdir(parents=True, exist_ok=True)
    base_img = Image.open(image_path).convert("RGB").resize((448, 448))
    deep_maps = {}
    image_files = []
    labels = []

    for name, query in QUERY_SPECS:
        out = router(query_tokens=query_tokens_map[name], image_tokens=image_tokens)
        deep = out.route_probs[0, :, 2].reshape(args.coarse_grid, args.coarse_grid).detach().cpu()
        deep_np = deep.numpy()
        deep_maps[name] = deep_np
        out_file = out_dir / f"heatmap_{name}.png"
        overlay_heatmap_on_image(base_img, deep_np, out_file)
        image_files.append(out_file)
        labels.append(f"{name}: {query}")

    panel_file = out_dir / "heatmap_panel.png"
    draw_panel(image_paths=image_files, labels=labels, out_path=panel_file)

    pairwise_l1 = {}
    for (a, _), (b, _) in itertools.combinations(QUERY_SPECS, 2):
        dist = float(np.mean(np.abs(deep_maps[a] - deep_maps[b])))
        pairwise_l1[f"{a}__vs__{b}"] = dist

    summary = {
        "model": str(model_path),
        "image": str(image_path),
        "coarse_grid": args.coarse_grid,
        "steps": args.steps,
        "mean_deep_prob": {
            name: float(np.mean(deep_maps[name])) for name, _ in QUERY_SPECS
        },
        "std_deep_prob": {
            name: float(np.std(deep_maps[name])) for name, _ in QUERY_SPECS
        },
        "pairwise_l1": pairwise_l1,
        "pairwise_l1_mean": float(np.mean(list(pairwise_l1.values()))),
        "pairwise_l1_max": float(np.max(list(pairwise_l1.values()))),
        "heatmap_panel": str(panel_file),
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("===== Query Routing Heatmap Visualization (Task 3.1) =====")
    print(f"model: {model_path}")
    print(f"image: {image_path}")
    print(f"coarse_grid: {args.coarse_grid}")
    print(f"steps: {args.steps}")
    for name, _ in QUERY_SPECS:
        print(
            f"{name}_mean_deep_prob: {summary['mean_deep_prob'][name]:.6f}, "
            f"{name}_std_deep_prob: {summary['std_deep_prob'][name]:.6f}"
        )
    print(f"pairwise_l1_mean: {summary['pairwise_l1_mean']:.6f}")
    print(f"pairwise_l1_max: {summary['pairwise_l1_max']:.6f}")
    print(f"heatmap_panel: {panel_file}")
    print(f"summary_json: {summary_json}")
    print("status: query-conditioned heatmap visualization succeeded")


if __name__ == "__main__":
    main()
