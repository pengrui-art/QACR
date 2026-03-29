#!/usr/bin/env python3
"""Phase 0.2 smoke test: lightweight query-adaptive depth-only router.

Validates:
1) Query + coarse image tokens -> {skip, shallow, deep} route weights.
2) One-step optimization of router parameters.
3) Router compute overhead ratio relative to a backbone MACs budget.
"""

from __future__ import annotations

import argparse
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

from qacr.routing import DepthOnlyRouter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Model/Qwen35-08B", help="Local model path")
    parser.add_argument(
        "--image", default="outputs/demo_phase01.png", help="Input image path"
    )
    parser.add_argument(
        "--query", default="请重点关注图中的几何形状，并判断哪些区域需要深层计算。"
    )
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--backbone-macs", type=float, default=25e9)
    parser.add_argument(
        "--coarse-grid",
        type=int,
        default=14,
        help="Produces coarse_grid^2 image tokens",
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
    tfm = transforms.Compose(
        [
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
        ]
    )
    pixel = tfm(image).unsqueeze(0)  # [1, 3, H, W]
    pooled = F.adaptive_avg_pool2d(pixel, output_size=(grid, grid))
    tokens = pooled.flatten(2).transpose(1, 2).contiguous()  # [1, grid*grid, 3]
    return tokens


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

    query_tokens = load_query_tokens(processor, model, args.query)
    image_tokens = load_coarse_image_tokens(image_path, grid=args.coarse_grid)

    router = DepthOnlyRouter(
        query_dim=query_tokens.size(-1),
        image_dim=image_tokens.size(-1),
        hidden_dim=args.hidden_dim,
        num_routes=3,
        temperature=args.temperature,
    )

    optimizer = torch.optim.AdamW(router.parameters(), lr=args.lr)

    out = router(query_tokens=query_tokens, image_tokens=image_tokens)

    deep_ratio = out.route_probs[..., 2].mean()
    shallow_ratio = out.route_probs[..., 1].mean()
    skip_ratio = out.route_probs[..., 0].mean()

    # Lightweight objective to verify differentiable training for router.
    target = torch.tensor([0.20, 0.45, 0.35], dtype=out.route_probs.dtype)
    observed = torch.stack([skip_ratio, shallow_ratio, deep_ratio])
    balance_loss = F.mse_loss(observed, target)
    entropy = (
        -(out.route_probs * (out.route_probs.clamp_min(1e-8).log())).sum(dim=-1).mean()
    )
    loss = balance_loss - 0.01 * entropy

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    token_count = image_tokens.size(1)
    router_macs = router.estimate_macs(batch_size=1, num_image_tokens=token_count)
    overhead_ratio = router.estimate_overhead_ratio(
        backbone_macs=args.backbone_macs,
        batch_size=1,
        num_image_tokens=token_count,
    )

    print("===== Depth-Only Router Smoke Test (Task 0.2) =====")
    print(f"model: {model_path}")
    print(f"query_token_shape: {tuple(query_tokens.shape)}")
    print(f"image_token_shape: {tuple(image_tokens.shape)}")
    print(f"router_hidden_dim: {args.hidden_dim}")
    print(f"route_mean_skip: {float(skip_ratio.detach()):.6f}")
    print(f"route_mean_shallow: {float(shallow_ratio.detach()):.6f}")
    print(f"route_mean_deep: {float(deep_ratio.detach()):.6f}")
    print(f"loss: {float(loss.detach()):.6f}")
    print(f"router_macs_estimate: {router_macs}")
    print(f"overhead_ratio: {overhead_ratio:.6%}")
    print(f"overhead_under_5_percent: {overhead_ratio < 0.05}")


if __name__ == "__main__":
    main()
