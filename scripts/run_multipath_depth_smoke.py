#!/usr/bin/env python3
"""Phase 0.3 smoke test: router + skip/shallow/deep multi-path execution."""

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
from qacr.vision import DepthMultiPathExecutor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Model/Qwen35-08B", help="Local model path")
    parser.add_argument(
        "--image", default="outputs/demo_phase01.png", help="Input image path"
    )
    parser.add_argument(
        "--query", default="请将注意力更多放在关键区域，给每个区域分配不同计算深度。"
    )
    parser.add_argument("--router-hidden", type=int, default=96)
    parser.add_argument("--executor-hidden", type=int, default=128)
    parser.add_argument("--deep-layers", type=int, default=3)
    parser.add_argument("--coarse-grid", type=int, default=14)
    parser.add_argument("--mode", choices=["soft", "hard"], default="soft")
    parser.add_argument("--lr", type=float, default=3e-4)
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
    pixel = tfm(image).unsqueeze(0)
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
        hidden_dim=args.router_hidden,
        num_routes=3,
        temperature=1.0,
    )
    executor = DepthMultiPathExecutor(
        token_dim=image_tokens.size(-1),
        hidden_dim=args.executor_hidden,
        deep_layers=args.deep_layers,
    )

    optimizer = torch.optim.AdamW(
        list(router.parameters()) + list(executor.parameters()), lr=args.lr
    )

    router_out = router(query_tokens=query_tokens, image_tokens=image_tokens)
    routed_tokens, stats = executor(
        image_tokens=image_tokens,
        route_probs=router_out.route_probs,
        route_indices=router_out.hard_routes,
        mode=args.mode,
    )

    # Keep objective simple: preserve coarse token content while allowing branch learning.
    loss = F.mse_loss(routed_tokens, image_tokens)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    print("===== Multi-Path Depth Execution Smoke Test (Task 0.3) =====")
    print(f"model: {model_path}")
    print(f"mode: {args.mode}")
    print(f"query_token_shape: {tuple(query_tokens.shape)}")
    print(f"image_token_shape: {tuple(image_tokens.shape)}")
    print(f"routed_token_shape: {tuple(routed_tokens.shape)}")
    print(f"skip_ratio: {stats['skip_ratio']:.6f}")
    print(f"shallow_ratio: {stats['shallow_ratio']:.6f}")
    print(f"deep_ratio: {stats['deep_ratio']:.6f}")
    print(f"loss: {float(loss.detach()):.6f}")
    print("status: skip/shallow/deep branches forward+backward succeeded")


if __name__ == "__main__":
    main()
