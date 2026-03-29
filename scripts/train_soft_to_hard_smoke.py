#!/usr/bin/env python3
"""Phase 0.4 smoke test: soft-to-hard routing with compute regularization."""

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
from qacr.routing.soft_hard import (
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
    parser.add_argument("--query", default="请将关键区域分配更深计算，背景尽量跳过。")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lambda-compute", type=float, default=0.7)
    parser.add_argument("--budget", type=float, default=0.45)
    parser.add_argument("--temp-start", type=float, default=1.4)
    parser.add_argument("--temp-end", type=float, default=0.35)
    parser.add_argument("--coarse-grid", type=int, default=14)
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
        query_dim=query_tokens.size(-1), image_dim=image_tokens.size(-1), hidden_dim=96
    )
    executor = DepthMultiPathExecutor(
        token_dim=image_tokens.size(-1), hidden_dim=128, deep_layers=3
    )

    optimizer = torch.optim.AdamW(
        list(router.parameters()) + list(executor.parameters()), lr=args.lr
    )
    route_costs = torch.tensor([0.0, 0.35, 1.0], dtype=torch.float32)

    last_soft = None
    for step in range(args.steps):
        temp = linear_temperature(step, args.steps, args.temp_start, args.temp_end)
        router_out = router(query_tokens=query_tokens, image_tokens=image_tokens)

        soft_probs = soft_routing_probs(
            router_out.logits, temperature=temp, use_gumbel=True
        )
        routed_tokens, _ = executor(
            image_tokens=image_tokens,
            route_probs=soft_probs,
            mode="soft",
        )

        task_loss = F.mse_loss(routed_tokens, image_tokens)
        comp_loss, expected_compute = compute_regularization_loss(
            route_probs=soft_probs,
            route_costs=route_costs,
            budget_ratio=args.budget,
        )
        loss = task_loss + args.lambda_compute * comp_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        last_soft = soft_probs.detach()

    # Infer-hard check after soft training.
    hard_idx = hard_routing_from_logits(router_out.logits.detach())
    hard_one_hot = F.one_hot(hard_idx, num_classes=3).to(last_soft.dtype)
    _, hard_stats = executor(
        image_tokens=image_tokens,
        route_probs=hard_one_hot,
        route_indices=hard_idx,
        mode="hard",
    )

    soft_stats = {
        "skip": float(last_soft[..., 0].mean()),
        "shallow": float(last_soft[..., 1].mean()),
        "deep": float(last_soft[..., 2].mean()),
    }

    print("===== Soft-to-Hard Routing Smoke Test (Task 0.4) =====")
    print(f"model: {model_path}")
    print(f"steps: {args.steps}")
    print(f"temperature_start_end: {args.temp_start} -> {args.temp_end}")
    print(f"budget_target: {args.budget:.4f}")
    print(f"expected_compute: {float(expected_compute.detach()):.6f}")
    print(f"task_loss: {float(task_loss.detach()):.6f}")
    print(f"compute_loss: {float(comp_loss.detach()):.6f}")
    print(f"total_loss: {float(loss.detach()):.6f}")
    print(f"soft_ratio_skip: {soft_stats['skip']:.6f}")
    print(f"soft_ratio_shallow: {soft_stats['shallow']:.6f}")
    print(f"soft_ratio_deep: {soft_stats['deep']:.6f}")
    print(f"hard_ratio_skip: {hard_stats['skip_ratio']:.6f}")
    print(f"hard_ratio_shallow: {hard_stats['shallow_ratio']:.6f}")
    print(f"hard_ratio_deep: {hard_stats['deep_ratio']:.6f}")
    print("status: soft training + hard inference switch succeeded")


if __name__ == "__main__":
    main()
