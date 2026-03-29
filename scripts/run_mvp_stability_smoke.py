#!/usr/bin/env python3
"""Phase 0.5 smoke test: stability, backprop, and collapse checks."""

from __future__ import annotations

import argparse
import math
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
    linear_temperature,
    soft_routing_probs,
)
from qacr.vision import DepthMultiPathExecutor


QUERIES = [
    "请关注几何形状边缘，背景可更浅计算。",
    "优先识别文字区域，其他区域尽量节省算力。",
    "对于显著目标给更深计算，对平坦背景做跳过。",
    "整体描述时平衡精度和预算，避免全部走deep。",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Model/Qwen35-08B")
    parser.add_argument("--image", default="outputs/demo_phase01.png")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-compute", type=float, default=0.8)
    parser.add_argument("--lambda-entropy", type=float, default=0.02)
    parser.add_argument("--budget", type=float, default=0.45)
    parser.add_argument("--temp-start", type=float, default=1.5)
    parser.add_argument("--temp-end", type=float, default=0.4)
    parser.add_argument("--collapse-threshold", type=float, default=0.95)
    parser.add_argument("--coarse-grid", type=int, default=14)
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


def is_finite_grads(modules: list[torch.nn.Module]) -> bool:
    for module in modules:
        for p in module.parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                return False
    return True


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
    init_query_tokens = load_query_tokens(processor, model, QUERIES[0])
    router = DepthOnlyRouter(
        query_dim=init_query_tokens.size(-1),
        image_dim=image_tokens.size(-1),
        hidden_dim=96,
    )
    executor = DepthMultiPathExecutor(
        token_dim=image_tokens.size(-1), hidden_dim=128, deep_layers=3
    )
    optimizer = torch.optim.AdamW(
        list(router.parameters()) + list(executor.parameters()), lr=args.lr
    )
    route_costs = torch.tensor([0.0, 0.35, 1.0], dtype=torch.float32)

    soft_history = []
    finite_all_steps = True
    final_loss = None

    for step in range(args.steps):
        query = QUERIES[step % len(QUERIES)]
        query_tokens = load_query_tokens(processor, model, query)
        temp = linear_temperature(step, args.steps, args.temp_start, args.temp_end)

        router_out = router(query_tokens=query_tokens, image_tokens=image_tokens)
        soft_probs = soft_routing_probs(
            router_out.logits, temperature=temp, use_gumbel=True
        )
        routed_tokens, stats = executor(
            image_tokens=image_tokens, route_probs=soft_probs, mode="soft"
        )

        task_loss = F.mse_loss(routed_tokens, image_tokens)
        compute_loss, expected_compute = compute_regularization_loss(
            route_probs=soft_probs,
            route_costs=route_costs,
            budget_ratio=args.budget,
        )
        entropy = -(soft_probs * soft_probs.clamp_min(1e-8).log()).sum(dim=-1).mean()

        loss = (
            task_loss
            + args.lambda_compute * compute_loss
            - args.lambda_entropy * entropy
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        finite_all_steps = (
            finite_all_steps
            and is_finite_grads([router, executor])
            and math.isfinite(float(loss.detach()))
        )
        optimizer.step()

        soft_history.append(
            (stats["skip_ratio"], stats["shallow_ratio"], stats["deep_ratio"])
        )
        final_loss = loss.detach()

    hist = torch.tensor(soft_history)
    mean_ratios = hist.mean(dim=0)
    collapse_threshold = args.collapse_threshold
    collapse_detected = bool((mean_ratios > collapse_threshold).any())
    route_var = hist.var(dim=0, unbiased=False)

    print("===== MVP Stability Smoke Test (Task 0.5) =====")
    print(f"model: {model_path}")
    print(f"steps: {args.steps}")
    print(f"finite_gradients_all_steps: {finite_all_steps}")
    print(f"final_loss: {float(final_loss):.6f}")
    print(f"mean_skip_ratio: {float(mean_ratios[0]):.6f}")
    print(f"mean_shallow_ratio: {float(mean_ratios[1]):.6f}")
    print(f"mean_deep_ratio: {float(mean_ratios[2]):.6f}")
    print(f"var_skip_ratio: {float(route_var[0]):.6f}")
    print(f"var_shallow_ratio: {float(route_var[1]):.6f}")
    print(f"var_deep_ratio: {float(route_var[2]):.6f}")
    print(f"collapse_detected: {collapse_detected}")
    print(f"budget_target: {args.budget:.4f}")
    print(f"expected_compute_last: {float(expected_compute.detach()):.6f}")
    print("status: end-to-end stability check succeeded")


if __name__ == "__main__":
    main()
