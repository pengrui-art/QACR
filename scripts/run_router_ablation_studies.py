#!/usr/bin/env python3
"""Phase 3.4: router design and hyper-parameter ablation studies."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import transforms
from transformers import AutoModelForImageTextToText, AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qacr.routing import (
    DepthOnlyRouter,
    compute_regularization_loss,
    hard_routing_from_logits,
    linear_temperature,
    soft_routing_probs,
)
from qacr.vision import DepthMultiPathExecutor


QUERIES = [
    "请关注左侧主体并弱化背景。",
    "请关注右侧主体并尽量节省预算。",
    "请识别文字区域并保持语义准确。",
    "平衡答案质量与延迟，不要全部走deep。",
]


class LinearRouter(nn.Module):
    def __init__(self, query_dim: int, image_dim: int, hidden_dim: int = 96) -> None:
        super().__init__()
        self.query_proj = nn.Linear(query_dim, hidden_dim, bias=False)
        self.image_proj = nn.Linear(image_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 3)

    def forward(self, query_tokens: torch.Tensor, image_tokens: torch.Tensor) -> torch.Tensor:
        q = self.query_proj(query_tokens.mean(dim=1)).unsqueeze(1)
        i = self.image_proj(image_tokens)
        x = self.norm(i + q)
        return self.head(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Model/Qwen35-08B")
    parser.add_argument("--image", default="outputs/demo_phase01.png")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--budget", type=float, default=0.45)
    parser.add_argument("--router-hidden", type=int, default=96)
    parser.add_argument("--executor-hidden", type=int, default=128)
    parser.add_argument("--deep-layers", type=int, default=3)
    parser.add_argument("--coarse-grid", type=int, default=14)
    parser.add_argument("--out-json", default="outputs/phase34_ablation_summary.json")
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


def has_nonfinite_grad(modules: list[nn.Module]) -> bool:
    for m in modules:
        for p in m.parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                return True
    return False


def train_eval_once(
    router_kind: str,
    image_tokens: torch.Tensor,
    query_cache: dict[str, torch.Tensor],
    steps: int,
    lr: float,
    budget: float,
    lambda_compute: float,
    lambda_entropy: float,
    temp_mode: str,
    router_hidden: int,
    executor_hidden: int,
    deep_layers: int,
) -> dict[str, float | bool]:
    q_dim = next(iter(query_cache.values())).size(-1)
    i_dim = image_tokens.size(-1)
    if router_kind == "mlp":
        router: nn.Module = DepthOnlyRouter(
            query_dim=q_dim, image_dim=i_dim, hidden_dim=router_hidden
        )
    elif router_kind == "linear":
        router = LinearRouter(query_dim=q_dim, image_dim=i_dim, hidden_dim=router_hidden)
    else:
        raise ValueError(router_kind)

    executor = DepthMultiPathExecutor(
        token_dim=i_dim, hidden_dim=executor_hidden, deep_layers=deep_layers
    )
    optimizer = torch.optim.AdamW(list(router.parameters()) + list(executor.parameters()), lr=lr)
    route_costs = torch.tensor([0.0, 0.35, 1.0], dtype=torch.float32)

    finite_grad_all = True
    final_task = None
    final_total = None
    final_expected = None
    final_logits = None
    final_soft = None

    for step in range(steps):
        q = query_cache[QUERIES[step % len(QUERIES)]]
        if router_kind == "mlp":
            out = router(query_tokens=q, image_tokens=image_tokens)
            logits = out.logits
        else:
            logits = router(query_tokens=q, image_tokens=image_tokens)

        if temp_mode == "anneal":
            temp = linear_temperature(step, steps, 1.5, 0.4)
        elif temp_mode == "fixed_high":
            temp = 1.5
        elif temp_mode == "fixed_low":
            temp = 0.4
        else:
            raise ValueError(temp_mode)

        soft_probs = soft_routing_probs(logits, temperature=temp, use_gumbel=True)
        routed_tokens, _ = executor(image_tokens=image_tokens, route_probs=soft_probs, mode="soft")
        task_loss = F.mse_loss(routed_tokens, image_tokens)
        comp_loss, expected_compute = compute_regularization_loss(
            route_probs=soft_probs, route_costs=route_costs, budget_ratio=budget
        )
        entropy = -(soft_probs * soft_probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
        total_loss = task_loss + lambda_compute * comp_loss - lambda_entropy * entropy

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        finite_grad_all = finite_grad_all and (not has_nonfinite_grad([router, executor]))
        finite_grad_all = finite_grad_all and math.isfinite(float(total_loss.detach()))
        optimizer.step()

        final_task = task_loss.detach()
        final_total = total_loss.detach()
        final_expected = expected_compute.detach()
        final_logits = logits.detach()
        final_soft = soft_probs.detach()

    hard_idx = hard_routing_from_logits(final_logits)
    hard_one_hot = F.one_hot(hard_idx, num_classes=3).to(final_soft.dtype)
    _, hard_stats = executor(
        image_tokens=image_tokens,
        route_probs=hard_one_hot,
        route_indices=hard_idx,
        mode="hard",
    )
    collapse_ratio = float(max(hard_stats.values()))
    budget_gap = float(abs(float(final_expected) - budget))

    return {
        "task_loss": float(final_task),
        "total_loss": float(final_total),
        "expected_compute": float(final_expected),
        "budget_gap": budget_gap,
        "soft_skip": float(final_soft[..., 0].mean()),
        "soft_shallow": float(final_soft[..., 1].mean()),
        "soft_deep": float(final_soft[..., 2].mean()),
        "hard_skip": float(hard_stats["skip_ratio"]),
        "hard_shallow": float(hard_stats["shallow_ratio"]),
        "hard_deep": float(hard_stats["deep_ratio"]),
        "hard_collapse_ratio": collapse_ratio,
        "finite_gradients_all_steps": finite_grad_all,
    }


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

    image_tokens = load_coarse_image_tokens(image_path, grid=args.coarse_grid)
    query_cache = {q: load_query_tokens(processor, model, q) for q in QUERIES}

    router_ablation = {
        "mlp": train_eval_once(
            router_kind="mlp",
            image_tokens=image_tokens,
            query_cache=query_cache,
            steps=args.steps,
            lr=args.lr,
            budget=args.budget,
            lambda_compute=0.8,
            lambda_entropy=0.02,
            temp_mode="anneal",
            router_hidden=args.router_hidden,
            executor_hidden=args.executor_hidden,
            deep_layers=args.deep_layers,
        ),
        "linear": train_eval_once(
            router_kind="linear",
            image_tokens=image_tokens,
            query_cache=query_cache,
            steps=args.steps,
            lr=args.lr,
            budget=args.budget,
            lambda_compute=0.8,
            lambda_entropy=0.02,
            temp_mode="anneal",
            router_hidden=args.router_hidden,
            executor_hidden=args.executor_hidden,
            deep_layers=args.deep_layers,
        ),
    }

    temp_ablation = {}
    for mode in ["anneal", "fixed_high", "fixed_low"]:
        temp_ablation[mode] = train_eval_once(
            router_kind="mlp",
            image_tokens=image_tokens,
            query_cache=query_cache,
            steps=args.steps,
            lr=args.lr,
            budget=args.budget,
            lambda_compute=0.8,
            lambda_entropy=0.02,
            temp_mode=mode,
            router_hidden=args.router_hidden,
            executor_hidden=args.executor_hidden,
            deep_layers=args.deep_layers,
        )

    compute_lambda_ablation = {}
    for lam in [0.3, 0.8, 1.5]:
        compute_lambda_ablation[str(lam)] = train_eval_once(
            router_kind="mlp",
            image_tokens=image_tokens,
            query_cache=query_cache,
            steps=args.steps,
            lr=args.lr,
            budget=args.budget,
            lambda_compute=lam,
            lambda_entropy=0.02,
            temp_mode="anneal",
            router_hidden=args.router_hidden,
            executor_hidden=args.executor_hidden,
            deep_layers=args.deep_layers,
        )

    summary = {
        "model": str(model_path),
        "image": str(image_path),
        "steps": args.steps,
        "budget": args.budget,
        "router_structure_ablation": router_ablation,
        "temperature_ablation": temp_ablation,
        "compute_lambda_ablation": compute_lambda_ablation,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("===== Router Ablation Studies (Task 3.4) =====")
    print(f"model: {model_path}")
    print(f"image: {image_path}")
    print(f"steps: {args.steps}")
    print(f"budget: {args.budget:.4f}")
    print("router_structure | task_loss | budget_gap | hard_collapse_ratio | finite_grad")
    for name, m in router_ablation.items():
        print(
            f"{name} | {m['task_loss']:.6f} | {m['budget_gap']:.6f} | "
            f"{m['hard_collapse_ratio']:.6f} | {m['finite_gradients_all_steps']}"
        )
    print("temperature_mode | task_loss | budget_gap | hard_collapse_ratio")
    for name, m in temp_ablation.items():
        print(
            f"{name} | {m['task_loss']:.6f} | {m['budget_gap']:.6f} | "
            f"{m['hard_collapse_ratio']:.6f}"
        )
    print("lambda_compute | task_loss | expected_compute | budget_gap")
    for lam, m in compute_lambda_ablation.items():
        print(
            f"{lam} | {m['task_loss']:.6f} | {m['expected_compute']:.6f} | "
            f"{m['budget_gap']:.6f}"
        )
    print(f"summary_json: {out_json}")
    print("status: ablation studies succeeded")


if __name__ == "__main__":
    main()

