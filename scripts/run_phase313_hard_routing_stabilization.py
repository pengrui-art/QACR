#!/usr/bin/env python3
"""Task 3.13: hard routing de-collapse and train/infer loop stabilization."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qacr.routing import (
    AttentionLevelRouter,
    compute_regularization_loss,
    hard_routing_from_logits,
    soft_routing_probs,
)
from qacr.vision import DepthMultiPathExecutor


SPECS = [
    ("left_focus", "请只关注左边蓝色区域，右边可以忽略。", "left_focus"),
    ("right_focus", "请只关注右边黄色区域，左边可以忽略。", "right_focus"),
    ("bottom_text", "请重点识别底部文字，其他几何区域降算力。", "bottom_text"),
    ("count_query", "请数清左右主体个数，并忽略背景。", "left_right_dual"),
    ("relation_query", "请判断左右两侧目标的关系。", "left_right_dual"),
    ("center_focus", "只看中心区域，不要关注左右边缘。", "center_focus"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="outputs/demo_phase01.png")
    parser.add_argument("--coarse-grid", type=int, default=14)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--router-hidden", type=int, default=96)
    parser.add_argument("--budget", type=float, default=0.45)
    parser.add_argument("--temp-start", type=float, default=1.5)
    parser.add_argument("--temp-end", type=float, default=0.20)
    parser.add_argument("--deep-threshold", type=float, default=0.33)
    parser.add_argument(
        "--summary-json", default="outputs/phase313_stabilization_summary.json"
    )
    return parser.parse_args()


def encode_query_tokens(
    text: str, device: torch.device, dim: int = 32, tlen: int = 8
) -> torch.Tensor:
    feat = torch.zeros(dim, dtype=torch.float32)
    feat[0] = 1.0 if "左" in text else 0.0
    feat[1] = 1.0 if "右" in text else 0.0
    feat[2] = 1.0 if "中心" in text else 0.0
    feat[3] = 1.0 if ("底" in text or "文字" in text) else 0.0
    feat[4] = 1.0 if ("数" in text or "个" in text) else 0.0
    feat[5] = 1.0 if ("关系" in text or "同时" in text) else 0.0
    feat[6] = 1.0 if "忽略" in text else 0.0
    feat[7] = float(len(text) / 40.0)
    feat[8] = float(sum(ord(c) for c in text) % 89) / 89.0
    feat[9] = 1.0
    for i in range(10, dim):
        feat[i] = math.sin((i + 1) * feat[8] * 0.8)
    tokens = feat.unsqueeze(0).repeat(tlen, 1)
    pos = torch.linspace(0.0, 1.0, tlen).unsqueeze(1)
    tokens = tokens + 0.04 * pos * torch.tanh(tokens)
    return tokens.unsqueeze(0).to(device)


def load_coarse_tokens(
    image_path: Path, grid: int, device: torch.device
) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    tfm = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor()])
    pixel = tfm(image).unsqueeze(0).to(device)
    pooled = F.adaptive_avg_pool2d(pixel, output_size=(grid, grid))
    return pooled.flatten(2).transpose(1, 2).contiguous()


def mesh_xy(grid: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, grid, device=device),
        torch.linspace(0.0, 1.0, grid, device=device),
        indexing="ij",
    )
    return yy, xx


def make_focus_mask(grid: int, mode: str, device: torch.device) -> torch.Tensor:
    yy, xx = mesh_xy(grid, device)
    if mode == "left_focus":
        mask = (xx < 0.45).float()
    elif mode == "right_focus":
        mask = (xx > 0.55).float()
    elif mode == "center_focus":
        dist = torch.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2)
        mask = (dist < 0.28).float()
    elif mode == "bottom_text":
        mask = ((yy > 0.62) & (xx > 0.05) & (xx < 0.6)).float()
    elif mode == "left_right_dual":
        mask = ((xx < 0.30) | (xx > 0.70)).float()
    else:
        raise ValueError(f"Unknown mode: {mode}")
    if float(mask.mean()) < 0.05:
        mask[grid // 2, grid // 2] = 1.0
    return mask.reshape(1, grid * grid, 1)


def build_target_route(mask: torch.Tensor) -> torch.Tensor:
    focus_probs = torch.tensor(
        [0.05, 0.15, 0.80], dtype=mask.dtype, device=mask.device
    ).view(1, 1, 3)
    nonfocus_probs = torch.tensor(
        [0.55, 0.35, 0.10], dtype=mask.dtype, device=mask.device
    ).view(1, 1, 3)
    return mask * focus_probs + (1.0 - mask) * nonfocus_probs


def temp_linear(step: int, total_steps: int, t_start: float, t_end: float) -> float:
    if total_steps <= 1:
        return float(t_end)
    alpha = min(max(step / (total_steps - 1), 0.0), 1.0)
    return float(t_start + alpha * (t_end - t_start))


def budget_curriculum(step: int, total_steps: int, target_budget: float) -> float:
    warm_budget = min(0.70, max(target_budget + 0.20, target_budget))
    if total_steps <= 1:
        return target_budget
    alpha = min(max(step / (total_steps - 1), 0.0), 1.0)
    return float(warm_budget + alpha * (target_budget - warm_budget))


def budget_matched_hard_from_logits(
    logits: torch.Tensor, budget: float
) -> torch.Tensor:
    """Convert logits to hard routes while matching compute budget approximately."""
    bsz, tokens, _ = logits.shape
    hard = torch.zeros(bsz, tokens, 3, dtype=torch.float32, device=logits.device)
    hard[..., 0] = 1.0

    deep_ratio = min(max(0.62 * budget, 0.0), 0.55)
    shallow_ratio = min(max((budget - deep_ratio) / 0.35, 0.0), 1.0 - deep_ratio)
    k_deep = int(round(tokens * deep_ratio))
    k_shallow = int(round(tokens * shallow_ratio))

    deep_scores = logits[..., 2]
    shallow_scores = logits[..., 1]

    for b in range(bsz):
        deep_rank = torch.argsort(deep_scores[b], descending=True)
        if k_deep > 0:
            deep_idx = deep_rank[:k_deep]
            hard[b, deep_idx] = torch.tensor([0.0, 0.0, 1.0], device=logits.device)

        if k_shallow > 0:
            remain = deep_rank[k_deep:]
            remain_scores = shallow_scores[b, remain]
            shallow_rank = torch.argsort(remain_scores, descending=True)
            shallow_idx = remain[shallow_rank[:k_shallow]]
            hard[b, shallow_idx] = torch.tensor([0.0, 1.0, 0.0], device=logits.device)
    return hard


def target_route_mix_from_budget(budget: float) -> torch.Tensor:
    deep_ratio = min(max(0.62 * budget, 0.0), 0.55)
    shallow_ratio = min(max((budget - deep_ratio) / 0.35, 0.0), 1.0 - deep_ratio)
    skip_ratio = max(0.0, 1.0 - deep_ratio - shallow_ratio)
    return torch.tensor([skip_ratio, shallow_ratio, deep_ratio], dtype=torch.float32)


def has_nonfinite_grad(module: torch.nn.Module) -> bool:
    for p in module.parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            return True
    return False


def train_once(
    image_tokens: torch.Tensor,
    query_tokens: dict[str, torch.Tensor],
    query_modes: dict[str, str],
    args: argparse.Namespace,
    stabilized: bool,
) -> tuple[AttentionLevelRouter, bool]:
    router = AttentionLevelRouter(
        query_dim=next(iter(query_tokens.values())).size(-1),
        image_dim=image_tokens.size(-1),
        hidden_dim=args.router_hidden,
    ).to(image_tokens.device)
    executor = DepthMultiPathExecutor(token_dim=image_tokens.size(-1)).to(
        image_tokens.device
    )
    optimizer = torch.optim.AdamW(
        list(router.parameters()) + list(executor.parameters()),
        lr=args.lr,
    )
    route_costs = torch.tensor(
        [0.0, 0.35, 1.0], dtype=torch.float32, device=image_tokens.device
    )
    finite_gradients_all_steps = True

    for step in range(args.steps):
        q_name, _, _ = SPECS[step % len(SPECS)]
        mode = query_modes[q_name]
        key_mask = make_focus_mask(args.coarse_grid, mode, image_tokens.device)
        target_route = build_target_route(key_mask)

        out = router(query_tokens=query_tokens[q_name], image_tokens=image_tokens)
        if stabilized:
            temp = temp_linear(step, args.steps, args.temp_start, args.temp_end)
            budget = budget_curriculum(step, args.steps, args.budget)
        else:
            temp = 1.20
            budget = args.budget

        soft = soft_routing_probs(out.logits, temperature=temp, use_gumbel=True)
        routed, _ = executor(image_tokens=image_tokens, route_probs=soft, mode="soft")
        task_loss = F.mse_loss(routed, image_tokens)

        compute_loss, _ = compute_regularization_loss(
            route_probs=soft,
            route_costs=route_costs,
            budget_ratio=budget,
        )

        if stabilized:
            mean_route = soft.mean(dim=(0, 1))
            target_mix = target_route_mix_from_budget(budget).to(
                device=soft.device, dtype=soft.dtype
            )
            load_balance_loss = F.mse_loss(mean_route, target_mix)
            collapse_penalty = torch.relu(mean_route.max() - 0.70) ** 2
            entropy = -(soft * soft.clamp_min(1e-8).log()).sum(dim=-1).mean()
            key_aux_loss = F.mse_loss(soft, target_route)
            hard_one_hot = budget_matched_hard_from_logits(
                out.logits, budget=budget
            ).to(soft.dtype)
            consistency_loss = F.mse_loss(soft, hard_one_hot.detach())

            total_loss = (
                task_loss
                + 2.40 * compute_loss
                + 0.70 * load_balance_loss
                + 0.50 * collapse_penalty
                + 0.20 * key_aux_loss
                + 0.20 * consistency_loss
                - 0.03 * entropy
            )
        else:
            total_loss = task_loss + 0.80 * compute_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        finite_gradients_all_steps = finite_gradients_all_steps and (
            not has_nonfinite_grad(router)
        )
        finite_gradients_all_steps = finite_gradients_all_steps and (
            not has_nonfinite_grad(executor)
        )
        finite_gradients_all_steps = finite_gradients_all_steps and math.isfinite(
            float(total_loss.detach())
        )
        optimizer.step()

    return router, finite_gradients_all_steps


def evaluate_router(
    router: AttentionLevelRouter,
    image_tokens: torch.Tensor,
    query_tokens: dict[str, torch.Tensor],
    query_modes: dict[str, str],
    args: argparse.Namespace,
    use_budget_hard: bool,
) -> dict:
    soft_hard_gaps = []
    hard_ratios = []
    key_recalls = []
    misses = []
    proxy_correct = 0

    for q_name, _, _ in SPECS:
        mode = query_modes[q_name]
        key_mask = make_focus_mask(args.coarse_grid, mode, image_tokens.device)
        target = build_target_route(key_mask)

        out = router(query_tokens=query_tokens[q_name], image_tokens=image_tokens)
        soft = soft_routing_probs(
            out.logits, temperature=args.temp_end, use_gumbel=False
        )
        if use_budget_hard:
            hard = budget_matched_hard_from_logits(out.logits, budget=args.budget).to(
                soft.dtype
            )
        else:
            hard_idx = hard_routing_from_logits(out.logits)
            hard = F.one_hot(hard_idx, num_classes=3).to(soft.dtype)

        soft_route_mean = soft.mean(dim=(0, 1))
        hard_route_mean = hard.mean(dim=(0, 1))
        soft_hard_gaps.append(
            float(torch.abs(hard_route_mean - soft_route_mean).mean().detach())
        )

        route_mean = hard.mean(dim=(0, 1))
        hard_ratios.append(route_mean)

        key = key_mask[0, :, 0] > 0.5
        deep_hard = hard[0, :, 2]
        deep_soft = soft[0, :, 2]
        key_recall = float(deep_hard[key].mean()) if key.any() else 0.0
        miss = (
            float((deep_soft[key] < args.deep_threshold).float().mean())
            if key.any()
            else 0.0
        )
        key_recalls.append(key_recall)
        misses.append(miss)

        # Proxy benchmark pass criterion: key tokens are mostly preserved and miss rate is low.
        proxy_correct += int((key_recall >= 0.60) and (miss <= 0.35))

    hard_ratio_mean = torch.stack(hard_ratios, dim=0).mean(dim=0)
    hard_collapse = float(torch.max(hard_ratio_mean))

    return {
        "mean_hard_minus_soft_gap": float(sum(soft_hard_gaps) / len(soft_hard_gaps)),
        "max_hard_minus_soft_gap": float(max(soft_hard_gaps)),
        "hard_collapse_ratio": hard_collapse,
        "final_hard_ratio_skip": float(hard_ratio_mean[0]),
        "final_hard_ratio_shallow": float(hard_ratio_mean[1]),
        "final_hard_ratio_deep": float(hard_ratio_mean[2]),
        "key_token_recall": float(sum(key_recalls) / len(key_recalls)),
        "miss_rate_key_tokens": float(sum(misses) / len(misses)),
        "proxy_benchmark_macro_accuracy": float(proxy_correct / len(SPECS)),
        "num_eval_cases": len(SPECS),
    }


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    summary_json = Path(args.summary_json)
    if not image_path.exists():
        raise FileNotFoundError(f"Image path not found: {image_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tokens = load_coarse_tokens(image_path, args.coarse_grid, device)

    query_tokens = {
        q_name: encode_query_tokens(query, device=device) for q_name, query, _ in SPECS
    }
    query_modes = {q_name: mode for q_name, _, mode in SPECS}

    baseline_router, baseline_grad_ok = train_once(
        image_tokens=image_tokens,
        query_tokens=query_tokens,
        query_modes=query_modes,
        args=args,
        stabilized=False,
    )
    stabilized_router, stabilized_grad_ok = train_once(
        image_tokens=image_tokens,
        query_tokens=query_tokens,
        query_modes=query_modes,
        args=args,
        stabilized=True,
    )

    baseline_metrics = evaluate_router(
        router=baseline_router,
        image_tokens=image_tokens,
        query_tokens=query_tokens,
        query_modes=query_modes,
        args=args,
        use_budget_hard=False,
    )
    stabilized_metrics = evaluate_router(
        router=stabilized_router,
        image_tokens=image_tokens,
        query_tokens=query_tokens,
        query_modes=query_modes,
        args=args,
        use_budget_hard=False,
    )

    deltas = {
        "delta_mean_hard_minus_soft_gap": float(
            stabilized_metrics["mean_hard_minus_soft_gap"]
            - baseline_metrics["mean_hard_minus_soft_gap"]
        ),
        "delta_hard_collapse_ratio": float(
            stabilized_metrics["hard_collapse_ratio"]
            - baseline_metrics["hard_collapse_ratio"]
        ),
        "delta_key_token_recall": float(
            stabilized_metrics["key_token_recall"]
            - baseline_metrics["key_token_recall"]
        ),
        "delta_proxy_benchmark_macro_accuracy": float(
            stabilized_metrics["proxy_benchmark_macro_accuracy"]
            - baseline_metrics["proxy_benchmark_macro_accuracy"]
        ),
    }

    pass_gate = bool(
        stabilized_metrics["mean_hard_minus_soft_gap"] <= 0.02
        and stabilized_metrics["hard_collapse_ratio"]
        < baseline_metrics["hard_collapse_ratio"]
        and (
            stabilized_metrics["key_token_recall"]
            > baseline_metrics["key_token_recall"]
            or stabilized_metrics["proxy_benchmark_macro_accuracy"]
            >= baseline_metrics["proxy_benchmark_macro_accuracy"]
        )
    )

    summary = {
        "task": "3.13_hard_routing_stabilization",
        "image": str(image_path),
        "coarse_grid": args.coarse_grid,
        "steps": args.steps,
        "budget": args.budget,
        "strategies": [
            "route entropy + load balancing regularization",
            "key-token auxiliary supervision",
            "soft-hard consistency loss (distillation style)",
            "budget annealing curriculum",
        ],
        "baseline": {
            **baseline_metrics,
            "finite_gradients_all_steps": baseline_grad_ok,
        },
        "stabilized": {
            **stabilized_metrics,
            "finite_gradients_all_steps": stabilized_grad_ok,
        },
        "deltas": deltas,
        "pass_gate": pass_gate,
    }

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("===== Phase 3.13 Hard Routing Stabilization =====")
    print(
        "setting | mean_gap | hard_collapse | hard_skip | hard_shallow | hard_deep | key_recall | proxy_acc"
    )
    print(
        "baseline | "
        f"{baseline_metrics['mean_hard_minus_soft_gap']:.6f} | "
        f"{baseline_metrics['hard_collapse_ratio']:.6f} | "
        f"{baseline_metrics['final_hard_ratio_skip']:.6f} | "
        f"{baseline_metrics['final_hard_ratio_shallow']:.6f} | "
        f"{baseline_metrics['final_hard_ratio_deep']:.6f} | "
        f"{baseline_metrics['key_token_recall']:.6f} | "
        f"{baseline_metrics['proxy_benchmark_macro_accuracy']:.6f}"
    )
    print(
        "stabilized | "
        f"{stabilized_metrics['mean_hard_minus_soft_gap']:.6f} | "
        f"{stabilized_metrics['hard_collapse_ratio']:.6f} | "
        f"{stabilized_metrics['final_hard_ratio_skip']:.6f} | "
        f"{stabilized_metrics['final_hard_ratio_shallow']:.6f} | "
        f"{stabilized_metrics['final_hard_ratio_deep']:.6f} | "
        f"{stabilized_metrics['key_token_recall']:.6f} | "
        f"{stabilized_metrics['proxy_benchmark_macro_accuracy']:.6f}"
    )
    print(f"pass_gate: {pass_gate}")
    print(f"summary_json: {summary_json}")


if __name__ == "__main__":
    main()
