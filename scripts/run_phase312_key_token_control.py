#!/usr/bin/env python3
"""Task 3.12: same-image multi-query control experiment for key-token protection."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qacr.routing import AttentionLevelRouter, soft_routing_probs


QUERIES = [
    ("left_focus", "请只关注左边蓝色区域，右边可以忽略。", "left_focus"),
    ("right_focus", "请只关注右边黄色区域，左边可以忽略。", "right_focus"),
    ("bottom_text", "请重点识别底部文字，其他几何区域降算力。", "bottom_text"),
    ("count_query", "请数清左右主体个数，并忽略背景。", "left_right_dual"),
    ("relation_query", "请判断左右两侧目标的关系。", "left_right_dual"),
    ("center_focus", "只看中心区域，不要关注左右边缘。", "center_focus"),
]


@dataclass
class MethodEval:
    name: str
    per_case: list[dict]
    aggregate: dict
    deep_maps: list[torch.Tensor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="outputs/demo_phase01.png")
    parser.add_argument("--coarse-grid", type=int, default=14)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--router-hidden", type=int, default=96)
    parser.add_argument("--budget", type=float, default=0.45)
    parser.add_argument("--temp-start", type=float, default=1.5)
    parser.add_argument("--temp-end", type=float, default=0.35)
    parser.add_argument("--deep-threshold", type=float, default=0.33)
    parser.add_argument("--skip-threshold", type=float, default=0.50)
    parser.add_argument(
        "--summary-json", default="outputs/phase312_key_token_control_summary.json"
    )
    return parser.parse_args()


def encode_query_tokens(
    text: str, device: torch.device, dim: int = 32, tlen: int = 8
) -> torch.Tensor:
    q = text
    feat = torch.zeros(dim, dtype=torch.float32)
    feat[0] = 1.0 if "左" in q else 0.0
    feat[1] = 1.0 if "右" in q else 0.0
    feat[2] = 1.0 if "中心" in q else 0.0
    feat[3] = 1.0 if ("底" in q or "文字" in q) else 0.0
    feat[4] = 1.0 if ("数" in q or "个" in q) else 0.0
    feat[5] = 1.0 if ("关系" in q or "同时" in q) else 0.0
    feat[6] = 1.0 if "忽略" in q else 0.0
    feat[7] = float(len(q) / 40.0)
    feat[8] = float(sum(ord(c) for c in q) % 97) / 97.0
    feat[9] = 1.0
    for i in range(10, dim):
        feat[i] = math.sin((i + 1) * feat[8] * 0.7)
    tokens = feat.unsqueeze(0).repeat(tlen, 1)
    pos = torch.linspace(0.0, 1.0, tlen).unsqueeze(1)
    tokens = tokens + 0.05 * pos * torch.tanh(tokens)
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


def temperature(step: int, total_steps: int, t_start: float, t_end: float) -> float:
    if total_steps <= 1:
        return t_end
    alpha = min(max(step / (total_steps - 1), 0.0), 1.0)
    return float(t_start + alpha * (t_end - t_start))


def infer_mode_from_query(query_name: str) -> str:
    if "left" in query_name:
        return "left_focus"
    if "right" in query_name:
        return "right_focus"
    if "text" in query_name:
        return "bottom_text"
    if "center" in query_name:
        return "center_focus"
    return "left_right_dual"


def budgeted_route_from_scores(
    scores: torch.Tensor, budget: float, use_shallow: bool
) -> torch.Tensor:
    t = scores.numel()
    route = torch.zeros(t, 3, dtype=torch.float32, device=scores.device)
    route[:, 0] = 1.0
    order = torch.argsort(scores, descending=True)

    if use_shallow:
        deep_ratio = min(max(budget * 0.60, 0.0), 0.55)
        shallow_ratio = min(max((budget - deep_ratio) / 0.35, 0.0), 1.0 - deep_ratio)
    else:
        deep_ratio = min(max(budget, 0.0), 1.0)
        shallow_ratio = 0.0

    k_deep = int(round(t * deep_ratio))
    k_shallow = int(round(t * shallow_ratio))
    k_deep = max(0, min(k_deep, t))
    k_shallow = max(0, min(k_shallow, t - k_deep))

    if k_deep > 0:
        idx_deep = order[:k_deep]
        route[idx_deep] = torch.tensor([0.0, 0.0, 1.0], device=scores.device)
    if k_shallow > 0:
        idx_shallow = order[k_deep : k_deep + k_shallow]
        route[idx_shallow] = torch.tensor([0.0, 1.0, 0.0], device=scores.device)
    return route.unsqueeze(0)


def lvpruning_score(grid: int, mode: str, image_tokens: torch.Tensor) -> torch.Tensor:
    yy, xx = mesh_xy(grid, image_tokens.device)
    sal = image_tokens[0].norm(dim=-1).reshape(grid, grid)
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-6)
    if mode == "left_focus":
        rel = 1.0 - xx
    elif mode == "right_focus":
        rel = xx
    elif mode == "center_focus":
        rel = torch.exp(-((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / 0.18)
    elif mode == "bottom_text":
        rel = (yy > 0.58).float() * (0.3 + 0.7 * (1.0 - (xx - 0.35).abs()))
    else:
        rel = ((xx < 0.35) | (xx > 0.65)).float()
    return (0.65 * rel + 0.35 * sal).reshape(-1)


def crop_like_route(
    grid: int, mode: str, budget: float, device: torch.device
) -> torch.Tensor:
    yy, xx = mesh_xy(grid, device)
    if mode == "left_focus":
        core = (xx < 0.42).float()
        ring = ((xx >= 0.42) & (xx < 0.56)).float()
    elif mode == "right_focus":
        core = (xx > 0.58).float()
        ring = ((xx <= 0.58) & (xx > 0.44)).float()
    elif mode == "center_focus":
        dist = torch.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2)
        core = (dist < 0.24).float()
        ring = ((dist >= 0.24) & (dist < 0.36)).float()
    elif mode == "bottom_text":
        core = ((yy > 0.66) & (xx > 0.10) & (xx < 0.56)).float()
        ring = ((yy > 0.56) & (yy <= 0.66)).float()
    else:
        core = ((xx < 0.28) | (xx > 0.72)).float()
        ring = ((xx >= 0.28) & (xx < 0.36) | ((xx > 0.64) & (xx <= 0.72))).float()

    route = torch.zeros(grid * grid, 3, device=device)
    route[:, 0] = 1.0
    core_flat = core.reshape(-1)
    ring_flat = ring.reshape(-1)
    route[core_flat > 0.5] = torch.tensor([0.0, 0.0, 1.0], device=device)
    route[ring_flat > 0.5] = torch.tensor([0.0, 1.0, 0.0], device=device)

    expected = (route[:, 1] * 0.35 + route[:, 2]).mean().item()
    if expected < budget:
        shortage = budget - expected
        add = int(round(shortage * grid * grid))
        skip_idx = torch.nonzero(route[:, 0] > 0.5, as_tuple=False).flatten()
        if skip_idx.numel() > 0 and add > 0:
            add = min(add, int(skip_idx.numel()))
            route[skip_idx[:add]] = torch.tensor([0.0, 1.0, 0.0], device=device)
    return route.unsqueeze(0).to(torch.float32)


def compute_case_metrics(
    route_probs: torch.Tensor,
    key_mask: torch.Tensor,
    deep_threshold: float,
    skip_threshold: float,
) -> dict:
    deep = route_probs[0, :, 2]
    skip = route_probs[0, :, 0]
    deep_pred = deep > deep_threshold
    key = key_mask[0, :, 0] > 0.5
    nonkey = ~key

    if key.any():
        key_recall = float(deep_pred[key].float().mean())
        miss_rate = float((deep[key] < deep_threshold).float().mean())
        early_skip = float((skip[key] > skip_threshold).float().mean())
        key_deep_mean = float(deep[key].mean())
    else:
        key_recall = 0.0
        miss_rate = 0.0
        early_skip = 0.0
        key_deep_mean = 0.0

    if deep_pred.any():
        deep_precision = float(key[deep_pred].float().mean())
    else:
        deep_precision = 0.0

    nonkey_deep_mean = float(deep[nonkey].mean()) if nonkey.any() else 0.0
    separation = key_deep_mean - nonkey_deep_mean
    flagged = bool((miss_rate > 0.40) or (early_skip > 0.25) or (separation <= 0.0))

    return {
        "key_token_recall": key_recall,
        "deep_route_precision": deep_precision,
        "deep_route_recall": key_recall,
        "miss_rate_key_tokens": miss_rate,
        "early_skip_rate_key_tokens": early_skip,
        "separation_key_minus_nonkey": separation,
        "flagged_as_error": flagged,
    }


def pairwise_shift_stats(
    deep_maps: list[torch.Tensor], masks: list[torch.Tensor]
) -> dict:
    pred_shift = []
    tgt_shift = []
    for i in range(len(deep_maps)):
        for j in range(i + 1, len(deep_maps)):
            pred_shift.append(float((deep_maps[i] - deep_maps[j]).abs().mean()))
            tgt_shift.append(float((masks[i] - masks[j]).abs().mean()))
    pred_t = torch.tensor(pred_shift, dtype=torch.float32)
    tgt_t = torch.tensor(tgt_shift, dtype=torch.float32)
    pred_center = pred_t - pred_t.mean()
    tgt_center = tgt_t - tgt_t.mean()
    denom = pred_center.norm() * tgt_center.norm()
    corr = (
        float((pred_center * tgt_center).sum() / denom) if float(denom) > 1e-8 else 0.0
    )
    return {
        "same_image_different_query_shift_l1_mean": float(pred_t.mean()),
        "same_image_different_query_shift_l1_max": float(pred_t.max()),
        "same_image_different_query_consistency_corr": corr,
    }


def train_qacr_router(
    image_tokens: torch.Tensor,
    query_tokens: dict[str, torch.Tensor],
    query_modes: dict[str, str],
    args: argparse.Namespace,
) -> AttentionLevelRouter:
    router = AttentionLevelRouter(
        query_dim=next(iter(query_tokens.values())).size(-1),
        image_dim=image_tokens.size(-1),
        hidden_dim=args.router_hidden,
    ).to(image_tokens.device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=args.lr)
    route_costs = torch.tensor(
        [0.0, 0.35, 1.0], dtype=torch.float32, device=image_tokens.device
    )

    for step in range(args.steps):
        q_name, _, _ = QUERIES[step % len(QUERIES)]
        mode = query_modes[q_name]
        target = build_target_route(
            make_focus_mask(args.coarse_grid, mode, image_tokens.device)
        )
        out = router(query_tokens=query_tokens[q_name], image_tokens=image_tokens)
        temp = temperature(step, args.steps, args.temp_start, args.temp_end)
        soft = soft_routing_probs(out.logits, temperature=temp, use_gumbel=True)
        task_loss = F.mse_loss(soft, target)
        expected = (soft * route_costs.view(1, 1, -1)).sum(dim=-1).mean()
        budget_loss = (expected - args.budget) ** 2
        entropy = -(soft * soft.clamp_min(1e-8).log()).sum(dim=-1).mean()
        loss = task_loss + 0.8 * budget_loss - 0.02 * entropy
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return router


def evaluate_methods(
    image_tokens: torch.Tensor,
    query_tokens: dict[str, torch.Tensor],
    query_modes: dict[str, str],
    router: AttentionLevelRouter,
    args: argparse.Namespace,
) -> dict:
    grid = args.coarse_grid
    device = image_tokens.device
    saliency = image_tokens[0].norm(dim=-1)
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-6)

    target_masks = [
        make_focus_mask(grid, query_modes[q_name], device)[0, :, 0]
        for q_name, _, _ in QUERIES
    ]

    method_store: dict[str, MethodEval] = {}
    for method in ["QACR-Attention", "TokenPruning", "LVPruning-like", "CROP-like"]:
        method_store[method] = MethodEval(
            name=method, per_case=[], aggregate={}, deep_maps=[]
        )

    for q_name, _, _ in QUERIES:
        mode = query_modes[q_name]
        key_mask = make_focus_mask(grid, mode, device)

        out = router(query_tokens=query_tokens[q_name], image_tokens=image_tokens)
        qacr_probs = soft_routing_probs(
            out.logits, temperature=args.temp_end, use_gumbel=False
        )

        tp_probs = budgeted_route_from_scores(
            saliency, budget=args.budget, use_shallow=False
        )
        lv_scores = lvpruning_score(grid, mode, image_tokens)
        lv_probs = budgeted_route_from_scores(
            lv_scores, budget=args.budget, use_shallow=False
        )
        crop_probs = crop_like_route(grid, mode, budget=args.budget, device=device)

        per_method_probs = {
            "QACR-Attention": qacr_probs,
            "TokenPruning": tp_probs,
            "LVPruning-like": lv_probs,
            "CROP-like": crop_probs,
        }
        for method_name, probs in per_method_probs.items():
            case_metrics = compute_case_metrics(
                route_probs=probs,
                key_mask=key_mask,
                deep_threshold=args.deep_threshold,
                skip_threshold=args.skip_threshold,
            )
            case_metrics["query"] = q_name
            method_store[method_name].per_case.append(case_metrics)
            method_store[method_name].deep_maps.append(probs[0, :, 2].detach().cpu())

    for method_name, record in method_store.items():
        n = len(record.per_case)
        agg = {
            "num_cases": n,
            "num_flagged_errors": int(
                sum(int(x["flagged_as_error"]) for x in record.per_case)
            ),
            "key_token_recall": float(
                sum(x["key_token_recall"] for x in record.per_case) / n
            ),
            "deep_route_precision": float(
                sum(x["deep_route_precision"] for x in record.per_case) / n
            ),
            "deep_route_recall": float(
                sum(x["deep_route_recall"] for x in record.per_case) / n
            ),
            "miss_rate_key_tokens": float(
                sum(x["miss_rate_key_tokens"] for x in record.per_case) / n
            ),
            "separation_key_minus_nonkey": float(
                sum(x["separation_key_minus_nonkey"] for x in record.per_case) / n
            ),
        }
        shift = pairwise_shift_stats(record.deep_maps, target_masks)
        record.aggregate = {**agg, **shift}

    return {
        k: {
            "aggregate": v.aggregate,
            "cases": v.per_case,
        }
        for k, v in method_store.items()
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
        q_name: encode_query_tokens(query, device=device)
        for q_name, query, _ in QUERIES
    }
    query_modes = {
        q_name: infer_mode_from_query(q_name if mode is None else mode)
        for q_name, _, mode in QUERIES
    }

    router = train_qacr_router(
        image_tokens=image_tokens,
        query_tokens=query_tokens,
        query_modes=query_modes,
        args=args,
    )
    eval_summary = evaluate_methods(
        image_tokens=image_tokens,
        query_tokens=query_tokens,
        query_modes=query_modes,
        router=router,
        args=args,
    )

    qacr_flagged = eval_summary["QACR-Attention"]["aggregate"]["num_flagged_errors"]
    gate = bool(qacr_flagged <= 2)

    summary = {
        "task": "3.12_same_image_multi_query_key_token_control",
        "image": str(image_path),
        "coarse_grid": args.coarse_grid,
        "steps": args.steps,
        "budget": args.budget,
        "metrics_definition": {
            "key_token_recall": "fraction of key tokens routed to deep",
            "deep_route_precision": "fraction of deep-routed tokens that are key",
            "deep_route_recall": "same as key_token_recall",
            "miss_rate_key_tokens": "fraction of key tokens whose deep prob < threshold",
            "same_image_different_query_consistency_corr": "correlation between route-shift and target mask shift",
        },
        "methods": eval_summary,
        "pass_gate": gate,
    }

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("===== Phase 3.12 Key-Token Control =====")
    print(
        "method | flagged | key_recall | deep_precision | miss_rate | separation | shift_corr"
    )
    for method_name, block in eval_summary.items():
        a = block["aggregate"]
        print(
            f"{method_name} | {a['num_flagged_errors']} | {a['key_token_recall']:.6f} | "
            f"{a['deep_route_precision']:.6f} | {a['miss_rate_key_tokens']:.6f} | "
            f"{a['separation_key_minus_nonkey']:.6f} | "
            f"{a['same_image_different_query_consistency_corr']:.6f}"
        )
    print(f"pass_gate(num_flagged<=2): {gate}")
    print(f"summary_json: {summary_json}")


if __name__ == "__main__":
    main()
