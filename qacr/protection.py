"""Query-conditioned key-token protection utilities for QACR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from qacr.analysis import classify_question_type


SENSITIVE_PROFILES: dict[str, dict[str, float]] = {
    "numeric_time": {
        "topk_ratio": 0.22,
        "keep_ratio": 0.12,
        "deep_ratio": 0.08,
        "deep_bonus": 0.55,
        "keep_bonus": 0.24,
        "skip_penalty": 0.20,
    },
    "direct_reading": {
        "topk_ratio": 0.18,
        "keep_ratio": 0.10,
        "deep_ratio": 0.06,
        "deep_bonus": 0.46,
        "keep_bonus": 0.20,
        "skip_penalty": 0.16,
    },
    "name_entity": {
        "topk_ratio": 0.16,
        "keep_ratio": 0.09,
        "deep_ratio": 0.05,
        "deep_bonus": 0.40,
        "keep_bonus": 0.18,
        "skip_penalty": 0.14,
    },
    "location": {
        "topk_ratio": 0.16,
        "keep_ratio": 0.08,
        "deep_ratio": 0.04,
        "deep_bonus": 0.34,
        "keep_bonus": 0.16,
        "skip_penalty": 0.12,
    },
    "url_email_address": {
        "topk_ratio": 0.18,
        "keep_ratio": 0.10,
        "deep_ratio": 0.06,
        "deep_bonus": 0.44,
        "keep_bonus": 0.18,
        "skip_penalty": 0.14,
    },
    "document_field": {
        "topk_ratio": 0.18,
        "keep_ratio": 0.10,
        "deep_ratio": 0.06,
        "deep_bonus": 0.46,
        "keep_bonus": 0.20,
        "skip_penalty": 0.14,
    },
}


@dataclass
class ProtectionPlan:
    enabled: bool
    question_type: str
    keep_indices: torch.Tensor
    deep_indices: torch.Tensor
    topk_ratio: float = 0.0
    keep_ratio: float = 0.0
    deep_ratio: float = 0.0
    deep_bonus: float = 0.0
    keep_bonus: float = 0.0
    skip_penalty: float = 0.0


def _normalize_question_type(
    question: str | None,
    provided: str | Sequence[str] | None,
    dataset_name: str,
) -> str:
    if isinstance(provided, str) and provided.strip():
        return provided.strip()
    if isinstance(provided, Sequence):
        for item in provided:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                return text
    return classify_question_type(question or "", dataset_name)


def _ocr_density_scale(ocr_tokens: Sequence[str] | None) -> float:
    if not ocr_tokens:
        return 1.0
    num_tokens = len([tok for tok in ocr_tokens if str(tok).strip()])
    return min(1.45, 1.0 + (num_tokens / 60.0))


def build_single_protection_plan(
    route_probs: torch.Tensor,
    question: str | None,
    question_type: str | Sequence[str] | None,
    dataset_name: str,
    ocr_tokens: Sequence[str] | None = None,
    topk_scale: float = 1.0,
    keep_scale: float = 1.0,
    deep_scale: float = 1.0,
) -> ProtectionPlan:
    if route_probs.ndim != 3 or route_probs.size(0) != 1 or route_probs.size(-1) != 3:
        raise ValueError("route_probs must have shape [1, T, 3] for single-sample protection planning")

    qtype = _normalize_question_type(question, question_type, dataset_name)
    profile = SENSITIVE_PROFILES.get(qtype)
    num_tokens = int(route_probs.size(1))
    device = route_probs.device

    if profile is None or num_tokens <= 0 or dataset_name not in {"textvqa", "docvqa"}:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return ProtectionPlan(False, qtype, empty, empty)

    density = _ocr_density_scale(ocr_tokens)
    topk_ratio = min(profile["topk_ratio"] * topk_scale * density, 0.35)
    keep_ratio = min(profile["keep_ratio"] * keep_scale * density, 0.20)
    deep_ratio = min(profile["deep_ratio"] * deep_scale * density, 0.12)

    topk_count = max(1, int(round(num_tokens * topk_ratio)))
    keep_count = max(1, int(round(num_tokens * keep_ratio)))
    deep_count = max(1, int(round(num_tokens * deep_ratio)))
    topk_count = min(num_tokens, max(topk_count, keep_count))
    keep_count = min(num_tokens, min(keep_count, topk_count))
    deep_count = min(keep_count, deep_count)

    probs = route_probs[0]
    deep_prob = probs[:, 2]
    shallow_prob = probs[:, 1]
    skip_prob = probs[:, 0]
    uncertainty = 1.0 - probs.max(dim=-1).values
    score = 1.15 * deep_prob + 0.55 * shallow_prob + 0.25 * uncertainty - 0.35 * skip_prob

    ranked = torch.topk(score, k=topk_count, dim=0, largest=True).indices
    keep_indices = ranked[:keep_count]

    deep_pool_scores = deep_prob[keep_indices] + 0.35 * uncertainty[keep_indices]
    deep_rank_local = torch.topk(deep_pool_scores, k=deep_count, dim=0, largest=True).indices
    deep_indices = keep_indices[deep_rank_local]

    return ProtectionPlan(
        enabled=True,
        question_type=qtype,
        keep_indices=keep_indices,
        deep_indices=deep_indices,
        topk_ratio=topk_count / max(num_tokens, 1),
        keep_ratio=keep_count / max(num_tokens, 1),
        deep_ratio=deep_count / max(num_tokens, 1),
        deep_bonus=float(profile["deep_bonus"]),
        keep_bonus=float(profile["keep_bonus"]),
        skip_penalty=float(profile["skip_penalty"]),
    )


def apply_protection_logit_bias(
    logits: torch.Tensor,
    plan: ProtectionPlan,
    logit_bias_scale: float = 1.0,
) -> torch.Tensor:
    if not plan.enabled or logits.ndim != 3 or logits.size(0) != 1:
        return logits
    updated = logits.clone()
    if plan.keep_indices.numel() > 0:
        updated[0, plan.keep_indices, 1] += plan.keep_bonus * logit_bias_scale
        updated[0, plan.keep_indices, 0] -= plan.skip_penalty * logit_bias_scale
    if plan.deep_indices.numel() > 0:
        updated[0, plan.deep_indices, 2] += plan.deep_bonus * logit_bias_scale
        updated[0, plan.deep_indices, 0] -= 0.5 * plan.skip_penalty * logit_bias_scale
    return updated


def compute_protection_aux_loss(
    route_probs: torch.Tensor,
    plan: ProtectionPlan,
) -> torch.Tensor:
    if not plan.enabled or route_probs.ndim != 3 or route_probs.size(0) != 1:
        return route_probs.new_zeros(())

    losses: list[torch.Tensor] = []
    probs = route_probs[0]
    if plan.keep_indices.numel() > 0:
        keep_non_skip = (probs[plan.keep_indices, 1] + probs[plan.keep_indices, 2]).clamp(1e-6, 1 - 1e-6)
        target = torch.ones_like(keep_non_skip)
        losses.append(F.binary_cross_entropy(keep_non_skip, target))
    if plan.deep_indices.numel() > 0:
        deep_prob = probs[plan.deep_indices, 2].clamp(1e-6, 1 - 1e-6)
        target = torch.full_like(deep_prob, 0.85)
        losses.append(F.binary_cross_entropy(deep_prob, target))
    if not losses:
        return route_probs.new_zeros(())
    return torch.stack(losses).mean()
