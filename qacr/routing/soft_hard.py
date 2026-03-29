"""Soft-to-hard routing utilities and compute regularization for QACR."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_routing_probs(
    logits: torch.Tensor,
    temperature: float,
    use_gumbel: bool = True,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if use_gumbel:
        return F.gumbel_softmax(logits, tau=temperature, hard=False, dim=-1)
    return torch.softmax(logits / temperature, dim=-1)


def hard_routing_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=-1)


def compute_regularization_loss(
    route_probs: torch.Tensor,
    route_costs: torch.Tensor,
    budget_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if route_probs.ndim != 3 or route_probs.size(-1) != route_costs.numel():
        raise ValueError("route_probs must be [B, T, R] and match route_costs length")
    if not (0.0 <= budget_ratio <= 1.0):
        raise ValueError("budget_ratio must be in [0, 1]")

    expected_compute = (route_probs * route_costs.view(1, 1, -1)).sum(dim=-1).mean()
    budget = torch.tensor(
        budget_ratio, device=route_probs.device, dtype=route_probs.dtype
    )
    loss = torch.square(expected_compute - budget)
    return loss, expected_compute


def linear_temperature(
    step: int, total_steps: int, t_start: float, t_end: float
) -> float:
    if total_steps <= 1:
        return float(t_end)
    alpha = min(max(step / (total_steps - 1), 0.0), 1.0)
    return float(t_start + alpha * (t_end - t_start))
