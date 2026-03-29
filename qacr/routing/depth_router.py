"""Lightweight depth-only router for Query-Adaptive Compute Routing (QACR)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class RoutingOutput:
    logits: torch.Tensor
    route_probs: torch.Tensor
    hard_routes: torch.Tensor


class DepthOnlyRouter(nn.Module):
    """Predict route weights for {skip, shallow, deep} per image token.

    Inputs:
    - query_tokens: [B, Tq, Dq]
    - image_tokens: [B, Ti, Di] (coarse visual tokens)
    """

    def __init__(
        self,
        query_dim: int,
        image_dim: int,
        hidden_dim: int = 96,
        num_routes: int = 3,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if num_routes != 3:
            raise ValueError(
                "Depth-only router expects exactly 3 routes: skip/shallow/deep"
            )

        self.query_dim = query_dim
        self.image_dim = image_dim
        self.hidden_dim = hidden_dim
        self.num_routes = num_routes
        self.temperature = temperature

        self.query_proj = nn.Linear(query_dim, hidden_dim, bias=False)
        self.image_proj = nn.Linear(image_dim, hidden_dim, bias=False)
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_routes),
        )

    def forward(
        self, query_tokens: torch.Tensor, image_tokens: torch.Tensor
    ) -> RoutingOutput:
        if query_tokens.ndim != 3:
            raise ValueError(
                f"query_tokens must be [B, Tq, Dq], got shape={tuple(query_tokens.shape)}"
            )
        if image_tokens.ndim != 3:
            raise ValueError(
                f"image_tokens must be [B, Ti, Di], got shape={tuple(image_tokens.shape)}"
            )
        if query_tokens.size(0) != image_tokens.size(0):
            raise ValueError(
                "Batch size mismatch between query_tokens and image_tokens"
            )

        # Mean-pool query tokens into one global query condition vector per sample.
        query_state = query_tokens.mean(dim=1)

        query_feat = self.query_proj(query_state).unsqueeze(1)  # [B, 1, H]
        image_feat = self.image_proj(image_tokens)  # [B, Ti, H]
        fused = self.fusion_norm(image_feat + query_feat)
        logits = self.head(fused)

        scaled = logits / max(self.temperature, 1e-6)
        route_probs = torch.softmax(scaled, dim=-1)
        hard_routes = torch.argmax(route_probs, dim=-1)

        return RoutingOutput(
            logits=logits, route_probs=route_probs, hard_routes=hard_routes
        )

    def estimate_macs(self, batch_size: int, num_image_tokens: int) -> int:
        """Rough MACs estimate for linear layers in the router."""
        macs_query_proj = batch_size * self.query_dim * self.hidden_dim
        macs_image_proj = (
            batch_size * num_image_tokens * self.image_dim * self.hidden_dim
        )
        macs_head_1 = batch_size * num_image_tokens * self.hidden_dim * self.hidden_dim
        macs_head_2 = batch_size * num_image_tokens * self.hidden_dim * self.num_routes
        return int(macs_query_proj + macs_image_proj + macs_head_1 + macs_head_2)

    def estimate_overhead_ratio(
        self, backbone_macs: float, batch_size: int, num_image_tokens: int
    ) -> float:
        if backbone_macs <= 0:
            raise ValueError("backbone_macs must be positive")
        router_macs = self.estimate_macs(
            batch_size=batch_size, num_image_tokens=num_image_tokens
        )
        return float(router_macs / backbone_macs)
