"""Attention-enhanced router for Phase 3.5 go/no-go validation."""

from __future__ import annotations

import math

import torch
from torch import nn

from .depth_router import RoutingOutput


class AttentionLevelRouter(nn.Module):
    """Add lightweight positional and self-attention refinement before routing."""

    def __init__(
        self,
        query_dim: int,
        image_dim: int,
        hidden_dim: int = 96,
        num_routes: int = 3,
        num_heads: int = 4,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if num_routes != 3:
            raise ValueError("AttentionLevelRouter expects exactly 3 routes")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.query_dim = query_dim
        self.image_dim = image_dim
        self.hidden_dim = hidden_dim
        self.num_routes = num_routes
        self.temperature = temperature

        self.query_proj = nn.Linear(query_dim, hidden_dim, bias=False)
        self.image_proj = nn.Linear(image_dim, hidden_dim, bias=False)
        self.coord_proj = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pre_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.attn_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.post_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_routes),
        )

    def _coord_features(self, num_tokens: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        grid = int(math.sqrt(num_tokens))
        if grid * grid != num_tokens:
            raise ValueError("image token count must form a square grid")
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, grid, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, grid, device=device, dtype=dtype),
            indexing="ij",
        )
        coords = torch.stack([xx, yy, xx * yy, xx.square(), yy.square()], dim=-1)
        return coords.view(1, num_tokens, 5)

    def forward(self, query_tokens: torch.Tensor, image_tokens: torch.Tensor) -> RoutingOutput:
        if query_tokens.ndim != 3:
            raise ValueError("query_tokens must be [B, Tq, Dq]")
        if image_tokens.ndim != 3:
            raise ValueError("image_tokens must be [B, Ti, Di]")
        if query_tokens.size(0) != image_tokens.size(0):
            raise ValueError("batch mismatch between query_tokens and image_tokens")

        query_state = query_tokens.mean(dim=1)
        query_feat = self.query_proj(query_state).unsqueeze(1)
        image_feat = self.image_proj(image_tokens)
        coord_feat = self.coord_proj(
            self._coord_features(
                num_tokens=image_tokens.size(1),
                device=image_tokens.device,
                dtype=image_feat.dtype,
            )
        )
        fused = self.pre_norm(image_feat + query_feat + coord_feat)
        attn_out, _ = self.attn(fused, fused, fused, need_weights=False)
        gate = torch.sigmoid(
            self.attn_gate(torch.cat([fused, query_feat.expand_as(fused)], dim=-1))
        )
        refined = self.post_norm(fused + gate * attn_out)
        logits = self.head(refined)

        scaled = logits / max(self.temperature, 1e-6)
        route_probs = torch.softmax(scaled, dim=-1)
        hard_routes = torch.argmax(route_probs, dim=-1)
        return RoutingOutput(
            logits=logits,
            route_probs=route_probs,
            hard_routes=hard_routes,
        )
