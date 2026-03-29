"""Image-only router baseline for QACR (without query conditioning)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ImageOnlyRoutingOutput:
    logits: torch.Tensor
    route_probs: torch.Tensor
    hard_routes: torch.Tensor


class ImageOnlyRouter(nn.Module):
    """Predict skip/shallow/deep route weights from image tokens only."""

    def __init__(
        self,
        image_dim: int,
        hidden_dim: int = 96,
        num_routes: int = 3,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if num_routes != 3:
            raise ValueError(
                "Image-only router expects exactly 3 routes: skip/shallow/deep"
            )

        self.image_dim = image_dim
        self.hidden_dim = hidden_dim
        self.num_routes = num_routes
        self.temperature = temperature

        self.image_proj = nn.Linear(image_dim, hidden_dim, bias=False)
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_routes),
        )

    def forward(self, image_tokens: torch.Tensor) -> ImageOnlyRoutingOutput:
        if image_tokens.ndim != 3:
            raise ValueError(
                f"image_tokens must be [B, Ti, Di], got shape={tuple(image_tokens.shape)}"
            )

        image_feat = self.image_proj(image_tokens)
        fused = self.fusion_norm(image_feat)
        logits = self.head(fused)

        scaled = logits / max(self.temperature, 1e-6)
        route_probs = torch.softmax(scaled, dim=-1)
        hard_routes = torch.argmax(route_probs, dim=-1)

        return ImageOnlyRoutingOutput(
            logits=logits,
            route_probs=route_probs,
            hard_routes=hard_routes,
        )

