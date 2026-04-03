"""Skip/Shallow/Deep multi-path depth execution for coarse visual tokens."""

from __future__ import annotations

import torch
from torch import nn


class _TokenBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm(x))


class DepthMultiPathExecutor(nn.Module):
    """Execute tokens through three depth branches: skip, shallow, deep.

    - skip branch: identity
    - shallow branch: one token block
    - deep branch: multiple token blocks
    """

    def __init__(
        self,
        token_dim: int,
        hidden_dim: int = 128,
        deep_layers: int = 3,
        output_alpha: float = 1.0,
    ) -> None:
        super().__init__()
        if deep_layers < 1:
            raise ValueError("deep_layers must be >= 1")
        if not (0.0 <= output_alpha <= 1.0):
            raise ValueError("output_alpha must be in [0, 1]")

        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.deep_layers = deep_layers
        # Blend routed output with original visual token to preserve base semantics.
        # 1.0 means full routed output (legacy behavior); lower values are more conservative.
        self.output_alpha = float(output_alpha)

        self.input_proj = nn.Linear(token_dim, hidden_dim)
        self.skip_proj = nn.Identity()
        self.shallow = _TokenBlock(hidden_dim)
        self.deep = nn.Sequential(
            *[_TokenBlock(hidden_dim) for _ in range(deep_layers)]
        )
        self.output_proj = nn.Linear(hidden_dim, token_dim)

    def forward(
        self,
        image_tokens: torch.Tensor,
        route_probs: torch.Tensor,
        route_indices: torch.Tensor | None = None,
        mode: str = "soft",
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if image_tokens.ndim != 3:
            raise ValueError("image_tokens must be [B, Ti, Di]")
        if route_probs.ndim != 3 or route_probs.size(-1) != 3:
            raise ValueError("route_probs must be [B, Ti, 3]")
        if image_tokens.shape[:2] != route_probs.shape[:2]:
            raise ValueError("image_tokens and route_probs must match on [B, Ti]")
        if mode not in {"soft", "hard", "hard_conditional"}:
            raise ValueError("mode must be 'soft', 'hard', or 'hard_conditional'")

        x = self.input_proj(image_tokens)

        if mode == "hard_conditional":
            if route_indices is None:
                route_indices = torch.argmax(route_probs, dim=-1)
            hidden = self._conditional_hidden_forward(x, route_indices.long())
            out = self.output_proj(hidden)
            if self.output_alpha < 1.0:
                out = image_tokens + self.output_alpha * (out - image_tokens)
            w = torch.nn.functional.one_hot(route_indices, num_classes=3).to(
                route_probs.dtype
            )
            route_stats = {
                "skip_ratio": float(w[..., 0].mean().detach()),
                "shallow_ratio": float(w[..., 1].mean().detach()),
                "deep_ratio": float(w[..., 2].mean().detach()),
            }
            return out, route_stats

        skip_out = self.skip_proj(x)
        shallow_out = self.shallow(x)
        deep_out = self.deep(x)

        if mode == "soft":
            w = route_probs
        else:
            if route_indices is None:
                route_indices = torch.argmax(route_probs, dim=-1)
            w = torch.nn.functional.one_hot(route_indices, num_classes=3).to(
                route_probs.dtype
            )

        fused = (
            w[..., 0:1] * skip_out + w[..., 1:2] * shallow_out + w[..., 2:3] * deep_out
        )

        out = self.output_proj(fused)
        if self.output_alpha < 1.0:
            out = image_tokens + self.output_alpha * (out - image_tokens)

        route_stats = {
            "skip_ratio": float(w[..., 0].mean().detach()),
            "shallow_ratio": float(w[..., 1].mean().detach()),
            "deep_ratio": float(w[..., 2].mean().detach()),
        }
        return out, route_stats

    def _conditional_hidden_forward(
        self, hidden_tokens: torch.Tensor, route_indices: torch.Tensor
    ) -> torch.Tensor:
        if route_indices.ndim != 2:
            raise ValueError("route_indices must be [B, Ti]")
        if route_indices.shape != hidden_tokens.shape[:2]:
            raise ValueError("route_indices must match hidden_tokens on [B, Ti]")

        out_hidden = torch.zeros_like(hidden_tokens)
        batch_size = hidden_tokens.size(0)

        for batch_idx in range(batch_size):
            token_hidden = hidden_tokens[batch_idx]
            token_routes = route_indices[batch_idx]
            out_sample = out_hidden[batch_idx]

            skip_idx = torch.nonzero(token_routes == 0, as_tuple=False).flatten()
            shallow_idx = torch.nonzero(token_routes == 1, as_tuple=False).flatten()
            deep_idx = torch.nonzero(token_routes == 2, as_tuple=False).flatten()

            if skip_idx.numel() > 0:
                out_sample.index_copy_(0, skip_idx, token_hidden.index_select(0, skip_idx))
            if shallow_idx.numel() > 0:
                shallow_in = token_hidden.index_select(0, shallow_idx).unsqueeze(0)
                shallow_out = self.shallow(shallow_in).squeeze(0)
                out_sample.index_copy_(0, shallow_idx, shallow_out)
            if deep_idx.numel() > 0:
                deep_in = token_hidden.index_select(0, deep_idx).unsqueeze(0)
                deep_out = self.deep(deep_in).squeeze(0)
                out_sample.index_copy_(0, deep_idx, deep_out)

        return out_hidden
