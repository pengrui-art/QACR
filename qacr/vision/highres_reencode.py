"""High-resolution re-encoding utilities for Phase 3.5 go/no-go validation."""

from __future__ import annotations

import math

import torch
from torch import nn


class HighResReEncoder(nn.Module):
    """Refine a small set of coarse tokens with local high-resolution sub-tokens."""

    def __init__(
        self,
        token_dim: int,
        hidden_dim: int = 96,
        patch_scale: int = 2,
        extra_compute_factor: float = 0.4,
    ) -> None:
        super().__init__()
        if patch_scale < 2:
            raise ValueError("patch_scale must be >= 2")
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.patch_scale = patch_scale
        self.extra_compute_factor = extra_compute_factor

        flat_dim = token_dim * patch_scale * patch_scale
        self.local_refine = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, token_dim),
        )
        self.blend_gate = nn.Sequential(
            nn.Linear(token_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _reshape_local_patches(self, highres_tokens: torch.Tensor, coarse_tokens: torch.Tensor) -> torch.Tensor:
        batch_size, coarse_count, _ = coarse_tokens.shape
        _, highres_count, _ = highres_tokens.shape
        coarse_grid = int(math.sqrt(coarse_count))
        highres_grid = int(math.sqrt(highres_count))
        if coarse_grid * coarse_grid != coarse_count:
            raise ValueError("coarse token count must form a square grid")
        if highres_grid * highres_grid != highres_count:
            raise ValueError("highres token count must form a square grid")
        if highres_grid != coarse_grid * self.patch_scale:
            raise ValueError("highres grid must equal coarse grid * patch_scale")

        highres_map = highres_tokens.view(
            batch_size,
            coarse_grid,
            self.patch_scale,
            coarse_grid,
            self.patch_scale,
            self.token_dim,
        )
        highres_map = highres_map.permute(0, 1, 3, 2, 4, 5).contiguous()
        return highres_map.view(
            batch_size,
            coarse_count,
            self.patch_scale * self.patch_scale * self.token_dim,
        )

    def forward(
        self,
        base_tokens: torch.Tensor,
        highres_tokens: torch.Tensor,
        selection_scores: torch.Tensor,
        selected_ratio: float = 0.15,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if not (0.0 < selected_ratio <= 1.0):
            raise ValueError("selected_ratio must be in (0, 1]")
        if base_tokens.ndim != 3 or highres_tokens.ndim != 3:
            raise ValueError("base_tokens and highres_tokens must be [B, T, D]")
        if selection_scores.ndim != 2:
            raise ValueError("selection_scores must be [B, T]")
        if base_tokens.shape[:2] != selection_scores.shape:
            raise ValueError("selection_scores must match base_tokens on [B, T]")

        batch_size, coarse_count, _ = base_tokens.shape
        local_patches = self._reshape_local_patches(
            highres_tokens=highres_tokens, coarse_tokens=base_tokens
        )
        refined_local = self.local_refine(local_patches)
        gate_input = torch.cat(
            [base_tokens, refined_local, selection_scores.unsqueeze(-1)], dim=-1
        )
        gate = torch.sigmoid(self.blend_gate(gate_input))

        topk = max(1, int(round(coarse_count * selected_ratio)))
        topk_idx = torch.topk(selection_scores, k=topk, dim=-1).indices
        select_mask = torch.zeros(
            batch_size,
            coarse_count,
            dtype=base_tokens.dtype,
            device=base_tokens.device,
        )
        batch_idx = torch.arange(batch_size, device=base_tokens.device).unsqueeze(1)
        select_mask[batch_idx, topk_idx] = 1.0

        refined = base_tokens + select_mask.unsqueeze(-1) * gate * (refined_local - base_tokens)
        selected_ratio_actual = float(select_mask.mean().detach())
        stats = {
            "selected_ratio": selected_ratio_actual,
            "extra_compute_ratio": float(selected_ratio_actual * self.extra_compute_factor),
        }
        return refined, stats
