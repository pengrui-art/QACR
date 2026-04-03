"""QACR routing integration with Qwen3.5-VL via forward hook.

The hook intercepts the visual encoder output, applies query-conditioned
depth routing, and returns the routed embeddings so the rest of the model
forward (token merging → LLM → loss) proceeds unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from qacr.protection import (
    apply_protection_logit_bias,
    build_single_protection_plan,
    compute_protection_aux_loss,
)
from qacr.routing import (
    DepthOnlyRouter,
    compute_regularization_loss,
    soft_routing_probs,
)
from qacr.vision import DepthMultiPathExecutor


# ── Constants ──────────────────────────────────────────────────────────────────
ROUTE_COSTS = torch.tensor([0.0, 0.35, 1.0])  # skip / shallow / deep
VISION_MERGE_SIZE = 2  # Qwen3.5-VL merges 2×2 patches


@dataclass
class RoutingStats:
    """Aggregated stats from one forward pass."""

    budget_loss: torch.Tensor = None
    expected_compute: float = 0.0
    mean_skip: float = 0.0
    mean_shallow: float = 0.0
    mean_deep: float = 0.0
    protection_loss: torch.Tensor = None
    protected_ratio: float = 0.0
    protected_deep_ratio: float = 0.0
    _all_probs: list = field(default_factory=list)


def _tokens_per_image(
    grid_thw: torch.Tensor, merge: int = VISION_MERGE_SIZE
) -> list[int]:
    """Return number of visual tokens per image after patch merging."""
    counts = []
    for row in grid_thw:
        t, h, w = int(row[0]), int(row[1]), int(row[2])
        counts.append(t * (h // merge) * (w // merge))
    return counts


def _module_device(m: nn.Module) -> torch.device:
    """Return the device of the first parameter for a module (or DDP-wrapped module)."""
    return next(m.parameters()).device


def budget_matched_route_indices(
    route_probs: torch.Tensor,
    budget: float,
    route_costs: torch.Tensor = ROUTE_COSTS,
    min_keep_ratio: float = 0.0,
    min_deep_ratio: float = 0.0,
    protected_keep_indices: list[torch.Tensor] | None = None,
    protected_deep_indices: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    """Convert soft route probabilities to hard routes under a compute budget."""
    if route_probs.ndim != 3 or route_probs.size(-1) != 3:
        raise ValueError("route_probs must be [B, T, 3]")
    if not (0.0 <= min_keep_ratio <= 1.0):
        raise ValueError("min_keep_ratio must be in [0, 1]")
    if not (0.0 <= min_deep_ratio <= 1.0):
        raise ValueError("min_deep_ratio must be in [0, 1]")

    route_costs = route_costs.to(device=route_probs.device, dtype=route_probs.dtype)
    batch_size, num_tokens, _ = route_probs.shape
    output = torch.zeros(batch_size, num_tokens, dtype=torch.long, device=route_probs.device)

    shallow_cost = float(route_costs[1].item())
    deep_cost = float(route_costs[2].item())

    for batch_idx in range(batch_size):
        probs = route_probs[batch_idx]
        target_cost = budget * num_tokens
        used_cost = 0.0
        min_keep_tokens = int(round(min_keep_ratio * num_tokens))
        min_deep_tokens = int(round(min_deep_ratio * num_tokens))
        assigned_tokens: set[int] = set()

        keep_hint = None if protected_keep_indices is None else protected_keep_indices[batch_idx]
        deep_hint = None if protected_deep_indices is None else protected_deep_indices[batch_idx]
        if deep_hint is not None and deep_hint.numel() > 0:
            for token_idx in deep_hint.tolist():
                if token_idx in assigned_tokens:
                    continue
                if used_cost + deep_cost > target_cost + 1e-6:
                    break
                output[batch_idx, token_idx] = 2
                assigned_tokens.add(int(token_idx))
                used_cost += deep_cost

        if keep_hint is not None and keep_hint.numel() > 0:
            deep_hint_set = set(deep_hint.tolist()) if deep_hint is not None else set()
            for token_idx in keep_hint.tolist():
                token_idx = int(token_idx)
                if token_idx in assigned_tokens:
                    continue
                prefer_deep = token_idx in deep_hint_set or float(probs[token_idx, 2].item()) >= float(probs[token_idx, 1].item())
                if prefer_deep and used_cost + deep_cost <= target_cost + 1e-6:
                    output[batch_idx, token_idx] = 2
                    used_cost += deep_cost
                elif used_cost + shallow_cost <= target_cost + 1e-6:
                    output[batch_idx, token_idx] = 1
                    used_cost += shallow_cost
                else:
                    continue
                assigned_tokens.add(token_idx)

        if min_deep_tokens > 0:
            deep_order = torch.argsort(probs[:, 2], descending=True)
            deep_assigned = 0
            for token_idx_tensor in deep_order:
                token_idx = int(token_idx_tensor.item())
                if used_cost + deep_cost > target_cost + 1e-6:
                    break
                output[batch_idx, token_idx] = 2
                assigned_tokens.add(token_idx)
                used_cost += deep_cost
                deep_assigned += 1
                if deep_assigned >= min_deep_tokens:
                    break

        current_keep = len(assigned_tokens)
        if min_keep_tokens > current_keep:
            keep_scores = probs[:, 1] + probs[:, 2] - probs[:, 0]
            keep_order = torch.argsort(keep_scores, descending=True)
            for token_idx_tensor in keep_order:
                if current_keep >= min_keep_tokens:
                    break
                token_idx = int(token_idx_tensor.item())
                if token_idx in assigned_tokens:
                    continue
                prefer_deep = float(probs[token_idx, 2].item()) >= float(probs[token_idx, 1].item())
                if prefer_deep and used_cost + deep_cost <= target_cost + 1e-6:
                    output[batch_idx, token_idx] = 2
                    used_cost += deep_cost
                elif used_cost + shallow_cost <= target_cost + 1e-6:
                    output[batch_idx, token_idx] = 1
                    used_cost += shallow_cost
                elif used_cost + deep_cost <= target_cost + 1e-6:
                    output[batch_idx, token_idx] = 2
                    used_cost += deep_cost
                else:
                    continue
                assigned_tokens.add(token_idx)
                current_keep += 1

        candidates = []
        for token_idx in range(num_tokens):
            skip_prob = float(probs[token_idx, 0].item())
            shallow_gain = float(probs[token_idx, 1].item()) - skip_prob
            deep_gain = float(probs[token_idx, 2].item()) - skip_prob
            candidates.append(
                (
                    shallow_gain / max(shallow_cost, 1e-6),
                    shallow_gain,
                    shallow_cost,
                    token_idx,
                    1,
                )
            )
            candidates.append(
                (
                    deep_gain / max(deep_cost, 1e-6),
                    deep_gain,
                    deep_cost,
                    token_idx,
                    2,
                )
            )

        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        for _, _, cost, token_idx, route_id in candidates:
            if token_idx in assigned_tokens:
                continue
            if used_cost + cost > target_cost + 1e-6:
                continue
            output[batch_idx, token_idx] = route_id
            assigned_tokens.add(token_idx)
            used_cost += cost

    return output


class QACRRoutingHook:
    """Callable hook for ``model.model.visual.register_forward_hook``.

    Before calling ``model.forward()``, the caller must set:
    - ``hook.query_embeds``  – [B, Tq, D]  (text embeddings, detached)
    - ``hook.grid_thw``      – [num_images, 3]
    - ``hook.budget``        – float
    - ``hook.temperature``   – float
    """

    def __init__(
        self,
        router: DepthOnlyRouter,
        executor: DepthMultiPathExecutor,
        lambda_compute: float = 0.8,
        lambda_entropy: float = 0.02,
    ) -> None:
        self.router = router
        self.executor = executor
        self.lambda_compute = lambda_compute
        self.lambda_entropy = lambda_entropy

        # These must be set before each forward
        self.query_embeds: torch.Tensor | None = None
        self.grid_thw: torch.Tensor | None = None
        self.budget: float = 0.45
        self.temperature: float = 1.0
        self.hard_inference: bool = False
        self.hard_budget_match: bool = False
        self.min_keep_ratio: float = 0.0
        self.min_deep_ratio: float = 0.0
        self.dataset_name: str = "vqav2"
        self.questions: list[str] | None = None
        self.question_types: list[str] | None = None
        self.ocr_tokens_batch: list[list[str]] | None = None
        self.protection_mode: str = "none"
        self.protection_topk_scale: float = 1.0
        self.protection_keep_scale: float = 1.0
        self.protection_deep_scale: float = 1.0
        self.protection_logit_bias: float = 1.0
        self.lambda_key_token: float = 0.0

        # Filled by the hook
        self.stats = RoutingStats()

    def __call__(self, module: nn.Module, _input: tuple, output):
        """Intercept visual encoder output and apply routing.

        The visual encoder returns ``BaseModelOutputWithPooling`` whose
        ``pooler_output`` holds the merged visual tokens as a flat
        ``[total_visual_tokens, hidden_dim]`` tensor.
        """
        from transformers.modeling_outputs import BaseModelOutputWithPooling

        # Extract the flat visual token tensor
        if isinstance(output, BaseModelOutputWithPooling):
            image_embeds = output.pooler_output  # [total_tokens, D]
        elif isinstance(output, torch.Tensor):
            image_embeds = output
        else:
            # Try dict-like access
            image_embeds = (
                output["pooler_output"] if "pooler_output" in output else output[0]
            )

        device = image_embeds.device
        dtype = image_embeds.dtype

        tokens_per_img = _tokens_per_image(self.grid_thw)
        chunks = image_embeds.split(tokens_per_img, dim=0)

        # Avoid per-token-chunk module migration overhead.
        if _module_device(self.router) != device:
            self.router.to(device)
        if _module_device(self.executor) != device:
            self.executor.to(device)

        routed_chunks: list[torch.Tensor] = []
        all_probs: list[torch.Tensor] = []
        protection_losses: list[torch.Tensor] = []
        protected_ratios: list[float] = []
        protected_deep_ratios: list[float] = []

        for i, chunk in enumerate(chunks):
            # chunk: [N_i, D] → [1, N_i, D]
            img_tokens = chunk.unsqueeze(0).float()

            # Query for this image (mean-pool the corresponding text embeds)
            if self.query_embeds.ndim == 3 and self.query_embeds.size(0) > i:
                q = self.query_embeds[i : i + 1]  # [1, Tq, D]
            else:
                q = (
                    self.query_embeds.unsqueeze(0)
                    if self.query_embeds.ndim == 2
                    else self.query_embeds[:1]
                )

            q = q.float().to(device)

            # Route
            route_out = self.router(query_tokens=q, image_tokens=img_tokens)
            logits = route_out.logits
            base_probs = soft_routing_probs(
                logits,
                temperature=self.temperature,
                use_gumbel=self.router.training,
            )
            question = None if self.questions is None or i >= len(self.questions) else self.questions[i]
            question_type = None if self.question_types is None or i >= len(self.question_types) else self.question_types[i]
            ocr_tokens = None if self.ocr_tokens_batch is None or i >= len(self.ocr_tokens_batch) else self.ocr_tokens_batch[i]
            protection_plan = build_single_protection_plan(
                route_probs=base_probs,
                question=question,
                question_type=question_type,
                dataset_name=self.dataset_name,
                ocr_tokens=ocr_tokens,
                topk_scale=self.protection_topk_scale,
                keep_scale=self.protection_keep_scale,
                deep_scale=self.protection_deep_scale,
            )
            if self.protection_mode in {"prior_only", "prior_aux"}:
                logits = apply_protection_logit_bias(
                    logits,
                    protection_plan,
                    logit_bias_scale=self.protection_logit_bias,
                )
            probs = soft_routing_probs(
                logits,
                temperature=self.temperature,
                use_gumbel=self.router.training,
            )
            if self.protection_mode in {"aux_only", "prior_aux"} and self.lambda_key_token > 0:
                protection_losses.append(compute_protection_aux_loss(probs, protection_plan))
            if protection_plan.enabled:
                protected_ratios.append(float(protection_plan.keep_ratio))
                protected_deep_ratios.append(float(protection_plan.deep_ratio))
            else:
                protected_ratios.append(0.0)
                protected_deep_ratios.append(0.0)
            if self.hard_inference:
                if self.hard_budget_match:
                    route_idx = budget_matched_route_indices(
                        probs,
                        budget=self.budget,
                        route_costs=ROUTE_COSTS,
                        min_keep_ratio=self.min_keep_ratio,
                        min_deep_ratio=self.min_deep_ratio,
                        protected_keep_indices=[protection_plan.keep_indices],
                        protected_deep_indices=[protection_plan.deep_indices],
                    )
                else:
                    route_idx = torch.argmax(probs, dim=-1)
                one_hot = F.one_hot(route_idx, num_classes=3).float()
                routed, _ = self.executor(
                    image_tokens=img_tokens,
                    route_probs=one_hot,
                    route_indices=route_idx,
                    mode="hard_conditional",
                )
                all_probs.append(one_hot)
            else:
                routed, _ = self.executor(
                    image_tokens=img_tokens, route_probs=probs, mode="soft"
                )
                all_probs.append(probs)

            routed_chunks.append(routed.squeeze(0).to(dtype))

        # Concat back to flat tensor
        routed_flat = torch.cat(routed_chunks, dim=0)

        # ── Compute budget regularisation ──────────────────────────────────
        stacked_probs = torch.cat(all_probs, dim=1)  # [1, total_tokens, 3]
        costs = ROUTE_COSTS.to(device=device, dtype=torch.float32)
        budget_loss, expected_compute = compute_regularization_loss(
            stacked_probs, costs, self.budget
        )

        self.stats = RoutingStats(
            budget_loss=budget_loss,
            expected_compute=float(expected_compute.detach()),
            mean_skip=float(stacked_probs[..., 0].mean().detach()),
            mean_shallow=float(stacked_probs[..., 1].mean().detach()),
            mean_deep=float(stacked_probs[..., 2].mean().detach()),
            protection_loss=(
                torch.stack(protection_losses).mean()
                if protection_losses
                else torch.zeros((), device=device, dtype=torch.float32)
            ),
            protected_ratio=(sum(protected_ratios) / max(len(protected_ratios), 1)),
            protected_deep_ratio=(sum(protected_deep_ratios) / max(len(protected_deep_ratios), 1)),
            _all_probs=all_probs,
        )

        # Return the same output type with routed embeddings
        if isinstance(output, BaseModelOutputWithPooling):
            output.pooler_output = routed_flat
            return output
        return routed_flat

    @property
    def routing_loss(self) -> torch.Tensor:
        """Combined budget + entropy loss (to be added to NTP loss)."""
        s = self.stats
        if s.budget_loss is None:
            return torch.tensor(0.0)
        entropy = torch.tensor(0.0, device=s.budget_loss.device)
        for p in s._all_probs:
            entropy = entropy - (p * p.clamp_min(1e-8).log()).sum(-1).mean()
        protection_loss = (
            s.protection_loss.to(s.budget_loss.device)
            if s.protection_loss is not None
            else torch.tensor(0.0, device=s.budget_loss.device)
        )
        return (
            self.lambda_compute * s.budget_loss
            - self.lambda_entropy * entropy
            + self.lambda_key_token * protection_loss
        )


def build_qacr_components(
    hidden_dim: int = 1024,
    router_hidden: int = 128,
    executor_hidden: int = 256,
    deep_layers: int = 3,
    executor_output_alpha: float = 1.0,
    device: str = "cuda",
) -> tuple[DepthOnlyRouter, DepthMultiPathExecutor]:
    """Create Router + Executor matched to Qwen3.5-VL's 1024‑dim visual tokens."""
    router = DepthOnlyRouter(
        query_dim=hidden_dim,
        image_dim=hidden_dim,
        hidden_dim=router_hidden,
    ).to(device)

    executor = DepthMultiPathExecutor(
        token_dim=hidden_dim,
        hidden_dim=executor_hidden,
        deep_layers=deep_layers,
        output_alpha=executor_output_alpha,
    ).to(device)

    return router, executor
