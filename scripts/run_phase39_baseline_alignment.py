#!/usr/bin/env python3
"""Task 3.9: export external baseline alignment matrix and table placement."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class BaselineSpec:
    name: str
    bucket: str
    family: str
    closest_to_qacr: str
    query_guided: str
    token_pruning: str
    region_compression: str
    layer_skipping: str
    multi_path_depth: str
    explicit_budget: str
    conditional_execution: str
    training_mode: str
    local_alignment: str
    table_placement: str
    reproduction_priority: str
    matched_budgets: list[float]
    source_title: str
    source_url: str
    venue_status: str
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-json",
        default="outputs/phase39_baseline_alignment.json",
    )
    parser.add_argument(
        "--out-md",
        default="outputs/phase39_baseline_alignment.md",
    )
    return parser.parse_args()


def build_specs() -> list[BaselineSpec]:
    budgets = [0.35, 0.45, 0.60]
    return [
        BaselineSpec(
            name="LowRes-9x9",
            bucket="internal",
            family="uniform_input_compression",
            closest_to_qacr="no",
            query_guided="no",
            token_pruning="no",
            region_compression="yes",
            layer_skipping="no",
            multi_path_depth="no",
            explicit_budget="implicit",
            conditional_execution="yes",
            training_mode="training-based",
            local_alignment="scripts/train_low_resolution_baseline.py",
            table_placement="main",
            reproduction_priority="P0",
            matched_budgets=budgets,
            source_title="Internal baseline",
            source_url="",
            venue_status="repo baseline",
            note="Strong compression baseline; mandatory for any honest QACR comparison.",
        ),
        BaselineSpec(
            name="TokenPruning-keep/drop",
            bucket="internal",
            family="heuristic_token_pruning",
            closest_to_qacr="partial",
            query_guided="no",
            token_pruning="yes",
            region_compression="no",
            layer_skipping="no",
            multi_path_depth="no",
            explicit_budget="fixed_keep_ratio",
            conditional_execution="partial",
            training_mode="training-based",
            local_alignment="scripts/train_token_pruning_baseline.py",
            table_placement="main",
            reproduction_priority="P0",
            matched_budgets=budgets,
            source_title="Internal baseline",
            source_url="",
            venue_status="repo baseline",
            note="Nearest generic pruning baseline already available in repo.",
        ),
        BaselineSpec(
            name="ImageOnlyRouting",
            bucket="internal",
            family="image_only_routing",
            closest_to_qacr="partial",
            query_guided="no",
            token_pruning="no",
            region_compression="no",
            layer_skipping="no",
            multi_path_depth="yes",
            explicit_budget="yes",
            conditional_execution="partial",
            training_mode="training-based",
            local_alignment="scripts/train_image_only_routing_baseline.py",
            table_placement="appendix",
            reproduction_priority="P1",
            matched_budgets=budgets,
            source_title="Internal baseline",
            source_url="",
            venue_status="repo baseline",
            note="Needed to isolate the value of query conditioning.",
        ),
        BaselineSpec(
            name="QACR-DepthOnly",
            bucket="internal",
            family="budgeted_multi_path_compute_allocation",
            closest_to_qacr="self",
            query_guided="yes",
            token_pruning="no",
            region_compression="no",
            layer_skipping="no",
            multi_path_depth="yes",
            explicit_budget="yes",
            conditional_execution="partial",
            training_mode="training-based",
            local_alignment="scripts/train_query_adaptive_budget_sweep.py",
            table_placement="main",
            reproduction_priority="P0",
            matched_budgets=budgets,
            source_title="Internal method",
            source_url="",
            venue_status="repo method",
            note="Core method; current novelty rests on compute allocation rather than keep/drop selection.",
        ),
        BaselineSpec(
            name="LVPruning",
            bucket="external",
            family="query_guided_token_pruning",
            closest_to_qacr="high",
            query_guided="yes",
            token_pruning="yes",
            region_compression="no",
            layer_skipping="no",
            multi_path_depth="no",
            explicit_budget="implicit",
            conditional_execution="yes",
            training_mode="plug-in / no backbone modification",
            local_alignment="text_guided_keep_drop_family",
            table_placement="main",
            reproduction_priority="P0",
            matched_budgets=budgets,
            source_title="LVPruning: An Effective yet Simple Language-Guided Vision Token Pruning Approach for Multi-modal Large Language Models",
            source_url="https://aclanthology.org/2025.findings-naacl.242/",
            venue_status="NAACL Findings 2025",
            note="Closest representative of language-guided keep/drop pruning; must be in main table.",
        ),
        BaselineSpec(
            name="CROP",
            bucket="external",
            family="query_relevant_region_compression",
            closest_to_qacr="high",
            query_guided="yes",
            token_pruning="yes",
            region_compression="yes",
            layer_skipping="partial",
            multi_path_depth="no",
            explicit_budget="implicit",
            conditional_execution="partial",
            training_mode="two-stage / mixed",
            local_alignment="region_compression_family",
            table_placement="main",
            reproduction_priority="P0",
            matched_budgets=budgets,
            source_title="CROP: Contextual Region-Oriented Visual Token Pruning",
            source_url="https://aclanthology.org/2025.emnlp-main.492/",
            venue_status="EMNLP 2025",
            note="Most relevant region-aware compression baseline; main-table representative for spatially localized compression.",
        ),
        BaselineSpec(
            name="SPIDER",
            bucket="external",
            family="prune_plus_sub_layer_skip",
            closest_to_qacr="high",
            query_guided="semantic / weak query dependence",
            token_pruning="yes",
            region_compression="no",
            layer_skipping="yes",
            multi_path_depth="partial",
            explicit_budget="implicit",
            conditional_execution="yes",
            training_mode="training-based",
            local_alignment="prune_then_skip_family",
            table_placement="main",
            reproduction_priority="P0",
            matched_budgets=budgets,
            source_title="SPIDER: Multi-Layer Semantic Token Pruning and Adaptive Sub-Layer Skipping in Multimodal Large Language Models",
            source_url="https://openreview.net/forum?id=aGpSK6QH3w",
            venue_status="ICLR 2026 submission",
            note="Best representative for compute-side skipping; main-table foil for the 'not just pruning' claim.",
        ),
        BaselineSpec(
            name="Script",
            bucket="external",
            family="query_conditioned_semantic_pruning",
            closest_to_qacr="high",
            query_guided="yes",
            token_pruning="yes",
            region_compression="no",
            layer_skipping="no",
            multi_path_depth="no",
            explicit_budget="implicit",
            conditional_execution="yes",
            training_mode="training-free",
            local_alignment="text_guided_keep_drop_family_plus_diversity",
            table_placement="appendix",
            reproduction_priority="P1",
            matched_budgets=budgets,
            source_title="Script: Graph-Structured and Query-Conditioned Semantic Token Pruning for Multimodal Large Language Models",
            source_url="https://openreview.net/forum?id=F6xKzbgcHq",
            venue_status="TMLR 2025",
            note="Very relevant but highly correlated with LVPruning-family; better placed in appendix unless it is the strongest reproduced pruning result.",
        ),
        BaselineSpec(
            name="FlashVLM",
            bucket="external",
            family="text_guided_token_selection_plus_diversity",
            closest_to_qacr="high",
            query_guided="yes",
            token_pruning="yes",
            region_compression="no",
            layer_skipping="no",
            multi_path_depth="no",
            explicit_budget="explicit token budget",
            conditional_execution="yes",
            training_mode="training-based",
            local_alignment="text_guided_selection_plus_diversity_family",
            table_placement="appendix",
            reproduction_priority="P2",
            matched_budgets=budgets,
            source_title="FlashVLM: Text-Guided Visual Token Selection for Large Multimodal Models",
            source_url="https://arxiv.org/abs/2512.20561",
            venue_status="arXiv preprint (submitted 2025-12-23)",
            note="Very relevant but still under submission; useful appendix stress test rather than mandatory main-table baseline.",
        ),
        BaselineSpec(
            name="DyRate",
            bucket="external",
            family="dynamic_pruning_ratio_during_generation",
            closest_to_qacr="medium",
            query_guided="no",
            token_pruning="yes",
            region_compression="no",
            layer_skipping="no",
            multi_path_depth="no",
            explicit_budget="dynamic rate schedule",
            conditional_execution="yes",
            training_mode="lightweight predictor",
            local_alignment="dynamic_pruning_ratio_family",
            table_placement="appendix",
            reproduction_priority="P2",
            matched_budgets=budgets,
            source_title="Dynamic Token Reduction during Generation for Vision Language Models",
            source_url="https://arxiv.org/abs/2501.14204",
            venue_status="arXiv preprint (submitted 2025-01-24)",
            note="Useful dynamic-rate reference, but less query-conditioned than QACR and less central than LVPruning/CROP/SPIDER.",
        ),
        BaselineSpec(
            name="TAMP",
            bucket="external",
            family="layerwise_adaptive_pruning",
            closest_to_qacr="medium",
            query_guided="no",
            token_pruning="partial",
            region_compression="no",
            layer_skipping="no",
            multi_path_depth="no",
            explicit_budget="sparsity schedule",
            conditional_execution="no",
            training_mode="post-training pruning",
            local_alignment="related_only_weight_or_layer_pruning",
            table_placement="related_only",
            reproduction_priority="P3",
            matched_budgets=budgets,
            source_title="TAMP: Token-Adaptive Layerwise Pruning in Multimodal Large Language Models",
            source_url="https://aclanthology.org/2025.findings-acl.359/",
            venue_status="ACL Findings 2025",
            note="Important related work, but not the right head-to-head baseline for token-level query-adaptive compute routing.",
        ),
    ]


def specs_to_markdown(specs: list[BaselineSpec]) -> str:
    lines: list[str] = []
    lines.append("# Phase 3.9 Baseline Alignment Matrix")
    lines.append("")
    lines.append("## Main Table Candidates")
    lines.append("")
    lines.append("| Method | Family | Query-guided | Region Compression | Layer Skipping | Multi-path Depth | Budget | Local Alignment | Why Main Table |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for spec in specs:
        if spec.table_placement != "main":
            continue
        lines.append(
            f"| {spec.name} | {spec.family} | {spec.query_guided} | {spec.region_compression} | "
            f"{spec.layer_skipping} | {spec.multi_path_depth} | {spec.explicit_budget} | "
            f"`{spec.local_alignment}` | {spec.note} |"
        )
    lines.append("")
    lines.append("## Appendix / Supplementary Candidates")
    lines.append("")
    lines.append("| Method | Placement Reason | Source |")
    lines.append("|---|---|---|")
    for spec in specs:
        if spec.table_placement != "appendix":
            continue
        source = spec.source_url if spec.source_url else "internal"
        lines.append(f"| {spec.name} | {spec.note} | {source} |")
    lines.append("")
    lines.append("## Related-Only References")
    lines.append("")
    lines.append("| Method | Why Not Direct Baseline |")
    lines.append("|---|---|")
    for spec in specs:
        if spec.table_placement != "related_only":
            continue
        lines.append(f"| {spec.name} | {spec.note} |")
    lines.append("")
    lines.append("## Fair Comparison Protocol")
    lines.append("")
    lines.append("- Backbone: use the same Qwen3.5-VL backbone and the same initial visual tokenization.")
    lines.append("- Budgets: compare at matched budgets `0.35 / 0.45 / 0.60` whenever possible.")
    lines.append("- Metrics: always report task score, compute ratio/FLOPs, latency, and peak memory.")
    lines.append("- Hardware: identical GPU, batch size, and timing protocol.")
    lines.append("- Interpretation: do not mix training-free and training-based baselines without explicit labeling.")
    lines.append("")
    lines.append("## Immediate Recommendation")
    lines.append("")
    lines.append("- Main text: `LowRes`, `TokenPruning`, `LVPruning`, `CROP`, `SPIDER`, `QACR`.")
    lines.append("- Appendix: `ImageOnly`, `Script`, `FlashVLM`, `DyRate`.")
    lines.append("- Related-only: `TAMP` and broader pruning-analysis papers.")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    specs = build_specs()
    payload = {
        "task": "3.9_external_baseline_alignment",
        "matched_budget_points": [0.35, 0.45, 0.60],
        "main_table_recommendation": [
            spec.name for spec in specs if spec.table_placement == "main"
        ],
        "appendix_recommendation": [
            spec.name for spec in specs if spec.table_placement == "appendix"
        ],
        "related_only": [
            spec.name for spec in specs if spec.table_placement == "related_only"
        ],
        "baselines": [asdict(spec) for spec in specs],
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(specs_to_markdown(specs), encoding="utf-8")

    print("===== Phase 3.9 Baseline Alignment =====")
    print("main_table:", ", ".join(payload["main_table_recommendation"]))
    print("appendix:", ", ".join(payload["appendix_recommendation"]))
    print("related_only:", ", ".join(payload["related_only"]))
    print(f"saved_json: {out_json}")
    print(f"saved_md: {out_md}")


if __name__ == "__main__":
    main()
