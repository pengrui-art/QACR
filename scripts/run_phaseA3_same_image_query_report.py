#!/usr/bin/env python3
"""Phase A.3: summarize same-image different-query control experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-json",
        type=str,
        default="outputs/phase312_key_token_control_summary.json",
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default="outputs/phaseA_same_image_query/phaseA3_same_image_query_report.md",
    )
    parser.add_argument(
        "--out-png",
        type=str,
        default="outputs/phaseA_same_image_query/phaseA3_same_image_query_report.png",
    )
    return parser.parse_args()


def _safe_method_name(name: str) -> str:
    return (
        name.replace("QACR-Attention", "QACR")
        .replace("LVPruning-like", "LVPruning-like")
        .replace("CROP-like", "CROP-like")
    )


def render_markdown(data: dict) -> str:
    methods = data["methods"]
    lines: list[str] = []
    lines.append("# Phase A.3 Same-Image / Different-Query Control Report")
    lines.append("")
    lines.append("## Experiment Goal")
    lines.append("")
    lines.append("This control experiment answers a specific question:")
    lines.append("")
    lines.append("> On the same image, does the routing pattern change with the query in a way that better tracks query-relevant regions?")
    lines.append("")
    lines.append("## Aggregate Summary")
    lines.append("")
    lines.append("| Method | Flagged Errors | Key Recall | Deep Precision | Miss Rate | Separation | Shift Corr |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for method_name, block in methods.items():
        a = block["aggregate"]
        lines.append(
            f"| {_safe_method_name(method_name)} | {a['num_flagged_errors']} | "
            f"{a['key_token_recall']:.4f} | {a['deep_route_precision']:.4f} | "
            f"{a['miss_rate_key_tokens']:.4f} | {a['separation_key_minus_nonkey']:.4f} | "
            f"{a['same_image_different_query_consistency_corr']:.4f} |"
        )
    lines.append("")
    lines.append("## Main Takeaways")
    lines.append("")

    qacr = methods["QACR-Attention"]["aggregate"]
    token_pruning = methods["TokenPruning"]["aggregate"]
    lvp = methods["LVPruning-like"]["aggregate"]
    crop = methods["CROP-like"]["aggregate"]
    lines.append(
        f"- `QACR` clearly outperforms `TokenPruning` on query-conditioned behavior: "
        f"`key_token_recall {qacr['key_token_recall']:.4f} vs {token_pruning['key_token_recall']:.4f}`, "
        f"`miss_rate {qacr['miss_rate_key_tokens']:.4f} vs {token_pruning['miss_rate_key_tokens']:.4f}`, "
        f"`shift_corr {qacr['same_image_different_query_consistency_corr']:.4f} vs {token_pruning['same_image_different_query_consistency_corr']:.4f}`."
    )
    lines.append(
        f"- `QACR` also has far fewer flagged failures than `TokenPruning` "
        f"(`{qacr['num_flagged_errors']} vs {token_pruning['num_flagged_errors']}`), "
        "which supports the claim that query-conditioned routing is doing something nontrivial."
    )
    lines.append(
        f"- `LVPruning-like` remains strong on this synthetic control and even exceeds `QACR` on "
        f"`shift_corr ({lvp['same_image_different_query_consistency_corr']:.4f})`, "
        "so A.3 should be used as mechanism evidence, not as a blanket claim that QACR dominates every heuristic."
    )
    lines.append(
        f"- `CROP-like` is the upper-bound style oracle on this toy setting "
        f"(`key_token_recall={crop['key_token_recall']:.4f}`), so it is best interpreted as a reference point rather than a real competing method."
    )
    lines.append("")
    lines.append("## Per-Query Cases")
    lines.append("")
    for method_name, block in methods.items():
        lines.append(f"### {_safe_method_name(method_name)}")
        lines.append("")
        lines.append("| Query | Key Recall | Deep Precision | Miss Rate | Early Skip | Separation | Flagged |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for case in block["cases"]:
            lines.append(
                f"| {case['query']} | {case['key_token_recall']:.4f} | {case['deep_route_precision']:.4f} | "
                f"{case['miss_rate_key_tokens']:.4f} | {case['early_skip_rate_key_tokens']:.4f} | "
                f"{case['separation_key_minus_nonkey']:.4f} | {int(case['flagged_as_error'])} |"
            )
        lines.append("")
    lines.append("## Usage Note")
    lines.append("")
    lines.append(
        "This A.3 report is best used as mechanism evidence for `same image, different query`, "
        "not as the final benchmark result. It complements A.1/A.2 by showing that the routing pattern actually shifts with the query."
    )
    lines.append("")
    return "\n".join(lines)


def render_png(data: dict, out_png: Path) -> None:
    import matplotlib.pyplot as plt

    methods = list(data["methods"].keys())
    pretty = [_safe_method_name(name) for name in methods]
    key_recall = [data["methods"][name]["aggregate"]["key_token_recall"] for name in methods]
    shift_corr = [
        data["methods"][name]["aggregate"]["same_image_different_query_consistency_corr"]
        for name in methods
    ]
    flagged = [data["methods"][name]["aggregate"]["num_flagged_errors"] for name in methods]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    colors = ["#1f4e79", "#b23a48", "#6a994e", "#bc6c25"]

    axes[0].bar(pretty, key_recall, color=colors)
    axes[0].set_title("Key-Token Recall")
    axes[0].set_ylim(0, 1.05)
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(pretty, shift_corr, color=colors)
    axes[1].set_title("Query-Shift Consistency Corr")
    axes[1].set_ylim(0, 1.05)
    axes[1].tick_params(axis="x", rotation=20)

    axes[2].bar(pretty, flagged, color=colors)
    axes[2].set_title("Flagged Error Count")
    axes[2].tick_params(axis="x", rotation=20)

    fig.suptitle("Phase A.3 Same-Image / Different-Query Control", fontsize=14)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary_json = Path(args.summary_json)
    out_md = Path(args.out_md)
    out_png = Path(args.out_png)

    with summary_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(data), encoding="utf-8")
    render_png(data, out_png)

    print(f"Saved Markdown report to {out_md}")
    print(f"Saved PNG report to {out_png}")


if __name__ == "__main__":
    main()
