#!/usr/bin/env python3
"""Task 3.8: aggregate go/no-go decision for Phase 4 launch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--highres-json",
        default="outputs/phase36_highres_reencoding_summary.json",
    )
    parser.add_argument(
        "--attention-json",
        default="outputs/phase37_attention_routing_summary.json",
    )
    parser.add_argument(
        "--reference-corner-json",
        default="outputs/phase33_corner_case_summary.json",
    )
    parser.add_argument("--reference-qacr-latency-ms", type=float, default=2.983424)
    parser.add_argument(
        "--out-json",
        default="outputs/phase38_go_no_go_review.json",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required summary not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    highres = load_json(Path(args.highres_json))
    attention = load_json(Path(args.attention_json))
    reference_corner = load_json(Path(args.reference_corner_json))
    out_json = Path(args.out_json)

    highres_perf = bool(
        highres["highres_reencode"]["proxy_task_loss"] < highres["depth_only"]["proxy_task_loss"]
    )
    highres_corner = False
    highres_latency = bool(
        highres["highres_reencode"]["latency_ms"] <= args.reference_qacr_latency_ms * 1.10
    )
    highres_criteria = int(highres_perf) + int(highres_corner) + int(highres_latency)

    attn_perf = bool(
        attention["attention_level"]["eval_proxy_route_loss"]
        < attention["depth_only"]["eval_proxy_route_loss"]
    )
    attn_corner = bool(
        attention["attention_level"]["num_flagged_errors"] < reference_corner["num_flagged_errors"]
        and attention["attention_level"]["mean_miss_rate_key_tokens"]
        < (
            sum(float(case["miss_rate_key_tokens"]) for case in reference_corner["cases"])
            / max(len(reference_corner["cases"]), 1)
        )
    )
    attn_latency = bool(
        attention["attention_level"]["latency_ms"] <= args.reference_qacr_latency_ms * 1.10
    )
    attn_criteria = int(attn_perf) + int(attn_corner) + int(attn_latency)

    if attn_criteria >= 2:
        recommendation = "GO_FOR_ATTENTION_AXIS_ONLY"
        rationale = (
            "attention-level routing satisfies at least two go/no-go criteria and is the more promising single-axis extension."
        )
    elif highres_criteria >= 2:
        recommendation = "GO_FOR_HIGHRES_AXIS_ONLY"
        rationale = (
            "high-resolution re-encoding satisfies at least two go/no-go criteria and is the stronger Phase 4 candidate."
        )
    else:
        recommendation = "NO_GO_CONTINUE_PHASE3"
        rationale = (
            "neither added axis satisfies two go/no-go criteria; strengthen Phase 3 evidence before expanding the system."
        )

    summary = {
        "task": "3.8_phase4_go_no_go_review",
        "highres_candidate": {
            "performance_improved": highres_perf,
            "corner_case_improved": highres_corner,
            "latency_within_10_percent": highres_latency,
            "criteria_met": highres_criteria,
        },
        "attention_candidate": {
            "performance_improved": attn_perf,
            "corner_case_improved": attn_corner,
            "latency_within_10_percent": attn_latency,
            "criteria_met": attn_criteria,
        },
        "reference": {
            "corner_json": args.reference_corner_json,
            "qacr_latency_ms": args.reference_qacr_latency_ms,
        },
        "recommendation": recommendation,
        "rationale": rationale,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("===== Phase 4 Go/No-Go Review (Task 3.8) =====")
    print("candidate | perf_improved | corner_improved | latency_within_10pct | criteria_met")
    print(
        f"HighRes-Reencode | {highres_perf} | {highres_corner} | {highres_latency} | {highres_criteria}"
    )
    print(
        f"Attention-Level | {attn_perf} | {attn_corner} | {attn_latency} | {attn_criteria}"
    )
    print(f"recommendation: {recommendation}")
    print(f"rationale: {rationale}")
    print(f"summary_json: {out_json}")


if __name__ == "__main__":
    main()
