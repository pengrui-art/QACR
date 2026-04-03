#!/usr/bin/env python3
"""Phase C promotion helper: summarize 200 -> 500 -> full TextVQA runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_metrics(path: str) -> dict:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    metrics = obj["metrics"]
    return {
        "accuracy": float(metrics["accuracy"]),
        "raw_accuracy": float(metrics.get("raw_accuracy", metrics["accuracy"])),
        "mean_compute": float(metrics["mean_compute"]),
        "total_evaluated": int(metrics["total_evaluated"]),
        "path": path,
    }


def parse_variant(text: str) -> tuple[str, str]:
    name, path = text.split("=", 1)
    return name.strip(), path.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, required=True, choices=["200", "500", "full"])
    parser.add_argument("--variant", action="append", default=[], help="name=/path/to/result.json")
    parser.add_argument("--baseline-name", type=str, default="baseline")
    parser.add_argument("--out-json", type=str, required=True)
    parser.add_argument("--out-md", type=str, required=True)
    args = parser.parse_args()

    rows = []
    for item in args.variant:
        name, path = parse_variant(item)
        row = {"variant": name, **load_metrics(path)}
        rows.append(row)

    if not rows:
        raise ValueError("At least one --variant is required")

    by_name = {row["variant"]: row for row in rows}
    baseline = by_name.get(args.baseline_name)
    if baseline is None:
        raise ValueError(f"Missing baseline variant: {args.baseline_name}")

    for row in rows:
        row["delta_accuracy_vs_baseline"] = row["accuracy"] - baseline["accuracy"]
        row["delta_compute_vs_baseline"] = row["mean_compute"] - baseline["mean_compute"]

    best = max(rows, key=lambda row: (row["accuracy"], -row["mean_compute"]))
    payload = {
        "stage": args.stage,
        "baseline_name": args.baseline_name,
        "best_variant": best["variant"],
        "rows": rows,
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# Phase C TextVQA Promotion Summary ({args.stage})",
        "",
        f"- baseline: `{args.baseline_name}`",
        f"- best_variant: `{best['variant']}`",
        "",
        "| Variant | Accuracy | Delta Acc vs Baseline | Mean Compute | Delta Compute | N |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (-item["accuracy"], item["mean_compute"], item["variant"])):
        lines.append(
            f"| {row['variant']} | {row['accuracy']:.5f} | {row['delta_accuracy_vs_baseline']:+.5f} | "
            f"{row['mean_compute']:.5f} | {row['delta_compute_vs_baseline']:+.5f} | {row['total_evaluated']} |"
        )
    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    print(f"best_variant: {best['variant']}")
    print(f"saved_json: {out_json}")
    print(f"saved_md: {out_md}")


if __name__ == "__main__":
    main()
