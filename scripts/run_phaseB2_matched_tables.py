#!/usr/bin/env python3
"""Phase B.2: merge benchmark accuracy with profiling into matched tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-json",
        type=str,
        default="outputs/phase6_full_benchmarks/phase62_final_summary.json",
    )
    parser.add_argument(
        "--profiling-json",
        type=str,
        default="outputs/phaseB_unified_profiling/phaseB1_unified_profiling.json",
    )
    parser.add_argument("--dataset", type=str, default="textvqa")
    parser.add_argument(
        "--select-profile-row",
        type=str,
        default="min_sample_latency",
        choices=["min_sample_latency", "max_throughput"],
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="outputs/phaseB_matched_tables/phaseB2_matched_tables.json",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="outputs/phaseB_matched_tables/phaseB2_matched_tables.csv",
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default="outputs/phaseB_matched_tables/phaseB2_matched_tables.md",
    )
    parser.add_argument(
        "--out-compute-png",
        type=str,
        default="outputs/phaseB_matched_tables/phaseB2_accuracy_compute.png",
    )
    parser.add_argument(
        "--out-latency-png",
        type=str,
        default="outputs/phaseB_matched_tables/phaseB2_accuracy_latency.png",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def choose_profile_rows(rows: list[dict], mode: str) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["method_key"], []).append(row)

    selected: dict[str, dict] = {}
    for method_key, candidates in grouped.items():
        if mode == "max_throughput":
            pick = max(
                candidates,
                key=lambda r: (r["throughput_samples_per_s"], -r["wall_clock_latency_per_sample_ms"], -r["batch_size"]),
            )
        else:
            pick = min(
                candidates,
                key=lambda r: (r["wall_clock_latency_per_sample_ms"], -r["throughput_samples_per_s"], -r["batch_size"]),
            )
        selected[method_key] = pick
    return selected


def merge_rows(summary_rows: list[dict], profile_rows: dict[str, dict], dataset: str) -> list[dict]:
    merged: list[dict] = []
    for row in summary_rows:
        if row["dataset"] != dataset:
            continue
        profile = profile_rows.get(row["method_key"])
        if profile is None:
            continue
        merged.append(
            {
                "method": row["method"],
                "method_key": row["method_key"],
                "dataset": dataset,
                "accuracy": float(row["accuracy"]),
                "raw_accuracy": float(row["raw_accuracy"]),
                "mean_compute": float(row["mean_compute"]),
                "profile_batch_size": int(profile["batch_size"]),
                "profile_sample_latency_ms": float(profile["wall_clock_latency_per_sample_ms"]),
                "profile_batch_latency_ms": float(profile["wall_clock_latency_ms"]),
                "profile_throughput_samples_per_s": float(profile["throughput_samples_per_s"]),
                "profile_peak_gpu_memory_mb": float(profile["peak_gpu_memory_mb"]),
                "profile_peak_gpu_memory_delta_mb": float(profile["peak_gpu_memory_delta_mb"]),
                "profile_mean_compute": float(profile["mean_compute_profiled"]),
            }
        )
    return merged


def build_markdown(rows: list[dict], dataset: str, selection_mode: str) -> str:
    compute_sorted = sorted(rows, key=lambda r: (r["mean_compute"], -r["accuracy"]))
    latency_sorted = sorted(rows, key=lambda r: (r["profile_sample_latency_ms"], -r["accuracy"]))

    lines: list[str] = []
    lines.append(f"# Phase B.2 Matched Tables ({dataset})")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(f"- Accuracy comes from the full benchmark summary for `{dataset}`.")
    lines.append("- Latency / throughput / peak memory come from Phase B.1 unified profiling.")
    lines.append(f"- Per method, the profiling row is selected by: `{selection_mode}`.")
    lines.append("")
    lines.append("## Matched Compute View")
    lines.append("")
    lines.append("| Method | Accuracy | Mean Compute | Profile Batch | Sample Latency (ms) | Throughput (samples/s) | Peak GPU Memory (MB) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in compute_sorted:
        lines.append(
            f"| {row['method']} | {row['accuracy']:.4f} | {row['mean_compute']:.4f} | {row['profile_batch_size']} | "
            f"{row['profile_sample_latency_ms']:.2f} | {row['profile_throughput_samples_per_s']:.2f} | {row['profile_peak_gpu_memory_mb']:.2f} |"
        )
    lines.append("")
    lines.append("## Matched Latency View")
    lines.append("")
    lines.append("| Method | Accuracy | Sample Latency (ms) | Throughput (samples/s) | Mean Compute | Peak GPU Memory (MB) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in latency_sorted:
        lines.append(
            f"| {row['method']} | {row['accuracy']:.4f} | {row['profile_sample_latency_ms']:.2f} | "
            f"{row['profile_throughput_samples_per_s']:.2f} | {row['mean_compute']:.4f} | {row['profile_peak_gpu_memory_mb']:.2f} |"
        )
    lines.append("")
    if rows:
        by_key = {row["method_key"]: row for row in rows}
        qacr = by_key.get("qacr_b045")
        orig = by_key.get("original")
        token = by_key.get("token_pruning")
        lowres = by_key.get("low_res")
        if qacr and orig:
            lines.append("## Key Comparisons")
            lines.append("")
            lat_gain = 100.0 * (orig["profile_sample_latency_ms"] - qacr["profile_sample_latency_ms"]) / orig["profile_sample_latency_ms"]
            th_gain = 100.0 * (qacr["profile_throughput_samples_per_s"] - orig["profile_throughput_samples_per_s"]) / orig["profile_throughput_samples_per_s"]
            lines.append(
                f"- `QACR b0.45` vs `Original`: sample latency improves by `{lat_gain:.1f}%`, throughput improves by `{th_gain:.1f}%`, while compute drops from `{orig['mean_compute']:.4f}` to `{qacr['mean_compute']:.4f}`."
            )
            if token:
                acc_gap = qacr["accuracy"] - token["accuracy"]
                lines.append(
                    f"- `QACR b0.45` vs `TokenPruning@0.45`: QACR is slower (`{qacr['profile_sample_latency_ms']:.2f}` vs `{token['profile_sample_latency_ms']:.2f} ms/sample`) but much more accurate (`{qacr['accuracy']:.4f}` vs `{token['accuracy']:.4f}`, gap `{acc_gap:+.4f}`)."
                )
            if lowres:
                acc_gap = qacr["accuracy"] - lowres["accuracy"]
                lines.append(
                    f"- `QACR b0.45` vs `LowRes-9x9`: QACR is slightly slower (`{qacr['profile_sample_latency_ms']:.2f}` vs `{lowres['profile_sample_latency_ms']:.2f} ms/sample`) but far more accurate (`{qacr['accuracy']:.4f}` vs `{lowres['accuracy']:.4f}`, gap `{acc_gap:+.4f}`)."
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def plot_scatter(rows: list[dict], x_key: str, x_label: str, out_png: Path, title: str) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.5, 6), dpi=160)
    for row in rows:
        x = row[x_key]
        y = row["accuracy"] * 100.0
        plt.scatter(x, y, s=60, alpha=0.9)
        plt.text(x + (0.005 if x_key == "mean_compute" else 3.0), y + 0.06, row["method"], fontsize=8)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Accuracy (%)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def main() -> None:
    args = parse_args()
    summary = load_json(Path(args.summary_json))
    profiling = load_json(Path(args.profiling_json))

    selected_profiles = choose_profile_rows(
        [row for row in profiling["rows"] if row["dataset"] == args.dataset],
        args.select_profile_row,
    )
    merged = merge_rows(summary["per_dataset_rows"], selected_profiles, args.dataset)

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    out_compute_png = Path(args.out_compute_png)
    out_latency_png = Path(args.out_latency_png)

    payload = {
        "dataset": args.dataset,
        "selection_mode": args.select_profile_row,
        "rows": merged,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(merged[0].keys()) if merged else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in merged:
            writer.writerow(row)

    out_md.write_text(build_markdown(merged, args.dataset, args.select_profile_row), encoding="utf-8")

    plot_scatter(
        merged,
        x_key="mean_compute",
        x_label="Mean Compute Ratio",
        out_png=out_compute_png,
        title=f"Phase B.2 Accuracy-Compute ({args.dataset})",
    )
    plot_scatter(
        merged,
        x_key="profile_sample_latency_ms",
        x_label="Sample Latency (ms)",
        out_png=out_latency_png,
        title=f"Phase B.2 Accuracy-Latency ({args.dataset})",
    )

    print(f"saved_json: {out_json}")
    print(f"saved_csv: {out_csv}")
    print(f"saved_md: {out_md}")
    print(f"saved_compute_png: {out_compute_png}")
    print(f"saved_latency_png: {out_latency_png}")


if __name__ == "__main__":
    main()
