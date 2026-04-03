#!/usr/bin/env python3
"""Phase 6.3: aggregate benchmark results and plot Pareto frontiers."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class MethodSpec:
    name: str
    key: str
    path_template: str


METHOD_SPECS: list[MethodSpec] = [
    MethodSpec("QACR b0.35", "qacr_b035", "checkpoints/qacr_vqav2_b0.35/eval_results_{dataset}.json"),
    MethodSpec("QACR b0.45", "qacr_b045", "checkpoints/qacr_vqav2_b0.45/eval_results_{dataset}.json"),
    MethodSpec("QACR b0.60", "qacr_b060", "checkpoints/qacr_vqav2_b0.60/eval_results_{dataset}.json"),
    MethodSpec("TokenPruning@0.45", "token_pruning", "checkpoints/token_pruning_kr0.45_vqav2/eval_results_{dataset}.json"),
    MethodSpec("ImageOnly@0.45", "image_only", "checkpoints/image_only_b0.45_vqav2/eval_results_{dataset}.json"),
    MethodSpec("LowRes-9x9", "low_res", "checkpoints/low_res_g9_vqav2/eval_results_{dataset}.json"),
    MethodSpec("FastV", "fastv", "checkpoints/sota_eval/fastv_{dataset}_results.json"),
    MethodSpec("LVPruning", "lvpruning", "checkpoints/sota_eval/lvpruning_{dataset}_results.json"),
    MethodSpec("Original", "original", "checkpoints/sota_eval/original_{dataset}_results.json"),
]

DEFAULT_EXPECTED_COUNTS: dict[str, int] = {
    "textvqa": 5000,
    "docvqa": 5349,
    "mmmu": 900,
}


def load_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    m = obj.get("metrics", {})
    if "accuracy" not in m or "mean_compute" not in m:
        return None
    return {
        "accuracy": float(m["accuracy"]),
        "raw_accuracy": float(m.get("raw_accuracy", m["accuracy"])),
        "mean_compute": float(m["mean_compute"]),
        "total_evaluated": int(m.get("total_evaluated", 0)),
        "source_path": str(path),
    }


def load_existing_rows(path: Path) -> dict[tuple[str, str], dict]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    rows = {}
    for row in obj.get("per_dataset_rows", []):
        rows[(row["method_key"], row["dataset"])] = row
    return rows


def parse_expected_counts(raw: str) -> dict[str, int]:
    expected = dict(DEFAULT_EXPECTED_COUNTS)
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        key, value = item.split("=", 1)
        expected[key.strip().lower()] = int(value.strip())
    return expected


def inspect_result_coverage(
    metrics: dict | None,
    dataset: str,
    expected_counts: dict[str, int],
    min_coverage_ratio: float,
) -> dict:
    expected_total = int(expected_counts.get(dataset, 0))
    if metrics is None:
        return {
            "status": "missing_or_invalid",
            "expected_total": expected_total,
            "observed_total": 0,
            "coverage_ratio": 0.0,
            "reason": "missing_or_invalid_metrics",
        }
    observed_total = int(metrics.get("total_evaluated", 0))
    coverage_ratio = (observed_total / expected_total) if expected_total > 0 else 1.0
    if expected_total > 0 and coverage_ratio < min_coverage_ratio:
        return {
            "status": "insufficient_coverage",
            "expected_total": expected_total,
            "observed_total": observed_total,
            "coverage_ratio": coverage_ratio,
            "reason": f"coverage_below_{min_coverage_ratio:.2f}",
        }
    return {
        "status": "accepted",
        "expected_total": expected_total,
        "observed_total": observed_total,
        "coverage_ratio": coverage_ratio,
        "reason": "accepted",
    }


def pareto_front(points: Iterable[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """Maximize accuracy, minimize compute."""
    sorted_pts = sorted(points, key=lambda x: (x[0], -x[1]))
    best_y = -1.0
    frontier: list[tuple[float, float, str]] = []
    for x, y, label in sorted_pts:
        if y > best_y:
            frontier.append((x, y, label))
            best_y = y
    return frontier


def plot_scatter_with_pareto(rows: list[dict], title: str, out_png: Path) -> None:
    if not rows:
        return
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 6), dpi=160)

    for r in rows:
        x = r["mean_compute"]
        y = r["accuracy"] * 100.0
        label = r["method"]
        plt.scatter(x, y, s=62, alpha=0.9)
        plt.text(x + 0.003, y + 0.06, label, fontsize=8)

    front = pareto_front([(r["mean_compute"], r["accuracy"] * 100.0, r["method"]) for r in rows])
    if len(front) >= 2:
        xs = [p[0] for p in front]
        ys = [p[1] for p in front]
        plt.plot(xs, ys, linestyle="--", linewidth=1.5)

    plt.title(title)
    plt.xlabel("Mean Compute Ratio")
    plt.ylabel("Accuracy (%)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def build_markdown(rows: list[dict], datasets: list[str]) -> str:
    hdr = [
        "# Phase 6.2/6.3 Final Summary",
        "",
        f"Datasets: {', '.join(datasets)}",
        "",
        "## Per-Dataset Results",
        "",
        "| Method | Dataset | Accuracy | Raw Accuracy | Mean Compute | N |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    body = []
    for r in sorted(rows, key=lambda x: (x["method"], x["dataset"])):
        body.append(
            f"| {r['method']} | {r['dataset']} | {r['accuracy']:.5f} | {r['raw_accuracy']:.5f} | {r['mean_compute']:.5f} | {r['total_evaluated']} |"
        )
    return "\n".join(hdr + body) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Phase 6.3 Pareto frontiers from final benchmark JSON files.")
    parser.add_argument("--datasets", type=str, default="textvqa,docvqa,mmmu")
    parser.add_argument("--out-dir", type=str, default="outputs/phase6_full_benchmarks")
    parser.add_argument(
        "--expected-counts",
        type=str,
        default="textvqa=5000,docvqa=5349,mmmu=900",
        help="Expected full-benchmark sample counts used to filter smoke or partial result files.",
    )
    parser.add_argument(
        "--min-coverage-ratio",
        type=float,
        default=0.95,
        help="Minimum fraction of the expected sample count required for a result file to be accepted as a full benchmark row.",
    )
    args = parser.parse_args()

    datasets = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing_rows = load_existing_rows(out_dir / "phase62_final_summary.json")
    expected_counts = parse_expected_counts(args.expected_counts)

    all_rows: list[dict] = []
    filter_records: list[dict] = []
    for spec in METHOD_SPECS:
        for ds in datasets:
            p = Path(spec.path_template.format(dataset=ds))
            m = load_metrics(p)
            prev = existing_rows.get((spec.key, ds))
            current_inspection = inspect_result_coverage(
                metrics=m,
                dataset=ds,
                expected_counts=expected_counts,
                min_coverage_ratio=args.min_coverage_ratio,
            )
            prev_inspection = inspect_result_coverage(
                metrics=prev,
                dataset=ds,
                expected_counts=expected_counts,
                min_coverage_ratio=args.min_coverage_ratio,
            ) if prev is not None else None
            # Explicitly filter out smoke / partial files for the full benchmark summary.
            if prev is not None and current_inspection["status"] != "accepted" and prev_inspection and prev_inspection["status"] == "accepted":
                row = dict(prev)
                row["source_path"] = f"{prev.get('source_path', str(p))} [from existing summary]"
                all_rows.append(row)
                filter_records.append(
                    {
                        "method": spec.name,
                        "method_key": spec.key,
                        "dataset": ds,
                        "candidate_path": str(p),
                        "candidate_status": current_inspection["status"],
                        "candidate_reason": current_inspection["reason"],
                        "candidate_total_evaluated": current_inspection["observed_total"],
                        "expected_total": current_inspection["expected_total"],
                        "fallback_used": True,
                        "fallback_source_path": prev.get("source_path", str(p)),
                    }
                )
                continue
            if current_inspection["status"] != "accepted":
                filter_records.append(
                    {
                        "method": spec.name,
                        "method_key": spec.key,
                        "dataset": ds,
                        "candidate_path": str(p),
                        "candidate_status": current_inspection["status"],
                        "candidate_reason": current_inspection["reason"],
                        "candidate_total_evaluated": current_inspection["observed_total"],
                        "expected_total": current_inspection["expected_total"],
                        "fallback_used": False,
                        "fallback_source_path": "",
                    }
                )
                continue
            all_rows.append(
                {
                    "method": spec.name,
                    "method_key": spec.key,
                    "dataset": ds,
                    **m,
                }
            )
            filter_records.append(
                {
                    "method": spec.name,
                    "method_key": spec.key,
                    "dataset": ds,
                    "candidate_path": str(p),
                    "candidate_status": current_inspection["status"],
                    "candidate_reason": current_inspection["reason"],
                    "candidate_total_evaluated": current_inspection["observed_total"],
                    "expected_total": current_inspection["expected_total"],
                    "fallback_used": False,
                    "fallback_source_path": "",
                }
            )

    # Per-dataset plots
    for ds in datasets:
        ds_rows = [r for r in all_rows if r["dataset"] == ds]
        plot_scatter_with_pareto(
            ds_rows,
            title=f"Phase 6.2 Pareto ({ds})",
            out_png=out_dir / f"pareto_{ds}.png",
        )

    # Macro across available datasets (partial coverage allowed)
    macro_rows: list[dict] = []
    for spec in METHOD_SPECS:
        rows = [r for r in all_rows if r["method_key"] == spec.key]
        if not rows:
            continue
        macro_rows.append(
            {
                "method": spec.name,
                "method_key": spec.key,
                "dataset_coverage": len(rows),
                "accuracy": sum(r["accuracy"] for r in rows) / len(rows),
                "raw_accuracy": sum(r["raw_accuracy"] for r in rows) / len(rows),
                "mean_compute": sum(r["mean_compute"] for r in rows) / len(rows),
            }
        )
    plot_scatter_with_pareto(
        macro_rows,
        title="Phase 6.2 Pareto (Macro over available datasets)",
        out_png=out_dir / "pareto_macro_available.png",
    )

    # Macro on complete methods only (must cover all requested datasets)
    macro_complete = [r for r in macro_rows if r["dataset_coverage"] == len(datasets)]
    plot_scatter_with_pareto(
        macro_complete,
        title="Phase 6.2 Pareto (Macro, complete coverage only)",
        out_png=out_dir / "pareto_macro_complete_only.png",
    )

    # Export tables
    summary = {
        "datasets": datasets,
        "expected_counts": expected_counts,
        "min_coverage_ratio": args.min_coverage_ratio,
        "per_dataset_rows": all_rows,
        "macro_rows_available": macro_rows,
        "macro_rows_complete_only": macro_complete,
        "filter_records": filter_records,
    }
    with (out_dir / "phase62_final_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with (out_dir / "phase62_final_summary.md").open("w", encoding="utf-8") as f:
        f.write(build_markdown(all_rows, datasets))
        f.write("\n## Macro (Available Coverage)\n\n")
        f.write("| Method | Coverage | Macro Acc | Macro Raw Acc | Macro Compute |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for r in sorted(macro_rows, key=lambda x: x["method"]):
            f.write(
                f"| {r['method']} | {r['dataset_coverage']}/{len(datasets)} | {r['accuracy']:.5f} | {r['raw_accuracy']:.5f} | {r['mean_compute']:.5f} |\n"
            )

    with (out_dir / "phase62_final_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "method_key",
                "dataset",
                "accuracy",
                "raw_accuracy",
                "mean_compute",
                "total_evaluated",
                "source_path",
            ],
        )
        writer.writeheader()
        for r in sorted(all_rows, key=lambda x: (x["dataset"], x["method"])):
            writer.writerow(r)

    missing = []
    for spec in METHOD_SPECS:
        for ds in datasets:
            p = Path(spec.path_template.format(dataset=ds))
            if not p.exists():
                missing.append(str(p))
    with (out_dir / "phase62_missing_files.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(missing) + ("\n" if missing else ""))

    with (out_dir / "phase62_filter_report.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "method_key",
                "dataset",
                "candidate_path",
                "candidate_status",
                "candidate_reason",
                "candidate_total_evaluated",
                "expected_total",
                "fallback_used",
                "fallback_source_path",
            ],
        )
        writer.writeheader()
        for row in sorted(filter_records, key=lambda x: (x["dataset"], x["method"])):
            writer.writerow(row)

    print(f"[done] out_dir={out_dir}")
    print(f"[done] per_dataset_rows={len(all_rows)}")
    print(f"[done] macro_rows_available={len(macro_rows)}")
    print(f"[done] macro_rows_complete_only={len(macro_complete)}")
    print(f"[done] missing_files={len(missing)}")
    print(f"[done] filter_records={len(filter_records)}")


if __name__ == "__main__":
    main()
