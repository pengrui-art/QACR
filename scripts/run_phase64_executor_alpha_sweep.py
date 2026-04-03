#!/usr/bin/env python3
"""Quick sweep of executor output alpha for QACR checkpoint evaluation."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def run_one(
    checkpoint_dir: str,
    model: str,
    dataset: str,
    local_data_dir: str,
    max_samples: int,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    alpha: float,
    out_file: Path,
) -> dict:
    cmd = [
        "python",
        "scripts/eval_qacr_benchmark.py",
        "--checkpoint-dir",
        checkpoint_dir,
        "--model",
        model,
        "--dataset",
        dataset,
        "--local-data-dir",
        local_data_dir,
        "--max-samples",
        str(max_samples),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--prefetch-factor",
        str(prefetch_factor),
        "--executor-output-alpha",
        str(alpha),
        "--out-file",
        str(out_file),
    ]
    subprocess.run(cmd, check=True)
    with out_file.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj["metrics"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep executor output alpha for quick method-improvement validation.")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="Model/Qwen35-08B")
    parser.add_argument("--datasets", type=str, default="textvqa,docvqa,mmmu")
    parser.add_argument("--alphas", type=str, default="1.0,0.8,0.6,0.4,0.2")
    parser.add_argument("--local-data-dir", type=str, default="/data1/pengrui/CCFA/QACR/data")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--out-dir", type=str, default="outputs/phase64_alpha_sweep")
    args = parser.parse_args()

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for alpha in alphas:
        for dataset in datasets:
            out_file = out_dir / f"{Path(args.checkpoint_dir).name}_{dataset}_a{alpha:.2f}.json"
            metrics = run_one(
                checkpoint_dir=args.checkpoint_dir,
                model=args.model,
                dataset=dataset,
                local_data_dir=args.local_data_dir,
                max_samples=args.max_samples,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
                alpha=alpha,
                out_file=out_file,
            )
            rows.append(
                {
                    "alpha": alpha,
                    "dataset": dataset,
                    "accuracy": float(metrics["accuracy"]),
                    "raw_accuracy": float(metrics.get("raw_accuracy", metrics["accuracy"])),
                    "mean_compute": float(metrics["mean_compute"]),
                    "total_evaluated": int(metrics.get("total_evaluated", 0)),
                    "result_file": str(out_file),
                }
            )

    summary = {"checkpoint_dir": args.checkpoint_dir, "datasets": datasets, "alphas": alphas, "rows": rows}
    summary_json = out_dir / "alpha_sweep_summary.json"
    summary_md = out_dir / "alpha_sweep_summary.md"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = [
        "# Phase 6.4 Executor Alpha Sweep",
        "",
        f"checkpoint: `{args.checkpoint_dir}`",
        "",
        "| alpha | dataset | accuracy | raw_accuracy | mean_compute | N |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for r in sorted(rows, key=lambda x: (x["alpha"], x["dataset"])):
        lines.append(
            f"| {r['alpha']:.2f} | {r['dataset']} | {r['accuracy']:.5f} | {r['raw_accuracy']:.5f} | {r['mean_compute']:.5f} | {r['total_evaluated']} |"
        )
    with summary_md.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"saved: {summary_json}")
    print(f"saved: {summary_md}")


if __name__ == "__main__":
    main()

