#!/usr/bin/env python3
"""Phase 2.3: compare efficiency-performance across baselines."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Model/Qwen35-08B")
    parser.add_argument("--image", default="outputs/demo_phase01.png")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--budget", type=float, default=0.45)
    parser.add_argument("--low-grid", type=int, default=9)
    parser.add_argument("--base-grid", type=int, default=14)
    parser.add_argument("--keep-ratio", type=float, default=0.45)
    return parser.parse_args()


def run_and_capture(cmd: list[str], cwd: Path) -> str:
    out = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )
    return out.stdout


def find_float(text: str, key: str) -> float:
    m = re.search(rf"{re.escape(key)}:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", text)
    if not m:
        raise ValueError(f"Cannot find key '{key}' in output")
    return float(m.group(1))


def parse_budget_row(text: str, budget: float) -> tuple[float, float, float]:
    pattern = (
        rf"^{budget:.2f}\s*\|\s*"
        r"([-+]?\d*\.?\d+)\s*\|\s*"
        r"([-+]?\d*\.?\d+)\s*\|\s*"
        r"([-+]?\d*\.?\d+)\s*\|\s*"
        r"([-+]?\d*\.?\d+)\s*\|\s*"
        r"([-+]?\d*\.?\d+)\s*\|\s*"
        r"([-+]?\d*\.?\d+)\s*$"
    )
    for line in text.splitlines():
        m = re.match(pattern, line.strip())
        if m:
            expected_compute = float(m.group(1))
            task_loss = float(m.group(2))
            latency_ms = float(m.group(6))
            return expected_compute, task_loss, latency_ms
    raise ValueError(f"Cannot find budget row for {budget:.2f}")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    py = sys.executable

    upper_out = run_and_capture(
        [
            py,
            "scripts/run_upper_bound_baseline.py",
            "--model",
            args.model,
            "--image",
            args.image,
            "--steps",
            str(args.steps),
            "--coarse-grid",
            str(args.base_grid),
        ],
        cwd=repo_root,
    )
    lowres_out = run_and_capture(
        [
            py,
            "scripts/train_low_resolution_baseline.py",
            "--model",
            args.model,
            "--image",
            args.image,
            "--steps",
            str(args.steps),
            "--base-grid",
            str(args.base_grid),
            "--low-grid",
            str(args.low_grid),
        ],
        cwd=repo_root,
    )
    pruning_out = run_and_capture(
        [
            py,
            "scripts/train_token_pruning_baseline.py",
            "--model",
            args.model,
            "--image",
            args.image,
            "--steps",
            str(args.steps),
            "--keep-ratio",
            str(args.keep_ratio),
            "--coarse-grid",
            str(args.base_grid),
        ],
        cwd=repo_root,
    )
    image_only_out = run_and_capture(
        [
            py,
            "scripts/train_image_only_routing_baseline.py",
            "--model",
            args.model,
            "--image",
            args.image,
            "--steps",
            str(args.steps),
            "--budget",
            str(args.budget),
            "--coarse-grid",
            str(args.base_grid),
        ],
        cwd=repo_root,
    )
    qacr_out = run_and_capture(
        [
            py,
            "scripts/train_query_adaptive_budget_sweep.py",
            "--model",
            args.model,
            "--image",
            args.image,
            "--budgets",
            f"{args.budget:.2f}",
            "--steps",
            str(args.steps),
            "--coarse-grid",
            str(args.base_grid),
            "--router-type",
            "depth",
        ],
        cwd=repo_root,
    )
    qacr_attn_out = run_and_capture(
        [
            py,
            "scripts/train_query_adaptive_budget_sweep.py",
            "--model",
            args.model,
            "--image",
            args.image,
            "--budgets",
            f"{args.budget:.2f}",
            "--steps",
            str(args.steps),
            "--coarse-grid",
            str(args.base_grid),
            "--router-hidden",
            "96",
            "--router-type",
            "attention",
        ],
        cwd=repo_root,
    )

    rows = []
    rows.append(
        (
            "UpperBound-Deep",
            find_float(upper_out, "expected_compute_ratio"),
            find_float(upper_out, "final_loss"),
            find_float(upper_out, "avg_forward_latency_ms"),
        )
    )
    rows.append(
        (
            f"LowRes-{args.low_grid}x{args.low_grid}",
            find_float(lowres_out, "compute_ratio_vs_base"),
            find_float(lowres_out, "final_loss"),
            find_float(lowres_out, "avg_forward_latency_ms"),
        )
    )
    rows.append(
        (
            f"TokenPruning-keep{args.keep_ratio:.2f}",
            find_float(pruning_out, "expected_compute_ratio"),
            find_float(pruning_out, "final_loss"),
            find_float(pruning_out, "avg_forward_latency_ms"),
        )
    )
    rows.append(
        (
            "ImageOnlyRouting",
            find_float(image_only_out, "expected_compute"),
            find_float(image_only_out, "task_loss"),
            find_float(image_only_out, "avg_forward_latency_ms"),
        )
    )
    qacr_compute, qacr_task_loss, qacr_latency = parse_budget_row(qacr_out, args.budget)
    rows.append(("QACR-QueryAdaptive", qacr_compute, qacr_task_loss, qacr_latency))
    qacr_attn_compute, qacr_attn_task_loss, qacr_attn_latency = parse_budget_row(qacr_attn_out, args.budget)
    rows.append(("QACR-AttentionLevel", qacr_attn_compute, qacr_attn_task_loss, qacr_attn_latency))

    print("===== Efficiency-Performance Comparison (Task 2.3) =====")
    print(f"model: {args.model}")
    print(f"image: {args.image}")
    print(f"steps: {args.steps}")
    print(f"budget_anchor: {args.budget:.2f}")
    print("")
    print("| Method | ComputeRatio | ProxyTaskLoss | Latency(ms) |")
    print("|---|---:|---:|---:|")
    for method, comp, loss, lat in rows:
        print(f"| {method} | {comp:.6f} | {loss:.6f} | {lat:.6f} |")

    target = None
    for row in rows:
        if row[0] == "QACR-QueryAdaptive":
            target = row
            break
    if target is None:
        raise RuntimeError("QACR row missing")
    _, t_comp, t_loss, _ = target

    print("")
    print("near_budget_methods_vs_qacr:")
    for method, comp, loss, _ in rows:
        if method == "QACR-QueryAdaptive":
            continue
        if abs(comp - t_comp) <= 0.08:
            print(
                f"{method}: delta_compute={comp - t_comp:+.6f}, "
                f"delta_task_loss={loss - t_loss:+.6f}"
            )
    print("status: multi-baseline comparison succeeded")


if __name__ == "__main__":
    main()

