"""
plot_pareto_vqav2.py  —  Generate Accuracy vs Compute Pareto plot from eval results.

Run after eval_all_models.sh completes:
    conda run -n qacr python scripts/plot_pareto_vqav2.py

Output: outputs/vqav2_pareto.png
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless / no-display safe
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Method registry ──────────────────────────────────────────────────────────
# (label, json_path, marker, color, is_qacr)
METHODS = [
    # QACR checkpoints
    ("QACR b=0.35",      "checkpoints/qacr_vqav2_b0.35/eval_results.json",          "o", "#e63946", True),
    ("QACR b=0.45",      "checkpoints/qacr_vqav2_b0.45/eval_results.json",          "o", "#c1121f", True),
    ("QACR b=0.60",      "checkpoints/qacr_vqav2_b0.60/eval_results.json",          "o", "#9b1d20", True),
    # Baselines
    ("TokenPruning",     "checkpoints/token_pruning_kr0.45_vqav2/eval_results.json","s", "#2196f3", False),
    ("ImageOnly",        "checkpoints/image_only_b0.45_vqav2/eval_results.json",    "^", "#9e9e9e", False),
    ("LowRes (9×9)",     "checkpoints/low_res_g9_vqav2/eval_results.json",          "v", "#ff9800", False),
    # External SOTA
    ("FastV",            "checkpoints/sota_eval/fastv_vqav2_results.json",           "D", "#7b2d8b", False),
    ("LVPruning",        "checkpoints/sota_eval/lvpruning_vqav2_results.json",       "*", "#795548", False),
    ("Original (Full)",  "checkpoints/sota_eval/original_vqav2_results.json",        "X", "#212121", False),
]

# LowRes true compute ratio: (9/14)^2 ≈ 0.413
LOWRES_COMPUTE = (9 / 14) ** 2


def _load_metrics(path: str) -> dict | None:
    """Load eval_results.json; return unified metrics dict or None if missing."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        res = json.load(f)
    # Support two layouts: {metrics: {...}, ...} or flat {accuracy: ..., ...}
    return res.get("metrics", res)


def main():
    collected = []              # (label, acc_pct, compute, marker, color, is_qacr)
    missing = []

    for label, path, marker, color, is_qacr in METHODS:
        m = _load_metrics(path)
        if m is None:
            missing.append(label)
            continue

        acc = m.get("accuracy", 0.0) * 100.0
        compute = m.get("mean_compute", 1.0)

        # LowRes hook outputs mean_compute=1.0 (all-deep executor for token subset).
        # Override with true compute ratio based on token count reduction.
        if "low_res" in path.lower() or "LowRes" in label:
            compute = LOWRES_COMPUTE

        # Sanity: skip obviously broken results (e.g., old gt=None runs)
        if acc < 0.5 and "Original" not in label:
            print(f"  [WARN] {label}: acc={acc:.2f}% is suspiciously low — included but flagged")

        collected.append((label, acc, compute, marker, color, is_qacr))
        print(f"  {label:25s}: Acc={acc:6.2f}%  Compute={compute:.4f}")

    if missing:
        print(f"\n[INFO] Missing results (not plotted): {missing}")

    if not collected:
        print("No eval_results.json found. Run eval_all_models.sh first.")
        sys.exit(0)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))

    # Draw QACR Pareto frontier line (connect QACR points sorted by compute)
    qacr_pts = sorted([(c, a) for lbl, a, c, mk, col, iq in collected if iq], key=lambda x: x[0])
    if len(qacr_pts) >= 2:
        xs, ys = zip(*qacr_pts)
        ax.plot(xs, ys, "--", color="#e63946", linewidth=1.8, alpha=0.7, zorder=1, label="_pareto")

    for label, acc, compute, marker, color, is_qacr in collected:
        size = 220 if is_qacr else 140
        zorder = 4 if is_qacr else 3
        ax.scatter(compute, acc, marker=marker, color=color, s=size,
                   zorder=zorder, edgecolors="white", linewidths=0.8, label=label)
        offset_y = 11 if is_qacr else -15
        ax.annotate(label, (compute, acc),
                    textcoords="offset points", xytext=(0, offset_y),
                    ha="center", fontsize=8,
                    color=color, fontweight="bold" if is_qacr else "normal")

    # Axes formatting
    ax.set_title("QACR vs Baselines — Accuracy vs Compute (VQAv2 Validation)",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Average Vision Token Compute Ratio  (lower = more efficient)", fontsize=11)
    ax.set_ylabel("VQAv2 Accuracy (%)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)

    # Legend (skip auto-duplicates and internal markers)
    handles, labels_leg = ax.get_legend_handles_labels()
    by_label = {l: h for h, l in zip(handles, labels_leg) if not l.startswith("_")}
    ax.legend(by_label.values(), by_label.keys(),
              bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/vqav2_pareto.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved Pareto plot → {out_path}")


if __name__ == "__main__":
    main()
