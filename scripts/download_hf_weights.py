#!/usr/bin/env python3
"""Download QACR checkpoints from Hugging Face Hub."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download QACR checkpoints from HF")
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Full repo id, e.g. TezBaby/QACR-Qwen35-08B-B0.45",
    )
    parser.add_argument("--namespace", type=str, default="TezBaby")
    parser.add_argument("--model-name", type=str, default="QACR-Qwen35-08B")
    parser.add_argument("--budget", type=float, default=0.45, help="Compute budget")
    parser.add_argument(
        "--router-type",
        type=str,
        default="depth",
        choices=["depth", "attention"],
    )
    parser.add_argument("--out-dir", type=str, default="checkpoints")
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional branch/tag/commit",
    )
    return parser.parse_args()


def format_budget(budget: float) -> str:
    return f"{budget:.2f}".rstrip("0").rstrip(".")


def build_repo_id(
    namespace: str,
    model_name: str,
    budget: float,
    router_type: str,
) -> str:
    budget_tag = format_budget(budget)
    if router_type == "attention":
        return f"{namespace}/{model_name}-Attn-B{budget_tag}"
    return f"{namespace}/{model_name}-B{budget_tag}"


def main() -> None:
    args = parse_args()
    repo_id = args.repo_id or build_repo_id(
        namespace=args.namespace,
        model_name=args.model_name,
        budget=args.budget,
        router_type=args.router_type,
    )

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    local_dir = out_root / repo_id.replace("/", "__")

    downloaded = snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        revision=args.revision,
    )

    expected_files = ["router.pt", "router_config.json"]
    missing = [f for f in expected_files if not (local_dir / f).exists()]

    print("===== QACR Checkpoint Download =====")
    print(f"repo_id: {repo_id}")
    print(f"downloaded_to: {downloaded}")
    if missing:
        print("warning: downloaded repo does not include expected files")
        print(f"missing: {missing}")
    else:
        print("status: router.pt and router_config.json found")


if __name__ == "__main__":
    main()
