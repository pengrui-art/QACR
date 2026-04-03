#!/usr/bin/env python3
"""Upload QACR checkpoints to Hugging Face Hub from a manifest file."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi


@dataclass
class UploadItem:
    local_dir: Path
    repo_id: str
    private: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload QACR checkpoints to HF")
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to JSON manifest describing checkpoint folders and repo ids",
    )
    parser.add_argument("--namespace", type=str, default="TezBaby")
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create repos as private by default",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate manifest and print upload plan",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload QACR checkpoint",
    )
    return parser.parse_args()


def load_manifest(
    path: Path,
    namespace: str,
    default_private: bool,
    allow_missing_local_dirs: bool,
) -> list[UploadItem]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("manifest must be a JSON array")

    items: list[UploadItem] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"manifest[{i}] must be an object")

        local_dir = Path(item["local_dir"])
        if not allow_missing_local_dirs and (
            not local_dir.exists() or not local_dir.is_dir()
        ):
            raise FileNotFoundError(
                f"local_dir not found or not a directory: {local_dir}"
            )

        repo_name = item.get("repo_name")
        repo_id = item.get("repo_id")
        if repo_id is None:
            if repo_name is None:
                raise ValueError(
                    f"manifest[{i}] must provide either repo_id or repo_name"
                )
            repo_id = f"{namespace}/{repo_name}"

        private = bool(item.get("private", default_private))
        items.append(UploadItem(local_dir=local_dir, repo_id=repo_id, private=private))

    return items


def validate_checkpoint_dir(path: Path) -> list[str]:
    expected = ["router.pt", "router_config.json", "README.md"]
    return [name for name in expected if not (path / name).exists()]


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    items = load_manifest(
        path=manifest_path,
        namespace=args.namespace,
        default_private=args.private,
        allow_missing_local_dirs=args.dry_run,
    )

    print("===== QACR HF Upload Plan =====")
    print(f"manifest: {manifest_path}")
    print(f"num_items: {len(items)}")

    for item in items:
        missing = validate_checkpoint_dir(item.local_dir)
        print(f"- repo_id: {item.repo_id}")
        print(f"  local_dir: {item.local_dir}")
        print(f"  private: {item.private}")
        if not item.local_dir.exists():
            print("  warning: local_dir does not exist yet")
            continue
        if missing:
            print(f"  warning: missing expected files: {missing}")

    if args.dry_run:
        print("dry_run=True, no upload performed")
        return

    api = HfApi()
    for item in items:
        api.create_repo(
            repo_id=item.repo_id, repo_type="model", private=item.private, exist_ok=True
        )
        api.upload_folder(
            repo_id=item.repo_id,
            repo_type="model",
            folder_path=str(item.local_dir),
            commit_message=args.commit_message,
        )
        print(f"uploaded: https://huggingface.co/{item.repo_id}")


if __name__ == "__main__":
    main()
