#!/usr/bin/env python3
"""Package a trained router checkpoint into a Hugging Face-ready folder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare QACR router checkpoint for HF"
    )
    parser.add_argument(
        "--input-ckpt", type=str, required=True, help="Path to .pt checkpoint"
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--router-type", choices=["depth", "attention"], default="depth"
    )
    parser.add_argument("--budget", type=float, required=True)
    parser.add_argument("--query-dim", type=int, required=True)
    parser.add_argument("--image-dim", type=int, required=True)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--notes", type=str, default="")
    return parser.parse_args()


def load_router_state(path: Path) -> dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        if "router_state_dict" in obj and isinstance(obj["router_state_dict"], dict):
            return obj["router_state_dict"]
        has_tensor_values = any(torch.is_tensor(v) for v in obj.values())
        if has_tensor_values:
            return obj
    raise ValueError(
        "Unsupported checkpoint format. Expected a state_dict or dict with key 'router_state_dict'."
    )


def build_model_card(config: dict[str, object]) -> str:
    return f"""# QACR Router Checkpoint

This repository stores a trained QACR router checkpoint.

## Metadata

- router_type: {config['router_type']}
- budget: {config['budget']}
- query_dim: {config['query_dim']}
- image_dim: {config['image_dim']}
- hidden_dim: {config['hidden_dim']}
- base_model: {config['base_model']}

## Load Example

```python
import json
from pathlib import Path
import torch

from qacr.routing import AttentionLevelRouter, DepthOnlyRouter

ckpt_dir = Path(".")
cfg = json.loads((ckpt_dir / "router_config.json").read_text(encoding="utf-8"))
router_cls = AttentionLevelRouter if cfg["router_type"] == "attention" else DepthOnlyRouter
router = router_cls(
    query_dim=cfg["query_dim"],
    image_dim=cfg["image_dim"],
    hidden_dim=cfg["hidden_dim"],
)
router.load_state_dict(torch.load(ckpt_dir / "router.pt", map_location="cpu"))
router.eval()
```
"""


def main() -> None:
    args = parse_args()
    input_ckpt = Path(args.input_ckpt)
    if not input_ckpt.exists():
        raise FileNotFoundError(f"input checkpoint not found: {input_ckpt}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state_dict = load_router_state(input_ckpt)
    torch.save(state_dict, output_dir / "router.pt")

    config = {
        "router_type": args.router_type,
        "budget": args.budget,
        "query_dim": args.query_dim,
        "image_dim": args.image_dim,
        "hidden_dim": args.hidden_dim,
        "base_model": args.base_model,
        "notes": args.notes,
    }
    (output_dir / "router_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "README.md").write_text(build_model_card(config), encoding="utf-8")

    print("===== Prepared HF Router Checkpoint =====")
    print(f"input_ckpt: {input_ckpt}")
    print(f"output_dir: {output_dir}")
    print(f"router_type: {args.router_type}")
    print(f"budget: {args.budget}")


if __name__ == "__main__":
    main()
