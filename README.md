# Query-Adaptive Compute Routing (QACR)

QACR is a query-conditioned compute allocation framework for MLLMs. It routes visual tokens to
`skip / shallow / deep` paths under explicit compute budget constraints, and supports both
depth-only routing and attention-level routing variants.

## Highlights

- Query-adaptive token routing with budget regularization.
- Soft-to-hard routing training and conditional execution profiling.
- Unified comparison scripts for LowRes, TokenPruning, ImageOnly, and QACR variants.
- Reproducible Phase-based experimental pipeline under `scripts/`.

## Repository Structure

```text
qacr/
  routing/          # routers: depth, attention, soft/hard utilities
  vision/           # multipath execution and vision-side modules
scripts/            # train/eval/profile/export utilities
Docx/               # plan table and experiment reports
outputs/            # generated figures/tables/json summaries
```

## Environment Setup

### Option A: One-click setup (recommended)

```bash
bash scripts/setup_env.sh qacr
```

### Option B: Manual setup

```bash
conda create -n qacr python=3.10 -y
conda activate qacr
pip install --upgrade pip
pip install -r requirements.txt
```

## One-click Run Entrypoint

Use `scripts/run_one_click.sh` for common actions:

```bash
# Inference smoke
bash scripts/run_one_click.sh infer

# Query-adaptive budget sweep training
bash scripts/run_one_click.sh train

# Baseline comparison table
bash scripts/run_one_click.sh compare
```

Optional environment variables:

- `MODEL_DIR` (default: `Model/Qwen35-08B`)
- `IMAGE_PATH` (default: `outputs/demo_phase01.png`)
- `STEPS` (default: `16`)
- `BUDGETS` (default: `0.35,0.45,0.60`)
- `BUDGET` (default: `0.45`)

## Core Commands (Manual)

### 1) Image+Text inference smoke

```bash
python scripts/run_qwen35_vl_infer.py \
  --model Model/Qwen35-08B \
  --image outputs/demo_phase01.png
```

### 2) Query-adaptive budget sweep

```bash
python scripts/train_query_adaptive_budget_sweep.py \
  --model Model/Qwen35-08B \
  --image outputs/demo_phase01.png \
  --budgets 0.35,0.45,0.60 \
  --steps 20
```

### 3) Efficiency-performance comparison

```bash
python scripts/run_efficiency_performance_comparison.py \
  --model Model/Qwen35-08B \
  --image outputs/demo_phase01.png \
  --budget 0.45 \
  --steps 16
```

## Hugging Face Checkpoint Workflow

### A. Upload local checkpoints to `TezBaby`

1. Login first:

```bash
huggingface-cli login
```

2. Prepare a manifest JSON (example: `scripts/hf_repos_manifest.example.json`) and run:

```bash
python scripts/upload_hf_weights.py \
  --manifest scripts/hf_repos_manifest.example.json \
  --namespace TezBaby
```

### B. Download checkpoints by budget/router type

```bash
python scripts/download_hf_weights.py \
  --budget 0.45 \
  --router-type depth \
  --out-dir checkpoints
```

You can also pass a full `--repo-id` directly.

### C. Load base VLM with `transformers` + local router checkpoint

```python
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForImageTextToText, AutoProcessor

from qacr.routing import AttentionLevelRouter, DepthOnlyRouter

base_model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
repo_id = "TezBaby/QACR-Qwen35-08B-B0.45"

router_dir = Path(snapshot_download(repo_id=repo_id))
cfg = json.loads((router_dir / "router_config.json").read_text(encoding="utf-8"))

router_cls = AttentionLevelRouter if cfg["router_type"] == "attention" else DepthOnlyRouter
router = router_cls(
    query_dim=cfg["query_dim"],
    image_dim=cfg["image_dim"],
    hidden_dim=cfg["hidden_dim"],
)
router.load_state_dict(torch.load(router_dir / "router.pt", map_location="cpu"))
router.eval()

processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)
vlm = AutoModelForImageTextToText.from_pretrained(base_model_id, trust_remote_code=True)
```

## Notes

- For strict reproducibility, keep model path and benchmark settings consistent with `Docx/项目代码计划表.md`.
- Some scripts assume local Qwen model weights under `Model/`.
- `outputs/` is git-ignored for large generated artifacts.

## Citation

If this repository is helpful, please cite the upcoming QACR paper once released.
