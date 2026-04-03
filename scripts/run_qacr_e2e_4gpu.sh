#!/usr/bin/env bash
set -euo pipefail

# 4-GPU launcher with realtime logs and tqdm progress from rank 0.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

mkdir -p logs

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OMP_NUM_THREADS

conda run -n qacr --no-capture-output \
  python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" \
  scripts/train_qacr_e2e.py \
  --model Model/Qwen35-08B \
  --dataset vqav2 \
  --local-data-dir data/VQAv2 \
  --max-samples 20000 \
  --epochs 3 \
  --num-workers 8 \
  --profile-every 10 \
  --budget 0.35 \
  "$@" 2>&1 | tee "logs/train_qacr_e2e_4gpu_$(date +%Y%m%d_%H%M%S).log"
