#!/usr/bin/env bash
# ============================================================
# Phase 6.4 P0 boost cycle:
# 1) Train QACR b0.45 with conservative executor output alpha
# 2) Run full benchmark on textvqa/docvqa/mmmu
# 3) Refresh summary + pareto
# ============================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

ALPHA="${ALPHA:-0.30}"
BUDGET="${BUDGET:-0.45}"
DATASETS="${DATASETS:-textvqa,docvqa,mmmu}"
MAX_SAMPLES="${MAX_SAMPLES:-10000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-12}"
DOCVQA_NUM_WORKERS="${DOCVQA_NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
DOCVQA_PREFETCH_FACTOR="${DOCVQA_PREFETCH_FACTOR:-1}"
TRAIN_SAVE_DIR="${TRAIN_SAVE_DIR:-checkpoints/qacr_vqav2_b0.45_alpha030}"

mkdir -p logs
LOG_FILE="logs/phase64_p0_boost_$(date +%Y%m%d_%H%M%S).log"
echo "===== Phase64 P0 boost started at $(date) =====" | tee "$LOG_FILE"
echo "ALPHA=$ALPHA BUDGET=$BUDGET SAVE_DIR=$TRAIN_SAVE_DIR" | tee -a "$LOG_FILE"

# 1) Training
echo "[1/3] Training QACR with executor_output_alpha=$ALPHA" | tee -a "$LOG_FILE"
bash scripts/run_qacr_e2e_4gpu.sh \
  --budget "$BUDGET" \
  --executor-output-alpha "$ALPHA" \
  --save-dir "$TRAIN_SAVE_DIR" 2>&1 | tee -a "$LOG_FILE"

# 2) Evaluation (current script evaluates fixed checkpoint names).
#    Keep this phase for protocol refresh; for custom checkpoint eval,
#    use scripts/eval_qacr_benchmark.py directly as needed.
echo "[2/3] Running full benchmark with alpha override=$ALPHA" | tee -a "$LOG_FILE"
EXECUTOR_OUTPUT_ALPHA="$ALPHA" \
NUM_WORKERS="$NUM_WORKERS" \
DOCVQA_NUM_WORKERS="$DOCVQA_NUM_WORKERS" \
PREFETCH_FACTOR="$PREFETCH_FACTOR" \
DOCVQA_PREFETCH_FACTOR="$DOCVQA_PREFETCH_FACTOR" \
BATCH_SIZE="$BATCH_SIZE" \
MAX_SAMPLES="$MAX_SAMPLES" \
DATASETS="$DATASETS" \
bash scripts/eval_phase62_multidataset.sh 2>&1 | tee -a "$LOG_FILE"

# 3) Summary
echo "[3/3] Refreshing final summary + pareto" | tee -a "$LOG_FILE"
conda run -n qacr python scripts/plot_phase63_pareto_frontiers.py \
  --datasets "$DATASETS" \
  --out-dir outputs/phase6_full_benchmarks 2>&1 | tee -a "$LOG_FILE"

echo "===== Phase64 P0 boost done at $(date) =====" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"

