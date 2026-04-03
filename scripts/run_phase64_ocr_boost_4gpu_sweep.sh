#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TS="${1:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="logs/phase64_ocr_boost"
OUT_DIR="outputs/phase64_ocr_boost/${TS}"
MODEL_DIR="${MODEL_DIR:-Model/Qwen35-08B}"
DATA_DIR="${DATA_DIR:-/data1/pengrui/CCFA/QACR/data}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/qacr_vqav2_b0.45}"
MAX_SAMPLES="${MAX_SAMPLES:-200}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TEXT_NUM_WORKERS="${TEXT_NUM_WORKERS:-4}"
DOC_NUM_WORKERS="${DOC_NUM_WORKERS:-2}"
TEXT_PREFETCH_FACTOR="${TEXT_PREFETCH_FACTOR:-1}"
DOC_PREFETCH_FACTOR="${DOC_PREFETCH_FACTOR:-1}"
EXECUTOR_OUTPUT_ALPHA="${EXECUTOR_OUTPUT_ALPHA:-0.30}"

mkdir -p "$LOG_DIR" "$OUT_DIR"
LAUNCH_LOG="${LOG_DIR}/${TS}_launcher.log"

run_pair() {
  local gpu="$1"
  local keep="$2"
  local deep="$3"
  local tag="$4"
  local log_file="${LOG_DIR}/${TS}_${tag}.log"

  {
    echo "[$(date +%F' '%T)] START gpu=${gpu} tag=${tag} keep=${keep} deep=${deep}"
    for dataset in textvqa docvqa; do
      local workers="$TEXT_NUM_WORKERS"
      local prefetch="$TEXT_PREFETCH_FACTOR"
      if [[ "$dataset" == "docvqa" ]]; then
        workers="$DOC_NUM_WORKERS"
        prefetch="$DOC_PREFETCH_FACTOR"
      fi
      local out_file="${OUT_DIR}/${tag}_${dataset}.json"
      echo "[$(date +%F' '%T)] dataset=${dataset} out=${out_file}"
      CUDA_VISIBLE_DEVICES="$gpu" stdbuf -oL -eL conda run -n qacr --no-capture-output \
        python scripts/eval_qacr_benchmark.py \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --model "$MODEL_DIR" \
        --dataset "$dataset" \
        --local-data-dir "$DATA_DIR" \
        --max-samples "$MAX_SAMPLES" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$workers" \
        --prefetch-factor "$prefetch" \
        --executor-output-alpha "$EXECUTOR_OUTPUT_ALPHA" \
        --min-keep-ratio "$keep" \
        --min-deep-ratio "$deep" \
        --out-file "$out_file"
    done
    echo "[$(date +%F' '%T)] END gpu=${gpu} tag=${tag}"
  } 2>&1 | tee "$log_file" &
}

run_pair 0 0.00 0.00 base
run_pair 1 0.08 0.02 keep008_deep002
run_pair 2 0.12 0.04 keep012_deep004
run_pair 3 0.16 0.06 keep016_deep006

echo "[$(date +%F' '%T)] launched 4-GPU OCR boost sweep: $TS" | tee -a "$LAUNCH_LOG"
echo "out_dir=$OUT_DIR" | tee -a "$LAUNCH_LOG"
echo "Use: tail -f ${LOG_DIR}/${TS}_base.log" | tee -a "$LAUNCH_LOG"
wait
