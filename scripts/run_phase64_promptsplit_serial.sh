#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TS="${1:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="logs/phase64_promptsplit_serial"
CONDA_BIN="${CONDA_BIN:-/home/pengr/miniconda/bin/conda}"

TEXTVQA_GPU="${TEXTVQA_GPU:-0}"
DOCVQA_GPU="${DOCVQA_GPU:-1}"
ALPHA="${EXECUTOR_OUTPUT_ALPHA:-0.30}"

TEXTVQA_BATCH_SIZE="${TEXTVQA_BATCH_SIZE:-8}"
TEXTVQA_NUM_WORKERS="${TEXTVQA_NUM_WORKERS:-12}"
TEXTVQA_PREFETCH_FACTOR="${TEXTVQA_PREFETCH_FACTOR:-2}"

DOCVQA_BATCH_SIZE="${DOCVQA_BATCH_SIZE:-8}"
DOCVQA_NUM_WORKERS="${DOCVQA_NUM_WORKERS:-4}"
DOCVQA_PREFETCH_FACTOR="${DOCVQA_PREFETCH_FACTOR:-1}"

mkdir -p "$LOG_DIR" "checkpoints/qacr_vqav2_b0.45/history"

LAUNCHER_LOG="${LOG_DIR}/${TS}_launcher.log"

log() {
  echo "[$(date +%F' '%T)] $*" | tee -a "$LAUNCHER_LOG"
}

backup_result() {
  local dataset="$1"
  local src="checkpoints/qacr_vqav2_b0.45/eval_results_${dataset}.json"
  local dst="checkpoints/qacr_vqav2_b0.45/history/eval_results_${dataset}.pre_serial_${TS}.json"
  if [[ -f "$src" ]]; then
    cp -f "$src" "$dst"
    log "backup $src -> $dst"
  fi
}

run_one() {
  local gpu="$1"
  local dataset="$2"
  local max_samples="$3"
  local batch_size="$4"
  local workers="$5"
  local prefetch="$6"

  local log_file="${LOG_DIR}/${TS}_${dataset}.log"
  local out_file="checkpoints/qacr_vqav2_b0.45/eval_results_${dataset}.json"

  log "START dataset=$dataset gpu=$gpu max_samples=$max_samples batch=$batch_size workers=$workers prefetch=$prefetch alpha=$ALPHA"
  log "log_file=$log_file"
  log "out_file=$out_file"

  export PYTHONUNBUFFERED=1
  CUDA_VISIBLE_DEVICES="$gpu" stdbuf -oL -eL \
    "$CONDA_BIN" run -n qacr --no-capture-output python scripts/eval_qacr_benchmark.py \
      --checkpoint-dir checkpoints/qacr_vqav2_b0.45 \
      --model Model/Qwen35-08B \
      --dataset "$dataset" \
      --local-data-dir data \
      --max-samples "$max_samples" \
      --batch-size "$batch_size" \
      --num-workers "$workers" \
      --prefetch-factor "$prefetch" \
      --executor-output-alpha "$ALPHA" \
      --out-file "$out_file" \
      2>&1 | tee "$log_file"

  log "END dataset=$dataset"
}

log "serial rerun start"
backup_result "textvqa"
backup_result "docvqa"

run_one "$TEXTVQA_GPU" "textvqa" "5000" "$TEXTVQA_BATCH_SIZE" "$TEXTVQA_NUM_WORKERS" "$TEXTVQA_PREFETCH_FACTOR"
run_one "$DOCVQA_GPU" "docvqa" "10000" "$DOCVQA_BATCH_SIZE" "$DOCVQA_NUM_WORKERS" "$DOCVQA_PREFETCH_FACTOR"

log "serial rerun all done"
