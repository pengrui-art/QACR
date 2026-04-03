#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
CONDA_BIN="${CONDA_BIN:-/home/pengr/miniconda/bin/conda}"
LOG_DIR="${LOG_DIR:-logs/phaseB3_stable_eval/${TS}}"
RESULT_ROOT="${RESULT_ROOT:-}"
DATASETS="${DATASETS:-textvqa,docvqa,mmmu}"
METHODS="${METHODS:-qacr_b035,qacr_b045,qacr_b060,token_pruning,image_only,low_res,fastv,lvpruning,original}"
DATA_DIR="${DATA_DIR:-data}"
MODEL_DIR="${MODEL_DIR:-Model/Qwen35-08B}"
MAX_SAMPLES="${MAX_SAMPLES:-10000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DOCVQA_NUM_WORKERS="${DOCVQA_NUM_WORKERS:-$NUM_WORKERS}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
DOCVQA_PREFETCH_FACTOR="${DOCVQA_PREFETCH_FACTOR:-$PREFETCH_FACTOR}"
NO_PERSISTENT_WORKERS="${NO_PERSISTENT_WORKERS:-0}"
NO_PIN_MEMORY="${NO_PIN_MEMORY:-0}"
EXECUTOR_OUTPUT_ALPHA="${EXECUTOR_OUTPUT_ALPHA:-}"
MIN_KEEP_RATIO="${MIN_KEEP_RATIO:-}"
MIN_DEEP_RATIO="${MIN_DEEP_RATIO:-}"
GPU_ID="${GPU_ID:-0}"

mkdir -p "$LOG_DIR"
LAUNCHER_LOG="${LOG_DIR}/launcher.log"

log() {
  echo "[$(date +%F' '%T)] $*" | tee -a "$LAUNCHER_LOG"
}

resolve_workers() {
  local dataset="$1"
  if [[ "$dataset" == "docvqa" ]]; then
    echo "$DOCVQA_NUM_WORKERS"
  else
    echo "$NUM_WORKERS"
  fi
}

resolve_prefetch() {
  local dataset="$1"
  if [[ "$dataset" == "docvqa" ]]; then
    echo "$DOCVQA_PREFETCH_FACTOR"
  else
    echo "$PREFETCH_FACTOR"
  fi
}

get_checkpoint_dir() {
  local method="$1"
  case "$method" in
    qacr_b035) echo "checkpoints/qacr_vqav2_b0.35" ;;
    qacr_b045) echo "checkpoints/qacr_vqav2_b0.45" ;;
    qacr_b060) echo "checkpoints/qacr_vqav2_b0.60" ;;
    token_pruning) echo "checkpoints/token_pruning_kr0.45_vqav2" ;;
    image_only) echo "checkpoints/image_only_b0.45_vqav2" ;;
    low_res) echo "checkpoints/low_res_g9_vqav2" ;;
    *) return 1 ;;
  esac
}

is_external_method() {
  case "$1" in
    fastv|lvpruning|original) return 0 ;;
    *) return 1 ;;
  esac
}

resolve_output_path() {
  local method="$1"
  local dataset="$2"
  if [[ -n "$RESULT_ROOT" ]]; then
    if is_external_method "$method"; then
      echo "${RESULT_ROOT}/sota_eval/${method}_${dataset}_results.json"
    else
      echo "${RESULT_ROOT}/${method}/eval_results_${dataset}.json"
    fi
    return 0
  fi
  if is_external_method "$method"; then
    echo "checkpoints/sota_eval/${method}_${dataset}_results.json"
  else
    local ckpt
    ckpt="$(get_checkpoint_dir "$method")"
    echo "${ckpt}/eval_results_${dataset}.json"
  fi
}

backup_existing_result() {
  local out_file="$1"
  if [[ ! -f "$out_file" ]]; then
    return 0
  fi
  local out_dir
  out_dir="$(dirname "$out_file")"
  local history_dir="${out_dir}/history"
  mkdir -p "$history_dir"
  local base
  base="$(basename "$out_file" .json)"
  local backup_path="${history_dir}/${base}.pre_b3_${TS}.json"
  cp -f "$out_file" "$backup_path"
  log "backup $out_file -> $backup_path"
}

run_internal_eval() {
  local dataset="$1"
  local method="$2"
  local ckpt_dir="$3"
  local out_file="$4"
  local log_file="$5"
  local workers="$6"
  local prefetch="$7"
  local tmp_out="${out_file}.tmp.${TS}"

  backup_existing_result "$out_file"
  mkdir -p "$(dirname "$out_file")"
  rm -f "$tmp_out"

  local cmd=(
    "$CONDA_BIN" run -n qacr --no-capture-output python scripts/eval_qacr_benchmark.py
    --checkpoint-dir "$ckpt_dir"
    --model "$MODEL_DIR"
    --dataset "$dataset"
    --local-data-dir "$DATA_DIR"
    --max-samples "$MAX_SAMPLES"
    --batch-size "$BATCH_SIZE"
    --num-workers "$workers"
    --prefetch-factor "$prefetch"
    --out-file "$tmp_out"
  )
  if [[ "$NO_PERSISTENT_WORKERS" == "1" ]]; then
    cmd+=(--no-persistent-workers)
  fi
  if [[ "$NO_PIN_MEMORY" == "1" ]]; then
    cmd+=(--no-pin-memory)
  fi
  if [[ -n "$EXECUTOR_OUTPUT_ALPHA" ]]; then
    cmd+=(--executor-output-alpha "$EXECUTOR_OUTPUT_ALPHA")
  fi
  if [[ -n "$MIN_KEEP_RATIO" ]]; then
    cmd+=(--min-keep-ratio "$MIN_KEEP_RATIO")
  fi
  if [[ -n "$MIN_DEEP_RATIO" ]]; then
    cmd+=(--min-deep-ratio "$MIN_DEEP_RATIO")
  fi

  log "START method=$method dataset=$dataset gpu=$GPU_ID out=$out_file"
  export PYTHONUNBUFFERED=1
  CUDA_VISIBLE_DEVICES="$GPU_ID" stdbuf -oL -eL "${cmd[@]}" 2>&1 | tee "$log_file"
  mv -f "$tmp_out" "$out_file"
  log "END method=$method dataset=$dataset out=$out_file"
}

run_external_eval() {
  local dataset="$1"
  local method="$2"
  local out_file="$3"
  local log_file="$4"
  local workers="$5"
  local prefetch="$6"
  local out_dir
  out_dir="$(dirname "$out_file")"
  local tmp_dir="${out_dir}/.tmp_${method}_${dataset}_${TS}"

  backup_existing_result "$out_file"
  mkdir -p "$out_dir"
  rm -rf "$tmp_dir"
  mkdir -p "$tmp_dir"

  local cmd=(
    "$CONDA_BIN" run -n qacr --no-capture-output python scripts/eval_external_sota.py
    --method "$method"
    --model "$MODEL_DIR"
    --dataset "$dataset"
    --local-data-dir "$DATA_DIR"
    --max-samples "$MAX_SAMPLES"
    --batch-size "$BATCH_SIZE"
    --num-workers "$workers"
    --prefetch-factor "$prefetch"
    --out-dir "$tmp_dir"
  )
  if [[ "$NO_PERSISTENT_WORKERS" == "1" ]]; then
    cmd+=(--no-persistent-workers)
  fi
  if [[ "$NO_PIN_MEMORY" == "1" ]]; then
    cmd+=(--no-pin-memory)
  fi

  log "START method=$method dataset=$dataset gpu=$GPU_ID out=$out_file"
  export PYTHONUNBUFFERED=1
  CUDA_VISIBLE_DEVICES="$GPU_ID" stdbuf -oL -eL "${cmd[@]}" 2>&1 | tee "$log_file"
  mv -f "${tmp_dir}/${method}_${dataset}_results.json" "$out_file"
  rm -rf "$tmp_dir"
  log "END method=$method dataset=$dataset out=$out_file"
}

IFS=',' read -r -a DATASET_LIST <<< "$DATASETS"
IFS=',' read -r -a METHOD_LIST <<< "$METHODS"

log "stable eval launcher start"
log "datasets=$DATASETS"
log "methods=$METHODS"
log "gpu_id=$GPU_ID batch_size=$BATCH_SIZE max_samples=$MAX_SAMPLES"
log "num_workers=$NUM_WORKERS docvqa_num_workers=$DOCVQA_NUM_WORKERS prefetch=$PREFETCH_FACTOR docvqa_prefetch=$DOCVQA_PREFETCH_FACTOR"
if [[ -n "$RESULT_ROOT" ]]; then
  log "result_root=$RESULT_ROOT"
fi

for dataset in "${DATASET_LIST[@]}"; do
  workers="$(resolve_workers "$dataset")"
  prefetch="$(resolve_prefetch "$dataset")"
  for method in "${METHOD_LIST[@]}"; do
    log_file="${LOG_DIR}/${method}_${dataset}.log"
    out_file="$(resolve_output_path "$method" "$dataset")"
    if is_external_method "$method"; then
      run_external_eval "$dataset" "$method" "$out_file" "$log_file" "$workers" "$prefetch"
    else
      ckpt_dir="$(get_checkpoint_dir "$method")"
      run_internal_eval "$dataset" "$method" "$ckpt_dir" "$out_file" "$log_file" "$workers" "$prefetch"
    fi
  done
done

log "stable eval launcher done"
