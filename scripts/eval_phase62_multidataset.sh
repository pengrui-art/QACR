#!/usr/bin/env bash
# ============================================================
# eval_phase62_multidataset.sh  —  Phase 6.2 full-scale eval
# ============================================================
# Runs fair-protocol evaluation on TextVQA / DocVQA / MMMU
# for QACR checkpoints, in-repo baselines, and external SOTA patches.
#
# Usage:
#   bash scripts/eval_phase62_multidataset.sh
#   # Optional env override:
#   #   DATASETS="textvqa,docvqa,mmmu" MAX_SAMPLES=10000 BATCH_SIZE=8 \
#   #   bash scripts/eval_phase62_multidataset.sh
# ============================================================
set -euo pipefail

DATASETS="${DATASETS:-textvqa,docvqa,mmmu}"
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
DATA_DIR="${DATA_DIR:-/data1/pengrui/CCFA/QACR/data}"
MODEL_DIR="${MODEL_DIR:-Model/Qwen35-08B}"

mkdir -p logs outputs/phase62_eval
LOG_FILE="logs/eval_phase62_$(date +%Y%m%d_%H%M%S).log"
echo "===== Phase 6.2 evaluation started at $(date) =====" | tee "$LOG_FILE"
echo "DATASETS=$DATASETS  MAX_SAMPLES=$MAX_SAMPLES  BATCH_SIZE=$BATCH_SIZE  NUM_WORKERS=$NUM_WORKERS  DOCVQA_NUM_WORKERS=$DOCVQA_NUM_WORKERS  EXECUTOR_OUTPUT_ALPHA=${EXECUTOR_OUTPUT_ALPHA:-default}  MIN_KEEP_RATIO=${MIN_KEEP_RATIO:-default}  MIN_DEEP_RATIO=${MIN_DEEP_RATIO:-default}" | tee -a "$LOG_FILE"

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

run_eval() {
    local gpu_id="$1"
    local dataset="$2"
    local ckpt="$3"
    local out_file="$4"
    local workers
    local prefetch
    workers="$(resolve_workers "$dataset")"
    prefetch="$(resolve_prefetch "$dataset")"
    echo "[GPU $gpu_id] dataset=$dataset ckpt=$ckpt" | tee -a "$LOG_FILE"
    local cmd=(
        conda run -n qacr --no-capture-output python scripts/eval_qacr_benchmark.py
        --checkpoint-dir "$ckpt"
        --model "$MODEL_DIR"
        --dataset "$dataset"
        --local-data-dir "$DATA_DIR"
        --max-samples "$MAX_SAMPLES"
        --batch-size "$BATCH_SIZE"
        --num-workers "$workers"
        --prefetch-factor "$prefetch"
        --out-file "$out_file"
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
    CUDA_VISIBLE_DEVICES="$gpu_id" "${cmd[@]}" < /dev/null 2>&1 | tee -a "$LOG_FILE" &
}

run_sota_eval() {
    local gpu_id="$1"
    local dataset="$2"
    local method="$3"
    local workers
    local prefetch
    workers="$(resolve_workers "$dataset")"
    prefetch="$(resolve_prefetch "$dataset")"
    echo "[GPU $gpu_id] dataset=$dataset sota=$method" | tee -a "$LOG_FILE"
    local cmd=(
        conda run -n qacr --no-capture-output python scripts/eval_external_sota.py
        --method "$method"
        --model "$MODEL_DIR"
        --dataset "$dataset"
        --local-data-dir "$DATA_DIR"
        --max-samples "$MAX_SAMPLES"
        --batch-size "$BATCH_SIZE"
        --num-workers "$workers"
        --prefetch-factor "$prefetch"
        --out-dir "checkpoints/sota_eval"
    )
    if [[ "$NO_PERSISTENT_WORKERS" == "1" ]]; then
        cmd+=(--no-persistent-workers)
    fi
    if [[ "$NO_PIN_MEMORY" == "1" ]]; then
        cmd+=(--no-pin-memory)
    fi
    CUDA_VISIBLE_DEVICES="$gpu_id" "${cmd[@]}" < /dev/null 2>&1 | tee -a "$LOG_FILE" &
}

IFS=',' read -r -a DATASET_LIST <<< "$DATASETS"
for dataset in "${DATASET_LIST[@]}"; do
    echo "==========================================" | tee -a "$LOG_FILE"
    echo "=== Dataset: $dataset ===" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"

    echo "--- Batch 1 ($dataset): QACR 0.35 / 0.45 / 0.60 / TokenPruning ---" | tee -a "$LOG_FILE"
    run_eval 0 "$dataset" "checkpoints/qacr_vqav2_b0.35" "checkpoints/qacr_vqav2_b0.35/eval_results_${dataset}.json"
    run_eval 1 "$dataset" "checkpoints/qacr_vqav2_b0.45" "checkpoints/qacr_vqav2_b0.45/eval_results_${dataset}.json"
    run_eval 2 "$dataset" "checkpoints/qacr_vqav2_b0.60" "checkpoints/qacr_vqav2_b0.60/eval_results_${dataset}.json"
    run_eval 3 "$dataset" "checkpoints/token_pruning_kr0.45_vqav2" "checkpoints/token_pruning_kr0.45_vqav2/eval_results_${dataset}.json"
    wait
    echo "--- Batch 1 complete ($dataset) ---" | tee -a "$LOG_FILE"

    echo "--- Batch 2 ($dataset): ImageOnly / LowRes / FastV / LVPruning ---" | tee -a "$LOG_FILE"
    run_eval 0 "$dataset" "checkpoints/image_only_b0.45_vqav2" "checkpoints/image_only_b0.45_vqav2/eval_results_${dataset}.json"
    run_eval 1 "$dataset" "checkpoints/low_res_g9_vqav2" "checkpoints/low_res_g9_vqav2/eval_results_${dataset}.json"
    run_sota_eval 2 "$dataset" "fastv"
    run_sota_eval 3 "$dataset" "lvpruning"
    wait
    echo "--- Batch 2 complete ($dataset) ---" | tee -a "$LOG_FILE"

    echo "--- Batch 3 ($dataset): Original baseline ---" | tee -a "$LOG_FILE"
    run_sota_eval 0 "$dataset" "original"
    wait
    echo "--- Batch 3 complete ($dataset) ---" | tee -a "$LOG_FILE"
done

echo "==========================================" | tee -a "$LOG_FILE"
echo "Phase 6.2 evaluation done at $(date)!" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
