#!/usr/bin/env bash
# ============================================================
# eval_phase62_docvqa_missing.sh
# Re-run only missing DocVQA methods for Phase 6.2.
# ============================================================
set -euo pipefail

MAX_SAMPLES="${MAX_SAMPLES:-10000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-12}"
DOCVQA_NUM_WORKERS="${DOCVQA_NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
DOCVQA_PREFETCH_FACTOR="${DOCVQA_PREFETCH_FACTOR:-1}"
NO_PERSISTENT_WORKERS="${NO_PERSISTENT_WORKERS:-0}"
NO_PIN_MEMORY="${NO_PIN_MEMORY:-0}"
DATA_DIR="${DATA_DIR:-/data1/pengrui/CCFA/QACR/data}"
MODEL_DIR="${MODEL_DIR:-Model/Qwen35-08B}"
DATASET="docvqa"

mkdir -p logs
LOG_FILE="logs/eval_phase62_docvqa_missing_$(date +%Y%m%d_%H%M%S).log"
echo "===== Phase 6.2 DocVQA missing rerun started at $(date) =====" | tee "$LOG_FILE"
echo "BATCH_SIZE=$BATCH_SIZE NUM_WORKERS=$NUM_WORKERS DOCVQA_NUM_WORKERS=$DOCVQA_NUM_WORKERS" | tee -a "$LOG_FILE"

build_common_args() {
    local -n _out_ref=$1
    _out_ref=(
        --model "$MODEL_DIR"
        --dataset "$DATASET"
        --local-data-dir "$DATA_DIR"
        --max-samples "$MAX_SAMPLES"
        --batch-size "$BATCH_SIZE"
        --num-workers "$DOCVQA_NUM_WORKERS"
        --prefetch-factor "$DOCVQA_PREFETCH_FACTOR"
    )
    if [[ "$NO_PERSISTENT_WORKERS" == "1" ]]; then
        _out_ref+=(--no-persistent-workers)
    fi
    if [[ "$NO_PIN_MEMORY" == "1" ]]; then
        _out_ref+=(--no-pin-memory)
    fi
}

run_eval() {
    local gpu_id="$1"
    local ckpt="$2"
    local out_file="$3"
    local common=()
    build_common_args common
    echo "[GPU $gpu_id] ckpt=$ckpt" | tee -a "$LOG_FILE"
    CUDA_VISIBLE_DEVICES="$gpu_id" conda run -n qacr --no-capture-output python scripts/eval_qacr_benchmark.py \
        --checkpoint-dir "$ckpt" \
        "${common[@]}" \
        --out-file "$out_file" < /dev/null 2>&1 | tee -a "$LOG_FILE" &
}

run_sota_eval() {
    local gpu_id="$1"
    local method="$2"
    local common=()
    build_common_args common
    echo "[GPU $gpu_id] sota=$method" | tee -a "$LOG_FILE"
    CUDA_VISIBLE_DEVICES="$gpu_id" conda run -n qacr --no-capture-output python scripts/eval_external_sota.py \
        --method "$method" \
        "${common[@]}" \
        --out-dir "checkpoints/sota_eval" < /dev/null 2>&1 | tee -a "$LOG_FILE" &
}

echo "--- Batch 1: QACR 0.35 / 0.45 / 0.60 / TokenPruning ---" | tee -a "$LOG_FILE"
run_eval 0 "checkpoints/qacr_vqav2_b0.35" "checkpoints/qacr_vqav2_b0.35/eval_results_docvqa.json"
run_eval 1 "checkpoints/qacr_vqav2_b0.45" "checkpoints/qacr_vqav2_b0.45/eval_results_docvqa.json"
run_eval 2 "checkpoints/qacr_vqav2_b0.60" "checkpoints/qacr_vqav2_b0.60/eval_results_docvqa.json"
run_eval 3 "checkpoints/token_pruning_kr0.45_vqav2" "checkpoints/token_pruning_kr0.45_vqav2/eval_results_docvqa.json"
wait
echo "--- Batch 1 complete ---" | tee -a "$LOG_FILE"

echo "--- Batch 2: ImageOnly / LVPruning ---" | tee -a "$LOG_FILE"
run_eval 0 "checkpoints/image_only_b0.45_vqav2" "checkpoints/image_only_b0.45_vqav2/eval_results_docvqa.json"
run_sota_eval 1 "lvpruning"
wait
echo "--- Batch 2 complete ---" | tee -a "$LOG_FILE"

echo "===== DocVQA missing rerun done at $(date) =====" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
