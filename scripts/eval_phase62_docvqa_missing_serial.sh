#!/usr/bin/env bash
# ============================================================
# eval_phase62_docvqa_missing_serial.sh
# Serial rerun for missing DocVQA files to maximize stability.
# ============================================================
set -euo pipefail

MAX_SAMPLES="${MAX_SAMPLES:-10000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DOCVQA_NUM_WORKERS="${DOCVQA_NUM_WORKERS:-4}"
DOCVQA_PREFETCH_FACTOR="${DOCVQA_PREFETCH_FACTOR:-1}"
NO_PERSISTENT_WORKERS="${NO_PERSISTENT_WORKERS:-0}"
NO_PIN_MEMORY="${NO_PIN_MEMORY:-0}"
DATA_DIR="${DATA_DIR:-/data1/pengrui/CCFA/QACR/data}"
MODEL_DIR="${MODEL_DIR:-Model/Qwen35-08B}"
DATASET="docvqa"
GPU_ID="${GPU_ID:-0}"

mkdir -p logs
LOG_FILE="logs/eval_phase62_docvqa_missing_serial_$(date +%Y%m%d_%H%M%S).log"
echo "===== Serial DocVQA missing rerun started at $(date) =====" | tee "$LOG_FILE"
echo "GPU_ID=$GPU_ID BATCH_SIZE=$BATCH_SIZE DOCVQA_NUM_WORKERS=$DOCVQA_NUM_WORKERS" | tee -a "$LOG_FILE"

common_args=(
    --model "$MODEL_DIR"
    --dataset "$DATASET"
    --local-data-dir "$DATA_DIR"
    --max-samples "$MAX_SAMPLES"
    --batch-size "$BATCH_SIZE"
    --num-workers "$DOCVQA_NUM_WORKERS"
    --prefetch-factor "$DOCVQA_PREFETCH_FACTOR"
)
if [[ "$NO_PERSISTENT_WORKERS" == "1" ]]; then
    common_args+=(--no-persistent-workers)
fi
if [[ "$NO_PIN_MEMORY" == "1" ]]; then
    common_args+=(--no-pin-memory)
fi

run_qacr_if_missing() {
    local ckpt="$1"
    local out_file="$2"
    if [[ -f "$out_file" ]]; then
        echo "[skip] exists: $out_file" | tee -a "$LOG_FILE"
        return 0
    fi
    echo "[run] eval_qacr_benchmark ckpt=$ckpt -> $out_file" | tee -a "$LOG_FILE"
    CUDA_VISIBLE_DEVICES="$GPU_ID" conda run -n qacr --no-capture-output python scripts/eval_qacr_benchmark.py \
        --checkpoint-dir "$ckpt" \
        "${common_args[@]}" \
        --out-file "$out_file" 2>&1 | tee -a "$LOG_FILE"
}

run_sota_if_missing() {
    local method="$1"
    local out_file="$2"
    if [[ -f "$out_file" ]]; then
        echo "[skip] exists: $out_file" | tee -a "$LOG_FILE"
        return 0
    fi
    echo "[run] eval_external_sota method=$method -> $out_file" | tee -a "$LOG_FILE"
    CUDA_VISIBLE_DEVICES="$GPU_ID" conda run -n qacr --no-capture-output python scripts/eval_external_sota.py \
        --method "$method" \
        "${common_args[@]}" \
        --out-dir "checkpoints/sota_eval" 2>&1 | tee -a "$LOG_FILE"
}

run_qacr_if_missing "checkpoints/qacr_vqav2_b0.35" "checkpoints/qacr_vqav2_b0.35/eval_results_docvqa.json"
run_qacr_if_missing "checkpoints/qacr_vqav2_b0.45" "checkpoints/qacr_vqav2_b0.45/eval_results_docvqa.json"
run_qacr_if_missing "checkpoints/qacr_vqav2_b0.60" "checkpoints/qacr_vqav2_b0.60/eval_results_docvqa.json"
run_qacr_if_missing "checkpoints/token_pruning_kr0.45_vqav2" "checkpoints/token_pruning_kr0.45_vqav2/eval_results_docvqa.json"
run_qacr_if_missing "checkpoints/image_only_b0.45_vqav2" "checkpoints/image_only_b0.45_vqav2/eval_results_docvqa.json"
run_sota_if_missing "lvpruning" "checkpoints/sota_eval/lvpruning_docvqa_results.json"

echo "===== Serial DocVQA missing rerun done at $(date) =====" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
