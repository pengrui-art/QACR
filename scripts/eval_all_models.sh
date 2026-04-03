#!/usr/bin/env bash
# ============================================================
# eval_all_models.sh  —  4-GPU concurrent evaluation
# ============================================================
# Evaluates all 6 trained checkpoints + FastV/LVPruning/Original
# across 4 GPUs.  Each GPU runs one job at a time; missing jobs
# queue up automatically in two batches.
#
# Usage:
#   bash scripts/eval_all_models.sh
#   # Check progress:
#   tail -f logs/eval_all_*.log
# ============================================================
set -e

export PATH="/home/pengr/.conda/envs/qacr/bin:$PATH"

MAX_SAMPLES=10000   # samples per model (controls speed vs. accuracy)
BATCH_SIZE=8
DATA_DIR="/data1/pengrui/CCFA/QACR/data"
MODEL_DIR="Model/Qwen35-08B"

mkdir -p logs
log_file="logs/eval_all_$(date +%Y%m%d_%H%M%S).log"
echo "===== Evaluation started at $(date) =====" | tee "$log_file"
echo "MAX_SAMPLES=$MAX_SAMPLES  BATCH_SIZE=$BATCH_SIZE" | tee -a "$log_file"

# ── Helper: eval one trained checkpoint ─────────────────────
run_eval() {
    local gpu_id=$1
    local ckpt=$2
    echo "[GPU $gpu_id] eval checkpoint: $ckpt" | tee -a "$log_file"
    CUDA_VISIBLE_DEVICES=$gpu_id python scripts/eval_qacr_benchmark.py \
        --checkpoint-dir "$ckpt" \
        --model "$MODEL_DIR" \
        --dataset vqav2 \
        --local-data-dir "$DATA_DIR" \
        --max-samples "$MAX_SAMPLES" \
        --batch-size "$BATCH_SIZE" < /dev/null 2>&1 | tee -a "$log_file" &
}

# ── Helper: eval one external SOTA method ───────────────────
run_sota_eval() {
    local gpu_id=$1
    local method=$2
    echo "[GPU $gpu_id] eval SOTA: $method" | tee -a "$log_file"
    CUDA_VISIBLE_DEVICES=$gpu_id python scripts/eval_external_sota.py \
        --method "$method" \
        --model "$MODEL_DIR" \
        --dataset vqav2 \
        --local-data-dir "$DATA_DIR" \
        --max-samples "$MAX_SAMPLES" \
        --batch-size "$BATCH_SIZE" < /dev/null 2>&1 | tee -a "$log_file" &
}

# ════════════════════════════════════════════════════════════
# Batch 1 — 4 QACR checkpoints  (GPUs 0-3)
# ════════════════════════════════════════════════════════════
echo "--- Batch 1: QACR 0.35 / 0.45 / 0.60 / TokenPruning ---" | tee -a "$log_file"
run_eval 0 "checkpoints/qacr_vqav2_b0.35"
run_eval 1 "checkpoints/qacr_vqav2_b0.45"
run_eval 2 "checkpoints/qacr_vqav2_b0.60"
run_eval 3 "checkpoints/token_pruning_kr0.45_vqav2"
wait
echo "--- Batch 1 complete ---" | tee -a "$log_file"

# ════════════════════════════════════════════════════════════
# Batch 2 — ImageOnly + LowRes + FastV + LVPruning  (GPUs 0-3)
# ════════════════════════════════════════════════════════════
echo "--- Batch 2: ImageOnly / LowRes / FastV / LVPruning ---" | tee -a "$log_file"
run_eval     0 "checkpoints/image_only_b0.45_vqav2"
run_eval     1 "checkpoints/low_res_g9_vqav2"
run_sota_eval 2 "fastv"
run_sota_eval 3 "lvpruning"
wait
echo "--- Batch 2 complete ---" | tee -a "$log_file"

# ════════════════════════════════════════════════════════════
# Batch 3 — Original (full model, no routing)  (GPU 0)
# ════════════════════════════════════════════════════════════
echo "--- Batch 3: Original (no routing baseline) ---" | tee -a "$log_file"
run_sota_eval 0 "original"
wait
echo "--- Batch 3 complete ---" | tee -a "$log_file"

echo "==========================================" | tee -a "$log_file"
echo "All evaluations done at $(date)!" | tee -a "$log_file"
echo "Results saved inside each checkpoint/sota_eval folder as eval_results.json" | tee -a "$log_file"
echo "Next step: python scripts/plot_pareto_vqav2.py" | tee -a "$log_file"
