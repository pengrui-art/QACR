#!/usr/bin/env bash
set -e

export PATH="/home/pengr/.conda/envs/qacr/bin:$PATH"
export PYTHONUNBUFFERED=1

LOG_FILE="logs/train_baselines_e2e_4gpu_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to ${LOG_FILE}"

echo "==================================" | tee -a $LOG_FILE
echo "Starting Token Pruning (keep_ratio=0.45)" | tee -a $LOG_FILE
python -m torch.distributed.run --standalone --nproc_per_node=4 scripts/train_baselines_e2e.py \
    --baseline token_pruning --keep-ratio 0.45 \
    --model Model/Qwen35-08B --dataset vqav2 \
    --local-data-dir /data1/pengrui/CCFA/QACR/data \
    --max-samples 20000 --epochs 3 < /dev/null 2>&1 | tee -a $LOG_FILE

echo "==================================" | tee -a $LOG_FILE
echo "Starting Image Only (budget=0.45)" | tee -a $LOG_FILE
python -m torch.distributed.run --standalone --nproc_per_node=4 scripts/train_baselines_e2e.py \
    --baseline image_only --budget 0.45 \
    --model Model/Qwen35-08B --dataset vqav2 \
    --local-data-dir /data1/pengrui/CCFA/QACR/data \
    --max-samples 20000 --epochs 3 < /dev/null 2>&1 | tee -a $LOG_FILE

echo "==================================" | tee -a $LOG_FILE
echo "Starting Low Res (grid=9)" | tee -a $LOG_FILE
python -m torch.distributed.run --standalone --nproc_per_node=4 scripts/train_baselines_e2e.py \
    --baseline low_res --low-res-grid 9 \
    --model Model/Qwen35-08B --dataset vqav2 \
    --local-data-dir /data1/pengrui/CCFA/QACR/data \
    --max-samples 20000 --epochs 3 < /dev/null 2>&1 | tee -a $LOG_FILE

echo "All baseline training finished!" | tee -a $LOG_FILE
