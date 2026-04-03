#!/usr/bin/env bash
set -euo pipefail
cd "/data1/pengrui/CCFA/QACR"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1
stdbuf -oL -eL "/home/pengr/miniconda/bin/conda" run -n qacr --no-capture-output python scripts/eval_qacr_benchmark.py   --checkpoint-dir "checkpoints/qacr_vqav2_b0.45"   --model Model/Qwen35-08B   --dataset "docvqa"   --local-data-dir data   --max-samples "10000"   --batch-size "4"   --num-workers "2"   --prefetch-factor "1"   --executor-output-alpha "0.30"   --out-file "outputs/tmp_eval/phase64_parallel_live/20260403_143000_b045_docvqa_a030.json" 2>&1 | tee -a "logs/phase64_parallel_live/20260403_143000_b045_docvqa_a030.log"
