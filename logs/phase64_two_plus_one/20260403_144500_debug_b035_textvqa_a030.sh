#!/usr/bin/env bash
set -euo pipefail
cd "/data1/pengrui/CCFA/QACR"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
stdbuf -oL -eL "/home/pengr/miniconda/bin/conda" run -n qacr --no-capture-output python scripts/eval_qacr_benchmark.py   --checkpoint-dir "checkpoints/qacr_vqav2_b0.35"   --model Model/Qwen35-08B   --dataset "textvqa"   --local-data-dir data   --max-samples "5000"   --batch-size "8"   --num-workers "12"   --prefetch-factor "2"   --executor-output-alpha "0.30"   --out-file "outputs/tmp_eval/phase64_two_plus_one/20260403_144500_debug_b035_textvqa_a030.json" 2>&1 | tee -a "logs/phase64_two_plus_one/20260403_144500_debug_b035_textvqa_a030.log"
