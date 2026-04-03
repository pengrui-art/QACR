#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TS="${1:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="logs/phase64_parallel_wait"
OUT_DIR="outputs/tmp_eval/phase64_parallel_wait"
CONDA_BIN="${CONDA_BIN:-/home/pengr/miniconda/bin/conda}"
mkdir -p "$LOG_DIR" "$OUT_DIR"

launch_eval() {
  local gpu="$1"
  local ckpt="$2"
  local dataset="$3"
  local max_samples="$4"
  local num_workers="$5"
  local prefetch_factor="$6"
  local batch_size="$7"
  local alpha="$8"
  local tag="$9"

  local out_file="${OUT_DIR}/${TS}_${tag}.json"
  local log_file="${LOG_DIR}/${TS}_${tag}.log"

  nohup env CUDA_VISIBLE_DEVICES="$gpu" \
    "$CONDA_BIN" run -n qacr --no-capture-output python scripts/eval_qacr_benchmark.py \
      --checkpoint-dir "$ckpt" \
      --model Model/Qwen35-08B \
      --dataset "$dataset" \
      --local-data-dir data \
      --max-samples "$max_samples" \
      --batch-size "$batch_size" \
      --num-workers "$num_workers" \
      --prefetch-factor "$prefetch_factor" \
      --executor-output-alpha "$alpha" \
      --out-file "$out_file" \
      >"$log_file" 2>&1 &

  local pid="$!"
  echo "[$(date +%F' '%T)] started pid=${pid} gpu=${gpu} tag=${tag}" | tee -a "${LOG_DIR}/${TS}_launcher.log"
  echo "  log: ${log_file}" | tee -a "${LOG_DIR}/${TS}_launcher.log"
  echo "  out: ${out_file}" | tee -a "${LOG_DIR}/${TS}_launcher.log"
}

# Fill idle GPU 0/2/3 while GPU1 continues DocVQA full-run.
launch_eval 0 checkpoints/qacr_vqav2_b0.35 textvqa 5000 12 2 8 0.30 b035_textvqa_a030
launch_eval 2 checkpoints/qacr_vqav2_b0.60 textvqa 5000 12 2 8 0.30 b060_textvqa_a030
launch_eval 3 checkpoints/qacr_vqav2_b0.35 mmmu 900 12 2 8 0.30 b035_mmmu_a030

echo
echo "Launched all jobs. Monitor with:"
echo "  watch -n 2 nvidia-smi"
echo "  watch -n 5 \"ps -eo pid,etime,pcpu,pmem,cmd | rg 'eval_qacr_benchmark.py' -S\""
echo "  tail -f ${LOG_DIR}/${TS}_launcher.log"
