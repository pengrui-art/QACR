#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TS="${1:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="logs/phase64_serial_queue"
OUT_DIR="outputs/tmp_eval/phase64_serial_queue"
CONDA_BIN="${CONDA_BIN:-/home/pengr/miniconda/bin/conda}"

# Keep your requested settings (no downscale).
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-12}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
DOCVQA_NUM_WORKERS="${DOCVQA_NUM_WORKERS:-4}"
DOCVQA_PREFETCH_FACTOR="${DOCVQA_PREFETCH_FACTOR:-1}"
ALPHA="${EXECUTOR_OUTPUT_ALPHA:-0.30}"

# Wait until currently running b0.45 DocVQA eval exits.
WAIT_PATTERN="${WAIT_PATTERN:-eval_qacr_benchmark.py --checkpoint-dir checkpoints/qacr_vqav2_b0.45 --model Model/Qwen35-08B --dataset docvqa}"

mkdir -p "$LOG_DIR" "$OUT_DIR"
LAUNCHER_LOG="${LOG_DIR}/${TS}_launcher.log"

echo "[$(date +%F' '%T)] queue launcher start" | tee -a "$LAUNCHER_LOG"
echo "wait_pattern=$WAIT_PATTERN" | tee -a "$LAUNCHER_LOG"
echo "batch_size=$BATCH_SIZE num_workers=$NUM_WORKERS prefetch_factor=$PREFETCH_FACTOR alpha=$ALPHA" | tee -a "$LAUNCHER_LOG"

while pgrep -af "$WAIT_PATTERN" >/dev/null 2>&1; do
  echo "[$(date +%F' '%T)] waiting current DocVQA to finish ..." | tee -a "$LAUNCHER_LOG"
  sleep 20
done

echo "[$(date +%F' '%T)] DocVQA finished, start queued jobs." | tee -a "$LAUNCHER_LOG"

run_one() {
  local gpu="$1"
  local ckpt="$2"
  local dataset="$3"
  local max_samples="$4"
  local tag="$5"

  local workers="$NUM_WORKERS"
  local prefetch="$PREFETCH_FACTOR"
  if [[ "$dataset" == "docvqa" ]]; then
    workers="$DOCVQA_NUM_WORKERS"
    prefetch="$DOCVQA_PREFETCH_FACTOR"
  fi

  local log_file="${LOG_DIR}/${TS}_${tag}.log"
  local out_file="${OUT_DIR}/${TS}_${tag}.json"

  {
    echo "[$(date +%F' '%T)] START tag=$tag gpu=$gpu"
    echo "  ckpt=$ckpt dataset=$dataset max_samples=$max_samples"
    echo "  workers=$workers prefetch=$prefetch batch=$BATCH_SIZE alpha=$ALPHA"
    echo "  log=$log_file"
    echo "  out=$out_file"
  } | tee -a "$LAUNCHER_LOG"

  set +e
  export PYTHONUNBUFFERED=1
  CUDA_VISIBLE_DEVICES="$gpu" stdbuf -oL -eL \
    "$CONDA_BIN" run -n qacr --no-capture-output python scripts/eval_qacr_benchmark.py \
      --checkpoint-dir "$ckpt" \
      --model Model/Qwen35-08B \
      --dataset "$dataset" \
      --local-data-dir data \
      --max-samples "$max_samples" \
      --batch-size "$BATCH_SIZE" \
      --num-workers "$workers" \
      --prefetch-factor "$prefetch" \
      --executor-output-alpha "$ALPHA" \
      --out-file "$out_file" \
      2>&1 | tee -a "$log_file"
  local ec=$?
  set -e

  echo "[$(date +%F' '%T)] END tag=$tag exit_code=$ec" | tee -a "$LAUNCHER_LOG"
}

# Queue items (sequential, no overlap):
run_one 0 checkpoints/qacr_vqav2_b0.35 textvqa 5000 b035_textvqa_a030
run_one 2 checkpoints/qacr_vqav2_b0.60 textvqa 5000 b060_textvqa_a030
run_one 3 checkpoints/qacr_vqav2_b0.35 mmmu 900 b035_mmmu_a030

echo "[$(date +%F' '%T)] queue all done" | tee -a "$LAUNCHER_LOG"
