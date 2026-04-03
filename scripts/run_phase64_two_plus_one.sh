#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TS="${1:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="logs/phase64_two_plus_one"
OUT_DIR="outputs/tmp_eval/phase64_two_plus_one"
CONDA_BIN="${CONDA_BIN:-/home/pengr/miniconda/bin/conda}"

# Keep original settings.
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-12}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
ALPHA="${EXECUTOR_OUTPUT_ALPHA:-0.30}"

mkdir -p "$LOG_DIR" "$OUT_DIR"
LAUNCH_LOG="${LOG_DIR}/${TS}_launcher.log"

run_bg() {
  local gpu="$1"
  local ckpt="$2"
  local dataset="$3"
  local max_samples="$4"
  local tag="$5"
  local log_file="${LOG_DIR}/${TS}_${tag}.log"
  local out_file="${OUT_DIR}/${TS}_${tag}.json"
  local job_file="${LOG_DIR}/${TS}_${tag}.sh"

  cat > "$job_file" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT_DIR"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=$gpu
stdbuf -oL -eL "$CONDA_BIN" run -n qacr --no-capture-output python scripts/eval_qacr_benchmark.py \
  --checkpoint-dir "$ckpt" \
  --model Model/Qwen35-08B \
  --dataset "$dataset" \
  --local-data-dir data \
  --max-samples "$max_samples" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --prefetch-factor "$PREFETCH_FACTOR" \
  --executor-output-alpha "$ALPHA" \
  --out-file "$out_file" 2>&1 | tee -a "$log_file"
EOF
  chmod +x "$job_file"
  nohup bash "$job_file" >/dev/null 2>&1 &
  local pid="$!"
  echo "[$(date +%F' '%T)] START pid=$pid gpu=$gpu tag=$tag log=$log_file" >> "$LAUNCH_LOG"
  echo "$pid"
}

echo "[$(date +%F' '%T)] launch two workers first (gpu0+gpu2)" | tee -a "$LAUNCH_LOG"
PID_A="$(run_bg 0 checkpoints/qacr_vqav2_b0.35 textvqa 5000 b035_textvqa_a030)"
sleep 20
PID_B="$(run_bg 2 checkpoints/qacr_vqav2_b0.60 textvqa 5000 b060_textvqa_a030)"

echo "[$(date +%F' '%T)] waiting one of first two to finish..." | tee -a "$LAUNCH_LOG"
while true; do
  alive_a=0; alive_b=0
  kill -0 "$PID_A" 2>/dev/null && alive_a=1 || true
  kill -0 "$PID_B" 2>/dev/null && alive_b=1 || true
  if [[ "$alive_a" -eq 0 || "$alive_b" -eq 0 ]]; then
    break
  fi
  sleep 10
done

echo "[$(date +%F' '%T)] one finished, start third on gpu3" | tee -a "$LAUNCH_LOG"
run_bg 3 checkpoints/qacr_vqav2_b0.35 mmmu 900 b035_mmmu_a030 >/dev/null

echo "[$(date +%F' '%T)] all jobs submitted (2+1 mode)." | tee -a "$LAUNCH_LOG"
