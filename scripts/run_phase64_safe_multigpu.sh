#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TS="${1:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="logs/phase64_parallel_live"
OUT_DIR="outputs/tmp_eval/phase64_parallel_live"
CONDA_BIN="${CONDA_BIN:-/home/pengr/miniconda/bin/conda}"

# Standard settings (user-requested, no downscale).
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-12}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
DOCVQA_NUM_WORKERS="${DOCVQA_NUM_WORKERS:-4}"
DOCVQA_PREFETCH_FACTOR="${DOCVQA_PREFETCH_FACTOR:-1}"
ALPHA="${EXECUTOR_OUTPUT_ALPHA:-0.30}"
STAGGER_SEC="${STAGGER_SEC:-45}"

mkdir -p "$LOG_DIR" "$OUT_DIR"

launch_job() {
  local gpu="$1"
  local ckpt="$2"
  local dataset="$3"
  local max_samples="$4"
  local tag="$5"

  local log_file="${LOG_DIR}/${TS}_${tag}.log"
  local out_file="${OUT_DIR}/${TS}_${tag}.json"
  local job_script="${LOG_DIR}/${TS}_${tag}.sh"
  local workers="$NUM_WORKERS"
  local prefetch="$PREFETCH_FACTOR"
  if [[ "$dataset" == "docvqa" ]]; then
    workers="$DOCVQA_NUM_WORKERS"
    prefetch="$DOCVQA_PREFETCH_FACTOR"
  fi

  cat > "$job_script" <<EOF
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
  --num-workers "$workers" \
  --prefetch-factor "$prefetch" \
  --executor-output-alpha "$ALPHA" \
  --out-file "$out_file" 2>&1 | tee -a "$log_file"
EOF
  chmod +x "$job_script"

  nohup bash "$job_script" >/dev/null 2>&1 &
  local pid="$!"

  {
    echo "[$(date +%F' '%T)] START pid=$pid gpu=$gpu tag=$tag"
    echo "  ckpt=$ckpt dataset=$dataset max_samples=$max_samples"
    echo "  batch_size=$BATCH_SIZE num_workers=$workers prefetch_factor=$prefetch alpha=$ALPHA"
    echo "  log=$log_file"
    echo "  out=$out_file"
  } | tee -a "${LOG_DIR}/${TS}_launcher.log"
}

echo "[$(date +%F' '%T)] launching safe multi-gpu evals ..." | tee -a "${LOG_DIR}/${TS}_launcher.log"

# Restart the interrupted main run with visible live log.
launch_job 1 checkpoints/qacr_vqav2_b0.45 docvqa 10000 b045_docvqa_a030
sleep "$STAGGER_SEC"

# Use remaining GPUs for side tasks while docvqa is running.
launch_job 0 checkpoints/qacr_vqav2_b0.35 textvqa 5000 b035_textvqa_a030
sleep "$STAGGER_SEC"
launch_job 2 checkpoints/qacr_vqav2_b0.60 textvqa 5000 b060_textvqa_a030
sleep "$STAGGER_SEC"
launch_job 3 checkpoints/qacr_vqav2_b0.35 mmmu 900 b035_mmmu_a030

cat <<EOF

Launched. Progress checks:
  watch -n 2 nvidia-smi
  watch -n 5 "ps -eo pid,etime,pcpu,pmem,cmd | rg 'eval_qacr_benchmark.py' -S"
  tail -f ${LOG_DIR}/${TS}_b045_docvqa_a030.log
  tail -f ${LOG_DIR}/${TS}_b035_textvqa_a030.log
  tail -f ${LOG_DIR}/${TS}_b060_textvqa_a030.log
  tail -f ${LOG_DIR}/${TS}_b035_mmmu_a030.log
  tail -f ${LOG_DIR}/${TS}_launcher.log
EOF
