#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-infer}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-Model/Qwen35-08B}"
IMAGE_PATH="${IMAGE_PATH:-outputs/demo_phase01.png}"
STEPS="${STEPS:-16}"
BUDGETS="${BUDGETS:-0.35,0.45,0.60}"
BUDGET="${BUDGET:-0.45}"

cd "$REPO_ROOT"

run_infer() {
  python scripts/run_qwen35_vl_infer.py \
    --model "$MODEL_DIR" \
    --image "$IMAGE_PATH"
}

run_train() {
  python scripts/train_query_adaptive_budget_sweep.py \
    --model "$MODEL_DIR" \
    --image "$IMAGE_PATH" \
    --budgets "$BUDGETS" \
    --steps "$STEPS"
}

run_compare() {
  python scripts/run_efficiency_performance_comparison.py \
    --model "$MODEL_DIR" \
    --image "$IMAGE_PATH" \
    --budget "$BUDGET" \
    --steps "$STEPS"
}

case "$MODE" in
  infer)
    run_infer
    ;;
  train)
    run_train
    ;;
  compare)
    run_compare
    ;;
  *)
    echo "Unsupported mode: $MODE"
    echo "Usage: bash scripts/run_one_click.sh [infer|train|compare]"
    exit 1
    ;;
esac
