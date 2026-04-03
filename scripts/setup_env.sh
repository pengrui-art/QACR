#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-qacr}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$REPO_ROOT"

if command -v conda >/dev/null 2>&1; then
  if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -n "$ENV_NAME" python=3.10 -y
  fi
  eval "$(conda shell.bash hook)"
  conda activate "$ENV_NAME"
else
  if [[ ! -d ".venv" ]]; then
    python3 -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "===== QACR Environment Ready ====="
echo "repo_root: $REPO_ROOT"
echo "python: $(command -v python)"
