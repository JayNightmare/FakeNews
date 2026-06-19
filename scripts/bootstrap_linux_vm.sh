#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"

echo "[bootstrap] repo root: $ROOT_DIR"
echo "[bootstrap] python: $PYTHON_BIN"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[bootstrap] python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[bootstrap] creating virtual environment at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

PIP_BIN="$VENV_DIR/bin/pip"
PYTHON_VENV_BIN="$VENV_DIR/bin/python"

echo "[bootstrap] upgrading pip"
"$PIP_BIN" install --upgrade pip

echo "[bootstrap] installing repository dependencies"
"$PIP_BIN" install -r "$ROOT_DIR/requirements.txt"

echo "[bootstrap] checking torch + cuda visibility"
"$PYTHON_VENV_BIN" - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={torch.cuda.device_count()}")
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        print(f"cuda_device_{index}={torch.cuda.get_device_name(index)}")
PY

echo "[bootstrap] done"