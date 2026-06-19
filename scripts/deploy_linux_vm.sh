#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <ssh-target> <remote-dir>" >&2
  echo "Example: $0 user@gpu-vm ~/FakeNews" >&2
  exit 1
fi

SSH_TARGET="$1"
REMOTE_DIR="$2"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[deploy] syncing repository to $SSH_TARGET:$REMOTE_DIR"
rsync -avz --delete \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude 'artifacts/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  "$ROOT_DIR/" "$SSH_TARGET:$REMOTE_DIR/"

echo "[deploy] bootstrapping remote environment"
ssh "$SSH_TARGET" "cd '$REMOTE_DIR' && bash scripts/bootstrap_linux_vm.sh"

echo "[deploy] complete"