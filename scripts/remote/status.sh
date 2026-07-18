#!/usr/bin/env bash
# Quick remote health check.
set -euo pipefail
REPO_LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_LOCAL}/scripts/remote/config.sh"

alab_ssh bash -s <<EOF
set -euo pipefail
export PATH="\$HOME/.local/bin:\$PATH"
echo "host=\$(hostname) user=\$(whoami)"
echo "root=${ALAB_REMOTE_ROOT}"
cd ${ALAB_REMOTE_ROOT} 2>/dev/null || { echo "MISSING ${ALAB_REMOTE_ROOT}"; exit 1; }
git rev-parse --short HEAD 2>/dev/null || true
git status -sb 2>/dev/null | head -5 || true
echo "uv=\$(uv --version 2>/dev/null || echo missing)"
for e in rl sft eval; do
  if [[ -x .venv-\$e/bin/python ]]; then
    echo "venv-\$e=\$(.venv-\$e/bin/python -V)"
  else
    echo "venv-\$e=MISSING"
  fi
done
echo "=== tmux ==="
tmux ls 2>/dev/null || echo "(no sessions)"
echo "=== GPUs ==="
nvidia-smi -L
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv
EOF
