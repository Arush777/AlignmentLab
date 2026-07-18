#!/usr/bin/env bash
# First-time (or repair) setup on the H200 worker.
# Run from Mac:  scripts/remote/bootstrap.sh
# Or on the box: bash scripts/remote/bootstrap.sh --local
set -euo pipefail

REPO_LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_LOCAL}/scripts/remote/config.sh"

LOCAL=0
SYNC_FLASH=1
for a in "$@"; do
  case "$a" in
    --local) LOCAL=1 ;;
    --no-flash-attn) SYNC_FLASH=0 ;;
    -h|--help)
      echo "Usage: scripts/remote/bootstrap.sh [--local] [--no-flash-attn]"
      exit 0
      ;;
  esac
done

remote_bootstrap() {
  alab_ssh bash -s <<EOF
set -euo pipefail
ROOT="${ALAB_REMOTE_ROOT}"
mkdir -p "\${ROOT}" /data/anupam/scratch /data/anupam/.cache/huggingface
if [[ ! -d "\${ROOT}/.git" ]]; then
  git clone git@github.com:Arush777/AlignmentLab.git "\${ROOT}"
fi
cd "\${ROOT}"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="\$HOME/.local/bin:\$PATH"
fi
export PATH="\$HOME/.local/bin:\$PATH"
# Prefer H200 cluster config when present
if [[ -f configs/cluster.h200.yaml ]]; then
  cp -f configs/cluster.h200.yaml configs/cluster.yaml
fi
mkdir -p scratch results/runs results/evals results/remote_jobs data/processed
tmux has-session -t alab-ctl 2>/dev/null || tmux new-session -d -s alab-ctl -n ctl "echo alab control session; bash"
echo "Host ready. Next: sync code, then scripts/alab sync ..."
hostname
nvidia-smi -L | head -3
uv --version
tmux ls || true
EOF
}

if [[ "${LOCAL}" -eq 1 ]]; then
  export PATH="${HOME}/.local/bin:${PATH}"
  cd "${REPO_LOCAL}"
  if [[ -f configs/cluster.h200.yaml ]]; then
    cp -f configs/cluster.h200.yaml configs/cluster.yaml
  fi
  mkdir -p scratch results/runs results/evals results/remote_jobs data/processed
  flash_flag=()
  [[ "${SYNC_FLASH}" -eq 1 ]] && flash_flag=(--flash-attn)
  bash scripts/alab sync all "${flash_flag[@]}"
else
  remote_bootstrap
  echo "Bootstrap SSH done. Pushing tree + syncing envs..."
  bash "${REPO_LOCAL}/scripts/remote/sync.sh" --rsync
  flash_flag=()
  [[ "${SYNC_FLASH}" -eq 1 ]] && flash_flag=(--flash-attn)
  alab_ssh "cd ${ALAB_REMOTE_ROOT} && export PATH=\$HOME/.local/bin:\$PATH && bash scripts/alab sync all ${flash_flag[*]}"
fi

echo "Bootstrap complete."
echo "  Control tmux: ssh ${ALAB_SSH} -t 'tmux attach -t alab-ctl'"
echo "  Start a job:  scripts/remote/job.sh start <name> -- <command>"
