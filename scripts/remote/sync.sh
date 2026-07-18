#!/usr/bin/env bash
# Push local AlignmentLab code to the H200 worker.
#
# Modes:
#   --rsync   (default) copy working tree; no git commit required — good for agent loops
#   --git     remote git pull --ff-only origin <branch>; requires you pushed first
#
# Usage:
#   scripts/remote/sync.sh
#   scripts/remote/sync.sh --git
#   scripts/remote/sync.sh --git main
set -euo pipefail

REPO_LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_LOCAL}/scripts/remote/config.sh"

MODE=rsync
BRANCH=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --rsync) MODE=rsync; shift ;;
    --git) MODE=git; shift; BRANCH="${1:-}"; [[ -n "${BRANCH}" && "${BRANCH}" != --* ]] && shift || true ;;
    -h|--help)
      echo "Usage: scripts/remote/sync.sh [--rsync|--git [branch]]"
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

case "${MODE}" in
  rsync)
    echo "rsync → ${ALAB_SSH}:${ALAB_REMOTE_ROOT}/"
    alab_rsync \
      "${REPO_LOCAL}/" \
      "${ALAB_SSH}:${ALAB_REMOTE_ROOT}/"
    ;;
  git)
    if [[ -z "${BRANCH}" ]]; then
      BRANCH="$(cd "${REPO_LOCAL}" && git rev-parse --abbrev-ref HEAD)"
    fi
    echo "remote git pull --ff-only origin ${BRANCH}"
    alab_ssh "cd ${ALAB_REMOTE_ROOT} && git fetch origin && git checkout ${BRANCH} && git pull --ff-only origin ${BRANCH}"
    ;;
esac

# Keep H200 cluster.yaml in place after sync (rsync may overwrite with CCC paths)
alab_ssh "cd ${ALAB_REMOTE_ROOT} && if [[ -f configs/cluster.h200.yaml ]]; then cp -f configs/cluster.h200.yaml configs/cluster.yaml; fi && mkdir -p results/remote_jobs scratch"

echo "sync ok"
