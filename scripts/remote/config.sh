#!/usr/bin/env bash
# Shared remote defaults for the H200 box.
# Override via env: ALAB_SSH, ALAB_REMOTE_ROOT, ALAB_SSH_OPTS

ALAB_SSH="${ALAB_SSH:-anupam@169.38.10.80}"
ALAB_REMOTE_ROOT="${ALAB_REMOTE_ROOT:-/data/anupam/AlignmentLab}"
ALAB_SSH_OPTS="${ALAB_SSH_OPTS:--o BatchMode=yes -o StrictHostKeyChecking=accept-new}"

alab_ssh() {
  # shellcheck disable=SC2086
  ssh ${ALAB_SSH_OPTS} "${ALAB_SSH}" "$@"
}

alab_rsync() {
  # shellcheck disable=SC2086
  rsync -az --delete \
    --exclude '.git/' \
    --exclude '.venv*/' \
    --exclude '__pycache__/' \
    --exclude '.pytest_cache/' \
    --exclude 'wandb/' \
    --exclude 'data/raw/' \
    --exclude 'data/processed/' \
    --exclude 'results/runs/*/samples.jsonl' \
    --exclude 'results/evals/*/passk_samples.jsonl' \
    --exclude 'third_party/' \
    --exclude '.env' \
    "$@"
}
