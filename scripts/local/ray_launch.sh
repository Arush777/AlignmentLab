#!/usr/bin/env bash
# Local (non-LSF) Ray + GRPO launcher for the H200 worker.
# Prefer: scripts/remote/job.sh start ... -- bash scripts/local/ray_launch.sh ...
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO}"
export PATH="${HOME}/.local/bin:${PATH}"

ALAB="${REPO}/scripts/alab"
GPUS="${CUDA_VISIBLE_DEVICES:-0}"
# Count commas+1
NGPU="$(awk -F',' '{print NF}' <<<"${GPUS}")"

FORWARD=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      # Ignored when CUDA_VISIBLE_DEVICES already set by job.sh; still forwarded to train_grpo
      FORWARD+=("$1" "$2"); shift 2
      ;;
    *) FORWARD+=("$1"); shift ;;
  esac
done

export ALAB_SCRATCH="${ALAB_SCRATCH:-${REPO}/scratch}"
export HF_HOME="${HF_HOME:-/data/anupam/.cache/huggingface}"
export ALAB_NODE_TMP="${ALAB_NODE_TMP:-/tmp/alab_${$}}"
export ALAB_NODE_LOCAL="${ALAB_NODE_LOCAL:-1}"
export ALAB_HUB_PUSH="${ALAB_HUB_PUSH:-1}"
mkdir -p "${ALAB_SCRATCH}" "${ALAB_NODE_TMP}" results/runs

# Rewrite hub: SFT ckpts to local paths
RESOLVED=()
i=0
while [[ $i -lt ${#FORWARD[@]} ]]; do
  arg="${FORWARD[$i]}"
  if [[ "${arg}" == "--sft-ckpt" ]]; then
    next="${FORWARD[$((i + 1))]:-}"
    if [[ "${next}" == hub:* ]]; then
      repo_id="${next#hub:}"
      dest="${ALAB_NODE_TMP}/sft_init"
      echo "Fetching ${repo_id} → ${dest}"
      fetched="$("${ALAB}" rl bash scripts/fetch_hub_ckpt.sh "${repo_id}" "${dest}" | awk 'NF{line=$0} END{print line}')"
      RESOLVED+=(--sft-ckpt "${fetched}")
      i=$((i + 2))
      continue
    fi
  fi
  RESOLVED+=("${arg}")
  i=$((i + 1))
done

RAY_PORT="${RAY_PORT:-$((6380 + RANDOM % 500))}"
# Ray AF_UNIX sockets must stay under ~107 bytes; keep temp under /tmp, not deep scratch paths.
RAY_TMP="${RAY_TMP:-/tmp/alab_ray_${$}}"
mkdir -p "${RAY_TMP}"

cleanup() {
  echo "=== tearing down Ray ==="
  "${ALAB}" rl ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo "Resolving reward env..."
ENV_LINES="$("${ALAB}" rl python -u src/rl/train_grpo.py --emit-env --gpus "${NGPU}" "${RESOLVED[@]}")"
while IFS= read -r kv; do
  case "$kv" in ALAB_*=*|HF_HOME=*) export "${kv?}" ;; esac
done <<< "${ENV_LINES}"
echo "Reward env: ALAB_REWARD_MODE=${ALAB_REWARD_MODE:-?} ALAB_RUN_ID=${ALAB_RUN_ID:-?}"

HEAD_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
HEAD_IP="${HEAD_IP:-127.0.0.1}"
echo "Starting Ray head ${HEAD_IP}:${RAY_PORT} gpus=${NGPU}"
"${ALAB}" rl ray start --head \
  --node-ip-address="${HEAD_IP}" \
  --port="${RAY_PORT}" \
  --num-gpus="${NGPU}" \
  --temp-dir="${RAY_TMP}" \
  --disable-usage-stats

export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
for i in $(seq 1 30); do
  if "${ALAB}" rl ray status >/dev/null 2>&1; then
    echo "Ray ready (${i})"
    break
  fi
  sleep 2
  [[ "$i" -eq 30 ]] && { echo "Ray failed to start" >&2; exit 1; }
done

exec "${ALAB}" rl python -u src/rl/train_grpo.py --gpus "${NGPU}" --run-id "${ALAB_RUN_ID}" "${RESOLVED[@]}"
