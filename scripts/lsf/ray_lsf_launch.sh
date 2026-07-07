#!/bin/bash
# ray_lsf_launch.sh — Ray-on-LSF launcher for OpenRLHF GRPO (AlignmentLab Track A).
#
# Dual mode, one file:
#   SUBMIT (run on login node, no $LSB_JOBID):
#       builds an inline-flag `bsub` (LSF house style) and submits THIS script as
#       the job body. `mkdir -p` all -o/-e dirs first (LSF will not create them).
#   RUN   (inside the allocation, $LSB_JOBID set):
#       start a Ray head on the primary host, wait until ready, exec train_grpo.py,
#       then tear Ray down on any exit. Multi-node via blaunch is auto-detected and
#       attempted, but single-node is the supported/primary path.
#
# Cluster params (queue, gmem, scratch, hf cache) come from configs/cluster.yaml —
# never hardcoded (contract §6).
#
# Usage (submit):
#   scripts/lsf/ray_lsf_launch.sh --config configs/grpo/q3-8b_grpo_math.yaml \
#       --reward-mode gt --gpus 8 --wall 24:00
#   scripts/lsf/ray_lsf_launch.sh --config configs/grpo/q3-8b_grpo_math.yaml --smoke
#
# Any flag not consumed below is forwarded verbatim to train_grpo.py.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CLUSTER_YAML="${REPO}/configs/cluster.yaml"
CONDA_ENV="alab-rl"

# --- tiny YAML scalar reader (flat `key: "value"` / `key: value`) -----------------
yget() {
  # yget <key> [default]
  local key="$1" def="${2:-}" val
  val="$(sed -nE "s/^${key}:[[:space:]]*\"?([^\"#]*)\"?.*/\1/p" "${CLUSTER_YAML}" 2>/dev/null \
         | head -n1 | sed -E 's/[[:space:]]+$//')"
  if [[ -z "${val}" ]]; then echo "${def}"; else echo "${val}"; fi
}

QUEUE="$(yget queue normal)"
GPUS_PER_NODE="$(yget gpus_per_node 8)"
SCRATCH="$(yget scratch "/u/${USER}/alignmentlab_scratch")"
HF_HOME_CFG="$(yget hf_home "/u/${USER}/.cache/huggingface")"

# --- arg parsing: peel off launcher-level flags, forward the rest ------------------
CONFIG=""
GPUS=""
WALL=""
JOBNAME=""
SMOKE=0
FORWARD=()          # passed through to train_grpo.py
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)      CONFIG="$2"; FORWARD+=("$1" "$2"); shift 2 ;;
    --gpus)        GPUS="$2"; shift 2 ;;
    --wall)        WALL="$2"; shift 2 ;;
    --jobname)     JOBNAME="$2"; shift 2 ;;
    --smoke)       SMOKE=1; FORWARD+=("$1"); shift ;;
    *)             FORWARD+=("$1"); shift ;;
  esac
done

if [[ -z "${CONFIG}" ]]; then
  echo "ERROR: --config <grpo yaml> is required" >&2
  exit 2
fi

# Smoke defaults: 1 GPU, short wall. Full runs default to a whole node / long wall.
if [[ "${SMOKE}" -eq 1 ]]; then
  GPUS="${GPUS:-1}"; WALL="${WALL:-01:00}"; JOBNAME="${JOBNAME:-grpo_smoke}"
else
  GPUS="${GPUS:-${GPUS_PER_NODE}}"; WALL="${WALL:-24:00}"; JOBNAME="${JOBNAME:-grpo_$(basename "${CONFIG}" .yaml)}"
fi

# ==================================================================================
# SUBMIT MODE — not yet inside LSF.
# ==================================================================================
if [[ -z "${LSB_JOBID:-}" ]]; then
  LOG_DIR="${REPO}/results/logs/rl"
  mkdir -p "${LOG_DIR}"                      # LSF won't create -o/-e dirs
  mkdir -p "${SCRATCH}/ray"                  # ray temp / spill root

  # CPU slots: give the trainer plenty of dataloader/host threads. span[hosts=1]
  # keeps a single-node GRPO run on one host (multi-node is opt-in below).
  local_cpus=$(( GPUS * 12 )); [[ ${local_cpus} -lt 8 ]] && local_cpus=8
  # mode=shared + j_exclusive=yes: the job still owns the GPUs outright, but the
  # CUDA compute mode stays DEFAULT. mode=exclusive_process allows only one CUDA
  # context per GPU, which kills colocated vLLM+DeepSpeed (job 798410 fail vs
  # 798513 pass, identical otherwise).
  GPU_REQ="num=${GPUS}:mode=shared:j_exclusive=yes:gmem=80G"

  echo "Submitting ${JOBNAME}: queue=${QUEUE} gpus=${GPUS} wall=${WALL}"
  bsub -q "${QUEUE}" \
       -J "${JOBNAME}" \
       -n "${local_cpus}" \
       -R "span[hosts=1]" \
       -R "rusage[mem=24G]" \
       -gpu "${GPU_REQ}" \
       -W "${WALL}" \
       -o "${LOG_DIR}/${JOBNAME}.%J.out" \
       -e "${LOG_DIR}/${JOBNAME}.%J.err" \
       bash "${BASH_SOURCE[0]}" --gpus "${GPUS}" "${FORWARD[@]}"
  exit $?
fi

# ==================================================================================
# RUN MODE — inside the LSF allocation.
# ==================================================================================
echo "=== ray_lsf_launch RUN mode | job ${LSB_JOBID} on $(hostname) ==="
export HF_HOME="${HF_HOME:-${HF_HOME_CFG}}"
export ALAB_SCRATCH="${SCRATCH}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# Compute nodes may be offline; assume the HF cache is pre-populated by a CPU job.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export ALAB_NODE_LOCAL="${ALAB_NODE_LOCAL:-1}"
export ALAB_NODE_TMP="${ALAB_NODE_TMP:-/tmp/alab_${LSB_JOBID}}"
mkdir -p "${ALAB_NODE_TMP}"
if [[ "${SMOKE}" -eq 1 ]]; then
  DEFAULT_ALAB_HUB_PUSH=0
else
  DEFAULT_ALAB_HUB_PUSH=1
fi
export ALAB_HUB_PUSH="${ALAB_HUB_PUSH:-${DEFAULT_ALAB_HUB_PUSH}}"

# GPUs actually available to this job.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  NGPU=$(awk -F, '{print NF}' <<<"${CUDA_VISIBLE_DEVICES}")
else
  NGPU="${GPUS}"
fi
echo "Visible GPUs: ${NGPU} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"

HEAD_IP="$(hostname -i | awk '{print $1}')"
# Ray head (gcs_server) port, derived from job id to avoid collisions between
# concurrent jobs on a host. MUST stay clear of Ray's other fixed/reserved ports:
# client_server=10001, worker_ports=10002-19999, dashboard=8265. So we pin it to
# the 6380-7979 band (below all of them). Picking inside 10002-19999 is what killed
# job 827416 (gcs_server 13416 collided with the worker range).
RAY_PORT=$(( 6380 + (LSB_JOBID % 1600) ))
RAY_TMP="${SCRATCH}/ray/${LSB_JOBID}"
mkdir -p "${RAY_TMP}"

cleanup() {
  echo "=== tearing down Ray (job ${LSB_JOBID}) ==="
  conda run -n "${CONDA_ENV}" ray stop --force >/dev/null 2>&1 || true
  if [[ -d "${ALAB_NODE_TMP}" ]]; then
    if find "${ALAB_NODE_TMP}" -name .keep -print -quit | grep -q .; then
      echo "Preserving ${ALAB_NODE_TMP} because a .keep sentinel was found" >&2
    else
      rm -rf "${ALAB_NODE_TMP}"
    fi
  fi
}
trap cleanup EXIT INT TERM

# Resolve the reward env (arm, run id, sample-log path) and export it BEFORE ray start,
# so the raylet passes it to every worker that imports reward.py. Without this the
# workers see reward.py's defaults and ALL arms silently run `gt` (samples land in
# results/runs/unknown/). Diagnostics from --emit-env go to stderr; stdout is KEY=VALUE.
echo "Resolving reward env (train_grpo.py --emit-env)..."
ENV_LINES="$(conda run -n "${CONDA_ENV}" python -u "${REPO}/src/rl/train_grpo.py" \
             --emit-env --gpus "${NGPU}" "${FORWARD[@]}")" \
  || { echo "ERROR: --emit-env failed" >&2; exit 1; }
while IFS= read -r kv; do
  case "$kv" in ALAB_*=*|HF_HOME=*) export "${kv?}" ;; esac
done <<< "${ENV_LINES}"
echo "Reward env: ALAB_REWARD_MODE=${ALAB_REWARD_MODE:-?} ALAB_RUN_ID=${ALAB_RUN_ID:-?}"

echo "Starting Ray head at ${HEAD_IP}:${RAY_PORT} (tmp ${RAY_TMP})"
conda run -n "${CONDA_ENV}" ray start --head \
  --node-ip-address="${HEAD_IP}" \
  --port="${RAY_PORT}" \
  --num-gpus="${NGPU}" \
  --temp-dir="${RAY_TMP}" \
  --disable-usage-stats

export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"

# --- Multi-node (STRETCH): fan Ray workers out with blaunch ------------------------
# LSB_MCPU_HOSTS = "host1 nslots1 host2 nslots2 ...". If we span >1 host, start a
# worker on each non-head host. Single-node is the tested path; multi-node is
# best-effort and prints a warning.
UNIQ_HOSTS=$(echo "${LSB_MCPU_HOSTS:-}" | tr ' ' '\n' | awk 'NR%2==1' | sort -u)
NHOSTS=$(echo "${UNIQ_HOSTS}" | grep -c . || true)
if [[ "${NHOSTS}" -gt 1 ]]; then
  echo "[WARN] Multi-node allocation (${NHOSTS} hosts) — blaunch worker fan-out is STRETCH/untested."
  for h in ${UNIQ_HOSTS}; do
    [[ "${h}" == "$(hostname)" ]] && continue
    echo "  blaunch Ray worker on ${h}"
    blaunch -z "${h}" conda run -n "${CONDA_ENV}" ray start \
      --address="${RAY_ADDRESS}" --num-gpus="${NGPU}" --temp-dir="${RAY_TMP}" &
  done
fi

# --- wait for Ray readiness -------------------------------------------------------
echo "Waiting for Ray to become ready..."
for i in $(seq 1 30); do
  if conda run -n "${CONDA_ENV}" ray status >/dev/null 2>&1; then
    echo "Ray is ready (after ${i} checks)."
    break
  fi
  sleep 2
  if [[ "${i}" -eq 30 ]]; then echo "ERROR: Ray head not ready after 60s" >&2; exit 1; fi
done

# --- exec the GRPO entrypoint (connects to the existing Ray via RAY_ADDRESS) ------
echo "=== launching train_grpo.py ==="
set -x
conda run -n "${CONDA_ENV}" python -u "${REPO}/src/rl/train_grpo.py" \
  --gpus "${NGPU}" --run-id "${ALAB_RUN_ID}" "${FORWARD[@]}"
