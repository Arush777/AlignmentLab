#!/bin/bash
# Submit a Phase-1 SFT GPU job from configs/sft/*.yaml. The human runs this; it calls bsub.
set -euo pipefail

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  cat <<'EOF'
Usage: scripts/submit_sft.sh [--dry-run] <configs/sft/experiment.yaml>

Submits one GPU LSF SFT job using config resources and configs/cluster.yaml.
The job runs:
  conda run -n <conda_env> accelerate launch --num_processes <n_gpus> src/train/sft.py --config <config>

Examples:
  scripts/submit_sft.sh --dry-run configs/sft/q3-0.6b_sft_smoke.yaml
  scripts/submit_sft.sh configs/sft/q3-0.6b_sft_smoke.yaml
  scripts/submit_sft.sh configs/sft/q3-8b_sft_tulu.yaml
EOF
  exit 0
fi

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
  DRY_RUN=1
  shift
fi

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 [--dry-run] <configs/sft/experiment.yaml>" >&2
  exit 2
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$1"
if [[ "${CONFIG}" != /* ]]; then
  CONFIG="${REPO}/${CONFIG}"
fi
CLUSTER_CONFIG="${REPO}/configs/cluster.yaml"

yaml_get() {
  local file="$1" key="$2" default="${3:-}"
  python3 - "$file" "$key" "$default" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
key = sys.argv[2]
default = sys.argv[3]

try:
    import yaml
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cur = data
    for part in key.split("."):
        cur = cur[part]
    print(cur)
    raise SystemExit(0)
except Exception:
    pass

def clean(value: str) -> str:
    value = value.split("#", 1)[0].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        value = value[1:-1]
    return value

parts = key.split(".")
parent = None
for raw in path.read_text(encoding="utf-8").splitlines():
    if not raw.strip() or raw.strip().startswith("#") or ":" not in raw:
        continue
    indent = len(raw) - len(raw.lstrip(" "))
    k, v = raw.strip().split(":", 1)
    if len(parts) == 1 and indent == 0 and k == parts[0]:
        print(clean(v))
        raise SystemExit(0)
    if len(parts) == 2:
        if indent == 0:
            parent = k if k == parts[0] else None
        elif parent == parts[0] and k == parts[1]:
            print(clean(v))
            raise SystemExit(0)
print(default)
PY
}

RUN_ID="$(yaml_get "${CONFIG}" run_id)"
CONDA_ENV="$(yaml_get "${CONFIG}" conda_env alab-sft)"
N_GPUS="$(yaml_get "${CONFIG}" resources.n_gpus 1)"
N_CPUS="$(yaml_get "${CONFIG}" resources.n_cpus 8)"
MEM="$(yaml_get "${CONFIG}" resources.mem 128G)"
WALL="$(yaml_get "${CONFIG}" resources.wall_time 01:00)"

QUEUE="$(yaml_get "${CLUSTER_CONFIG}" queue normal)"
GPU_TYPE="$(yaml_get "${CLUSTER_CONFIG}" gpu_type a100_80gb)"
GPUS_PER_NODE="$(yaml_get "${CLUSTER_CONFIG}" gpus_per_node 8)"
SCRATCH="$(yaml_get "${CLUSTER_CONFIG}" scratch /u/arushh/alignmentlab_scratch)"
HF_HOME_CFG="$(yaml_get "${CLUSTER_CONFIG}" hf_home /u/arushh/.cache/huggingface)"
WANDB_ENTITY="$(yaml_get "${CLUSTER_CONFIG}" wandb_entity CHANGE_ME)"

if [ -z "${RUN_ID}" ]; then
  echo "Config is missing run_id: ${CONFIG}" >&2
  exit 2
fi
if [ "${N_GPUS}" -gt "${GPUS_PER_NODE}" ]; then
  echo "Single-node Phase-1 SFT supports at most gpus_per_node=${GPUS_PER_NODE}; got ${N_GPUS}" >&2
  exit 2
fi

RUN_DIR="${REPO}/results/runs/${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
if [ "${DRY_RUN}" != "1" ]; then
  mkdir -p "${RUN_DIR}" "${LOG_DIR}" "${HF_HOME_CFG}"
fi

gpu_req="num=${N_GPUS}:mode=exclusive_process"
if [[ "${GPU_TYPE}" == *"80"* || "${GPU_TYPE}" == *"a100_80gb"* ]]; then
  gpu_req="${gpu_req}:gmem=80G"
fi

submit_job() {
  local job="$1"
  local cleanup_trap="trap 'rm -rf /tmp/alab_\${LSB_JOBID}' EXIT"
  local cmd="cd ${REPO} && \
${cleanup_trap} && \
mkdir -p /tmp/alab_\${LSB_JOBID} && \
source scripts/lsf/env.sh && \
export ALAB_SCRATCH=${SCRATCH} HF_HOME=${HF_HOME_CFG} WANDB_ENTITY=${WANDB_ENTITY} && \
export ALAB_NODE_LOCAL=1 ALAB_HUB_PUSH=1 ALAB_NODE_TMP=/tmp/alab_\${LSB_JOBID} && \
conda run -n ${CONDA_ENV} accelerate launch --num_processes ${N_GPUS} src/train/sft.py --config ${CONFIG}"

  echo "run_id=${RUN_ID}"
  echo "log_path=${LOG_DIR}/${job}.%J.out"
  if [ "${DRY_RUN}" = "1" ]; then
    echo "cleanup_trap=${cleanup_trap}"
    echo "bsub_job=${job}"
    echo "bsub_queue=${QUEUE}"
    echo "bsub_gpu=${gpu_req}"
    echo "job_command=${cmd}"
    LSB_JOBID=DRYRUN \
    ALAB_NODE_LOCAL=1 \
    ALAB_HUB_PUSH=1 \
    ALAB_NODE_TMP=/tmp/alab_DRYRUN \
    ALAB_SCRATCH="${SCRATCH}" \
    HF_HOME="${HF_HOME_CFG}" \
      python3 "${REPO}/src/train/sft.py" --config "${CONFIG}" --dry-run
    return 0
  fi
  bsub -q "${QUEUE}" \
       -J "${job}" \
       -n "${N_CPUS}" \
       -gpu "${gpu_req}" \
       -R "rusage[mem=${MEM}]" \
       -R "span[hosts=1]" \
       -W "${WALL}" \
       -o "${LOG_DIR}/${job}.%J.out" \
       -e "${LOG_DIR}/${job}.%J.err" \
       bash -lc "${cmd}"
}

submit_job "${RUN_ID}"
