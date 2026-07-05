#!/bin/bash
# Submit the Phase-1 HF cache + preprocessing CPU job. The human runs this; it calls bsub.
set -euo pipefail

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  cat <<'EOF'
Usage: scripts/submit_download.sh

Submits one CPU LSF job that:
  1. caches Phase-1 datasets and model snapshots into HF_HOME
  2. preprocesses data/processed/*.jsonl offline

Environment overrides:
  CONDA_ENV=alab-sft   Conda env used through conda run -n
  PREPROCESS=1         Set PREPROCESS=0 to cache only
  WALL=12:00           LSF wall time
  N_CPUS=8             CPU slots
  MEM=128G             LSF memory rusage
EOF
  exit 0
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLUSTER_CONFIG="${REPO}/configs/cluster.yaml"

yaml_get() {
  local file="$1" key="$2" default="${3:-}"
  python3 - "$file" "$key" "$default" <<'PY'
import re
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

QUEUE="$(yaml_get "${CLUSTER_CONFIG}" queue normal)"
SCRATCH="$(yaml_get "${CLUSTER_CONFIG}" scratch /u/arushh/alignmentlab_scratch)"
HF_HOME_CFG="$(yaml_get "${CLUSTER_CONFIG}" hf_home /u/arushh/.cache/huggingface)"
WANDB_ENTITY="$(yaml_get "${CLUSTER_CONFIG}" wandb_entity CHANGE_ME)"
CONDA_ENV="${CONDA_ENV:-alab-sft}"
RUN_ID="download_preprocess_$(date +%m%d)"
RUN_DIR="${REPO}/results/runs/${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
WALL="${WALL:-12:00}"
N_CPUS="${N_CPUS:-8}"
MEM="${MEM:-128G}"
PREPROCESS="${PREPROCESS:-1}"

mkdir -p "${RUN_DIR}" "${LOG_DIR}" "${SCRATCH}" "${HF_HOME_CFG}"

submit_job() {
  local job="$1"
  local cmd="cd ${REPO} && \
source scripts/lsf/env.sh && \
export ALAB_SCRATCH=${SCRATCH} HF_HOME=${HF_HOME_CFG} WANDB_ENTITY=${WANDB_ENTITY} && \
conda run -n ${CONDA_ENV} python src/data/download.py --cluster-config configs/cluster.yaml"

  if [ "${PREPROCESS}" = "1" ]; then
    cmd="${cmd} && export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 && conda run -n ${CONDA_ENV} python src/data/preprocess.py"
  fi

  echo "run_id=${RUN_ID}"
  echo "log_path=${LOG_DIR}/${job}.%J.out"
  bsub -q "${QUEUE}" \
       -J "${job}" \
       -n "${N_CPUS}" \
       -R "rusage[mem=${MEM}]" \
       -R "span[hosts=1]" \
       -W "${WALL}" \
       -o "${LOG_DIR}/${job}.%J.out" \
       -e "${LOG_DIR}/${job}.%J.err" \
       bash -lc "${cmd}"
}

submit_job "${RUN_ID}"
