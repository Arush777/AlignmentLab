#!/bin/bash
# Submit a 1-GPU AlignmentLab eval job for a Hub repo id or local checkpoint.
set -euo pipefail

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  cat <<'EOF'
Usage: scripts/submit_eval.sh [options] <hub-repo-id-or-local-path> [task-set]

Task set:
  all                         gsm8k,math-500,gsm_plus,passk (default)
  gsm8k,math-500,gsm_plus     lm_eval.json tasks
  passk                       AIME24 pass@k only

Options:
  --run-id RUN_ID             results/evals/<run_id> directory
  --limit N                   per-task row limit for cheap smoke evals
  --passk-limit N             AIME24 problem limit for smoke evals
  --passk-n-samples N         samples per AIME24 problem (default: 256)
  --passk-k-values CSV        default: 1,8,64,256
  --wall HH:MM                LSF wall time (default: 06:00)
  --dry-run                   print job script and do not submit

Examples:
  scripts/submit_eval.sh --run-id q3-0.6b_base_0707 Qwen/Qwen3-0.6B all
  scripts/submit_eval.sh --limit 8 --passk-limit 2 --passk-n-samples 2 \
    --passk-k-values 1,2 --run-id q3-0.6b_eval_smoke_0707 Qwen/Qwen3-0.6B all
EOF
  exit 0
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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

DRY_RUN=0
RUN_ID=""
TASK_SET="all"
LIMIT=""
PASSK_LIMIT=""
PASSK_N_SAMPLES="${PASSK_N_SAMPLES:-256}"
PASSK_K_VALUES="${PASSK_K_VALUES:-1,8,64,256}"
WALL="${WALL:-06:00}"
UV_ENV="${UV_ENV:-eval}"
N_CPUS="${N_CPUS:-8}"
MEM="${MEM:-96G}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
SAMPLES_PER_CALL="${SAMPLES_PER_CALL:-16}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --passk-limit) PASSK_LIMIT="$2"; shift 2 ;;
    --passk-n-samples) PASSK_N_SAMPLES="$2"; shift 2 ;;
    --passk-k-values) PASSK_K_VALUES="$2"; shift 2 ;;
    --wall) WALL="$2"; shift 2 ;;
    --uv-env) UV_ENV="$2"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --gpu-memory-utilization) GPU_MEM_UTIL="$2"; shift 2 ;;
    --samples-per-call) SAMPLES_PER_CALL="$2"; shift 2 ;;
    --*) echo "Unknown option: $1" >&2; exit 2 ;;
    *) break ;;
  esac
done

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: $0 [options] <hub-repo-id-or-local-path> [task-set]" >&2
  exit 2
fi

MODEL="$1"
if [ "$#" -eq 2 ]; then
  TASK_SET="$2"
fi

infer_run_id() {
  local model="$1" date_tag
  date_tag="$(date +%m%d)"
  case "${model}" in
    Qwen/Qwen3-0.6B) echo "q3-0.6b_base_${date_tag}" ;;
    Qwen/Qwen3-8B) echo "q3-8b_base_${date_tag}" ;;
    Qwen/Qwen3-14B) echo "q3-14b_base_${date_tag}" ;;
    Qwen/Qwen3-32B) echo "q3-32b_base_${date_tag}" ;;
    meta-llama/Llama-3.1-8B) echo "llama31-8b_base_${date_tag}" ;;
    *)
      basename "${model}" | tr -cs 'A-Za-z0-9._-' '_' | sed "s/_$//; s/$/_eval_${date_tag}/"
      ;;
  esac
}

if [ -z "${RUN_ID}" ]; then
  RUN_ID="$(infer_run_id "${MODEL}")"
fi

QUEUE="$(yaml_get "${CLUSTER_CONFIG}" queue normal)"
GPU_TYPE="$(yaml_get "${CLUSTER_CONFIG}" gpu_type a100_80gb)"
SCRATCH="$(yaml_get "${CLUSTER_CONFIG}" scratch "/u/${USER}/alignmentlab_scratch")"
HF_HOME_CFG="$(yaml_get "${CLUSTER_CONFIG}" hf_home "/u/${USER}/.cache/huggingface")"
WANDB_ENTITY="$(yaml_get "${CLUSTER_CONFIG}" wandb_entity CHANGE_ME)"

GPU_REQ="num=1:mode=exclusive_process"
if [[ "${GPU_TYPE}" == *"80"* || "${GPU_TYPE}" == *"a100_80gb"* ]]; then
  GPU_REQ="${GPU_REQ}:gmem=80G"
fi

case "${TASK_SET}" in
  all) LM_TASKS="gsm8k,math-500,gsm_plus"; DO_PASSK=1 ;;
  passk|pass@k|aime24) LM_TASKS=""; DO_PASSK=1 ;;
  *)
    DO_PASSK=0
    LM_TASKS=""
    IFS=',' read -ra TASKS <<< "${TASK_SET}"
    for task in "${TASKS[@]}"; do
      case "${task}" in
        passk|pass@k|aime24) DO_PASSK=1 ;;
        gsm8k|math-500|math_500|gsm_plus|gsm-plus)
          if [ -z "${LM_TASKS}" ]; then LM_TASKS="${task}"; else LM_TASKS="${LM_TASKS},${task}"; fi
          ;;
        "") ;;
        *) echo "Unsupported task in task-set: ${task}" >&2; exit 2 ;;
      esac
    done
    ;;
esac

RUN_DIR="${REPO}/results/evals/${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${RUN_DIR}" "${LOG_DIR}" "${HF_HOME_CFG}" "${SCRATCH}"

shell_quote() {
  printf "%q" "$1"
}

emit_job_script() {
  local q_repo q_model q_run_id q_out q_scratch q_hf_home q_wandb_entity q_uv_env
  q_repo="$(shell_quote "${REPO}")"
  q_model="$(shell_quote "${MODEL}")"
  q_run_id="$(shell_quote "${RUN_ID}")"
  q_out="$(shell_quote "${RUN_DIR}")"
  q_scratch="$(shell_quote "${SCRATCH}")"
  q_hf_home="$(shell_quote "${HF_HOME_CFG}")"
  q_wandb_entity="$(shell_quote "${WANDB_ENTITY}")"
  q_uv_env="$(shell_quote "${UV_ENV}")"

  cat <<EOF
#!/bin/bash
set -euo pipefail

cd ${q_repo}
source scripts/lsf/env.sh
export ALAB_SCRATCH=${q_scratch}
export HF_HOME=${q_hf_home}
export WANDB_ENTITY=${q_wandb_entity}
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="\${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="\${TRANSFORMERS_OFFLINE:-0}"

mkdir -p ${q_out}

MODEL_INPUT=${q_model}
MODEL_FOR_EVAL="\${MODEL_INPUT}"
EVAL_NODE_TMP=""

cleanup_eval_node_tmp() {
  if [ -n "\${EVAL_NODE_TMP}" ] && [ -d "\${EVAL_NODE_TMP}" ]; then
    rm -rf "\${EVAL_NODE_TMP}"
  fi
}
trap cleanup_eval_node_tmp EXIT
trap 'cleanup_eval_node_tmp; exit 130' INT
trap 'cleanup_eval_node_tmp; exit 143' TERM

if [ -e "\${MODEL_INPUT}" ]; then
  echo "Using local eval model \${MODEL_INPUT}"
else
  EVAL_NODE_TMP="/tmp/alab_eval_\${LSB_JOBID}"
  EVAL_MODEL_DEST="\${EVAL_NODE_TMP}/model"
  mkdir -p "\${EVAL_MODEL_DEST}"
  echo "Fetching eval model Hub repo \${MODEL_INPUT} into \${EVAL_MODEL_DEST}"
  if ! fetch_out="\$(HF_HUB_OFFLINE=0 ${q_repo}/scripts/alab ${q_uv_env} bash scripts/fetch_hub_ckpt.sh "\${MODEL_INPUT}" "\${EVAL_MODEL_DEST}")"; then
    echo "ERROR: failed to fetch eval model Hub repo \${MODEL_INPUT} into \${EVAL_MODEL_DEST}" >&2
    if [ -n "\${fetch_out:-}" ]; then
      printf '%s\n' "\${fetch_out}" >&2
    fi
    exit 1
  fi
  MODEL_FOR_EVAL="\$(printf '%s\n' "\${fetch_out}" | awk 'NF { line = \$0 } END { print line }')"
  if [ -z "\${MODEL_FOR_EVAL}" ] || [ ! -d "\${MODEL_FOR_EVAL}" ]; then
    echo "ERROR: fetch for eval model \${MODEL_INPUT} did not return a valid local path" >&2
    echo "Last stdout line: \${MODEL_FOR_EVAL:-<empty>}" >&2
    exit 1
  fi
  echo "Fetched eval model Hub repo \${MODEL_INPUT} to \${MODEL_FOR_EVAL}"
fi
EOF

  if [ -n "${LM_TASKS}" ]; then
    local q_lm_tasks limit_arg
    q_lm_tasks="$(shell_quote "${LM_TASKS}")"
    limit_arg=""
    if [ -n "${LIMIT}" ]; then
      limit_arg=" --limit $(shell_quote "${LIMIT}")"
    fi
    cat <<EOF
${q_repo}/scripts/alab ${q_uv_env} python -u src/evals/run_lm_eval.py \\
  --model "\${MODEL_FOR_EVAL}" \\
  --run-id ${q_run_id} \\
  --tasks ${q_lm_tasks} \\
  --output-dir ${q_out} \\
  --max-new-tokens ${MAX_NEW_TOKENS} \\
  --max-model-len ${MAX_MODEL_LEN} \\
  --gpu-memory-utilization ${GPU_MEM_UTIL}${limit_arg}
EOF
  fi

  if [ "${DO_PASSK}" = "1" ]; then
    local passk_limit_arg
    passk_limit_arg=""
    if [ -n "${PASSK_LIMIT}" ]; then
      passk_limit_arg=" --limit $(shell_quote "${PASSK_LIMIT}")"
    fi
    cat <<EOF
${q_repo}/scripts/alab ${q_uv_env} python -u src/evals/passk_generate.py \\
  --model "\${MODEL_FOR_EVAL}" \\
  --run-id ${q_run_id} \\
  --output-dir ${q_out} \\
  --n-samples ${PASSK_N_SAMPLES} \\
  --samples-per-call ${SAMPLES_PER_CALL} \\
  --max-new-tokens ${MAX_NEW_TOKENS} \\
  --max-model-len ${MAX_MODEL_LEN} \\
  --gpu-memory-utilization ${GPU_MEM_UTIL} \\
  --overwrite${passk_limit_arg}

${q_repo}/scripts/alab ${q_uv_env} python -u src/evals/passk_score.py \\
  --run-id ${q_run_id} \\
  --input ${q_out}/passk_samples.jsonl \\
  --output ${q_out}/passk.json \\
  --k-values $(shell_quote "${PASSK_K_VALUES}")
EOF
  fi
}

submit_job() {
  local job="$1"
  local job_script="${LOG_DIR}/${job}.job.sh"

  echo "run_id=${RUN_ID}"
  echo "output_dir=${RUN_DIR}"
  echo "log_path=${LOG_DIR}/${job}.%J.out"
  if [ "${DRY_RUN}" = "1" ]; then
    echo "bsub_job=${job}"
    echo "bsub_queue=${QUEUE}"
    echo "bsub_gpu=${GPU_REQ}"
    echo "job_script_body<<'ALAB_JOB'"
    emit_job_script
    echo "ALAB_JOB"
    return 0
  fi

  emit_job_script > "${job_script}"
  chmod 700 "${job_script}"
  bsub -q "${QUEUE}" \
       -J "${job}" \
       -n "${N_CPUS}" \
       -gpu "${GPU_REQ}" \
       -R "rusage[mem=${MEM}]" \
       -R "span[hosts=1]" \
       -W "${WALL}" \
       -o "${LOG_DIR}/${job}.%J.out" \
       -e "${LOG_DIR}/${job}.%J.err" \
       < "${job_script}"
}

submit_job "eval_${RUN_ID}"
