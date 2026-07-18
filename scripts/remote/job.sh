#!/usr/bin/env bash
# Tmux job runner for the H200 worker. Code stays local; this only executes.
#
# Usage:
#   scripts/remote/job.sh start NAME [--gpus CSV] -- CMD...
#   scripts/remote/job.sh stop NAME
#   scripts/remote/job.sh logs NAME [-f]
#   scripts/remote/job.sh status [NAME]
#   scripts/remote/job.sh list
#   scripts/remote/job.sh attach NAME
#   scripts/remote/job.sh sync-and-start NAME [--gpus CSV] -- CMD...
#
# Example:
#   scripts/remote/job.sh start smoke --gpus 0 -- \
#     bash scripts/alab rl python -u src/rl/train_grpo.py --smoke --gpus 1 \
#       --config configs/grpo/q3-8b_grpo_math_1gpu_h200_kl01.yaml --reward-mode gt
set -euo pipefail

REPO_LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_LOCAL}/scripts/remote/config.sh"

SESSION_PREFIX="alab-"

usage() {
  sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

cmd="${1:-}"
[[ -n "${cmd}" ]] || usage 2
shift || true

session_name() {
  local name="$1"
  [[ "${name}" == alab-* ]] && echo "${name}" || echo "${SESSION_PREFIX}${name}"
}

short_name() {
  local name="$1"
  echo "${name#${SESSION_PREFIX}}"
}

do_start() {
  local name="${1:?name required}"
  shift
  local gpus=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --gpus) gpus="$2"; shift 2 ;;
      --) shift; break ;;
      -h|--help) usage 0 ;;
      *) break ;;
    esac
  done
  if [[ $# -eq 0 ]]; then
    echo "start requires -- <command...>" >&2
    exit 2
  fi

  local sess short tmpdir
  sess="$(session_name "${name}")"
  short="$(short_name "${sess}")"
  tmpdir="$(mktemp -d)"
  trap 'rm -rf "${tmpdir}"' RETURN

  # Serialize argv as NUL-separated for the remote runner.
  printf '%s\0' "$@" > "${tmpdir}/argv.bin"
  base64 < "${tmpdir}/argv.bin" | tr -d '\n' > "${tmpdir}/argv.b64"

  cat > "${tmpdir}/run.sh" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail
export PATH="${HOME}/.local/bin:${PATH}"
# results/remote_jobs/<name>.run.sh → repo root
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
SHORT="$(basename "$0" .run.sh)"
JOB_DIR="${ROOT}/results/remote_jobs"
LOG="${JOB_DIR}/${SHORT}.log"
META="${JOB_DIR}/${SHORT}.meta"
ARGV_B64_FILE="${JOB_DIR}/${SHORT}.argv.b64"

# shellcheck disable=SC1090
source "${JOB_DIR}/${SHORT}.meta.env"

if [[ -n "${ALAB_JOB_GPUS:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${ALAB_JOB_GPUS}"
fi
export ALAB_SCRATCH="${ROOT}/scratch"
export HF_HOME="${HF_HOME:-/data/anupam/.cache/huggingface}"
export ALAB_NODE_TMP="/tmp/alab_${$}"
export ALAB_NODE_LOCAL=1
mkdir -p "${ALAB_NODE_TMP}" "${ALAB_SCRATCH}"

mapfile -d '' -t CMD < <(base64 -d < "${ARGV_B64_FILE}")
echo "[$(date -Is)] START cwd=$(pwd) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} cmd=${CMD[*]}" | tee -a "${LOG}"
set +e
"${CMD[@]}" 2>&1 | tee -a "${LOG}"
ec=${PIPESTATUS[0]}
set -e
echo "[$(date -Is)] EXIT ${ec}" | tee -a "${LOG}"
{
  echo "exit_code=${ec}"
  echo "ended=$(date -Is)"
} >> "${META}"
exit "${ec}"
EOS

  cat > "${tmpdir}/meta.env" <<EOF
ALAB_JOB_GPUS=${gpus}
EOF

  cat > "${tmpdir}/meta" <<EOF
name=${short}
session=${sess}
gpus=${gpus}
started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
log=${ALAB_REMOTE_ROOT}/results/remote_jobs/${short}.log
EOF

  alab_ssh "mkdir -p ${ALAB_REMOTE_ROOT}/results/remote_jobs"
  # shellcheck disable=SC2086
  scp -q ${ALAB_SSH_OPTS} "${tmpdir}/run.sh" "${ALAB_SSH}:${ALAB_REMOTE_ROOT}/results/remote_jobs/${short}.run.sh"
  # shellcheck disable=SC2086
  scp -q ${ALAB_SSH_OPTS} "${tmpdir}/argv.b64" "${ALAB_SSH}:${ALAB_REMOTE_ROOT}/results/remote_jobs/${short}.argv.b64"
  # shellcheck disable=SC2086
  scp -q ${ALAB_SSH_OPTS} "${tmpdir}/meta" "${ALAB_SSH}:${ALAB_REMOTE_ROOT}/results/remote_jobs/${short}.meta"
  # shellcheck disable=SC2086
  scp -q ${ALAB_SSH_OPTS} "${tmpdir}/meta.env" "${ALAB_SSH}:${ALAB_REMOTE_ROOT}/results/remote_jobs/${short}.meta.env"

  alab_ssh bash -s <<EOF
set -euo pipefail
ROOT=$(printf '%q' "${ALAB_REMOTE_ROOT}")
SESS=$(printf '%q' "${sess}")
SHORT=$(printf '%q' "${short}")
chmod +x "\${ROOT}/results/remote_jobs/\${SHORT}.run.sh"
: > "\${ROOT}/results/remote_jobs/\${SHORT}.log"
if tmux has-session -t "\${SESS}" 2>/dev/null; then
  echo "ERROR: tmux session \${SESS} already exists" >&2
  exit 1
fi
tmux new-session -d -s "\${SESS}" -n job "\${ROOT}/results/remote_jobs/\${SHORT}.run.sh"
echo "started \${SESS}"
echo "log \${ROOT}/results/remote_jobs/\${SHORT}.log"
EOF
}

do_stop() {
  local name="${1:?name required}"
  local sess
  sess="$(session_name "${name}")"
  alab_ssh "tmux kill-session -t $(printf '%q' "${sess}") 2>/dev/null && echo stopped ${sess} || echo not-running ${sess}"
}

do_logs() {
  local name="${1:?name required}"
  shift || true
  local follow=0
  [[ "${1:-}" == "-f" || "${1:-}" == "--follow" ]] && follow=1
  local short
  short="$(short_name "$(session_name "${name}")")"
  local log="${ALAB_REMOTE_ROOT}/results/remote_jobs/${short}.log"
  if [[ "${follow}" -eq 1 ]]; then
    alab_ssh "tail -n 100 -F $(printf '%q' "${log}")"
  else
    alab_ssh "tail -n 80 $(printf '%q' "${log}") 2>/dev/null || echo 'no log yet'"
  fi
}

do_status() {
  local name="${1:-}"
  if [[ -n "${name}" ]]; then
    local sess short
    sess="$(session_name "${name}")"
    short="$(short_name "${sess}")"
    alab_ssh bash -s <<EOF
tmux has-session -t $(printf '%q' "${sess}") 2>/dev/null && echo "session=RUNNING ${sess}" || echo "session=DEAD ${sess}"
[[ -f ${ALAB_REMOTE_ROOT}/results/remote_jobs/${short}.meta ]] && cat ${ALAB_REMOTE_ROOT}/results/remote_jobs/${short}.meta
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader | head -8
EOF
  else
    alab_ssh bash -s <<EOF
echo "=== tmux (alab-*) ==="
tmux ls 2>/dev/null | grep '^alab-' || echo "(none)"
echo "=== recent jobs ==="
ls -lt ${ALAB_REMOTE_ROOT}/results/remote_jobs/*.meta 2>/dev/null | head -10 || true
echo "=== GPUs ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
EOF
  fi
}

do_list() {
  alab_ssh "tmux ls 2>/dev/null | grep '^alab-' || true; echo '---'; ls -1 ${ALAB_REMOTE_ROOT}/results/remote_jobs/*.meta 2>/dev/null | xargs -n1 basename 2>/dev/null || true"
}

do_attach() {
  local name="${1:?name required}"
  local sess
  sess="$(session_name "${name}")"
  # shellcheck disable=SC2086
  ssh -t ${ALAB_SSH_OPTS} "${ALAB_SSH}" "tmux attach -t $(printf '%q' "${sess}")"
}

case "${cmd}" in
  -h|--help|help) usage 0 ;;
  start) do_start "$@" ;;
  stop) do_stop "$@" ;;
  logs) do_logs "$@" ;;
  status) do_status "$@" ;;
  list) do_list ;;
  attach) do_attach "$@" ;;
  sync-and-start)
    bash "${REPO_LOCAL}/scripts/remote/sync.sh" --rsync
    do_start "$@"
    ;;
  *)
    echo "Unknown command: ${cmd}" >&2
    usage 2
    ;;
esac
