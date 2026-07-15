#!/bin/bash
# monitor_gt_job.sh — cron-friendly babysitter for AlignmentLab gt GRPO.
# Safe on login node: only bjobs/stat/light TB read; no GPU compute.
#
# Env overrides:
#   ALAB_GT_JOBID   default 1045679
#   ALAB_GT_RUN_ID  default q3-8b_grpo-gt_math4gpu_0714_kl01b
#   ALAB_REPO       default this repo
set -euo pipefail

REPO="${ALAB_REPO:-/u/arushh/Arush/Project/AlignmentLab}"
JOBID="${ALAB_GT_JOBID:-1045679}"
RUN_ID="${ALAB_GT_RUN_ID:-q3-8b_grpo-gt_math4gpu_0714_kl01b}"
LOG_DIR="${REPO}/results/logs/rl"
TICK_LOG="${LOG_DIR}/monitor_ticks.log"
STATE_DIR="${LOG_DIR}/monitor_state"
mkdir -p "${STATE_DIR}" "${LOG_DIR}"

TS="$(date -Is)"
cd "${REPO}"

OUT="$(bjobs -w "${JOBID}" 2>&1 || true)"
STAT="$(printf '%s\n' "${OUT}" | awk -v id="${JOBID}" '$1==id{print $3; exit}')"
STAT="${STAT:-UNKNOWN}"

{
  echo "----- ${TS} cron -----"
  printf '%s\n' "${OUT}"
  echo "SUMMARY ${JOBID}:${STAT}"
} >> "${TICK_LOG}"

echo "${STAT}" > "${STATE_DIR}/last_stat"
echo "${TS}" > "${STATE_DIR}/last_tick"

case "${STAT}" in
  PEND)
    # Quiet while waiting; pending reason once per hour marker.
    REASON="$(bjobs -p "${JOBID}" 2>&1 | tail -n +2 | tr '\n' ' ' | sed 's/  */ /g' || true)"
    echo "${TS} PEND ${REASON}" >> "${STATE_DIR}/pend.log"
    ;;
  RUN)
    if [[ ! -f "${STATE_DIR}/saw_run" ]]; then
      echo "${TS} FIRST_RUN job=${JOBID} run_id=${RUN_ID}" | tee -a "${STATE_DIR}/events.log" >> "${TICK_LOG}"
      touch "${STATE_DIR}/saw_run"
      # Host for diagnostics
      HOST="$(bjobs -w "${JOBID}" 2>&1 | awk -v id="${JOBID}" '$1==id{print $6; exit}')"
      echo "${HOST}" > "${STATE_DIR}/exec_host"
    fi
    SAMPLES="${REPO}/results/runs/${RUN_ID}/samples.jsonl"
    if [[ -f "${SAMPLES}" ]]; then
      MTIME="$(stat -c '%Y' "${SAMPLES}")"
      AGE=$(( $(date +%s) - MTIME ))
      echo "${TS} samples_age_s=${AGE}" >> "${STATE_DIR}/health.log"
      if [[ "${AGE}" -gt 1200 ]]; then
        echo "${TS} WARN samples stale ${AGE}s — possible hang" | tee -a "${STATE_DIR}/events.log" >> "${TICK_LOG}"
        echo "STALE_SAMPLES age=${AGE}" > "${STATE_DIR}/ALERT"
      fi
    else
      echo "${TS} RUN but no samples.jsonl yet" >> "${STATE_DIR}/health.log"
    fi
    # Light TB peek if events exist (alab-rl; skip if conda missing)
    TB="$(ls -td "${REPO}/results/runs/${RUN_ID}/tb"/* 2>/dev/null | head -1 || true)"
    if [[ -n "${TB}" ]] && command -v conda >/dev/null 2>&1; then
      conda run -n alab-rl python - <<PY >> "${STATE_DIR}/health.log" 2>/dev/null || true
from tensorboard.backend.event_processing import event_accumulator
ea=event_accumulator.EventAccumulator("${TB}", size_guidance={"scalars":0})
ea.Reload()
tags=ea.Tags().get("scalars", [])
for t in ["train/gt_accuracy","train/kl","train/reward_mean"]:
    if t in tags:
        ev=ea.Scalars(t)[-3:]
        print("${TS}", t, [(e.step, round(e.value,4)) for e in ev])
PY
    fi
    ;;
  DONE)
    echo "${TS} DONE job=${JOBID}" | tee -a "${STATE_DIR}/events.log" >> "${TICK_LOG}"
    echo "DONE ${TS}" > "${STATE_DIR}/ALERT"
    # Kick autonomous completion → eval → push → Llama (idempotent).
    if [[ -x "${REPO}/scripts/lsf/gt_completion_pipeline.sh" ]]; then
      ALAB_GT_JOBID="${JOBID}" ALAB_GT_RUN_ID="${RUN_ID}" \
        bash "${REPO}/scripts/lsf/gt_completion_pipeline.sh" \
        >> "${LOG_DIR}/pipeline.log" 2>&1 || true
    fi
    ;;
  EXIT|USUSP|SSUSP|ZOMBI|UNKNOWN)
    echo "${TS} ${STAT} job=${JOBID} — needs attention" | tee -a "${STATE_DIR}/events.log" >> "${TICK_LOG}"
    echo "${STAT} ${TS}" > "${STATE_DIR}/ALERT"
    ERR="${LOG_DIR}/grpo_q3-8b_grpo_math_4gpu_kl01.${JOBID}.err"
    OUTF="${LOG_DIR}/grpo_q3-8b_grpo_math_4gpu_kl01.${JOBID}.out"
    for f in "${ERR}" "${OUTF}"; do
      if [[ -f "${f}" ]]; then
        echo "---- tail ${f} ----" >> "${STATE_DIR}/events.log"
        tail -n 40 "${f}" >> "${STATE_DIR}/events.log" || true
      fi
    done
    # Auto-resubmit ONLY obvious Ray MetricsHead startup flake (not OOM / hang).
    if [[ "${STAT}" == "EXIT" ]] && [[ ! -f "${STATE_DIR}/auto_resubmitted" ]]; then
      if rg -q "MetricsHead|Received EOF from pipe" "${ERR}" "${OUTF}" 2>/dev/null; then
        echo "${TS} auto-resubmit kl01c after Ray host flake" | tee -a "${STATE_DIR}/events.log"
        cd "${REPO}"
        bash scripts/lsf/ray_lsf_launch.sh --gpus 4 --cpus 16 --wall 96:00 \
          --exclude-hosts cccxc716,cccxc708 \
          --config configs/grpo/q3-8b_grpo_math_4gpu_kl01.yaml --reward-mode gt \
          --sft-ckpt hub:Arushhh/alab-q3-8b_sft_tulu_0705 \
          --run-id q3-8b_grpo-gt_math4gpu_0714_kl01c \
          >> "${STATE_DIR}/events.log" 2>&1 || true
        touch "${STATE_DIR}/auto_resubmitted"
      fi
    fi
    ;;
  *)
    echo "${TS} unexpected STAT=${STAT}" >> "${STATE_DIR}/events.log"
    ;;
esac

# Advance autonomous pipeline whenever training metrics exist (covers DONE
# race where bjobs forgets the job id, and multi-tick eval wait).
if [[ -f "${REPO}/results/runs/${RUN_ID}/metrics.json" ]] && \
   [[ ! -f "${STATE_DIR}/pipeline_cleaned" ]]; then
  ALAB_GT_JOBID="${JOBID}" ALAB_GT_RUN_ID="${RUN_ID}" \
    bash "${REPO}/scripts/lsf/gt_completion_pipeline.sh" \
    >> "${LOG_DIR}/pipeline.log" 2>&1 || true
fi

exit 0
