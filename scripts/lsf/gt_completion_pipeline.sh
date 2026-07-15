#!/bin/bash
# gt_completion_pipeline.sh — autonomous post-DONE path for Qwen gt → push → Llama.
# Idempotent stages under results/logs/rl/monitor_state/pipeline_*.
set -euo pipefail

REPO="${ALAB_REPO:-/u/arushh/Arush/Project/AlignmentLab}"
RUN_ID="${ALAB_GT_RUN_ID:-q3-8b_grpo-gt_math4gpu_0714_kl01b}"
HUB_MODEL="${ALAB_GT_HUB:-Arushhh/alab-${RUN_ID}}"
STATE_DIR="${REPO}/results/logs/rl/monitor_state"
LOG="${REPO}/results/logs/rl/pipeline.log"
mkdir -p "${STATE_DIR}"
cd "${REPO}"

log() { echo "$(date -Is) $*" | tee -a "${LOG}"; }
mark() { date -Is > "${STATE_DIR}/pipeline_$1"; }
have() { [[ -f "${STATE_DIR}/pipeline_$1" ]]; }

if ! have verified; then
  METRICS="${REPO}/results/runs/${RUN_ID}/metrics.json"
  if [[ ! -f "${METRICS}" ]]; then
    log "WAIT metrics.json missing for ${RUN_ID}"
    exit 0
  fi
  if ! python3 -c "
import json,sys
m=json.load(open('${METRICS}'))
sys.exit(0 if m.get('returncode')==0 and m.get('hub_push_ok') else 1)
"; then
    log "FAIL bad metrics — ALERT"
    echo "BAD_METRICS ${RUN_ID}" > "${STATE_DIR}/ALERT"
    exit 1
  fi
  mark verified
  log "OK metrics verified"
fi

if ! have eval_submitted; then
  log "Submitting eval ${HUB_MODEL}"
  bash scripts/submit_eval.sh --run-id "${RUN_ID}" "${HUB_MODEL}" all >> "${LOG}" 2>&1
  mark eval_submitted
  log "OK eval submitted"
  exit 0
fi

EVAL_DIR="${REPO}/results/evals/${RUN_ID}"
if ! have eval_done; then
  if [[ -f "${EVAL_DIR}/lm_eval.json" && -f "${EVAL_DIR}/passk.json" ]]; then
    python3 -c "
import json
from pathlib import Path
d=json.load(open('${EVAL_DIR}/lm_eval.json'))
def find_gsm(obj, path=''):
    if isinstance(obj, dict):
        for k,v in obj.items():
            lk=str(k).lower()
            if 'gsm8k' in lk and isinstance(v,(int,float)):
                return float(v)
            if isinstance(v,dict) and 'exact_match' in v and 'gsm8k' in (lk+path).lower():
                return float(v['exact_match'])
            r=find_gsm(v, path+'/'+str(k))
            if r is not None: return r
    return None
gsm=find_gsm(d)
print('gsm8k', gsm)
if gsm is not None and gsm < 0.25:
    open('${STATE_DIR}/ALERT','w').write(f'LOW_GSM8K {gsm}\\n')
" || true
    mark eval_done
    log "OK eval artifacts present"
  else
    log "WAIT eval outputs"
    exit 0
  fi
fi

if ! have collected; then
  python scripts/collect_results.py >> "${LOG}" 2>&1 || true
  conda run -n alab-eval python scripts/plot_arms.py >> "${LOG}" 2>&1 || true
  mark collected
  log "OK collect/plot"
fi

if ! have pushed; then
  git add \
    docs/experiment_cards.md docs/rq1_findings.md docs/rq1_validity.md \
    docs/implementation_notes.md docs/RUNBOOK.md docs/figs \
    results/summary.csv "results/evals/${RUN_ID}" \
    README.md 2>/dev/null || true
  git reset HEAD -- '**/passk_samples.jsonl' 2>/dev/null || true
  if git diff --cached --quiet 2>/dev/null; then
    log "WARN nothing staged for results commit"
  else
    git commit -m "$(cat <<'EOF'
Add Qwen gt kl01b eval results and update RQ1 cards.

EOF
)"
    git push origin HEAD
    log "OK results pushed"
  fi
  mark pushed
fi

if ! have rematch_submitted; then
  log "Submitting KL-matched format+random rematch"
  bash scripts/lsf/ray_lsf_launch.sh --gpus 4 --cpus 16 --wall 96:00 \
    --exclude-hosts cccxc716,cccxc708 \
    --config configs/grpo/q3-8b_grpo_math_4gpu_kl01.yaml \
    --reward-mode format \
    --sft-ckpt hub:Arushhh/alab-q3-8b_sft_tulu_0705 \
    --run-id q3-8b_grpo-fmt_math4gpu_kl01_rematch >> "${LOG}" 2>&1 || log "WARN format rematch submit failed"
  bash scripts/lsf/ray_lsf_launch.sh --gpus 4 --cpus 16 --wall 96:00 \
    --exclude-hosts cccxc716,cccxc708 \
    --config configs/grpo/q3-8b_grpo_math_4gpu_kl01.yaml \
    --reward-mode random \
    --sft-ckpt hub:Arushhh/alab-q3-8b_sft_tulu_0705 \
    --run-id q3-8b_grpo-rand_math4gpu_kl01_rematch >> "${LOG}" 2>&1 || log "WARN random rematch submit failed"
  mark rematch_submitted
fi

if ! have llama_sft_submitted; then
  rm -rf /u/arushh/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B || true
  rm -rf /u/arushh/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B || true
  rm -rf /u/arushh/.cache/huggingface/datasets/* || true
  log "Submitting Llama-3.1-8B SFT"
  bash scripts/submit_sft.sh configs/sft/llama31-8b_sft_tulu.yaml >> "${LOG}" 2>&1
  mark llama_sft_submitted
  log "OK Llama SFT submitted"
fi

if ! have cleaned; then
  rm -rf \
    results/runs/q3-8b_grpo-gt_math4gpu_0711_rfix* \
    results/runs/q3-8b_grpo-gt_math4gpu_test \
    results/runs/q3-0.6b_* \
    results/runs/check \
    scripts/_tmp_* 2>/dev/null || true
  if have llama_sft_submitted; then
    rm -rf /u/arushh/.cache/huggingface/hub/models--Qwen--Qwen3-8B || true
    log "Removed home-cache Qwen3-8B (Hub remains source of truth)"
  fi
  mark cleaned
  log "OK cleanup"
fi

log "PIPELINE IDLE"
exit 0
