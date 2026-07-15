# Runbook — AlignmentLab (IBM CCC / LSF)

All GPU work via `bsub`. Never run training/eval on the login node.
Conda envs: `alab-rl` (GRPO), `alab-sft` (SFT), `alab-eval` (plots / some eval helpers).

Host exclusions (Ray MetricsHead flake): `cccxc716`, `cccxc708`.

---

## Setup

```bash
cd /u/arushh/Arush/Project/AlignmentLab
# envs from envs/*.yml if needed
conda activate alab-rl   # or: conda run -n alab-rl ...
```

Cluster knobs: `configs/cluster.yaml` (queue, scratch, hf_home).

---

## Download / preprocess data

```bash
bash scripts/submit_download.sh   # or project-local data scripts under src/data/
# Expected: data/processed/sft.jsonl, data/processed/rlvr_math.jsonl
```

---

## Smoke test (1 GPU)

```bash
bash scripts/lsf/ray_lsf_launch.sh \
  --smoke --gpus 1 --wall 01:00 \
  --exclude-hosts cccxc716,cccxc708 \
  --config configs/grpo/q3-8b_grpo_math_4gpu_test.yaml \
  --reward-mode gt \
  --sft-ckpt hub:Arushhh/alab-q3-8b_sft_tulu_0705
```

---

## SFT 8B

```bash
bash scripts/submit_sft.sh   # see script for config path / run-id
# Landed init: Arushhh/alab-q3-8b_sft_tulu_0705 (public)
```

## DPO 8B

Not implemented (`NotImplementedError`). Deferred until Phase 1 control battery is clean.

---

## GRPO 8B — current gt race (KL=0.1)

Monitor only; first of three to RUN wins → `bkill` still-PEND losers:

| job | GPUs | config | run-id |
|---|---|---|---|
| 951566 | 6 (4+2) | `configs/grpo/q3-8b_grpo_math_6gpu_kl01.yaml` | `q3-8b_grpo-gt_math6gpu_0712_kl01` |
| 961251 | 5 (4+1) | `configs/grpo/q3-8b_grpo_math_5gpu_kl01.yaml` | `q3-8b_grpo-gt_math5gpu_0713_kl01` |
| 1007448 | 4 (3+1) | `configs/grpo/q3-8b_grpo_math_4gpu_kl01.yaml` | `q3-8b_grpo-gt_math4gpu_0714_kl01` |

```bash
bjobs -w 951566 961251 1007448
```

### Resubmit after startup EXIT (Ray host flake only)

```bash
# 6-GPU example (increment run-id kl01 → kl01b)
bash scripts/lsf/ray_lsf_launch.sh \
  --gpus 6 --wall 96:00 \
  --exclude-hosts cccxc716,cccxc708 \
  --config configs/grpo/q3-8b_grpo_math_6gpu_kl01.yaml \
  --reward-mode gt \
  --sft-ckpt hub:Arushhh/alab-q3-8b_sft_tulu_0705 \
  --run-id q3-8b_grpo-gt_math6gpu_0712_kl01b
```

### Rematch format / random (after successful gt eval)

Exact recipes: [`docs/rq1_validity.md`](rq1_validity.md).

### Health checks (RUN jobs)

```bash
# Liveness — trust this over quiet LSF .out (Ray dashboard EOF is often benign)
stat -c '%y' results/runs/<RUN_ID>/samples.jsonl

# TB scalars
p=$(ls -td results/runs/<RUN_ID>/tb/* | head -1)
conda run -n alab-rl python -c "
from tensorboard.backend.event_processing import event_accumulator
ea=event_accumulator.EventAccumulator('$p',size_guidance={'scalars':0});ea.Reload()
for t in ['train/gt_accuracy','train/reward_mean','train/kl']:
    if t in ea.Tags()['scalars']:
        ev=ea.Scalars(t);print(t,[(e.step,round(e.value,4)) for e in ev[-5:]])"
```

KL-watch (steps ~100–150): success if KL ≲0.15 and accuracy holds; escalate if KL>~0.3
with accuracy drop. Mid-run hang on non-colocated: diagnose, do **not** blind-resubmit.

---

## Evals

```bash
scripts/submit_eval.sh --run-id <run_id> Arushhh/alab-<run_id> all
# Outputs: results/evals/<run_id>/lm_eval.json, passk.json
```

Custom vLLM path writing harness-shaped JSON (not raw `lm-evaluation-harness` CLI).

---

## Collect + plot

```bash
python scripts/collect_results.py
conda run -n alab-eval python scripts/plot_arms.py
# Updates results/summary.csv and docs/figs/
```

---

## pass@k sweep

Submitted via `submit_eval.sh` task-set `all` (includes pass@k). Tune limits with
`--passk-limit` / `--passk-n-samples` on that script if needed.

---

## Hard rules

- Never login-node GPU compute.
- Never kill healthy jobs (exception: race-loser still PEND).
- No git commit/push unless user asks in-session.
- Pending jobs read `reward.py` / launch scripts from disk at launch — keep reward
  **values** behavior-preserving while race is PEND.
- Validity / rematch: [`docs/rq1_validity.md`](rq1_validity.md).
- Live loop state: [`results/logs/rl/LOOP_CONTEXT.md`](../results/logs/rl/LOOP_CONTEXT.md).
