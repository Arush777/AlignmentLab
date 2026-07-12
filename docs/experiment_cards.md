# Experiment Cards — Phase 1 (RQ1 control battery)

One row per planned run in Phase 1: the **six GRPO arms** (3 reward modes × 2
model families) plus the **two SFT runs** that produce the checkpoints GRPO
inits from. Run IDs follow the frozen naming contract
`{model}_{method}_{data}_{MMDD}` (PLAN.md, Interface contracts item 1). Dates
are **planned / indicative** — update them and the `status` column as runs land.

- **Reward arms** (identical data, steps, KL, group size across arms — that is
  the whole point of the control):
  - `grpo-gt` — math-verify equivalence vs gold answer (+1/0) + small format bonus.
  - `grpo-rand` — Bernoulli(0.5) per response, content-independent, seeded per run.
  - `grpo-fmt` — +1 iff a parseable `\boxed{}` answer exists, correctness ignored.
- **init_from** — base HF model for SFT; the matching family's SFT hub repo for
  every GRPO arm (the `sft_ckpt` field of `configs/grpo/*.yaml`).
- **hub_repo** — `Arushhh/alab-<run_id>`, private by default (PLAN.md, Storage).
- **data** — SFT: `allenai/tulu-3-sft-mixture` → `data/processed/sft.jsonl`.
  GRPO: `DeepScaleR-Preview-Dataset` (+ GSM8K *train*) →
  `data/processed/rlvr_math.jsonl`.

| run_id | model | method | init_from | data | status | hub_repo |
|---|---|---|---|---|---|---|
| q3-8b_sft_tulu_0705 | Qwen/Qwen3-8B | sft | Qwen/Qwen3-8B | tulu-3-sft-mixture (sft.jsonl) | RUNNING (job 828197) | Arushhh/alab-q3-8b_sft_tulu_0705 |
| q3-8b_grpo-gt_math_0712 | Qwen/Qwen3-8B | grpo-gt | Arushhh/alab-q3-8b_sft_tulu_0705 | rlvr_math (DeepScaleR+GSM8K-train) | TODO | Arushhh/alab-q3-8b_grpo-gt_math_0712 |
| q3-8b_grpo-rand_math_0712 | Qwen/Qwen3-8B | grpo-rand | Arushhh/alab-q3-8b_sft_tulu_0705 | rlvr_math (DeepScaleR+GSM8K-train) | TODO | Arushhh/alab-q3-8b_grpo-rand_math_0712 |
| q3-8b_grpo-fmt_math_0712 | Qwen/Qwen3-8B | grpo-fmt | Arushhh/alab-q3-8b_sft_tulu_0705 | rlvr_math (DeepScaleR+GSM8K-train) | TODO | Arushhh/alab-q3-8b_grpo-fmt_math_0712 |
| llama31-8b_sft_tulu_0716 | meta-llama/Llama-3.1-8B | sft | meta-llama/Llama-3.1-8B | tulu-3-sft-mixture (sft.jsonl) | TODO | Arushhh/alab-llama31-8b_sft_tulu_0716 |
| llama31-8b_grpo-gt_math_0719 | meta-llama/Llama-3.1-8B | grpo-gt | Arushhh/alab-llama31-8b_sft_tulu_0716 | rlvr_math (DeepScaleR+GSM8K-train) | TODO | Arushhh/alab-llama31-8b_grpo-gt_math_0719 |
| llama31-8b_grpo-rand_math_0719 | meta-llama/Llama-3.1-8B | grpo-rand | Arushhh/alab-llama31-8b_sft_tulu_0716 | rlvr_math (DeepScaleR+GSM8K-train) | TODO | Arushhh/alab-llama31-8b_grpo-rand_math_0719 |
| llama31-8b_grpo-fmt_math_0719 | meta-llama/Llama-3.1-8B | grpo-fmt | Arushhh/alab-llama31-8b_sft_tulu_0716 | rlvr_math (DeepScaleR+GSM8K-train) | TODO | Arushhh/alab-llama31-8b_grpo-fmt_math_0719 |

## Phase 1 — actual runs as landed (2026-07-12)

Actual run-ids differ from the planned names above (config/KL iterations). Completed
Qwen arms + baselines with eval numbers (lm-eval exact_match). Full analysis in
`docs/rq1_findings.md`; figures in `docs/figs/`.

| run_id | arm | status | gsm8k | math500 | gsm_plus | hub_repo (public) |
|---|---|---|---|---|---|---|
| q3-8b_base_none_0707 | base | DONE (eval) | 0.916 | 0.688 | 0.752 | — |
| q3-8b_sft_tulu_0705 | sft init | DONE (eval) | 0.861 | 0.430 | 0.701 | Arushhh/alab-q3-8b_sft_tulu_0705 |
| q3-8b_grpo-fmt_math4gpu_0710_k3 | grpo-fmt | DONE + eval | 0.498 | 0.248 | 0.385 | Arushhh/alab-q3-8b_grpo-fmt_math4gpu_0710_k3 |
| q3-8b_grpo-rand_math4gpu_0710_k3_v3 | grpo-rand | DONE + eval | 0.563 | 0.252 | 0.455 | Arushhh/alab-q3-8b_grpo-rand_math4gpu_0710_k3_v3 |
| q3-8b_grpo-gt_math6gpu_0712_ncol1 | grpo-gt | TRAINING (6-GPU non-colocated, job 947060) | — | — | — | (pending) |

Notes: gt required a 6-GPU **non-colocated** config (`q3-8b_grpo_math_6gpu.yaml`) after the
4-GPU colocated hybrid-engine deadlocked at ~step 57 three times. All `alab-*` model repos
are **public** (private HF storage quota was exhausted).

## Status legend
- **TODO** — planned, not yet submitted. Replace with `RUNNING` / `DONE` /
  `FAILED` once a run launches, and fill a per-run card from
  `docs/EXPERIMENT_CARD.template.md` (LSF job id, W&B link, anomalies).
- The Llama phase starts only after all Qwen arms complete (PLAN.md, Storage
  item 5: the Qwen HF cache may rotate out once Llama SFT begins).

## Compute-matching note
All six GRPO arms run under the same GPU-hour budget, and the two SFT runs are
matched across families. `metrics.json::gpu_hours` is the compute-matching
currency (PLAN.md, Interface contracts item 4). Before drawing any RQ1
conclusion, verify parity in `results/summary.csv` (produced by
`scripts/collect_results.py`); the headline figures are
`docs/figs/rq1_arms.png` and `docs/figs/rq2_perturb.png` (produced by
`scripts/plot_arms.py`).
