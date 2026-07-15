# Experiment Cards — Phase 1 (RQ1 control battery)

One row per Phase 1 run. Run IDs follow `{model}_{method}_{data}_{MMDD}` (PLAN.md).
**Matching debt:** completed format/random used KL=0.04 / 4-GPU colocated; in-flight gt
uses KL=0.1 / non-colocated. See [`rq1_validity.md`](rq1_validity.md) before claiming RQ1.

- **Reward arms** (design intent: identical data/steps/KL/group — currently KL unmatched):
  - `grpo-gt` — math-verify vs gold (+1/0) + small format bonus.
  - `grpo-rand` — Bernoulli(0.5), content-independent.
  - `grpo-fmt` — +1 iff parseable `\boxed{}`, correctness ignored.
- **init_from** — family SFT hub repo (`sft_ckpt` in `configs/grpo/*.yaml`).
- **hub_repo** — `Arushhh/alab-<run_id>` (**public**; private HF quota exhausted).

## Current status (2026-07-13)

| run_id | arm | status | gsm8k | math500 | gsm_plus | hub_repo |
|---|---|---|---|---|---|---|
| q3-8b_base_none_0707 | base | DONE (eval) | 0.916 | 0.688 | 0.752 | — |
| q3-8b_sft_tulu_0705 | sft init | DONE (eval) | 0.861 | 0.430 | 0.701 | Arushhh/alab-q3-8b_sft_tulu_0705 |
| q3-8b_grpo-fmt_math4gpu_0710_k3 | grpo-fmt | DONE + eval (KL=0.04, 4gpu colocated) | 0.498 | 0.248 | 0.385 | Arushhh/alab-q3-8b_grpo-fmt_math4gpu_0710_k3 |
| q3-8b_grpo-rand_math4gpu_0710_k3_v3 | grpo-rand | DONE + eval (KL=0.04, 4gpu colocated) | 0.563 | 0.252 | 0.455 | Arushhh/alab-q3-8b_grpo-rand_math4gpu_0710_k3_v3 |
| q3-8b_grpo-gt_math6gpu_0712_ncol1 | grpo-gt | KILLED (KL runaway @ KL=0.04) | — | — | — | (not published) |
| q3-8b_grpo-gt_math6gpu_0712_kl01 | grpo-gt | **KILLED** job 951566 (user cancelled 5/6 race 2026-07-15) | — | — | — | — |
| q3-8b_grpo-gt_math5gpu_0713_kl01 | grpo-gt | **KILLED** job 961251 (user cancelled 5/6 race 2026-07-15) | — | — | — | — |
| q3-8b_grpo-gt_math4gpu_0714_kl01 | grpo-gt | **KILLED** job 1007448 (fat `-n 48`; replaced by lean resubmit) | — | — | — | — |
| q3-8b_grpo-gt_math4gpu_0714_kl01b | grpo-gt | **PEND** job **1045679** (4gpu 3+1, KL=0.1, `-n 16`) | — | — | — | (pending) |
| llama31-8b_* | Llama phase | **BLOCKED** until gt+rematch (or explicit caveat) | — | — | — | — |

**Active gt job:** **1045679** (`…_kl01b`, lean CPU slots). After eval → rematch format+random
at 4gpu KL=0.1 ([`rq1_validity.md`](rq1_validity.md)).

Notes: gt left 4-GPU colocated after three deadlocks at ~step 57. Non-colocated works
for training but ncol1 Goodharted; kl01 race is the stronger-KL retry. Analysis:
[`rq1_findings.md`](rq1_findings.md). Figures: `docs/figs/`.

## Status legend
- **TODO / BLOCKED** — not submitted yet.
- **RACE PEND / RUNNING / DONE / KILLED / FAILED** — as landed.
- Per-run cards: `docs/EXPERIMENT_CARD.template.md` when useful.

## Compute-matching note
`metrics.json::gpu_hours` is the compute-matching currency. Before any RQ1 conclusion,
verify parity in `results/summary.csv` **and** KL/placement match per
[`rq1_validity.md`](rq1_validity.md). Headline figures: `docs/figs/rq1_arms.png`.
