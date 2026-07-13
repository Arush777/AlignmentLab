# AlignmentLab

AlignmentLab is a matched-compute comparison of SFT, DPO, and GRPO/RLVR post-training on Qwen3-8B and Qwen3-14B.
The core question is whether reinforcement learning from verifiable rewards (RLVR) actually adds capability or
only sharpens the sampling distribution — measured as pass@1 versus pass@k (k up to 256) on math reasoning.
Every method is run under the same GPU-hour budget so that gains are attributable to the learning signal rather
than to extra compute. Training uses TRL (SFT, DPO) and OpenRLHF (GRPO with Ray + vLLM); evaluation uses
lm-evaluation-harness for standard benchmarks and math-verify for verifiable reward and pass@k scoring.
Experiments run on A100/H100 nodes managed by IBM LSF (`bsub`/`bjobs`); no compute is ever run on the login node.
Runs are logged to Weights & Biases (public project `alignmentlab`).

## Research questions
- **RQ1:** SFT vs DPO vs GRPO on Qwen3-8B under the *same GPU-hour budget* — who wins on standard benchmarks?
- **RQ2:** Does GRPO/RLVR raise pass@1 on math while *lowering* pass@k (k up to 256)? (sharpening vs learning)
- **RQ3:** Preference-data ablation: UltraFeedback vs HelpSteer2 vs 50/50 mix at fixed N pairs (DPO).

## Repo structure
```
AlignmentLab/
├── PLAN.md  README.md  LICENSE  .gitignore
├── envs/                  # conda env yml per env: alab-sft.yml, alab-rl.yml, alab-eval.yml
├── configs/
│   ├── sft/  dpo/  grpo/  # one YAML per experiment (see run-ID convention)
│   └── cluster.yaml       # queue name, gpu type, scratch path, wandb entity
├── src/
│   ├── data/              # download + preprocess (Codex)
│   ├── train/             # TRL SFT/DPO entrypoints (Codex)
│   ├── rl/                # OpenRLHF GRPO launch + reward fn (Sonnet)
│   └── evals/             # lm-eval orchestration + pass@k harness (Codex)
├── scripts/
│   ├── lsf/               # bsub templates (GLM), ray_lsf launcher (Sonnet)
│   └── submit_*.sh        # one submitter per experiment type (Codex)
├── data/                  # raw/ processed/ (gitignored)
├── results/
│   ├── runs/<run_id>/     # config.yaml, metrics.json, lsf.log
│   └── evals/<run_id>/    # lm_eval.json, passk.json
├── docs/                  # experiment cards, report draft (GLM)
└── third_party/           # cloned repos (gitignored)
```

## Quickstart
See docs/RUNBOOK.md (coming soon).

## Results

### Phase 1 — RLVR reward control battery (Qwen3-8B)

Three GRPO arms that differ **only** in the reward signal, all initialised from the same SFT
checkpoint (`Arushhh/alab-q3-8b_sft_tulu_0705`), same data/steps/KL/group size. Isolates what
the *learning signal* does, independent of compute. Metrics are lm-eval `exact_match`.

| Run / model | Reward | GPU-hrs | GSM8K | MATH-500 | GSM-Plus | Status |
|---|---|---|---|---|---|---|
| base Qwen3-8B | — | — | 0.916 | 0.688 | 0.752 | reference |
| SFT init (RL start) | — | 8.2 | 0.861 | 0.430 | 0.701 | done |
| `grpo-fmt` | +1 iff `\boxed{}` exists | ~98 | 0.498 | 0.248 | 0.385 | done |
| `grpo-rand` | Bernoulli(0.5) placebo | ~98 | 0.563 | 0.252 | 0.455 | done |
| `grpo-gt` | correctness (math-verify) | — | — | — | — | re-running (stronger KL) |

**Headline so far:** both correctness-*blind* controls degrade sharply from the SFT start
(GSM8K 0.86 → 0.50/0.56). KL stayed tiny, so this is behavioural erosion (repetition,
non-termination, fewer parseable answers) rather than knowledge collapse — the expected
control signal that a bad/no reward hurts. The `grpo-gt` arm (real reward) is being re-run
after its first full run hit a **KL runaway / reward over-optimization** (accuracy peaked
~step 110 then fell as KL blew up); the headline gt-vs-controls comparison lands when it
finishes. MMLU / IFEval / pass@k columns and the SFT-vs-DPO-vs-GRPO (RQ1/RQ3) table follow
in later phases.

- **Detailed findings + sample-level analysis:** [`docs/rq1_findings.md`](docs/rq1_findings.md)
- **Engineering notes (deadlock fix, reward-thread bug, KL control, ops):** [`docs/implementation_notes.md`](docs/implementation_notes.md)
- **Run status matrix:** [`docs/experiment_cards.md`](docs/experiment_cards.md)
- Model checkpoints: public on the HF Hub under [`Arushhh/alab-*`](https://huggingface.co/Arushhh).
