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

| Run ID | Method | Data | GPU-hrs | GSM8K | MATH-500 | MMLU | IFEval | pass@1 | pass@256 |
|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |
