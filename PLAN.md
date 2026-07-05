# AlignmentLab — Master Plan (v2, 2026-07-05) — "RLVR Control Battery"

**One-liner:** A compute-matched control battery for RLVR: GRPO with ground-truth vs random vs
format-only rewards on Qwen3-8B AND Llama-3.1-8B, judged on perturbation-robust evals — which
post-training gains are real signal, and which are Qwen-specific memorization artifacts?

Motivation: Spurious Rewards (arXiv 2506.10947) showed random-reward RLVR recovers most math gains
on Qwen but not Llama/OLMo; follow-up (arXiv 2601.11061) implicates memorization shortcuts. Most
RLVR papers benchmark Qwen-only with no controls. We ship the standard control harness + verdict.

## Research questions
- **RQ1 (headline):** GRPO with reward ∈ {ground-truth, random, format-only}, compute-matched, on
  Qwen3-8B and Llama-3.1-8B: which gains survive the placebo arms, on which model family?
- **RQ2:** Do surviving gains persist on perturbed test sets (GSM-Plus / MATH-Perturb style), and in
  pass@k (k→256)? Gains that vanish under perturbation = memorization, not capability.
- **RQ3 (systems backbone):** SFT vs DPO vs GRPO at matched GPU-hours + preference-data ablation
  (UltraFeedback vs HelpSteer2 vs 50/50 mix at fixed N pairs).

## Models
| Role | Model | Method scope |
|---|---|---|
| Smoke test | Qwen/Qwen3-0.6B | everything, 1 GPU |
| Main science | Qwen/Qwen3-8B | all RQs, full-param |
| Family contrast | meta-llama/Llama-3.1-8B | SFT → GRPO reward arms (RQ1/RQ2) |
| Headline scale | Qwen/Qwen3-14B | best config per method |
| Stretch | Qwen/Qwen3-32B | LoRA only, one run |

## GRPO reward arms (RQ1) — spec
- `gt`: math-verify equivalence vs gold answer (+1/0) + small format bonus.
- `random`: Bernoulli(0.5) per response, independent of content (seeded per run).
- `format`: +1 iff a parseable `\boxed{}` answer exists, correctness ignored.
- Identical everything else (data, steps, KL, group size). Run-ID method field encodes arm:
  `grpo-gt`, `grpo-rand`, `grpo-fmt`.

## Stack
- TRL (SFT, DPO) · OpenRLHF (GRPO, Ray + vLLM) · lm-evaluation-harness (evals)
- math-verify (verifiable reward + pass@k scoring) · W&B (logging, public project `alignmentlab`)
- Cluster: LSF only. **Every GPU/heavy-CPU command runs via `bsub`; monitor with `bjobs`. Never run compute on the login node.**

## Repo layout (all tracks must follow exactly)
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

## Interface contracts (frozen — all agents code against these)
1. **Run ID:** `{model}_{method}_{data}_{MMDD}` e.g. `q3-8b_dpo_uf_0708`, `q3-8b_grpo_math_0712`.
2. **Processed data (JSONL in `data/processed/`):**
   - SFT: `{"messages": [{"role":..., "content":...}, ...]}`
   - Preference: `{"prompt": [messages], "chosen": [messages], "rejected": [messages], "source": "uf"|"hs2"}`
   - RLVR prompts: `{"prompt": [messages], "answer": "<gold final answer string>"}`
3. **Eval results:** `results/evals/<run_id>/lm_eval.json` (raw harness output) and
   `passk.json`: list of `{"task", "temperature", "n_samples", "k", "pass_at_k", "stderr"}` using the unbiased estimator.
4. **Every training run logs:** GPU count × wall-clock hours → `results/runs/<run_id>/metrics.json`
   field `gpu_hours` (this is the compute-matching currency).
5. **Checkpoints:** `$ALAB_SCRATCH/checkpoints/<run_id>/` — never inside the repo.
6. **Cluster params:** never hardcode queue/GPU/paths; read `configs/cluster.yaml`.
7. **Held-out (NEVER in training data):** GSM8K test, MATH-500, AIME24/25, MMLU, IFEval, BBH.
   All RLVR/SFT training data must pass 8-gram decontamination against these.

## Datasets
- SFT: `allenai/tulu-3-sft-mixture` (~150k subset, seed 42)
- DPO: `HuggingFaceH4/ultrafeedback_binarized`; `nvidia/HelpSteer2` (convert: max-vs-min helpfulness rating per prompt, drop ties)
- RLVR train prompts: `agentica-org/DeepScaleR-Preview-Dataset` (+ GSM8K *train* only)
- Eval: GSM8K test, `HuggingFaceH4/MATH-500`, MMLU, IFEval, BBH (harness), AIME24 (pass@k only)
- Perturbation-robust eval (RQ2): `qintongli/GSM-Plus` (adversarial GSM8K variants) and a
  MATH-Perturb-style set (`kaixuanhuang/MATH-Perturb` or nearest available HF mirror; if none,
  numeric-perturbation of MATH-500 is a fallback task for Codex — spec before building).

## LSF house style (all tracks)
Reference script: `/u/arushh/Arush/RQAOA-LR/code/submit_hamlib_arms.sh`. Follow its pattern:
inline `bsub` flags (`-q -J -n -R -W -o -e`) from a `submit_job()` bash function, `conda run -n <env>`,
and `mkdir -p` all output/log dirs BEFORE submitting (LSF will not create `-o/-e` dirs).

## Schedule
- **Days 1–2:** scaffold (GLM) ∥ data + SFT smoke on 0.6B (Codex) ∥ env + Ray-on-LSF design (Sonnet). First `bsub` training job by day 2.
- **Days 3–7:** Qwen3-8B SFT + DPO (UF, HS2, mix) + eval pipeline live.
- **Week 2:** GRPO on 8B (Sonnet critical path). Budget 3–4 failed launches.
- **Week 3:** pass@k sweeps (bsub array jobs, 1 GPU each), compute-matched re-runs.
- **Week 4:** 14B headline runs; 32B LoRA if smooth.
- **Week 5–6:** analysis, plots, report, release, resume bullets.

## Storage-phased execution (home quota 100GB hard — added 2026-07-05)
Full scientific scope, run strictly sequentially. Rules:
1. Cache only the current phase's models (`--skip-llama` until the Llama phase).
2. SFT/DPO checkpoints save model-only (`save_only_model`, default true) — no optimizer state.
3. Per arm: train → ALL evals (lm-eval, GSM-Plus, pass@k generation JSONLs) → push final model
   to the user's HF Hub account (free archive, public artifact) → delete local weights. Numbers,
   samples, and completions are always retained; only local weights are rotated.
4. Never delete the current base SFT checkpoint until every arm derived from it has trained.
5. Llama phase starts only after all Qwen arms complete and Qwen weights are rotated out.
6. `alab-eval` env deferred until eval phase (another ~10GB of vllm/torch).

## Track ownership (no cross-editing)
- **Sonnet (Track A):** `src/rl/`, `scripts/lsf/ray_*`, env ymls, distributed debugging, final stats.
- **Codex (Track B):** `src/data/`, `src/train/`, `src/evals/`, `scripts/submit_*.sh`.
- **GLM (Track C):** repo root files, `configs/`, `scripts/lsf/*.template`, `docs/`, plotting.
- Integrator (you+Fable): merge, run `bsub`, report failures upward (GLM→Codex→Sonnet).
