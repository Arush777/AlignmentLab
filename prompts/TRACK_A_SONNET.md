# Track A — Sonnet/Opus (hard 20%)

Paste this as the first message. Keep later messages minimal: one problem per message, tracebacks + config paths only, never whole files.

---

You are the distributed-training lead on AlignmentLab, at `/u/arushh/Arush/Project/AlignmentLab`.
Read `PLAN.md` first — the repo layout, interface contracts, and run-ID convention there are frozen; follow them exactly.
You own ONLY: `src/rl/`, `scripts/lsf/ray_*`, `envs/*.yml`. Do not touch other directories.

Cluster facts:
- IBM LSF. All GPU work via `bsub`; monitor `bjobs`; login node is CPU-only, may lack internet on compute nodes (assume HF cache pre-populated by a CPU job).
- A100-80GB nodes (assume up to 8 GPUs/node; multi-node possible via `blaunch` but treat single-node as primary).
- Read queue/GPU/scratch settings from `configs/cluster.yaml` (GLM creates it; if missing, create a stub with fields: `queue`, `gpu_type`, `gpus_per_node`, `scratch`, `wandb_entity`).

Deliverables, in order:

1. **Conda env specs** — `envs/alab-rl.yml` (OpenRLHF + ray + vllm, pinned mutually-compatible versions; check OpenRLHF's setup for the vllm pin), `envs/alab-sft.yml` (trl, transformers, accelerate, deepspeed, datasets, wandb), `envs/alab-eval.yml` (lm-eval[vllm], math-verify). Pin versions that are known-compatible as of mid-2026.

2. **Ray-on-LSF launcher** — `scripts/lsf/ray_lsf_launch.sh`: a bsub-submittable script that, inside a single N-GPU allocation, starts a Ray head, waits for readiness, then execs the OpenRLHF GRPO entrypoint; clean shutdown on exit/failure. Multi-node via blaunch is stretch — design for it, implement single-node first.

3. **GRPO training entrypoint** — `src/rl/train_grpo.py` + `configs/grpo/q3-8b_grpo_math.yaml` wiring OpenRLHF's GRPO (advantage_estimator group_norm / their current GRPO flag) with:
   - Policy Qwen3-8B (init from our SFT checkpoint path arg; must also work with meta-llama/Llama-3.1-8B — no Qwen-specific assumptions), vLLM rollout engines colocated, ZeRO-3.
   - Reward: `src/rl/reward.py` with a `reward_mode` YAML field (PLAN.md "GRPO reward arms" spec):
     `gt` = math-verify equivalence vs the `answer` field of RLVR JSONL (contract §2), +1/0, small `\boxed{}` format bonus;
     `random` = seeded Bernoulli(0.5) per response, content ignored;
     `format` = +1 iff parseable `\boxed{}` exists, correctness ignored.
     All modes MUST log every (prompt, response, reward) sample at 1% rate to `results/runs/<run_id>/samples.jsonl` for later reward-hacking analysis.
   - Group size 8, KL coef ~1e-3 (expose in YAML), context 4k prompt / 8k response.
   - Log `gpu_hours` to `results/runs/<run_id>/metrics.json` on exit (contract §4).
   - A `--smoke` flag: Qwen3-0.6B, 1 GPU, 20 steps, tiny data — must run end-to-end before any 8B launch.

4. **Later, on request:** debugging failed launches (you'll get tracebacks), and the final pass@k statistical analysis (unbiased estimator, bootstrap CIs) — do not start these now.

LSF house style: follow PLAN.md "LSF house style" — inline bsub flags in a submit function, `conda run -n`, and `mkdir -p` every `-o/-e` log dir before submitting (LSF won't create them).

Order of work: envs → smoke GRPO on 0.6B → 8B config. Write nothing outside your directories. When done, print exact `bsub` commands for (a) the 0.6B smoke, (b) the 8B `gt` run, (c) the 8B `random` control, using placeholders from cluster.yaml.
