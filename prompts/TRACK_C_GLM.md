# Track C — GLM (scaffold, configs, docs — mechanical, exact spec)

Paste as first message. Give it ONE numbered task at a time if it drifts. Every follow-up must state exact paths and exact content.

---

You are the repo-infrastructure engineer on AlignmentLab. Work ONLY inside `/u/arushh/Arush/Project/AlignmentLab`.
Rules: create EXACTLY the files listed below, nothing more. Do not write training code. Do not modify `PLAN.md`, `prompts/`, `src/`. If a directory already has files, leave them untouched. After finishing, print a checklist of every file created with a one-line description.

Create these files:

1. `.gitignore` — with entries: `data/`, `third_party/`, `results/runs/*/samples.jsonl`, `wandb/`, `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`, `*.log`, `.env`, `checkpoints/`

2. `LICENSE` — full Apache-2.0 text, copyright 2026 Arush.

3. `configs/cluster.yaml` — exactly these keys with these placeholder values and a comment header saying "EDIT before first run":
   ```
   queue: "gpu_queue"        # bsub -q value
   gpu_type: "a100_80gb"
   gpus_per_node: 8
   scratch: "/scratch/arushh/alignmentlab"   # $ALAB_SCRATCH
   wandb_entity: "CHANGE_ME"
   hf_home: "/scratch/arushh/hf_cache"
   ```

4. Empty dirs with `.gitkeep`: `configs/sft/`, `configs/dpo/`, `configs/grpo/`, `src/data/`, `src/train/`, `src/rl/`, `src/evals/`, `scripts/lsf/`, `data/raw/`, `data/processed/`, `results/runs/`, `results/evals/`, `docs/`, `envs/`, `third_party/`.

5. `scripts/lsf/gpu_job.template.bsub` — an LSF script template with placeholders `{{JOB_NAME}} {{QUEUE}} {{N_GPUS}} {{WALL_HOURS}} {{CONDA_ENV}} {{COMMAND}}`, containing: `#BSUB -J {{JOB_NAME}}`, `#BSUB -q {{QUEUE}}`, `#BSUB -gpu "num={{N_GPUS}}:mode=exclusive_process"`, `#BSUB -W {{WALL_HOURS}}:00`, `#BSUB -o results/runs/{{JOB_NAME}}/lsf.%J.log`, `#BSUB -e results/runs/{{JOB_NAME}}/lsf.%J.err`, then lines to `source activate {{CONDA_ENV}}`, `export HF_HUB_OFFLINE=1`, `export HF_HOME=$(read from configs/cluster.yaml hf_home via python -c ...)` — if reading YAML in bash is awkward, hardcode a `source scripts/lsf/env.sh` line and create `scripts/lsf/env.sh` exporting `ALAB_SCRATCH`, `HF_HOME`, `WANDB_PROJECT=alignmentlab` with the same placeholder paths as cluster.yaml.
   Also `scripts/lsf/cpu_job.template.bsub` — same but no `-gpu` line and 4 cores (`#BSUB -n 4`).

6. `README.md` — sections in this order (write real prose, ~15 lines total for now):
   - Title "AlignmentLab" + one-paragraph description: matched-compute comparison of SFT/DPO/GRPO post-training on Qwen3-8B/14B, measuring pass@1-vs-pass@k sharpening from RLVR. Mention: OpenRLHF, TRL, lm-evaluation-harness, math-verify, A100/H100, LSF.
   - "Research questions" — copy RQ1/RQ2/RQ3 verbatim from PLAN.md.
   - "Repo structure" — the tree from PLAN.md.
   - "Quickstart" — placeholder text "See docs/RUNBOOK.md (coming soon)".
   - "Results" — empty table with columns: Run ID | Method | Data | GPU-hrs | GSM8K | MATH-500 | MMLU | IFEval | pass@1 | pass@256.

7. `docs/EXPERIMENT_CARD.template.md` — markdown template with fields: Run ID, Date, Model, Method, Dataset, Config path, GPU-hours, W&B link, LSF job ID, Result summary, Notes/anomalies.

8. `docs/RUNBOOK.md` — headings only for now: Setup, Download data, Smoke test, SFT 8B, DPO 8B, GRPO 8B, Evals, pass@k sweep. Under each heading write "TBD".

9. `third_party/CLONE.sh` — a script that git-clones (depth 1) into `third_party/`: OpenRLHF/OpenRLHF, EleutherAI/lm-evaluation-harness, huggingface/math-verify, QwenLM/Qwen3. Echo each clone.

Later tasks (wait for instruction): plotting scripts from finished CSVs, results-table updates, experiment cards per run.
