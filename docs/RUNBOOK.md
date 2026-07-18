# Runbook — AlignmentLab

Envs are **uv** (not conda): `rl` (GRPO), `sft` (TRL), `eval` (plots / harness).
Runner: `scripts/alab <rl|sft|eval> <command...>`

---

## H200 worker (preferred for new runs)

Host: `anupam@169.38.10.80` · repo: `/data/anupam/AlignmentLab` · **8× H200**  
Coding tools stay on your Mac (proprietary policy). Only sync + execute on the box.

### One-time bootstrap (from Mac)

```bash
# from local clone
scripts/remote/bootstrap.sh          # SSH setup + rsync + uv sync all (+ flash-attn)
scripts/remote/status.sh             # sanity: uv, venvs, GPUs, tmux
```

Persistent control session on the box: `tmux` session `alab-ctl` (created by bootstrap).

### Everyday loop

```bash
# 1) edit locally on Mac
# 2) push code (no commit required)
scripts/remote/sync.sh               # rsync working tree
#    or: scripts/remote/sync.sh --git   # after you git push

# 3) start a long job in its own tmux session
scripts/remote/job.sh sync-and-start smoke --gpus 0 -- \
  bash scripts/local/ray_launch.sh \
    --config configs/grpo/q3-8b_grpo_math_1gpu_h200_kl01.yaml \
    --reward-mode gt --smoke

# 4) monitor
scripts/remote/job.sh status
scripts/remote/job.sh logs smoke -f
scripts/remote/job.sh attach smoke   # interactive tmux attach
```

Full gt arm example:

```bash
scripts/remote/job.sh sync-and-start gt-kl01 --gpus 0 -- \
  bash scripts/local/ray_launch.sh \
    --config configs/grpo/q3-8b_grpo_math_1gpu_h200_kl01.yaml \
    --reward-mode gt \
    --sft-ckpt hub:Arushhh/alab-q3-8b_sft_tulu_0705 \
    --run-id q3-8b_grpo-gt_math1h200_kl01
```

Agent collaboration: the Mac agent can `sync` + `job.sh start/logs/status` over SSH; it never needs an IDE on the GPU box.

---

## uv setup (any machine)

```bash
# install uv if needed: https://docs.astral.sh/uv/
scripts/alab sync all --flash-attn   # creates .venv-rl .venv-sft .venv-eval
scripts/alab which rl
scripts/alab rl python -c "import openrlhf, vllm; print('ok')"
```

Do **not** `uv sync --all-extras` — sft and rl pin different `transformers` lines.

Cluster knobs: `configs/cluster.yaml` (on H200 this is overwritten from `configs/cluster.h200.yaml` on sync).

---

## Data

```bash
scripts/alab sft python src/data/download.py --cluster-config configs/cluster.yaml
scripts/alab sft python src/data/preprocess.py
# Expected: data/processed/sft.jsonl, data/processed/rlvr_math.jsonl
```

---

## Smoke GRPO (local / H200)

```bash
bash scripts/local/ray_launch.sh \
  --config configs/grpo/q3-8b_grpo_math_1gpu_h200_kl01.yaml \
  --reward-mode gt --smoke
```

---

## SFT 8B

```bash
# CCC: bash scripts/submit_sft.sh
# H200: use scripts/remote/job.sh + scripts/alab sft accelerate launch ...
# Landed init: Arushhh/alab-q3-8b_sft_tulu_0705 (public)
```

## DPO 8B

Not implemented (`NotImplementedError`). Deferred until Phase 1 control battery is clean.

---

## Evals

```bash
# Prefer wrapping in job.sh on H200; or CCC submit_eval.sh
scripts/alab eval python -u src/evals/run_lm_eval.py --help
```

## Collect + plot

```bash
python scripts/collect_results.py
scripts/alab eval python scripts/plot_arms.py
```

---

## CCC / LSF (legacy path)

All GPU work via `bsub`. Never run training/eval on the login node.
Host exclusions (Ray MetricsHead flake): `cccxc716`, `cccxc708`.

```bash
bash scripts/lsf/ray_lsf_launch.sh \
  --gpus 4 --wall 96:00 \
  --exclude-hosts cccxc716,cccxc708 \
  --config configs/grpo/q3-8b_grpo_math_4gpu_kl01.yaml \
  --reward-mode gt \
  --sft-ckpt hub:Arushhh/alab-q3-8b_sft_tulu_0705
```

Health checks: `samples.jsonl` mtime; TensorBoard `train/gt_accuracy`, `train/kl`.
Validity / rematch: [`docs/rq1_validity.md`](rq1_validity.md).

---

## Hard rules

- No coding assistants / IDE agents on the H200 host — sync from Mac only.
- Never kill healthy jobs (exception: race-loser still PEND on LSF).
- No git commit/push unless user asks in-session.
- Validity / rematch: [`docs/rq1_validity.md`](rq1_validity.md).
