# Implementation notes — Qwen3-8B GRPO control battery

Engineering decisions and bugs that shaped the RQ1 runs. These are the non-obvious things
future runs (incl. the Llama phase) must not re-derive. Stack: OpenRLHF + Ray + vLLM +
DeepSpeed ZeRO-3, IBM CCC LSF cluster, model Qwen3-8B.

## Config matrix (`configs/grpo/`)
| config | GPUs | placement | KL | purpose |
|---|---|---|---|---|
| `q3-8b_grpo_math.yaml` | 8 | colocate_all | 0.04 / k3 | canonical full-node |
| `q3-8b_grpo_math_4gpu_test.yaml` | 4 | colocate_all + offload | 0.04 / k3 | dispatch test — **deadlocks** |
| `q3-8b_grpo_math_4gpu_kl01.yaml` | 4 | non-colocated 3+1 | **0.1** / k3 | schedule-friendly deadlock+KL fix |
| `q3-8b_grpo_math_6gpu.yaml` | 6 | non-colocated 4+2 | 0.04 / k3 | deadlock fix — **over-optimizes** |
| `q3-8b_grpo_math_6gpu_kl01.yaml` | 6 | non-colocated 4+2 | **0.1** / k3 | deadlock + KL fix |
| `q3-8b_grpo_math_5gpu_kl01.yaml` | 5 | non-colocated 4+1 | 0.1 / k3 | 5-GPU twin (easier to schedule) |

The three reward arms (gt / random / format) share one config; only `--reward-mode` differs.
That is the whole point of the control — identical data, steps, KL, group size across arms.

## Bug 1 — dead correctness reward (the big one)
`math_verify.parse()` uses `signal.alarm()`, which only works on the **main thread**, but
OpenRLHF runs the reward function in a **worker thread** → `parse()` raised, a bare
`except: return False` swallowed it → gt correctness *never scored*, gt_accuracy pinned at 0.
gt was silently identical to a degenerate format arm for a whole prior experiment.

**Fix attempt 1 (insufficient):** call `parse(..., parsing_timeout=None)`. On math-verify
**0.7.0** this does **not** disable the timeout wrapper — it still runs
`signal.alarm(None)` → TypeError → parse catches internally → returns `[]` →
`_is_correct` returns False with `parse_fail_rate` still 0. The H200 full-scale run
`q3-8b_grpo-gt_math5h200_kl01c` hit this: boxed_rate ~0.9, rewards only `{0.0, 0.1}`,
gt_accuracy = 0 for 30 steps. See `docs/h200_gt_kl01_status.md`.

**Fix (current):** patch `math_verify.utils.timeout` (and the `parser`/`grader` aliases)
to a no-op decorator at gt-mode import time (`_patch_math_verify_timeout` in
`src/rl/reward.py`). Also log `empty_parse_rate` so silent `[]` returns are visible.

**Verification signal:** a healthy gt now shows `train/gt_accuracy > 0` (≈0.25–0.40) within
the first few steps. Regression: `tests/test_reward.py::test_is_correct_from_worker_thread`.

## Bug 2 — the 4-GPU colocation deadlock
On the 4-GPU `colocate_all` config, the driver deadlocks in `ray.get()` (GPUs at 0% util)
inside the vLLM colocate-sleep + DeepSpeed-offload cycle. It froze reproducibly at ~step 57
across three attempts; py-spy confirmed it is **not** the reward code (no thread stuck in the
verifier). random/format ran fine on the identical config → the hang is gt-workload-specific
and intermittent. LSF `mode=shared:j_exclusive=yes` (not `exclusive_process`) is required so
the colocated vLLM+DeepSpeed can share GPU context at all (a separate earlier fix).

**Fix — non-colocated placement** (`build_argv()` in `src/rl/train_grpo.py`): a new branch
for `colocate_all: false` splits the GPU pool into a training sub-pool (`train_gpus`, actor+ref
under `--train.colocate_actor_ref`) and a **disjoint** vLLM sub-pool (`vllm.num_engines`),
dropping `--vllm.enable_sleep --ds.enable_sleep`. No sleep/offload cycle → no deadlock.
6-GPU = 4 train + 2 vLLM; 5-GPU = 4 train + 1 vLLM (kept train_gpus=4 for a
familiar shard — **not** because 8B cannot fit on fewer). A **4-GPU** non-colocated
config (`…_4gpu_kl01.yaml`, 3+1) is the right CCC default: format/random already
**completed** on 4×80G; gt's 4-GPU failure was the colocate **sleep deadlock**, not
OOM. 8B bf16 weights ≈16GB; Adam (~64GB) is CPU-offloaded. If you ever OOM, shrink
`micro_batch_size` / `max_tokens_per_gpu` / `response_max_len` — do not buy more GPUs.
Verify splits with `python src/rl/train_grpo.py --dry-run --gpus N --config ...`.

## Bug 3 — KL runaway / reward over-optimization (gt)
Even with the deadlock fixed, gt's first full run over-optimized: gt_accuracy peaked ~step
110 then fell as KL blew up 0.06 → 0.46. `kl_init_coef` is a **fixed** coefficient; k3/0.04
delayed but did not prevent the runaway. OpenRLHF's `AdaptiveKLController` (the Ziegler-2019
standard fix) is incompatible with the GRPO KL-as-loss path, so the lever is a stronger fixed
leash: `kl_init_coef` 0.04 → 0.1. Full analysis + trajectory table in `docs/rq1_findings.md`.
KL estimator note: DeepSeekMath's β=0.04 was calibrated against **k3** (`exp(q-p)-1-(q-p)`),
not k2 — k2 + 0.04 let KL run away (0.0004→0.43 by step 113 with text degradation observed).

## Ops gotchas
- **Ray dashboard startup flake:** hosts **cccxc716** and **cccxc708** fail Ray startup with
  `Module MetricsHead failed to start. Received EOF from pipe.` Submit gt via raw `bsub` with
  `-R "select[hname!=cccxc716 && hname!=cccxc708]"`; the launcher's own submit path can't pass
  host exclusions. If a new host startup-EXITs the same way, add it to the exclusion.
- **No mid-run checkpoints:** `save_steps=-1`. Periodic DeepSpeed checkpoints carry ~100 GB of
  optimizer state and would blow the 100 GB home quota; resume is unsupported anyway. So every
  restart begins at step 0, and only the **final** model is saved → an over-optimizing run
  cannot have its peak-step model recovered (motivates fixing regularization, not checkpointing).
- **Checkpoints never touch home:** train to node-local `/tmp/alab_<jobid>` → push to HF Hub.
  Home stays ~78/100 GB. On a failed push, a `.keep` sentinel holds the ckpt on the node for
  manual `hf upload` (recovered the random arm this way).
- **HF storage:** private-repo quota is small; all `alab-*` model repos are **public**
  (free/unlimited) so gt + the 5 Llama pushes (~130 GB total) don't hit the cap.
- **CUDA check:** `export DS_SKIP_CUDA_CHECK=1` in `scripts/lsf/ray_lsf_launch.sh` to avoid a
  CUDA-mismatch crash under `--ds.adam_offload`.

## Result pipeline
On an arm reaching DONE: verify `results/runs/<id>/metrics.json` (`returncode:0`,
`hub_push_ok:true`) → `scripts/submit_eval.sh --run-id <id> Arushhh/alab-<id> all` →
`scripts/collect_results.py` → `scripts/plot_arms.py` → update `docs/experiment_cards.md`.
Raw per-sample eval dumps (`results/evals/*/passk_samples.jsonl`, 29–41 MB each) are
gitignored; only the small metric JSONs, `results/summary.csv`, and `docs/figs/` are tracked.
