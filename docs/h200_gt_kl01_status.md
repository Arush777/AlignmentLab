# H200 live run status — `gt@KL0.1` full-scale GRPO

**Report date:** 2026-07-18 ~15:45 UTC (21:15 IST)  
**Host:** `dgre2` (`anupam@169.38.10.80`)  
**Repo path on host:** `/data/anupam/AlignmentLab`  
**Verdict:** Job is **alive and stepping**, but the **gt correctness signal is dead**. Treat this run as **INVALID for RQ1 gt** until the math-verify / worker-thread timeout bug is fixed and a smoke gate passes. See §5–§7.

Related: [`rq1_validity.md`](rq1_validity.md), [`rq1_findings.md`](rq1_findings.md), [`implementation_notes.md`](implementation_notes.md) (Bug 1).

---

## 1. What is running right now

| Field | Value |
|---|---|
| Intent | Full-scale **gt** arm rematch at **KL = 0.1** (science critical path after CCC KL runaway at 0.04) |
| Job name | `gt-kl01` |
| tmux session | `alab-gt-kl01` — **RUNNING** |
| Run ID | `q3-8b_grpo-gt_math5h200_kl01c` |
| Started | `2026-07-18T13:01:51Z` |
| Log | `results/remote_jobs/gt-kl01.log` |
| Config | `configs/grpo/q3-8b_grpo_math_5gpu_h200_kl01.yaml` |
| Reward mode | `ALAB_REWARD_MODE=gt` |
| Model | Qwen3-8B from SFT init `Arushhh/alab-q3-8b_sft_tulu_0705` (local `scratch/sft_init`) |
| Data | `data/processed/rlvr_math.jsonl` — **45,467** prompts (DeepScaleR + GSM8K-train, decontaminated) |
| Stack | OpenRLHF + Ray + vLLM + DeepSpeed ZeRO-3; launch via `scripts/local/ray_launch.sh` / remote tmux |

### Recipe (intended science match)

| Knob | Value |
|---|---|
| Advantage | GRPO `group_norm` |
| `n_samples_per_prompt` | 8 |
| Rollout / train batch | 128 prompts/step |
| Episodes / epochs | 1 / 1 → **~355 steps** |
| KL | `kl_init_coef=0.1`, estimator **k3**, `kl_use_loss=true` |
| Placement | **non-colocated**: `train_gpus=4` + `vllm.num_engines=1` |
| Adam | on-GPU (`adam_offload: false`, `torch_adam` DeepSpeed patch — H200 avoids CPUAdam/FusedAdam breakage) |
| Checkpoints | `save_steps=-1` (final only; Hub push at end) |

### GPU assignment (this job)

| Role | Physical GPUs (`CUDA_VISIBLE_DEVICES`) |
|---|---|
| Train (policy + ref, ZeRO-3) | **3, 4, 5, 6** |
| vLLM rollout | **7** |
| **Total used by `anupam`** | **5× H200** |

Host occupancy at report time (other users):

| GPU | Occupant | Notes |
|---|---|---|
| 0–1 | free | — |
| 2 | `kdas3` | Jupyter / `weather_ai` (~59 GB) |
| 3–7 | **`anupam`** | this AlignmentLab job |

---

## 2. Progress and ETA

Snapshot from TensorBoard `results/runs/q3-8b_grpo-gt_math5h200_kl01c/tb/ppo_0718T13:02`:

| Metric | Value |
|---|---|
| Steps logged | **30 / ~355** (~8.5%) |
| Wall time so far | **~2.65 h** |
| Mean step time | **~318 s/step** (~5.3 min; `train/timing/step_total` last ≈ 337 s) |
| ETA remaining | **~28–30 h** if rate holds |
| Implied finish | **~2026-07-19 ~20:00–22:00 UTC** (±20% if response length drifts) |
| `samples.jsonl` | **332** lines (~1% sample log), growing |

Training microbatches are advancing (Train epoch bars in the log; GPUs 3–6 at high util during PPO train; GPU 7 holds ~125 GB vLLM even at 0% util between gens).

**Important:** this ETA is wall-clock for a run that is **not delivering a gt learning signal** (§5). Completing it does not unlock a publishable gt arm.

---

## 3. Live training metrics (steps 1 → 30)

| Scalar | First | Last (step 30) | Reading |
|---|---|---|---|
| `train/gt_accuracy` | **0.0** | **0.0** (all 30 steps) | **Broken** — docs expect ≈0.25–0.40 in the first few steps |
| `train/boxed_rate` | 0.957 | 0.872 | Format detection works; mild decline |
| `train/reward_mean` | 0.096 | 0.087 | ≈ `boxed_rate × 0.1` = format bonus only |
| `train/kl` | 0.00036 | **0.0056** | Healthy vs Goodhart; far below 0.15 leash |
| `train/parse_fail_rate` | 0.0 | 0.0 | **Misleading** — see §5 |
| `train/response_length` | ~872 | ~899 | Stable / mild lengthening |

### Sample-log audit (`samples.jsonl`)

| Check | Result |
|---|---|
| `correct` field | **332 / 332 = False** |
| Observed rewards | **only `{0.0, 0.1}`** — never `1.0` or `1.1` |
| Mean reward | ~0.090 |
| Interpretation | Policy is paid **only** `ALAB_FORMAT_BONUS` (default 0.1) when `\boxed{...}` parses; **correctness never credits** |

So under the hood this job behaves like a **soft format arm** wearing a gt run ID.

---

## 4. Engineering health (what is fine)

These are **not** the failure mode:

- tmux session stable; Ray actors alive (`PolicyModelActor`, `ReferenceModelActor`, `VLLM::EngineCore`).
- Non-colocated 4+1 placement avoided the old colocate sleep deadlock.
- H200 Adam path (on-GPU `torch_adam`) avoided DeepSpeedCPUAdam / FusedAdam JIT failures from earlier launch attempts.
- Full `rlvr_math.jsonl` present (45,467 rows); not a smoke / truncated-data run.
- KL trajectory looks like a well-leashed early GRPO run (contrast CCC gt@0.04 Goodhart: KL → 0.46 after ~step 110).
- Remote workflow (`scripts/remote/*`, uv envs `.venv-rl`) is the intended Mac→H200 ops path.

---

## 5. Science failure — dead correctness reward (Bug 1 recurrence)

### Symptom

`gt_accuracy` pinned at 0 while `boxed_rate` is high and rewards stay in `{0.0, 0.1}`. This matches the failure mode documented in [`implementation_notes.md`](implementation_notes.md) **Bug 1** (“dead correctness reward”).

### Confirmed mechanism on this host

1. OpenRLHF runs `reward_func` in a **worker thread**.
2. Installed `math_verify.parse()` always wraps extraction in `signal.alarm`-based `timeout(...)`.
3. In a worker thread: `ValueError: signal only works in main thread of the main interpreter` (seen repeatedly in `gt-kl01.log` under `RolloutRayActor`).
4. Our code passes `parsing_timeout=None`, intending to disable timeouts. On this math-verify build that **does not disable** the alarm path — it triggers `TypeError: 'NoneType' object cannot be interpreted as an integer` inside `signal.alarm(None)`.
5. `math_verify.parse` **catches** the exception internally and returns **`[]`**.
6. `src/rl/reward.py` `_is_correct` treats empty parse as `False` **without** incrementing `parse_fail_rate` — so TensorBoard looks “clean” while scoring is dead.

### Offline autopsy (main-thread repro on `dgre2`)

- With `parsing_timeout=None`: parse returns `[]`.
- With default / integer timeout on main thread: parse can succeed.
- Across logged samples: many have extractable `\boxed{...}`; string-equal boxed-vs-label cases exist; **verify path still yields 0 corrects** under the live timeout/`None` behavior.

### Consequence for RQ1

| Action | Allowed? |
|---|---|
| Call this a gt arm result | **No** |
| Eval / Hub-push as `alab-*-gt*` for the KL-matched table | **No** |
| Start format/random rematch against this checkpoint | **No** |
| Keep burning ~28h to “finish” | **Strongly discouraged** — wastes H200 time under a false run ID |

This is the same class of silent failure that previously made gt “format-only” before the `parsing_timeout=None` fix was believed to work. The fix is **stale relative to the installed math-verify API**.

---

## 6. How to re-check status (ops cheat sheet)

```bash
# session alive?
ssh anupam@169.38.10.80 'tmux has-session -t alab-gt-kl01 && echo RUNNING || echo DEAD'

# who owns which GPUs (prefer compute-apps, not fuser)
ssh anupam@169.38.10.80 'nvidia-smi'
ssh anupam@169.38.10.80 'nvidia-smi --query-compute-apps=pid,used_gpu_memory,process_name --format=csv'

# map PID → user
#   ps -o user=,pid=,args= -p <PID>

# tail job log
ssh anupam@169.38.10.80 'tail -f /data/anupam/AlignmentLab/results/remote_jobs/gt-kl01.log'

# TensorBoard scalars of interest
#   train/gt_accuracy, train/reward_mean, train/kl, train/boxed_rate, train/parse_fail_rate
```

Local helpers: `scripts/remote/status.sh`, meta at `results/remote_jobs/gt-kl01.meta`.

---

## 7. Recommended next steps (gates)

1. **Abort** `alab-gt-kl01` / run `q3-8b_grpo-gt_math5h200_kl01c`. Preserve TB, `samples.jsonl`, and `gt-kl01.log`. Tag mentally as `INVALID_gt_verifier`.
2. **Fix** `src/rl/reward.py` for threaded math-verify (do not rely on `parsing_timeout=None`). Options: process-pool verify, patch/bypass `signal.alarm`, or in-thread sympy `grade_answer` fallback that cannot silently no-op. Add **`empty_parse_rate`** (or similar) so falsy `[]` is visible.
3. **Smoke gate (must pass before full restart):** 1–3 GRPO steps on H200 → `train/gt_accuracy` ∈ ~[0.15, 0.50], `reward_mean` ≳ 0.2, sample rewards include **1.0 or 1.1**.
4. **Restart** full gt@KL0.1 only after smoke green (same config / data / 5-GPU recipe).
5. **KL watch** at steps 100–150: KL ≲ 0.15; escalate if KL ≳ 0.3 **and** accuracy falling from peak ([`rq1_validity.md`](rq1_validity.md)).
6. **Only then** rematch format + random at **KL=0.1** on the same recipe; then eval + RQ1 table.

---

## 8. Scientific context (why this job exists)

RQ1 is a compute-matched control battery: GRPO with reward ∈ `{gt, random, format}` on Qwen3-8B from the same SFT init. format/random at KL=0.04 already showed sharp GSM8K degradation (≈0.86 → 0.50/0.56). The first gt full run Goodharted under KL=0.04. Publishable gt-vs-controls requires a **KL-matched rematch at 0.1** — but only if gt actually scores correctness.

**Bottom line:** We are on the right *scientific step* (H200 full-scale gt@KL0.1). This *execution* is not delivering that step until the verifier works in the reward worker thread.

---

## 9. Snapshot provenance

| Item | Value |
|---|---|
| Host clock at capture | 2026-07-18 ~15:43–15:45 UTC |
| TB dir | `results/runs/q3-8b_grpo-gt_math5h200_kl01c/tb/ppo_0718T13:02` |
| Steps in TB | 30 |
| samples.jsonl lines | 332 |
| Report authoring | Cursor agent status pull via SSH; committed to `docs/h200_gt_kl01_status.md` |
