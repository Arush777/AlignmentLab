# H200 operator status — 2026-07-18/19

## Gates

| Gate | Status |
|---|---|
| ≤2 GPU constraint | **honored** — smoke on GPU 0; full gt on GPUs **0,1** |
| Verifier smoke | **PASS** — `q3-8b_grpo-gt_math1h200_smoke_0718`: `gt_accuracy` 0.875→0.69, `reward_mean` 0.93→0.73, rewards 1.0/1.1, `empty_parse_rate=0` |
| Full gt@KL0.1 | **RUNNING** — `q3-8b_grpo-gt_math2h200_kl01b` on `alab-gt-2h200` |
| Rematch / eval / Llama | pending after gt DONE |

## Full run details

| Field | Value |
|---|---|
| Config | `configs/grpo/q3-8b_grpo_math_2gpu_h200_kl01.yaml` (1 train + 1 vLLM, KL=0.1) |
| GPUs | `CUDA_VISIBLE_DEVICES=0,1` (leave 2=`kdas3`, 7=`bhadresh` alone) |
| SFT init | `hub:Arushhh/alab-q3-8b_sft_tulu_0705` |
| Checkpoints | `save_steps: 50`, `ALAB_NODE_LOCAL=0` → under scratch on `/data/anupam` |
| First attempt `…_kl01` | **FAILED** OOM at `optimizer.step` with on-device Adam (verifier OK: sample reward 1.1) |
| Fix | `adam_offload: true`, `ref_offload: true` (torch_adam patch in `ray_launch`) |
| Live signal | train microbatches show `reward=1.1` / `1.0`; samples include `correct=True` |

## ETA (rough)

First PPO train pass alone is ~25 min (1024 microbatches @ ~1.4 s on 1 train GPU). Expect **~25–40 min/step** wall → **~6–10 days** for ~355 steps on 2×H200. Science recipe (bs=128, n=8) unchanged; wall-clock is the ≤2-GPU tax.

## Monitor

```bash
scripts/remote/job.sh status gt-2h200
scripts/remote/job.sh logs gt-2h200 -f
# TB: results/runs/q3-8b_grpo-gt_math2h200_kl01b/tb/
# KL watch steps 100–150: ≲0.15; escalate if ≳0.3 and accuracy falling
```

Do **not** treat `…_math5h200_kl01c` or failed `…_math2h200_kl01` as publishable gt.
