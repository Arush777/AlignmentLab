# Handoff — AlignmentLab on shared H200 (≤2 GPUs)

**Operator constraint:** at most **2× H200** for AlignmentLab; CCC LSF is not the path for this phase.

## Why ≤2 GPUs

8B weights fit on one H200. Multi-GPU in OpenRLHF/GRPO is mostly **policy+ref train + dedicated vLLM**, not “model too big.” On a shared box we trade wall-clock for neighborliness.

| Recipe | Split | Use |
|---|---|---|
| `q3-8b_grpo_math_1gpu_h200_kl01.yaml` | colocated 1 | smoke / spare-GPU-friendly full |
| `q3-8b_grpo_math_2gpu_h200_kl01.yaml` | **1 train + 1 vLLM** | default full gt / rematch |
| ~~5gpu_h200~~ | 4+1 | **retired for production** |

## Critical path

1. Smoke-prove verifier (`gt_accuracy > 0`) after Bug 1 fix in `src/rl/reward.py`
2. Full **gt@KL=0.1** on ≤2 GPUs with `save_steps: 50`
3. Rematch **format** then **random** on the **same** 2-GPU kl01 config
4. Eval + push cards/figs
5. Llama SFT (serial)

Do **not** treat `q3-8b_grpo-gt_math5h200_kl01c` as gt (verifier dead). See `docs/h200_gt_kl01_status.md`.

## Ops

- Sync: `scripts/remote/sync.sh` (Mac → `/data/anupam/AlignmentLab`)
- Jobs: `scripts/remote/job.sh start|stop|logs|status`
- Always `nvidia-smi` before launch; pin free indices only
- Checkpoints: prefer `/data/anupam/AlignmentLab/scratch` (set `ALAB_NODE_LOCAL=0` or point `ALAB_NODE_TMP` under scratch) so mid-run saves survive `/tmp` cleanup
