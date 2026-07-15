# RQ1 validity — matching debt and rematch checklist

**Status (2026-07-13):** Controls (format, random) are done at **KL=0.04 / 4-GPU colocated**.
The gt arm is racing two **KL=0.1 / non-colocated** jobs. Until rematch (or an explicit
caveat), do **not** treat gt-vs-controls as a clean causal RQ1 claim.

Related: [`rq1_findings.md`](rq1_findings.md) (interim numbers), [`implementation_notes.md`](implementation_notes.md) (ops), [`experiment_cards.md`](experiment_cards.md).

---

## What is matched today

| Axis | format / random (done) | gt kl01 (in flight) | Matched? |
|---|---|---|---|
| Model + SFT init | Qwen3-8B ← `alab-q3-8b_sft_tulu_0705` | same | yes |
| Data / batch / group / steps | rlvr_math, bs=128, n=8, ~355 steps | same | yes |
| Reward | format / random | gt (math-verify) | **only intentional difference** |
| `kl_init_coef` | **0.04** | **0.1** | **NO — scientific confound** |
| Placement | 4-GPU colocated | 5- or 6-GPU non-colocated | bookkeeping only (see below) |

---

## The real confound: KL

Completed placebos used `kl_init_coef=0.04`. Pending gt uses `0.1` because the first
full non-colocated gt run (`…_ncol1`) hit KL runaway / Goodhart (~step 110): accuracy
peaked then fell while KL blew up ~0.06→0.46.

Raising KL on the **treatment** arm alone changes the regularization. A reviewer can
correctly say the controls and gt were not under the same leash. That is the matching
debt that blocks a publishable RQ1 headline.

**Rematch is for defensibility of the causal claim**, not because we expect the control
numbers to move much (their observed KL never approached 0.1). Still required before
claiming “identical everything else except reward.”

---

## Placement / GPU count — bookkeeping only

4→5/6 GPU and colocated→non-colocated were forced by infra (colocated deadlock at ~step 57;
Ray MetricsHead host flakes). Batch size, group size, episodes, and optimizer settings
are unchanged. Placement affects **GPU-hour accounting and step wall-time**, not the
learning signal by design.

**Do not rematch solely for placement.** Rematch for **KL=0.1** (and whatever GPU split
the winning gt job actually used, so the table is one recipe).

---

## Interim framing (until rematch)

Safe to say:

- Controls degrade from SFT under KL=0.04 (behavioral erosion; tiny KL).
- gt@KL0.04 over-optimized (trajectory in `rq1_findings.md`).
- gt@KL0.1 is the attempted fix; race jobs still pending/running.

**Not** safe to say without caveat: “gt beat / matched / failed relative to format+random
under matched compute.”

---

## Rematch checklist (after successful kl01 gt + eval)

Gate: winning gt job reaches DONE, `metrics.json` has `returncode:0` and hub push OK,
eval completes, and KL-watch through steps ~100–150 looks healthy (KL ≲0.15, accuracy
holds/climbs). Then rematch **format** and **random** on the **winning** config only.

### If winner is 6-GPU (`q3-8b_grpo_math_6gpu_kl01.yaml`)

```bash
cd /u/arushh/Arush/Project/AlignmentLab

# format rematch
bash scripts/lsf/ray_lsf_launch.sh \
  --gpus 6 --wall 96:00 \
  --exclude-hosts cccxc716,cccxc708 \
  --config configs/grpo/q3-8b_grpo_math_6gpu_kl01.yaml \
  --reward-mode format \
  --sft-ckpt hub:Arushhh/alab-q3-8b_sft_tulu_0705 \
  --run-id q3-8b_grpo-fmt_math6gpu_kl01_rematch

# random rematch
bash scripts/lsf/ray_lsf_launch.sh \
  --gpus 6 --wall 96:00 \
  --exclude-hosts cccxc716,cccxc708 \
  --config configs/grpo/q3-8b_grpo_math_6gpu_kl01.yaml \
  --reward-mode random \
  --sft-ckpt hub:Arushhh/alab-q3-8b_sft_tulu_0705 \
  --run-id q3-8b_grpo-rand_math6gpu_kl01_rematch
```

### If winner is 4-GPU (`q3-8b_grpo_math_4gpu_kl01.yaml`)

Same as above but `--gpus 4`, config `..._4gpu_kl01.yaml`, run-ids
`q3-8b_grpo-{fmt,rand}_math4gpu_kl01_rematch`.

### If winner is 5-GPU (`q3-8b_grpo_math_5gpu_kl01.yaml`)

Same as above but `--gpus 5`, config `..._5gpu_kl01.yaml`, run-ids
`q3-8b_grpo-{fmt,rand}_math5gpu_kl01_rematch`.

### After rematch DONE

1. `scripts/submit_eval.sh --run-id <run_id> Arushhh/alab-<run_id> all` for each arm.
2. `python scripts/collect_results.py`
3. `conda run -n alab-eval python scripts/plot_arms.py`
4. Update `docs/experiment_cards.md` + `docs/rq1_findings.md` with the **KL-matched** table.
5. Git push only if the user explicitly asks.

### Explicitly deferred

- **Llama-3.1-8B** — not until rematch is done **or** RQ1 is written with an explicit
  unmatched-recipe caveat.
- Re-running controls at KL=0.04 on 6-GPU — unnecessary if kl01-gt is the published treatment.

---

## Success criteria for the in-flight gt race

| Signal | Success | Escalate |
|---|---|---|
| KL through steps 100–150 | ≲ 0.15, flat or mild rise | > ~0.3 **and** gt_accuracy declining from peak → tell user (next lever 0.2 or shorter horizon) |
| Mid-run hang (non-colocated) | samples.jsonl advancing | RUN but samples stale >15–20 min → diagnose, **do not** blind-resubmit |
| Startup EXIT (Ray MetricsHead EOF) | rare | resubmit same config with run-id `…_kl01b` + host exclusions |
