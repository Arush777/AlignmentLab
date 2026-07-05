# Track B — Phase 1.5: node-local checkpoints + HF Hub weight archive (Codex)

Context: home quota is 100GB HARD (currently ~92GB used) and one 8B checkpoint is 16GB, so
weights must never be written to home. Verified facts: compute nodes have ~311GB free local
disk on `/` (so `/tmp` works) and outbound internet (HF Hub reachable). `hf auth login` token
is at `~/.cache/huggingface/token` and is readable from compute nodes.

Read `PLAN.md` section "Storage-phased execution (rev 2)" first. Modify ONLY your owned files
(`src/train/`, `scripts/submit_*.sh`). Requirements:

1. `src/train/sft.py`:
   - `output_dir` moves to `${ALAB_NODE_TMP:-/tmp/alab_${LSB_JOBID}}/ckpt` when the env var
     `ALAB_NODE_LOCAL=1` is set (default in the submit script). Home path fallback otherwise.
   - After training: (a) write metrics.json + trainer_state.json + a copy of the final config
     to `results/runs/<run_id>/` in the repo (small files only); (b) if `ALAB_HUB_PUSH=1`
     (default), push the final model+tokenizer to HF Hub repo `alab-<run_id>` under the
     authenticated user's namespace, `private=True` (flag `hub_private: false` in YAML to
     flip). Use huggingface_hub upload with the trained model dir; print the hub URL.
   - The push must be retried up to 3 times on network errors, and its failure must NOT
     mark the training run as failed — print a loud warning with the exact manual
     `hf upload` command to run from a compute node instead.
2. `scripts/submit_sft.sh`:
   - export `ALAB_NODE_LOCAL=1 ALAB_HUB_PUSH=1 ALAB_NODE_TMP=/tmp/alab_${LSB_JOBID}` inside
     the job command; add a `trap 'rm -rf /tmp/alab_${LSB_JOBID}' EXIT` inside the job body.
   - Remove `HF_HUB_OFFLINE=1` etc. from the SFT job (compute nodes are online; the home HF
     cache stays the model source so downloads should still be cache hits — keep `HF_HOME`
     pointed at home).
3. New `scripts/fetch_hub_ckpt.sh <hub_repo_id> <dest_dir>` helper (used by later DPO/GRPO
   jobs): downloads a Hub model snapshot to a node-local dir with `local_dir=` semantics,
   NOT into the home HF cache. Print resolved path.
4. `src/train/dpo.py` (Phase 2 file, when you build it): accept `--sft-ckpt` as either a
   local path or a `hub:<repo_id>` reference; in the latter case call fetch into
   `${ALAB_NODE_TMP}/sft_init` before loading.
5. Smoke config (0.6B) keeps working unchanged; it may keep home output (1.5GB is fine) —
   gate everything on the env vars, not on model size.

Verify: py_compile, bash -n, and a dry-run mode that prints the resolved output_dir, hub repo
id, and cleanup trap without training. Print the exact submission commands when done.
