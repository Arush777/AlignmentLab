#!/usr/bin/env python
"""AlignmentLab GRPO entrypoint (Track A / Sonnet).

Thin driver around OpenRLHF 0.10.4's `openrlhf.cli.train_ppo_ray`. It:
  1. loads a YAML experiment config (configs/grpo/*.yaml),
  2. resolves the run id (contract §1) and cluster params (configs/cluster.yaml),
  3. exports the env vars that src/rl/reward.py reads (reward arm + sample logging),
  4. maps the config to OpenRLHF's nested `--section.key` CLI flags,
  5. runs it, and on exit writes gpu_hours to results/runs/<run_id>/metrics.json (§4).

It expects to run on the Ray head with RAY_ADDRESS already set by
scripts/lsf/ray_lsf_launch.sh (OpenRLHF's ray.init connects to the existing cluster).

Reward arm is chosen by `reward_mode` (gt|random|format) → method tag grpo-gt /
grpo-rand / grpo-fmt. `--smoke` overrides model/data/sizes for a 1-GPU 0.6B dry run.

Use `--dry-run` to print the assembled OpenRLHF command without launching.
"""

from __future__ import annotations

import argparse
import copy
import datetime as _dt
import json
import os
import subprocess
import sys
import time

try:
    import yaml
except ImportError:  # pragma: no cover
    print("ERROR: PyYAML required (conda run -n alab-rl ...).", file=sys.stderr)
    raise

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REWARD_PY = os.path.join(REPO, "src", "rl", "reward.py")


def env_true(name, default=False):
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")

METHOD_TAG = {"gt": "grpo-gt", "random": "grpo-rand", "format": "grpo-fmt"}
MODEL_SHORT = {
    "Qwen/Qwen3-8B": "q3-8b",
    "Qwen/Qwen3-0.6B": "q3-0.6b",
    "Qwen/Qwen3-14B": "q3-14b",
    "meta-llama/Llama-3.1-8B": "llama31-8b",
}


# ---------------------------------------------------------------------------------
def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def cluster_cfg():
    path = os.path.join(REPO, "configs", "cluster.yaml")
    if os.path.exists(path):
        return load_yaml(path) or {}
    return {}


def model_short(model, cfg):
    return cfg.get("model_short") or MODEL_SHORT.get(model) or model.split("/")[-1].lower()


def apply_smoke(cfg):
    """Override config for a self-contained 1-GPU / Qwen3-0.6B / ~20-step run."""
    cfg = copy.deepcopy(cfg)
    cfg["model"] = "Qwen/Qwen3-0.6B"
    cfg["model_short"] = "q3-0.6b"
    cfg["sft_ckpt"] = None
    d = cfg.setdefault("data", {})
    d["prompt_dataset"] = os.path.join(
        os.environ.get("ALAB_SCRATCH", os.path.join(REPO, "data")), "smoke_data", "rlvr_smoke.jsonl"
    )
    d["data_tag"] = "smoke"
    d["max_samples"] = 40
    r = cfg.setdefault("rollout", {})
    r.update(prompt_max_len=512, response_max_len=512, batch_size=2, micro_batch_size=1)
    t = cfg.setdefault("train", {})
    t.update(batch_size=8, micro_batch_size=1, num_episodes=1, max_epochs=1,
             max_tokens_per_gpu=4096, zero_stage=3, colocate_all=True, packing_samples=True)
    g = cfg.setdefault("grpo", {})
    g.setdefault("n_samples_per_prompt", 8)
    v = cfg.setdefault("vllm", {})
    v.update(gpu_memory_utilization=0.40, enforce_eager=True, tensor_parallel_size=1)
    c = cfg.setdefault("ckpt", {})
    c.update(save_steps=-1, save_hf=False)  # don't bother saving in a smoke
    return cfg


def ensure_smoke_data(path):
    """Generate a tiny arithmetic RLVR set (contract §2 format) so --smoke is
    self-contained and does not depend on Track B's data pipeline."""
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = []
    for i in range(40):
        a, b = 7 + i, 13 + (i * 3) % 17
        q = (f"What is {a} + {b}? Show brief reasoning and put the final answer "
             f"in \\boxed{{}}.")
        rows.append({"prompt": [{"role": "user", "content": q}], "answer": str(a + b)})
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[train_grpo] wrote smoke data: {path} ({len(rows)} rows)", file=sys.stderr)


# ---------------------------------------------------------------------------------
def hub_repo_id(run_id, cfg, namespace=None):
    explicit = str(cfg.get("hub_repo_id") or "").strip()
    if explicit:
        return explicit
    ns = namespace or os.environ.get("HF_USERNAME") or os.environ.get("HF_NAMESPACE") or "Arush777"
    return f"{ns}/alab-{run_id}"


def push_to_hub_with_retries(ckpt_dir, run_id, cfg, attempts=3):
    """Upload the final HF model folder (mirror of src/train/sft.py).

    Never raises; returns (ok, repo_id, url). The only durable copy of a GRPO
    checkpoint is this push — node-local ckpt_dir is trap-cleaned on job exit.
    """
    # The launcher exports HF_HUB_OFFLINE=1 so training loads from the local cache
    # only; huggingface_hub freezes that env at first import, and it would make
    # every API call raise OfflineModeIsEnabled. Clear it here (first import site).
    os.environ["HF_HUB_OFFLINE"] = "0"
    repo_id = hub_repo_id(run_id, cfg)
    last_error = None
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        try:
            ns = str(api.whoami().get("name") or "").strip()
            if ns:
                repo_id = hub_repo_id(run_id, cfg, namespace=ns)
        except Exception as exc:
            last_error = exc

        for attempt in range(1, attempts + 1):
            try:
                api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
                api.upload_folder(
                    repo_id=repo_id,
                    repo_type="model",
                    folder_path=str(ckpt_dir),
                    path_in_repo="",
                    ignore_patterns=["ckpt/*", "*.log"],  # ckpt/ = DeepSpeed state dir
                    commit_message=f"Upload AlignmentLab GRPO checkpoint {run_id}",
                )
                url = f"https://huggingface.co/{repo_id}"
                print(f"[train_grpo] HF Hub checkpoint: {url}", flush=True)
                return True, repo_id, url
            except Exception as exc:
                last_error = exc
                if attempt < attempts:
                    wait_s = 10 * attempt
                    print(f"[train_grpo] WARNING: Hub upload attempt {attempt}/{attempts} "
                          f"failed: {exc}. Retrying in {wait_s}s.", file=sys.stderr, flush=True)
                    time.sleep(wait_s)
    except Exception as exc:
        last_error = exc

    print("\n[train_grpo] WARNING: HF HUB PUSH FAILED AFTER TRAINING COMPLETED. "
          f"ckpt_dir is kept on the node (.keep sentinel) — upload manually:\n"
          f"  hf repo create {repo_id} --type model --private || true\n"
          f"  hf upload {repo_id} {ckpt_dir} . --repo-type model --exclude 'ckpt/*'\n"
          f"Last upload error: {last_error}\n", file=sys.stderr, flush=True)
    try:
        # Tell the launcher's trap not to rm the node-local dir, so a manual
        # upload from this node is still possible.
        with open(os.path.join(ckpt_dir, ".keep"), "w") as f:
            f.write(f"hub push failed for {run_id}: {last_error}\n")
    except OSError:
        pass
    return False, repo_id, None


# ---------------------------------------------------------------------------------
def build_argv(cfg, ngpu, run_id, ckpt_dir, run_dir, ccfg):
    d = cfg.get("data", {})
    g = cfg.get("grpo", {})
    r = cfg.get("rollout", {})
    t = cfg.get("train", {})
    v = cfg.get("vllm", {})
    c = cfg.get("ckpt", {})

    tp = int(v.get("tensor_parallel_size", 1))
    num_engines = max(1, ngpu // tp)
    model = cfg.get("sft_ckpt") or cfg["model"]
    prompt_max = int(r.get("prompt_max_len", 4096))
    resp_max = int(r.get("response_max_len", 8192))

    a = ["python", "-m", "openrlhf.cli.train_ppo_ray",
         "--actor.model_name_or_path", str(model),
         "--actor.num_nodes", "1", "--actor.num_gpus_per_node", str(ngpu),
         "--ref.num_nodes", "1", "--ref.num_gpus_per_node", str(ngpu),
         # vLLM rollout engines, colocated with policy (hybrid engine).
         "--vllm.num_engines", str(num_engines),
         "--vllm.tensor_parallel_size", str(tp),
         "--vllm.gpu_memory_utilization", str(v.get("gpu_memory_utilization", 0.55)),
         # GRPO: group-normalised advantage, KL loss.
         "--algo.advantage.estimator", str(g.get("advantage_estimator", "group_norm")),
         "--algo.kl.init_coef", str(g.get("kl_init_coef", 1e-3)),
         "--algo.kl.estimator", str(g.get("kl_estimator", "k2")),
         "--rollout.n_samples_per_prompt", str(g.get("n_samples_per_prompt", 8)),
         "--rollout.batch_size", str(r.get("batch_size", 128)),
         "--rollout.micro_batch_size", str(r.get("micro_batch_size", 8)),
         "--rollout.max_new_tokens", str(resp_max),
         "--rollout.temperature", str(r.get("temperature", 1.0)),
         "--rollout.top_p", str(r.get("top_p", 1.0)),
         "--train.batch_size", str(t.get("batch_size", 128)),
         "--train.micro_batch_size", str(t.get("micro_batch_size", 2)),
         "--train.num_episodes", str(t.get("num_episodes", 1)),
         "--train.max_epochs", str(t.get("max_epochs", 1)),
         "--train.max_tokens_per_gpu", str(t.get("max_tokens_per_gpu", 16384)),
         "--train.seed", str(cfg.get("seed", 42)),
         "--data.max_len", str(prompt_max + resp_max),
         "--data.prompt_dataset", str(d["prompt_dataset"]),
         "--data.input_key", str(d.get("input_key", "prompt")),
         "--data.label_key", str(d.get("label_key", "answer")),
         "--data.max_samples", str(d.get("max_samples", int(1e8))),
         "--ds.zero_stage", str(t.get("zero_stage", 3)),
         "--ds.param_dtype", "bf16",
         "--actor.adam.lr", str(t.get("actor_lr", 1e-6)),
         "--reward.remote_url", REWARD_PY,
         "--ckpt.output_dir", ckpt_dir,
         "--ckpt.path", os.path.join(ckpt_dir, "ckpt"),
         # save_steps=-1: final HF save only. Periodic DeepSpeed checkpoints carry
         # optimizer states (~100 GB for 8B) and resume is unsupported anyway since
         # the only durable copy is the final Hub push.
         "--ckpt.save_steps", str(c.get("save_steps", -1)),
         "--ckpt.max_num", str(c.get("max_num", 1)),
         "--logger.tensorboard_dir", os.path.join(run_dir, "tb"),
         "--logger.logging_steps", "1"]

    # store-true flags
    if d.get("apply_chat_template", True):
        a.append("--data.apply_chat_template")
    if g.get("kl_use_loss", True):
        a.append("--algo.kl.use_loss")
    if t.get("colocate_all", True):
        a += ["--train.colocate_all", "--vllm.enable_sleep", "--ds.enable_sleep"]
    if t.get("packing_samples", True):
        a.append("--ds.packing_samples")
    if t.get("gradient_checkpointing", True):
        a.append("--actor.gradient_checkpointing_enable")
    if v.get("enforce_eager", False):
        a.append("--vllm.enforce_eager")
    if c.get("save_hf", True):
        a.append("--ckpt.save_hf")

    # W&B only if a key is available and the entity has been configured.
    wandb_entity = ccfg.get("wandb_entity", "CHANGE_ME")
    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key and wandb_entity and wandb_entity != "CHANGE_ME":
        a += ["--logger.wandb.key", wandb_key,
              "--logger.wandb.org", wandb_entity,
              "--logger.wandb.project", "alignmentlab",
              "--logger.wandb.run_name", run_id]
    return a


# ---------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to configs/grpo/*.yaml")
    ap.add_argument("--gpus", type=int, default=None,
                    help="GPUs for this run; default 1 under --smoke, else $ALAB_NUM_GPUS or 8")
    ap.add_argument("--reward-mode", choices=["gt", "random", "format"], default=None,
                    help="override reward_mode from config")
    ap.add_argument("--model", default=None, help="override policy model")
    ap.add_argument("--sft-ckpt", default=None, help="init policy from this SFT checkpoint")
    ap.add_argument("--data", default=None, help="override prompt_dataset path")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the OpenRLHF command and exit")
    ap.add_argument("--emit-env", action="store_true",
                    help="resolve run_id + reward env, print KEY=VALUE lines to stdout, exit. "
                         "The launcher exports these BEFORE `ray start` so Ray workers "
                         "(which run reward.py) inherit ALAB_REWARD_MODE etc. Without this the "
                         "workers see reward.py's defaults and every arm silently runs `gt`.")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    if args.smoke:
        cfg = apply_smoke(cfg)
    if args.model:
        cfg["model"] = args.model
    if args.sft_ckpt:
        cfg["sft_ckpt"] = args.sft_ckpt
    if args.reward_mode:
        cfg["reward_mode"] = args.reward_mode
    if args.data:
        cfg.setdefault("data", {})["prompt_dataset"] = args.data

    reward_mode = cfg.get("reward_mode", "gt")
    assert reward_mode in METHOD_TAG, f"bad reward_mode {reward_mode!r}"
    ccfg = cluster_cfg()

    # Run id: {model}_{method}_{data}_{MMDD} (contract §1).
    ms = model_short(cfg["model"], cfg)
    data_tag = cfg.get("data", {}).get("data_tag", "math")
    mmdd = _dt.date.today().strftime("%m%d")
    run_id = args.run_id or f"{ms}_{METHOD_TAG[reward_mode]}_{data_tag}_{mmdd}"

    run_dir = os.path.join(REPO, "results", "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    scratch = os.environ.get("ALAB_SCRATCH") or ccfg.get("scratch", os.path.join(REPO, "scratch"))
    if args.smoke:
        # $ALAB_SCRATCH is on the quota'd home fileset; a smoke checkpoint is throwaway,
        # so write it to node-local $TMPDIR (ephemeral, off-quota) to avoid the home
        # 100GB cap.
        ckpt_dir = os.path.join(os.environ.get("TMPDIR", "/tmp"), "alab_smoke_ckpt", run_id)
    elif env_true("ALAB_NODE_LOCAL"):
        # Home is a 100 GB hard quota and $ALAB_SCRATCH lives on it; one 8B HF
        # checkpoint (~16 GB) blows it (job 827583). Real runs write to node-local
        # disk and push the final model to HF Hub; the launcher trap-cleans the dir.
        node_tmp = os.environ.get("ALAB_NODE_TMP") or f"/tmp/alab_{os.environ.get('LSB_JOBID', os.getpid())}"
        ckpt_dir = os.path.join(node_tmp, "grpo", run_id)
    else:
        ckpt_dir = os.path.join(scratch, "checkpoints", run_id)
        print("[train_grpo] WARNING: ALAB_NODE_LOCAL not set — checkpoint goes to "
              f"{ckpt_dir} on the quota'd home fileset; an 8B run WILL die on quota. "
              "Export ALAB_NODE_LOCAL=1 (ray_lsf_launch.sh does this).", file=sys.stderr)
    os.makedirs(ckpt_dir, exist_ok=True)

    if args.smoke:
        ensure_smoke_data(cfg["data"]["prompt_dataset"])

    if args.gpus is not None:
        ngpu = args.gpus
    elif args.smoke:
        ngpu = 1
    else:
        ngpu = int(os.environ.get("ALAB_NUM_GPUS", "8"))

    # Snapshot the resolved config (contract: results/runs/<run_id>/config.yaml).
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump({"run_id": run_id, "gpus": ngpu, "reward_mode": reward_mode,
                        "resolved": cfg}, f, sort_keys=False)

    # Env for reward.py (it is imported standalone by OpenRLHF workers).
    env = os.environ.copy()
    env["ALAB_REWARD_MODE"] = reward_mode
    env["ALAB_RUN_ID"] = run_id
    env["ALAB_RESULTS_DIR"] = os.path.join(REPO, "results", "runs")
    env["ALAB_SAMPLE_LOG_RATE"] = str(cfg.get("sample_log_rate", 0.01))
    env["ALAB_REWARD_SEED"] = str(cfg.get("seed", 42))
    env["HF_HOME"] = env.get("HF_HOME", ccfg.get("hf_home", os.path.expanduser("~/.cache/huggingface")))

    # --emit-env: print only the reward env (stdout) for the launcher to export before
    # `ray start`, so Ray workers running reward.py inherit the arm + run id + log path.
    if args.emit_env:
        for k in ("ALAB_REWARD_MODE", "ALAB_RUN_ID", "ALAB_RESULTS_DIR",
                  "ALAB_SAMPLE_LOG_RATE", "ALAB_REWARD_SEED", "HF_HOME"):
            print(f"{k}={env[k]}")
        return 0

    argv = build_argv(cfg, ngpu, run_id, ckpt_dir, run_dir, ccfg)

    print(f"[train_grpo] run_id={run_id} reward_mode={reward_mode} gpus={ngpu}")
    print(f"[train_grpo] ckpt_dir={ckpt_dir}")
    print("[train_grpo] OpenRLHF command:\n  " + " ".join(argv))

    if args.dry_run:
        print("[train_grpo] --dry-run: not launching.")
        return 0

    start = time.time()
    rc = 1
    try:
        rc = subprocess.call(argv, env=env, cwd=REPO)
    finally:
        hub_ok = False
        hub_repo = hub_repo_id(run_id, cfg)
        hub_url = None
        if rc == 0 and env_true("ALAB_HUB_PUSH", default=not args.smoke):
            hub_ok, hub_repo, hub_url = push_to_hub_with_retries(ckpt_dir, run_id, cfg)

        wall_s = time.time() - start
        wall_h = wall_s / 3600.0
        metrics = {
            "run_id": run_id,
            "reward_mode": reward_mode,
            "gpus": ngpu,
            "wall_seconds": round(wall_s, 1),
            "wall_hours": round(wall_h, 4),
            "gpu_hours": round(ngpu * wall_h, 4),  # compute-matching currency (§4)
            "returncode": rc,
            "hub_push_ok": hub_ok,
            "hub_repo_id": hub_repo,
            "hub_url": hub_url,
            "output_dir": ckpt_dir,
            "start": _dt.datetime.fromtimestamp(start).isoformat(),
            "end": _dt.datetime.now().isoformat(),
        }
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[train_grpo] wrote metrics.json: gpu_hours={metrics['gpu_hours']} rc={rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
