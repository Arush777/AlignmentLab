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
    print(f"[train_grpo] wrote smoke data: {path} ({len(rows)} rows)")


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
         "--ckpt.save_steps", str(c.get("save_steps", 20)),
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
    ckpt_dir = os.path.join(scratch, "checkpoints", run_id)  # contract §5: never in repo
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
            "start": _dt.datetime.fromtimestamp(start).isoformat(),
            "end": _dt.datetime.now().isoformat(),
        }
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[train_grpo] wrote metrics.json: gpu_hours={metrics['gpu_hours']} rc={rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
