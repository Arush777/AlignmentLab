#!/usr/bin/env python3
"""TRL SFT training entrypoint for AlignmentLab."""

from __future__ import annotations

import argparse
import atexit
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Any


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if "#" in value:
        value = value.split("#", 1)[0].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path} must contain a mapping")
        return data
    except ModuleNotFoundError:
        data: dict[str, Any] = {}
        current_parent: str | None = None
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip() or raw.strip().startswith("#"):
                continue
            indent = len(raw) - len(raw.lstrip(" "))
            line = raw.strip()
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            if indent == 0:
                if value.strip():
                    data[key.strip()] = _strip_quotes(value)
                    current_parent = None
                else:
                    current_parent = key.strip()
                    data[current_parent] = {}
            elif current_parent:
                data[current_parent][key.strip()] = _strip_quotes(value)
        return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a TRL SFTTrainer job from a YAML config.")
    parser.add_argument("--config", type=Path, required=True, help="SFT experiment YAML.")
    parser.add_argument(
        "--cluster-config",
        type=Path,
        default=repo_root_from_script() / "configs" / "cluster.yaml",
        help="Path to configs/cluster.yaml.",
    )
    parser.add_argument("--run-id", default=None, help="Override config run_id.")
    parser.add_argument("--model", default=None, help="Override model_name_or_path.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override training.max_steps.")
    return parser.parse_args()


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def rank0() -> bool:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0


class MetricsWriter:
    def __init__(self, run_dir: Path, run_id: str, n_gpus: int, start_time: float) -> None:
        self.run_dir = run_dir
        self.run_id = run_id
        self.n_gpus = n_gpus
        self.start_time = start_time
        self.status = "failed"
        self._written = False

    def write(self) -> None:
        if self._written or not rank0():
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        wall_hours = (time.time() - self.start_time) / 3600.0
        metrics = {
            "run_id": self.run_id,
            "status": self.status,
            "n_gpus": self.n_gpus,
            "wall_hours": wall_hours,
            "gpu_hours": self.n_gpus * wall_hours,
            "updated_at_unix": int(time.time()),
        }
        path = self.run_dir / "metrics.json"
        path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self._written = True


def build_deepspeed_config(run_dir: Path, zero_stage: int) -> Path:
    config = {
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": zero_stage,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_gather_16bit_weights_on_model_save": True,
        },
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "train_batch_size": "auto",
    }
    path = run_dir / "deepspeed_zero3.json"
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    root = repo_root_from_script()
    config = load_yaml(args.config)
    cluster = load_yaml(args.cluster_config)

    run_id = args.run_id or str(config["run_id"])
    model_name = args.model or str(config["model_name_or_path"])
    training = config.get("training", {}) or {}
    resources = config.get("resources", {}) or {}

    scratch = Path(os.environ.get("ALAB_SCRATCH") or cluster.get("scratch") or "")
    if not scratch:
        raise RuntimeError("ALAB_SCRATCH/configs.cluster scratch is required")
    checkpoint_dir = scratch / "checkpoints" / run_id
    if root.resolve() in checkpoint_dir.resolve().parents:
        raise RuntimeError(f"Checkpoint dir must not be inside repo: {checkpoint_dir}")

    run_dir = root / "results" / "runs" / run_id
    if rank0():
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.config, run_dir / "config.yaml")

    world_size = int(os.environ.get("WORLD_SIZE") or resources.get("n_gpus") or 1)
    metrics = MetricsWriter(run_dir=run_dir, run_id=run_id, n_gpus=world_size, start_time=time.time())
    atexit.register(metrics.write)

    def signal_handler(signum: int, _frame: Any) -> None:
        metrics.status = f"signal_{signum}"
        metrics.write()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("WANDB_PROJECT", "alignmentlab")
    wandb_entity = str(cluster.get("wandb_entity", "") or "")
    if wandb_entity and wandb_entity != "CHANGE_ME":
        os.environ.setdefault("WANDB_ENTITY", wandb_entity)

    from datasets import load_dataset
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    dataset_path = Path(str(config.get("dataset_path", "data/processed/sft.jsonl")))
    if not dataset_path.is_absolute():
        dataset_path = root / dataset_path
    if not dataset_path.exists():
        raise FileNotFoundError(f"Processed SFT dataset not found: {dataset_path}")

    train_dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    max_train_samples = training.get("max_train_samples")
    if max_train_samples:
        train_dataset = train_dataset.select(range(min(int(max_train_samples), len(train_dataset))))

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
    }
    attn_implementation = training.get("attn_implementation")
    if attn_implementation:
        model_kwargs["attn_implementation"] = str(attn_implementation)
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.config.use_cache = False
    if as_bool(training.get("gradient_checkpointing"), default=True):
        model.gradient_checkpointing_enable()

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    max_steps = args.max_steps if args.max_steps is not None else training.get("max_steps", -1)
    num_train_epochs = training.get("num_train_epochs", 1)
    use_deepspeed = as_bool(training.get("deepspeed"), default=False) or int(training.get("zero_stage", 0) or 0) > 0
    deepspeed_path = None
    if use_deepspeed:
        zero_stage = int(training.get("zero_stage", 3))
        deepspeed_path = str(build_deepspeed_config(run_dir, zero_stage))

    sft_args = SFTConfig(
        output_dir=str(checkpoint_dir),
        overwrite_output_dir=as_bool(training.get("overwrite_output_dir"), default=True),
        max_length=int(training.get("max_seq_length", 4096)),
        packing=as_bool(training.get("packing"), default=True),
        assistant_only_loss=as_bool(training.get("assistant_only_loss"), default=True),
        bf16=as_bool(training.get("bf16"), default=True),
        learning_rate=float(training.get("learning_rate", 2.0e-5)),
        lr_scheduler_type=str(training.get("lr_scheduler_type", "cosine")),
        warmup_ratio=float(training.get("warmup_ratio", 0.03)),
        weight_decay=float(training.get("weight_decay", 0.0)),
        per_device_train_batch_size=int(training.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 8)),
        num_train_epochs=float(num_train_epochs),
        max_steps=int(max_steps),
        logging_steps=int(training.get("logging_steps", 10)),
        save_steps=int(training.get("save_steps", 500)),
        save_total_limit=int(training.get("save_total_limit", 2)),
        # Home-quota constraint: skip optimizer/scheduler state in checkpoints
        # (~100GB extra per 8B ckpt). No mid-run resume; wall time must cover the run.
        save_only_model=as_bool(training.get("save_only_model"), default=True),
        gradient_checkpointing=as_bool(training.get("gradient_checkpointing"), default=True),
        report_to=["wandb"],
        run_name=run_id,
        seed=int(training.get("seed", config.get("seed", 42))),
        dataloader_num_workers=int(training.get("dataloader_num_workers", 4)),
        deepspeed=deepspeed_path,
    )

    trainer_kwargs = {
        "model": model,
        "args": sft_args,
        "train_dataset": train_dataset,
    }
    try:
        trainer = SFTTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = SFTTrainer(tokenizer=tokenizer, **trainer_kwargs)

    trainer.train()
    trainer.save_model(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))
    metrics.status = "complete"
    metrics.write()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
