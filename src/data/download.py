#!/usr/bin/env python3
"""Cache AlignmentLab datasets and model snapshots for offline cluster jobs."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_DATASETS = (
    ("allenai/tulu-3-sft-mixture", None),
    ("HuggingFaceH4/ultrafeedback_binarized", None),
    ("nvidia/HelpSteer2", None),
    ("agentica-org/DeepScaleR-Preview-Dataset", None),
    ("openai/gsm8k", "main"),
    ("HuggingFaceH4/MATH-500", None),
    ("qintongli/GSM-Plus", None),
    # Needed for Phase-1 8-gram decontamination against AIME24.
    ("Maxwell-Jia/AIME_2024", None),
)

REQUIRED_MODELS = (
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-8B",
)

LLAMA_PRIMARY = "meta-llama/Llama-3.1-8B"
LLAMA_FALLBACK = "meta-llama/Meta-Llama-3-8B"


@dataclass(frozen=True)
class CacheResult:
    name: str
    ok: bool
    detail: str


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
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or ":" not in line or raw[:1].isspace():
                continue
            key, value = line.split(":", 1)
            data[key.strip()] = _strip_quotes(value)
        return data


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def cache_dataset(path: str, name: str | None, revision: str | None = None) -> CacheResult:
    from datasets import load_dataset

    label = path if name is None else f"{path}/{name}"
    kwargs: dict[str, Any] = {}
    if revision:
        kwargs["revision"] = revision
    try:
        if name:
            load_dataset(path, name, **kwargs)
        else:
            load_dataset(path, **kwargs)
        return CacheResult(label, True, "cached")
    except Exception as exc:  # pragma: no cover - cluster/runtime dependent
        return CacheResult(label, False, repr(exc))


def cache_model(repo_id: str, token: str | None, full_model: bool) -> CacheResult:
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoTokenizer

    try:
        AutoTokenizer.from_pretrained(repo_id, token=token, trust_remote_code=True)
        AutoConfig.from_pretrained(repo_id, token=token, trust_remote_code=True)
        if full_model:
            snapshot_download(repo_id=repo_id, token=token, resume_download=True)
        return CacheResult(repo_id, True, "cached")
    except Exception as exc:  # pragma: no cover - cluster/runtime dependent
        return CacheResult(repo_id, False, repr(exc))


def loud_warning(message: str) -> None:
    print(f"\nWARNING: {message}\n", file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download/cache AlignmentLab Phase-1 datasets and model snapshots in HF_HOME. "
            "Run this only through scripts/submit_download.sh on the LSF CPU queue."
        )
    )
    parser.add_argument(
        "--cluster-config",
        type=Path,
        default=repo_root_from_script() / "configs" / "cluster.yaml",
        help="Path to configs/cluster.yaml.",
    )
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=None,
        help="Override HF_HOME. Defaults to configs/cluster.yaml hf_home, then existing HF_HOME.",
    )
    parser.add_argument(
        "--tokenizer-only",
        action="store_true",
        help="Cache tokenizers/configs only. Default caches full model snapshots for offline training.",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Cache datasets only.",
    )
    parser.add_argument(
        "--skip-datasets",
        action="store_true",
        help="Cache models only.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cluster = load_yaml(args.cluster_config)

    hf_home = args.hf_home or Path(
        os.environ.get("HF_HOME") or cluster.get("hf_home") or Path.home() / ".cache" / "huggingface"
    )
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    print(f"HF_HOME={hf_home}", flush=True)
    print(f"cluster_config={args.cluster_config}", flush=True)

    failures: list[CacheResult] = []

    if not args.skip_datasets:
        print("Caching datasets:", flush=True)
        for dataset_path, dataset_name in REQUIRED_DATASETS:
            result = cache_dataset(dataset_path, dataset_name)
            print(f"  [{'OK' if result.ok else 'FAIL'}] {result.name}: {result.detail}", flush=True)
            if not result.ok:
                failures.append(result)

    if not args.skip_models:
        token = os.environ.get("HF_TOKEN")
        full_model = not args.tokenizer_only
        print("Caching model snapshots:" if full_model else "Caching model tokenizers/configs:", flush=True)
        for model in REQUIRED_MODELS:
            result = cache_model(model, token=token, full_model=full_model)
            print(f"  [{'OK' if result.ok else 'FAIL'}] {result.name}: {result.detail}", flush=True)
            if not result.ok:
                failures.append(result)

        if not token:
            loud_warning(
                f"HF_TOKEN is unset; gated {LLAMA_PRIMARY} will likely fail. "
                f"Trying {LLAMA_FALLBACK} if access is denied."
            )
        primary = cache_model(LLAMA_PRIMARY, token=token, full_model=full_model)
        print(f"  [{'OK' if primary.ok else 'FAIL'}] {primary.name}: {primary.detail}", flush=True)
        if not primary.ok:
            loud_warning(
                f"Could not cache gated {LLAMA_PRIMARY}. Falling back to {LLAMA_FALLBACK}."
            )
            fallback = cache_model(LLAMA_FALLBACK, token=token, full_model=full_model)
            print(f"  [{'OK' if fallback.ok else 'FAIL'}] {fallback.name}: {fallback.detail}", flush=True)
            if not fallback.ok:
                loud_warning(
                    f"Skipping Llama cache entirely because both {LLAMA_PRIMARY} and "
                    f"{LLAMA_FALLBACK} failed. Primary error: {primary.detail}. "
                    f"Fallback error: {fallback.detail}"
                )

    if failures:
        print("\nRequired cache failures:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure.name}: {failure.detail}", file=sys.stderr)
        return 1

    print("All required Phase-1 datasets/models are cached.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
