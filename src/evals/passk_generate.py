#!/usr/bin/env python3
"""Generate AIME24 samples for unbiased pass@k scoring."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer
from vllm import SamplingParams

try:
    from .common import (
        append_jsonl,
        apply_chat_template,
        build_math_prompt,
        load_eval_rows,
        load_vllm,
        normalize_local_tokenizer_config,
        repo_root_from_script,
    )
except ImportError:  # pragma: no cover - direct script execution
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.evals.common import (  # type: ignore
        append_jsonl,
        apply_chat_template,
        build_math_prompt,
        load_eval_rows,
        load_vllm,
        normalize_local_tokenizer_config,
        repo_root_from_script,
    )


def parse_args() -> argparse.Namespace:
    root = repo_root_from_script()
    parser = argparse.ArgumentParser(description="Generate pass@k samples on AIME24 with vLLM.")
    parser.add_argument("--model", required=True, help="HF repo id or local model path.")
    parser.add_argument("--run-id", required=True, help="Eval run id / results subdirectory.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to results/evals/<run_id>.")
    parser.add_argument("--output", type=Path, default=None, help="Defaults to <output-dir>/passk_samples.jsonl.")
    parser.add_argument("--n-samples", type=int, default=256, help="Samples per problem.")
    parser.add_argument("--samples-per-call", type=int, default=16, help="vLLM n= chunk size.")
    parser.add_argument("--limit", type=int, default=None, help="Optional problem limit for smoke evals.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.n_samples < 1:
        raise ValueError("--n-samples must be >= 1")
    if args.samples_per_call < 1:
        raise ValueError("--samples-per-call must be >= 1")
    return args


def main() -> int:
    args = parse_args()
    root = repo_root_from_script()
    out_dir = args.output_dir or root / "results" / "evals" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output or out_dir / "passk_samples.jsonl"
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"{out_path} exists; pass --overwrite to replace it")
    if out_path.exists():
        out_path.unlink()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    rows = load_eval_rows("aime24", limit=args.limit)
    normalize_local_tokenizer_config(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = [apply_chat_template(tokenizer, build_math_prompt(row["question"])) for row in rows]
    llm = load_vllm(
        args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    start = time.time()
    sample_offset = 0
    remaining = args.n_samples
    while remaining > 0:
        chunk_n = min(args.samples_per_call, remaining)
        sampling = SamplingParams(
            n=chunk_n,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
            seed=args.seed + sample_offset,
        )
        outputs = llm.generate(prompts, sampling)
        jsonl_rows: list[dict[str, Any]] = []
        for row, request_output in zip(rows, outputs):
            for local_idx, completion in enumerate(request_output.outputs):
                jsonl_rows.append(
                    {
                        "task": "aime24",
                        "model": args.model,
                        "run_id": args.run_id,
                        "problem_id": row["problem_id"],
                        "prompt": row["question"],
                        "answer": row["answer"],
                        "sample_index": sample_offset + local_idx,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "completion": completion.text,
                    }
                )
        append_jsonl(out_path, jsonl_rows)
        sample_offset += chunk_n
        remaining -= chunk_n
        print(
            f"generated {sample_offset}/{args.n_samples} samples per problem "
            f"({len(rows)} problems, elapsed {time.time() - start:.1f}s)",
            flush=True,
        )

    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
