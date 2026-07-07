#!/usr/bin/env python3
"""Run held-out math evals and write results/evals/<run_id>/lm_eval.json."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer
from vllm import SamplingParams

try:
    from .common import (
        apply_chat_template,
        build_math_prompt,
        load_eval_rows,
        load_vllm,
        math_verify_equal,
        normalize_local_tokenizer_config,
        repo_root_from_script,
        task_display_name,
        write_json,
    )
except ImportError:  # pragma: no cover - direct script execution
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.evals.common import (  # type: ignore
        apply_chat_template,
        build_math_prompt,
        load_eval_rows,
        load_vllm,
        math_verify_equal,
        normalize_local_tokenizer_config,
        repo_root_from_script,
        task_display_name,
        write_json,
    )


SUPPORTED_TASKS = ("gsm8k", "math-500", "gsm_plus")


def parse_tasks(value: str) -> list[str]:
    if value.strip().lower() in {"all", "lm-eval", "lm_eval"}:
        return list(SUPPORTED_TASKS)
    tasks: list[str] = []
    for raw in value.split(","):
        task = raw.strip()
        if not task:
            continue
        normalized = task_display_name(task)
        if normalized == "math500":
            tasks.append("math-500")
        elif normalized in {"gsm8k", "gsm_plus"}:
            tasks.append(normalized)
        else:
            raise ValueError(f"Unsupported lm eval task: {task}")
    return tasks


def parse_args() -> argparse.Namespace:
    root = repo_root_from_script()
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a Hub repo id or local checkpoint on AlignmentLab held-out math tasks. "
            "Outputs a raw lm-eval-compatible JSON object at results/evals/<run_id>/lm_eval.json."
        )
    )
    parser.add_argument("--model", required=True, help="HF repo id or local model path.")
    parser.add_argument("--run-id", required=True, help="Eval run id / results subdirectory.")
    parser.add_argument("--tasks", default="gsm8k,math-500,gsm_plus", help="Comma tasks or 'all'.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to results/evals/<run_id>.")
    parser.add_argument("--limit", type=int, default=None, help="Optional per-task row limit for smoke evals.")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write-samples", action="store_true", help="Also write lm_eval_samples.jsonl.")
    return parser.parse_args()


def batched(seq: list[Any], size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def evaluate_task(
    llm: Any,
    tokenizer: Any,
    model: str,
    task: str,
    args: argparse.Namespace,
    samples_path: Path | None,
) -> dict[str, Any]:
    rows = load_eval_rows(task, limit=args.limit)
    prompts = [apply_chat_template(tokenizer, build_math_prompt(row["question"])) for row in rows]
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    correct = 0
    total = 0
    sample_lines: list[dict[str, Any]] = []
    for offset, prompt_batch in enumerate(batched(prompts, args.batch_size)):
        batch_rows = rows[offset * args.batch_size : offset * args.batch_size + len(prompt_batch)]
        outputs = llm.generate(prompt_batch, sampling)
        for row, request_output in zip(batch_rows, outputs):
            completion = request_output.outputs[0].text if request_output.outputs else ""
            ok = math_verify_equal(completion, row["answer"])
            correct += int(ok)
            total += 1
            if samples_path is not None:
                sample_lines.append(
                    {
                        "task": task_display_name(task),
                        "problem_id": row["problem_id"],
                        "prompt": row["question"],
                        "answer": row["answer"],
                        "completion": completion,
                        "correct": ok,
                    }
                )
        if samples_path is not None and sample_lines:
            with samples_path.open("a", encoding="utf-8") as f:
                for line in sample_lines:
                    f.write(json.dumps(line, ensure_ascii=False, sort_keys=False) + "\n")
            sample_lines.clear()

    acc = correct / total if total else 0.0
    stderr = (acc * (1.0 - acc) / total) ** 0.5 if total else 0.0
    return {
        "alias": task_display_name(task),
        "source": rows[0]["source"],
        "exact_match,none": acc,
        "exact_match_stderr,none": stderr,
        "acc,none": acc,
        "acc_stderr,none": stderr,
        "samples": total,
        "correct": correct,
        "model": model,
    }


def main() -> int:
    args = parse_args()
    tasks = parse_tasks(args.tasks)
    root = repo_root_from_script()
    out_dir = args.output_dir or root / "results" / "evals" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "lm_eval.json"
    samples_path = out_dir / "lm_eval_samples.jsonl" if args.write_samples else None
    if samples_path and samples_path.exists():
        samples_path.unlink()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    start = time.time()
    normalize_local_tokenizer_config(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = load_vllm(
        args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    results: dict[str, Any] = {}
    for task in tasks:
        results[task_display_name(task)] = evaluate_task(llm, tokenizer, args.model, task, args, samples_path)

    versions: dict[str, Any] = {}
    n_shot: dict[str, Any] = {}
    configs: dict[str, Any] = {}
    for task in tasks:
        task_name = task_display_name(task)
        versions.setdefault(task_name, 0)
        n_shot.setdefault(task_name, 0)
        configs.setdefault(
            task_name,
            {
                "task": task_name,
                "dataset_path": results[task_name].get("source", ""),
                "output_type": "generate_until",
            },
        )

    payload = {
        "results": results,
        "versions": versions,
        "n-shot": n_shot,
        "configs": configs,
        "metadata": {
            "run_id": args.run_id,
            "model": args.model,
            "backend": "alignmentlab-vllm",
            "script": "src/evals/run_lm_eval.py",
            "limit": args.limit,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "elapsed_seconds": round(time.time() - start, 3),
        },
    }
    write_json(out_path, payload)
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
