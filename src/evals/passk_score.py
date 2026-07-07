#!/usr/bin/env python3
"""Score AIME24 pass@k samples with the unbiased estimator and bootstrap CIs."""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path
from typing import Any

try:
    from .common import (
        bootstrap_mean,
        math_verify_equal,
        pass_at_k_unbiased,
        read_jsonl,
        repo_root_from_script,
        write_json,
    )
except ImportError:  # pragma: no cover - direct script execution
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.evals.common import (  # type: ignore
        bootstrap_mean,
        math_verify_equal,
        pass_at_k_unbiased,
        read_jsonl,
        repo_root_from_script,
        write_json,
    )


def parse_k_values(raw: str) -> list[int]:
    values = sorted({int(x.strip()) for x in raw.split(",") if x.strip()})
    if not values or values[0] < 1:
        raise ValueError("--k-values must contain positive integers")
    return values


def parse_args() -> argparse.Namespace:
    root = repo_root_from_script()
    parser = argparse.ArgumentParser(description="Score pass@k JSONL generations.")
    parser.add_argument("--run-id", required=True, help="Eval run id / results subdirectory.")
    parser.add_argument("--input", type=Path, default=None, help="Defaults to results/evals/<run_id>/passk_samples.jsonl.")
    parser.add_argument("--output", type=Path, default=None, help="Defaults to results/evals/<run_id>/passk.json.")
    parser.add_argument("--k-values", default="1,8,64,256")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root_from_script()
    eval_dir = root / "results" / "evals" / args.run_id
    input_path = args.input or eval_dir / "passk_samples.jsonl"
    output_path = args.output or eval_dir / "passk.json"
    k_values = parse_k_values(args.k_values)

    rows = read_jsonl(input_path)
    grouped: dict[tuple[str, float, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        task = str(row.get("task") or "aime24")
        temperature = float(row.get("temperature", 1.0))
        problem_id = str(row.get("problem_id"))
        grouped[(task, temperature, problem_id)].append(row)

    if not grouped:
        raise RuntimeError(f"No pass@k samples found in {input_path}")

    per_task_temp: dict[tuple[str, float], list[dict[str, Any]]] = collections.defaultdict(list)
    for (task, temperature, problem_id), samples in grouped.items():
        gold = str(samples[0].get("answer", ""))
        correct = 0
        for sample in samples:
            correct += int(math_verify_equal(str(sample.get("completion", "")), gold))
        per_task_temp[(task, temperature)].append(
            {
                "problem_id": problem_id,
                "n": len(samples),
                "correct": correct,
            }
        )

    out: list[dict[str, Any]] = []
    for (task, temperature), problems in sorted(per_task_temp.items()):
        n_samples = min(int(p["n"]) for p in problems)
        for k in k_values:
            if k > n_samples:
                raise ValueError(f"k={k} exceeds minimum n_samples={n_samples} for {task}")
            estimates = [pass_at_k_unbiased(int(p["n"]), int(p["correct"]), k) for p in problems]
            pass_at_k = sum(estimates) / len(estimates)
            stderr, ci_low, ci_high = bootstrap_mean(estimates, args.n_bootstrap, args.seed + k)
            out.append(
                {
                    "task": task,
                    "temperature": temperature,
                    "n_samples": n_samples,
                    "k": k,
                    "pass_at_k": pass_at_k,
                    "stderr": stderr,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "n_problems": len(problems),
                    "n_bootstrap": args.n_bootstrap,
                }
            )

    write_json(output_path, out)
    print(f"wrote {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
