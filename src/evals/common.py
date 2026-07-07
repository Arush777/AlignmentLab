#!/usr/bin/env python3
"""Shared helpers for AlignmentLab evaluation scripts."""

from __future__ import annotations

import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Iterable, Sequence


BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", re.DOTALL)
NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def normalize_ws(text: Any) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return " ".join(text.split())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(obj)
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def normalize_local_tokenizer_config(model: str) -> None:
    """Patch writable local snapshots whose tokenizer config predates this Transformers build."""
    config_path = Path(model) / "tokenizer_config.json"
    if not config_path.is_file():
        return
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(data, dict) or not isinstance(data.get("extra_special_tokens"), list):
        return

    data["extra_special_tokens"] = {}
    tmp = config_path.with_suffix(config_path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    tmp.replace(config_path)
    print(f"normalized tokenizer_config extra_special_tokens at {config_path}", flush=True)


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")
            count += 1
    return count


def choose_split(dataset_dict: Any, preferred: Sequence[str]) -> Any:
    if not hasattr(dataset_dict, "keys"):
        return dataset_dict
    keys = set(dataset_dict.keys())
    for name in preferred:
        if name in keys:
            return dataset_dict[name]
    return dataset_dict[sorted(keys)[0]]


def extract_question(row: dict[str, Any]) -> str:
    for key in ("question", "Question", "problem", "Problem", "prompt", "input"):
        text = normalize_ws(row.get(key))
        if text:
            return text
    return normalize_ws(row)


def extract_answer(row: dict[str, Any]) -> str:
    for key in ("answer", "Answer", "final_answer", "target", "label"):
        text = normalize_ws(row.get(key))
        if text:
            return text
    solution = normalize_ws(row.get("solution", row.get("Solution", "")))
    if "####" in solution:
        return normalize_ws(solution.rsplit("####", 1)[1])
    return solution


def task_display_name(task: str) -> str:
    aliases = {
        "math-500": "math500",
        "math_500": "math500",
        "gsm-plus": "gsm_plus",
        "gsmplus": "gsm_plus",
        "pass@k": "aime24",
        "passk": "aime24",
    }
    return aliases.get(task.strip().lower(), task.strip().lower())


def load_eval_rows(task: str, limit: int | None = None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    normalized = task_display_name(task)
    if normalized == "gsm8k":
        dataset = choose_split(load_dataset("openai/gsm8k", "main"), ("test",))
        source = "openai/gsm8k"
    elif normalized == "math500":
        dataset = choose_split(load_dataset("HuggingFaceH4/MATH-500"), ("test",))
        source = "HuggingFaceH4/MATH-500"
    elif normalized == "gsm_plus":
        dataset = choose_split(load_dataset("qintongli/GSM-Plus"), ("test", "testmini"))
        source = "qintongli/GSM-Plus"
    elif normalized == "aime24":
        dataset = choose_split(load_dataset("Maxwell-Jia/AIME_2024"), ("train", "test"))
        source = "Maxwell-Jia/AIME_2024"
    else:
        raise ValueError(f"Unsupported eval task: {task}")

    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(dataset):
        row_dict = dict(row)
        problem_id = (
            row_dict.get("unique_id")
            or row_dict.get("ID")
            or row_dict.get("id")
            or f"{normalized}-{idx}"
        )
        question = extract_question(row_dict)
        answer = extract_answer(row_dict)
        if not question or not answer:
            continue
        rows.append(
            {
                "task": normalized,
                "source": source,
                "problem_id": str(problem_id),
                "question": question,
                "answer": answer,
                "metadata": {
                    k: v
                    for k, v in row_dict.items()
                    if k not in {"question", "Question", "problem", "Problem", "answer", "Answer"}
                },
            }
        )
        if limit is not None and len(rows) >= limit:
            break
    if not rows:
        raise RuntimeError(f"No usable rows loaded for task {task}")
    return rows


def build_math_prompt(question: str) -> str:
    return (
        "Solve the following problem. Show your reasoning, and put only the final "
        "answer in \\boxed{}.\n\n"
        f"{question}"
    )


def apply_chat_template(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return prompt
    except Exception:
        return prompt


def load_vllm(model: str, tensor_parallel_size: int, max_model_len: int, gpu_memory_utilization: float) -> Any:
    from vllm import LLM

    normalize_local_tokenizer_config(model)
    return LLM(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        dtype="auto",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )


def extract_final_answer(text: str) -> str:
    text = text.strip()
    boxed = BOXED_RE.findall(text)
    if boxed:
        return normalize_ws(boxed[-1])
    if "####" in text:
        return normalize_ws(text.rsplit("####", 1)[1])
    lower = text.lower()
    for marker in ("final answer is", "answer is", "therefore"):
        pos = lower.rfind(marker)
        if pos >= 0:
            tail = text[pos + len(marker) :]
            nums = NUMBER_RE.findall(tail)
            if nums:
                return nums[-1].replace(",", "")
            return normalize_ws(tail.strip(" .:$"))
    nums = NUMBER_RE.findall(text)
    if nums:
        return nums[-1].replace(",", "")
    return normalize_ws(text[-200:])


def _simple_normalize_answer(text: Any) -> str:
    value = extract_final_answer(normalize_ws(text))
    value = value.strip().strip("$").strip()
    value = value.replace(",", "")
    if re.fullmatch(r"[-+]?\d+\.0+", value):
        value = value.split(".", 1)[0]
    if re.fullmatch(r"\d+", value) and len(value) <= 3:
        value = str(int(value))
    return value.lower()


def math_verify_equal(prediction: str, gold: str) -> bool:
    pred_final = extract_final_answer(prediction)
    try:
        from math_verify import parse, verify

        extraction_config = []
        for name in ("LatexExtractionConfig", "ExprExtractionConfig", "StringExtractionConfig"):
            try:
                cls = getattr(__import__("math_verify", fromlist=[name]), name)
                extraction_config.append(cls())
            except Exception:
                pass
        kwargs = {"extraction_config": extraction_config} if extraction_config else {}
        gold_parsed = parse(str(gold), **kwargs)
        pred_parsed = parse(prediction, **kwargs)
        pred_final_parsed = parse(pred_final, **kwargs)
        return bool(verify(gold_parsed, pred_parsed) or verify(gold_parsed, pred_final_parsed))
    except Exception:
        return _simple_normalize_answer(pred_final) == _simple_normalize_answer(gold)


def pass_at_k_unbiased(n: int, c: int, k: int) -> float:
    if k < 1:
        raise ValueError("k must be >= 1")
    if n < k:
        raise ValueError(f"Need at least k={k} samples, got n={n}")
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def bootstrap_mean(values: Sequence[float], n_bootstrap: int, seed: int) -> tuple[float, float, float]:
    if not values:
        return (0.0, 0.0, 0.0)
    if len(values) == 1 or n_bootstrap <= 1:
        return (0.0, float(values[0]), float(values[0]))
    rng = random.Random(seed)
    means: list[float] = []
    n = len(values)
    for _ in range(n_bootstrap):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    mean = sum(means) / len(means)
    variance = sum((x - mean) ** 2 for x in means) / (len(means) - 1)
    lo = means[int(0.025 * (len(means) - 1))]
    hi = means[int(0.975 * (len(means) - 1))]
    return (math.sqrt(variance), lo, hi)
