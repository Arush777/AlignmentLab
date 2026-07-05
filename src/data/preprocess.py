#!/usr/bin/env python3
"""Build AlignmentLab processed JSONL files from cached HF datasets."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, Iterator, Sequence


MESSAGE_ROLES = {"system", "user", "assistant", "tool"}
WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    root = repo_root_from_script()
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess cached AlignmentLab datasets into PLAN.md contract JSONL schemas. "
            "This is CPU-heavy and should be run through scripts/submit_download.sh."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=root, help="AlignmentLab repository root.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "data" / "processed",
        help="Directory for processed JSONL outputs.",
    )
    parser.add_argument("--sft-size", type=int, default=150_000, help="Tulu SFT sample count.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument(
        "--aime24-local",
        type=Path,
        default=None,
        help="Optional local JSON/JSONL/TXT file of AIME24 questions to add to decontamination.",
    )
    return parser.parse_args()


def normalize_ws(text: Any) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False, sort_keys=True)
    return " ".join(text.split())


def tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(text)]


def eightgrams(text: str) -> set[tuple[str, ...]]:
    toks = tokens(text)
    if len(toks) < 8:
        return set()
    return {tuple(toks[i : i + 8]) for i in range(len(toks) - 7)}


def message_text(messages: Sequence[dict[str, str]]) -> str:
    return "\n".join(normalize_ws(msg.get("content", "")) for msg in messages)


def has_heldout_overlap(text: str, heldout_grams: set[tuple[str, ...]]) -> bool:
    sample_grams = eightgrams(text)
    return bool(sample_grams and sample_grams.intersection(heldout_grams))


def normalize_role(role: Any) -> str:
    role_text = str(role or "").strip().lower()
    if role_text in {"human", "prompter"}:
        return "user"
    if role_text in {"gpt", "bot", "model"}:
        return "assistant"
    if role_text in MESSAGE_ROLES:
        return role_text
    return "user"


def normalize_messages(value: Any) -> list[dict[str, str]]:
    if isinstance(value, str):
        return [{"role": "user", "content": normalize_ws(value)}] if normalize_ws(value) else []
    if not isinstance(value, list):
        return []

    messages: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = item.get("role", item.get("from", item.get("speaker", "user")))
        content = item.get("content", item.get("value", item.get("text", "")))
        content_text = normalize_ws(content)
        if not content_text:
            continue
        messages.append({"role": normalize_role(role), "content": content_text})
    return messages


def prompt_messages_from_text(text: Any) -> list[dict[str, str]]:
    text = normalize_ws(text)
    return [{"role": "user", "content": text}] if text else []


def validate_messages(messages: Any, field: str) -> list[dict[str, str]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{field} must be a non-empty list of messages")
    normalized = normalize_messages(messages)
    if len(normalized) != len(messages):
        raise ValueError(f"{field} contains invalid/empty messages")
    return normalized


def validate_sft(row: dict[str, Any]) -> dict[str, Any]:
    messages = validate_messages(row.get("messages"), "messages")
    return {"messages": messages}


def validate_pref(row: dict[str, Any]) -> dict[str, Any]:
    source = row.get("source")
    if source not in {"uf", "hs2"}:
        raise ValueError("preference source must be 'uf' or 'hs2'")
    prompt = validate_messages(row.get("prompt"), "prompt")
    chosen = validate_messages(row.get("chosen"), "chosen")
    rejected = validate_messages(row.get("rejected"), "rejected")
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected, "source": source}


def validate_rlvr(row: dict[str, Any]) -> dict[str, Any]:
    prompt = validate_messages(row.get("prompt"), "prompt")
    answer = normalize_ws(row.get("answer"))
    if not answer:
        raise ValueError("RLVR answer must be non-empty")
    return {"prompt": prompt, "answer": answer}


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]], validator) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        for row in rows:
            checked = validator(row)
            tmp.write(json.dumps(checked, ensure_ascii=False) + "\n")
            count += 1
    tmp_path.replace(path)
    return count


def load_dataset_cached(path: str, *args: Any, split: str | None = None):
    from datasets import load_dataset

    if split is None:
        return load_dataset(path, *args)
    return load_dataset(path, *args, split=split)


def choose_split(dataset_dict: Any, preferred: Sequence[str]) -> Any:
    if not hasattr(dataset_dict, "keys"):
        return dataset_dict
    keys = set(dataset_dict.keys())
    for name in preferred:
        if name in keys:
            return dataset_dict[name]
    return dataset_dict[sorted(keys)[0]]


def heldout_questions_from_local(path: Path) -> Iterator[str]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            yield extract_question(obj)
    elif suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        records = obj if isinstance(obj, list) else obj.get("data", [])
        for record in records:
            yield extract_question(record)
    else:
        for block in path.read_text(encoding="utf-8").split("\n\n"):
            text = normalize_ws(block)
            if text:
                yield text


def extract_question(row: dict[str, Any]) -> str:
    for key in ("question", "Question", "problem", "Problem", "prompt", "input"):
        value = row.get(key)
        if isinstance(value, list):
            text = message_text(normalize_messages(value))
        else:
            text = normalize_ws(value)
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


def build_heldout_grams(aime24_local: Path | None) -> set[tuple[str, ...]]:
    heldout_texts: list[str] = []

    gsm8k = load_dataset_cached("openai/gsm8k", "main")
    for row in choose_split(gsm8k, ("test",)):
        question = extract_question(row)
        if question:
            heldout_texts.append(question)

    math500 = load_dataset_cached("HuggingFaceH4/MATH-500")
    for row in choose_split(math500, ("test", "train")):
        question = extract_question(row)
        if question:
            heldout_texts.append(question)

    try:
        aime24 = load_dataset_cached("Maxwell-Jia/AIME_2024")
        for row in choose_split(aime24, ("train", "test")):
            question = extract_question(row)
            if question:
                heldout_texts.append(question)
    except Exception as exc:
        if aime24_local is None:
            raise RuntimeError(
                "AIME24 HF cache is unavailable and --aime24-local was not provided; "
                "cannot satisfy Phase-1 decontamination."
            ) from exc

    if aime24_local is not None:
        heldout_texts.extend(text for text in heldout_questions_from_local(aime24_local) if text)

    grams: set[tuple[str, ...]] = set()
    for text in heldout_texts:
        grams.update(eightgrams(text))

    print(
        f"Built decontamination index: {len(heldout_texts)} held-out questions, "
        f"{len(grams)} unique 8-grams.",
        flush=True,
    )
    if not grams:
        raise RuntimeError("Held-out 8-gram index is empty; refusing to preprocess.")
    return grams


def decontaminate(
    rows: Iterable[dict[str, Any]],
    text_fn,
    heldout_grams: set[tuple[str, ...]],
    label: str,
) -> Iterator[dict[str, Any]]:
    total = 0
    removed = 0
    for row in rows:
        total += 1
        if has_heldout_overlap(text_fn(row), heldout_grams):
            removed += 1
            continue
        yield row
    print(f"Decontamination {label}: removed {removed} / {total}", flush=True)


def iter_sft_rows(seed: int, sft_size: int) -> Iterator[dict[str, Any]]:
    dataset = choose_split(load_dataset_cached("allenai/tulu-3-sft-mixture"), ("train",))
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    emitted = 0
    for idx in indices:
        row = dataset[int(idx)]
        messages = normalize_messages(row.get("messages") or row.get("conversation") or row.get("conversations"))
        if not messages:
            prompt = row.get("prompt") or row.get("instruction")
            response = row.get("response") or row.get("output")
            if normalize_ws(prompt) and normalize_ws(response):
                messages = [
                    {"role": "user", "content": normalize_ws(prompt)},
                    {"role": "assistant", "content": normalize_ws(response)},
                ]
        if messages:
            emitted += 1
            yield {"messages": messages}
        if emitted >= sft_size:
            break

    if emitted < sft_size:
        raise RuntimeError(f"Only emitted {emitted} valid SFT rows; requested {sft_size}.")


def split_preference_messages(row: dict[str, Any]) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    prompt = normalize_messages(row.get("prompt"))
    chosen = normalize_messages(row.get("chosen"))
    rejected = normalize_messages(row.get("rejected"))

    if not prompt and chosen and rejected:
        common_len = 0
        for left, right in zip(chosen, rejected):
            if left == right:
                common_len += 1
            else:
                break
        prompt = chosen[:common_len]
        chosen = chosen[common_len:]
        rejected = rejected[common_len:]

    if prompt and chosen and chosen[: len(prompt)] == prompt:
        chosen = chosen[len(prompt) :]
    if prompt and rejected and rejected[: len(prompt)] == prompt:
        rejected = rejected[len(prompt) :]

    return prompt, chosen, rejected


def iter_ultrafeedback_rows() -> Iterator[dict[str, Any]]:
    dataset = choose_split(
        load_dataset_cached("HuggingFaceH4/ultrafeedback_binarized"),
        ("train_prefs", "train"),
    )
    for row in dataset:
        prompt, chosen, rejected = split_preference_messages(row)
        if prompt and chosen and rejected:
            yield {"prompt": prompt, "chosen": chosen, "rejected": rejected, "source": "uf"}


def iter_helpsteer2_rows() -> Iterator[dict[str, Any]]:
    dataset = choose_split(load_dataset_cached("nvidia/HelpSteer2"), ("train",))
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dataset:
        prompt = normalize_ws(row.get("prompt") or row.get("instruction") or row.get("question"))
        response = normalize_ws(row.get("response") or row.get("answer") or row.get("completion"))
        if not prompt or not response:
            continue
        try:
            helpfulness = float(row.get("helpfulness"))
        except (TypeError, ValueError):
            continue
        by_prompt[prompt].append({"response": response, "helpfulness": helpfulness})

    for prompt, candidates in by_prompt.items():
        if len(candidates) < 2:
            continue
        sorted_candidates = sorted(candidates, key=lambda item: item["helpfulness"])
        lowest = sorted_candidates[0]["helpfulness"]
        highest = sorted_candidates[-1]["helpfulness"]
        if lowest == highest:
            continue
        if sum(item["helpfulness"] == highest for item in candidates) != 1:
            continue
        if sum(item["helpfulness"] == lowest for item in candidates) != 1:
            continue
        chosen = sorted_candidates[-1]["response"]
        rejected = sorted_candidates[0]["response"]
        yield {
            "prompt": [{"role": "user", "content": prompt}],
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}],
            "source": "hs2",
        }


def final_gsm8k_answer(answer: str) -> str:
    answer = normalize_ws(answer)
    if "####" in answer:
        return normalize_ws(answer.rsplit("####", 1)[1])
    return answer


def iter_rlvr_rows() -> Iterator[dict[str, Any]]:
    deep_scaler = choose_split(
        load_dataset_cached("agentica-org/DeepScaleR-Preview-Dataset"),
        ("train",),
    )
    for row in deep_scaler:
        question = extract_question(row)
        answer = extract_answer(row)
        if question and answer:
            yield {"prompt": prompt_messages_from_text(question), "answer": answer}

    gsm8k_train = choose_split(load_dataset_cached("openai/gsm8k", "main"), ("train",))
    for row in gsm8k_train:
        question = extract_question(row)
        answer = final_gsm8k_answer(row.get("answer", ""))
        if question and answer:
            yield {"prompt": prompt_messages_from_text(question), "answer": answer}


def pref_text(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            message_text(row["prompt"]),
            message_text(row["chosen"]),
            message_text(row["rejected"]),
        ]
    )


def main() -> int:
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    heldout_grams = build_heldout_grams(args.aime24_local)

    sft_rows = list(
        decontaminate(
            iter_sft_rows(seed=args.seed, sft_size=args.sft_size),
            lambda row: message_text(row["messages"]),
            heldout_grams,
            "sft",
        )
    )
    sft_count = write_jsonl(args.output_dir / "sft.jsonl", sft_rows, validate_sft)
    print(f"Wrote sft.jsonl: {sft_count} rows", flush=True)

    uf_rows = list(
        decontaminate(iter_ultrafeedback_rows(), pref_text, heldout_grams, "pref_uf")
    )
    uf_count = write_jsonl(args.output_dir / "pref_uf.jsonl", uf_rows, validate_pref)
    print(f"Wrote pref_uf.jsonl: {uf_count} rows", flush=True)

    hs2_rows = list(
        decontaminate(iter_helpsteer2_rows(), pref_text, heldout_grams, "pref_hs2")
    )
    hs2_count = write_jsonl(args.output_dir / "pref_hs2.jsonl", hs2_rows, validate_pref)
    print(f"Wrote pref_hs2.jsonl: {hs2_count} rows", flush=True)

    rng = random.Random(args.seed)
    total_mix = (min(len(uf_rows), len(hs2_rows)) // 2) * 2
    half_mix = total_mix // 2
    mix_rows = rng.sample(uf_rows, half_mix) + rng.sample(hs2_rows, half_mix)
    rng.shuffle(mix_rows)
    mix_count = write_jsonl(args.output_dir / "pref_mix.jsonl", mix_rows, validate_pref)
    print(
        f"Wrote pref_mix.jsonl: {mix_count} rows "
        f"({half_mix} UF + {half_mix} HS2, seed={args.seed})",
        flush=True,
    )

    rlvr_rows = list(
        decontaminate(
            iter_rlvr_rows(),
            lambda row: message_text(row["prompt"]) + "\n" + normalize_ws(row["answer"]),
            heldout_grams,
            "rlvr_math",
        )
    )
    rlvr_count = write_jsonl(args.output_dir / "rlvr_math.jsonl", rlvr_rows, validate_rlvr)
    print(f"Wrote rlvr_math.jsonl: {rlvr_count} rows", flush=True)

    if not all((sft_count, uf_count, hs2_count, mix_count, rlvr_count)):
        raise RuntimeError("One or more processed datasets are empty; refusing to continue.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
