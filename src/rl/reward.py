"""GRPO reward function for AlignmentLab's RLVR control battery (Track A).

OpenRLHF (0.10.4) loads this file via `--reward.remote_url src/rl/reward.py` and
calls `reward_func(queries, prompts, labels, **kwargs) -> dict` per response
(SingleTurnAgentExecutor). Because the file is imported standalone, its behaviour
is configured through environment variables set by train_grpo.py — NOT CLI flags:

    ALAB_REWARD_MODE    gt | random | format          (required; PLAN.md "GRPO reward arms")
    ALAB_RUN_ID         run id, for the samples log     (default "unknown")
    ALAB_RESULTS_DIR    root of results/runs            (default "results/runs")
    ALAB_SAMPLE_LOG_RATE fraction of (prompt,response,reward) to log  (default 0.01)
    ALAB_REWARD_SEED    seed for random-mode coin & sampler          (default 42)
    ALAB_FORMAT_BONUS   gt-mode bonus for a parseable \\boxed{}       (default 0.1)

Reward arms (identical everything else — this is the only thing that differs):
    gt      math-verify equivalence vs the `answer` field (+1/0) + small \\boxed bonus
    random  seeded Bernoulli(0.5) per response, content ignored (placebo)
    format  +1 iff a parseable \\boxed{} exists, correctness ignored

Every mode logs a ~1% sample of (prompt, response, reward) to
`<ALAB_RESULTS_DIR>/<ALAB_RUN_ID>/samples.jsonl` for later reward-hacking analysis.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from typing import List

import torch

# ---------------------------------------------------------------------------------
# Config from environment (read once at import).
# ---------------------------------------------------------------------------------
REWARD_MODE = os.environ.get("ALAB_REWARD_MODE", "gt").strip().lower()
RUN_ID = os.environ.get("ALAB_RUN_ID", "unknown")
RESULTS_DIR = os.environ.get("ALAB_RESULTS_DIR", "results/runs")
SAMPLE_LOG_RATE = float(os.environ.get("ALAB_SAMPLE_LOG_RATE", "0.01"))
REWARD_SEED = int(os.environ.get("ALAB_REWARD_SEED", "42"))
FORMAT_BONUS = float(os.environ.get("ALAB_FORMAT_BONUS", "0.1"))

assert REWARD_MODE in ("gt", "random", "format"), f"bad ALAB_REWARD_MODE={REWARD_MODE!r}"

SAMPLES_PATH = os.path.join(RESULTS_DIR, RUN_ID, "samples.jsonl")

# math-verify is required for `gt`; import lazily so `random`/`format` never need it.
_MV = None
if REWARD_MODE == "gt":
    try:
        from math_verify import parse as _mv_parse, verify as _mv_verify

        _MV = (_mv_parse, _mv_verify)
    except Exception as e:  # pragma: no cover - env issue surfaced at runtime
        print(f"[reward] WARNING: math_verify import failed ({e}); falling back to openrlhf grade_answer")
        try:
            from openrlhf.utils import grade_answer as _grade_answer  # sympy-based fallback
        except Exception:
            _grade_answer = None

_BOXED_RE = re.compile(r"\\boxed\s*\{")


# ---------------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------------
def _response_of(query: str, prompt) -> str:
    """Strip the prompt prefix from the full decoded query to isolate the response."""
    if isinstance(prompt, str) and prompt and prompt in query:
        return query[query.index(prompt) + len(prompt):]
    return query


def _has_boxed(text: str) -> bool:
    """True iff the text contains a balanced-looking \\boxed{...} we can parse."""
    if not _BOXED_RE.search(text):
        return False
    return _extract_boxed(text) is not None


def _extract_boxed(text: str):
    """Return the content of the LAST \\boxed{...} with brace matching, else None."""
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    i = text.find("{", idx)
    if i == -1:
        return None
    depth = 0
    for j in range(i, len(text)):
        c = text[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None


def _is_correct(response: str, label: str) -> bool:
    """math-verify equivalence between the model's boxed answer and the gold label."""
    if label is None:
        return False
    gold = str(label)
    if _MV is not None:
        parse, verify = _MV
        try:
            # Parse gold; wrap bare gold in \boxed so the parser treats it as an answer.
            gold_parsed = parse(gold if "\\boxed" in gold or "$" in gold else f"${gold}$")
            pred_parsed = parse(response)
            if not gold_parsed or not pred_parsed:
                return False
            # math_verify.verify(gold, target) — order matters; try both directions.
            return bool(verify(gold_parsed, pred_parsed)) or bool(verify(pred_parsed, gold_parsed))
        except Exception:
            return False
    # Fallback: sympy-based grader on the extracted boxed answer.
    boxed = _extract_boxed(response) or response
    if "_grade_answer" in globals() and _grade_answer is not None:
        try:
            return bool(_grade_answer(boxed, gold))
        except Exception:
            return False
    return boxed.strip() == gold.strip()


def _random_coin(query: str) -> float:
    """Content-independent Bernoulli(0.5), keyed on (seed, query) so it's stable and
    reproducible across distributed workers regardless of call order."""
    h = hashlib.sha256(f"{REWARD_SEED}:{query}".encode("utf-8")).digest()
    # First 8 bytes -> uniform in [0,1); the mapping is uncorrelated with correctness.
    u = int.from_bytes(h[:8], "big") / float(1 << 64)
    return 1.0 if u < 0.5 else 0.0


# Per-process sampler for the 1% log (no shared RNG needed across workers).
import random as _random  # noqa: E402

_sampler = _random.Random(REWARD_SEED ^ (os.getpid() * 2654435761 & 0xFFFFFFFF))


def _maybe_log(prompt, response: str, label, reward: float, correct):
    if SAMPLE_LOG_RATE <= 0 or _sampler.random() >= SAMPLE_LOG_RATE:
        return
    rec = {
        "run_id": RUN_ID,
        "mode": REWARD_MODE,
        "ts": time.time(),
        "prompt": (prompt if isinstance(prompt, str) else str(prompt))[:4000],
        "response": response[:4000],
        "label": None if label is None else str(label)[:512],
        "reward": reward,
        "correct": correct,
    }
    line = json.dumps(rec, ensure_ascii=False) + "\n"
    try:
        os.makedirs(os.path.dirname(SAMPLES_PATH), exist_ok=True)
        import fcntl

        with open(SAMPLES_PATH, "a", encoding="utf-8") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.write(line)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception as e:  # logging must never crash training
        print(f"[reward] sample-log write failed: {e}")


# ---------------------------------------------------------------------------------
# Entry point called by OpenRLHF.
# ---------------------------------------------------------------------------------
def reward_func(queries: List[str], prompts: List[str], labels: List[str], **kwargs) -> dict:
    rewards: List[float] = []
    n_correct = 0
    n_boxed = 0

    for i, query in enumerate(queries):
        prompt = prompts[i] if i < len(prompts) else ""
        label = labels[i] if labels is not None and i < len(labels) else None
        response = _response_of(query, prompt)
        has_boxed = _has_boxed(response)
        n_boxed += int(has_boxed)

        correct = None
        if REWARD_MODE == "gt":
            correct = _is_correct(response, label)
            r = (1.0 if correct else 0.0) + (FORMAT_BONUS if has_boxed else 0.0)
            n_correct += int(correct)
        elif REWARD_MODE == "format":
            r = 1.0 if has_boxed else 0.0
        else:  # random
            r = _random_coin(query)

        rewards.append(r)
        _maybe_log(prompt, response, label, r, correct)

    rewards_tensor = torch.tensor(rewards, dtype=torch.float)
    n = max(1, len(rewards))
    extra_logs = {
        "reward_mean": rewards_tensor.mean(),
        "boxed_rate": torch.tensor(n_boxed / n, dtype=torch.float),
    }
    if REWARD_MODE == "gt":
        extra_logs["gt_accuracy"] = torch.tensor(n_correct / n, dtype=torch.float)

    return {"rewards": rewards_tensor, "scores": rewards_tensor, "extra_logs": extra_logs}
