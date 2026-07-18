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
import threading
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
_grade_answer = None


def _patch_math_verify_timeout() -> None:
    """Disable math-verify's signal.alarm timeouts (required for OpenRLHF worker threads).

    math-verify 0.7 wraps parse/verify in ``utils.timeout``, which calls
    ``signal.alarm``. That only works on the main thread. OpenRLHF runs
    ``reward_func`` in a worker thread, so every parse raised and — in older
    math-verify — was swallowed into False. Passing ``parsing_timeout=None``
    does **not** disable the wrapper in 0.7: it still calls ``signal.alarm(None)``
    (TypeError), which parse catches internally and returns ``[]``. Result:
    gt_accuracy stuck at 0 while format bonus still pays (soft format arm).

    Patch the timeout factory to a no-op in utils + the parser/grader aliases
    that already bound the name at import time.
    """
    import math_verify.grader as mv_grader
    import math_verify.parser as mv_parser
    import math_verify.utils as mv_utils

    if getattr(mv_utils, "_alab_timeout_disabled", False):
        return

    def _noop_timeout(timeout_seconds: int = 10):  # noqa: ARG001
        def decorator(func):
            return func

        return decorator

    mv_utils.timeout = _noop_timeout
    mv_parser.timeout = _noop_timeout
    mv_grader.timeout = _noop_timeout
    mv_utils._alab_timeout_disabled = True


if REWARD_MODE == "gt":
    try:
        _patch_math_verify_timeout()
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


# Parse-failure telemetry (logging only — does NOT change reward values).
# OpenRLHF once swallowed every math_verify thread error into False; this makes
# silent zeros visible via train/parse_fail_rate without altering scoring.
# empty_parse_rate covers the math-verify-0.7 path where parse catches timeout
# errors internally and returns [] (never raises into our except).
_PARSE_FAIL_LOCK = threading.Lock()
_PARSE_FAIL_COUNT = 0
_EMPTY_PARSE_COUNT = 0
_PARSE_FAIL_LOG_EVERY = int(os.environ.get("ALAB_PARSE_FAIL_LOG_EVERY", "50"))


def _record_parse_fail(exc: BaseException) -> None:
    """Count + rate-limited log of verifier exceptions. Reward path still returns False."""
    global _PARSE_FAIL_COUNT
    with _PARSE_FAIL_LOCK:
        _PARSE_FAIL_COUNT += 1
        n = _PARSE_FAIL_COUNT
    if n == 1 or n % _PARSE_FAIL_LOG_EVERY == 0:
        print(
            f"[reward] parse_fail #{n}: {type(exc).__name__}: {exc}",
            flush=True,
        )


def _record_empty_parse() -> None:
    """Count silent empty parse results (math-verify returned [] without raising)."""
    global _EMPTY_PARSE_COUNT
    with _PARSE_FAIL_LOCK:
        _EMPTY_PARSE_COUNT += 1
        n = _EMPTY_PARSE_COUNT
    if n == 1 or n % _PARSE_FAIL_LOG_EVERY == 0:
        print(f"[reward] empty_parse #{n}: math_verify.parse returned []", flush=True)


def _is_correct(response: str, label: str) -> bool:
    """math-verify equivalence between the model's boxed answer and the gold label."""
    if label is None:
        return False
    gold = str(label)
    if _MV is not None:
        parse, verify = _MV
        try:
            # Timeouts are disabled via _patch_math_verify_timeout(); pass a
            # positive int so math-verify never hits signal.alarm(None).
            gold_parsed = parse(
                gold if "\\boxed" in gold or "$" in gold else f"${gold}$",
                parsing_timeout=5,
            )
            pred_parsed = parse(response, parsing_timeout=5)
            if not gold_parsed or not pred_parsed:
                _record_empty_parse()
                return False
            # math_verify.verify(gold, target) — order matters; try both directions.
            return bool(verify(gold_parsed, pred_parsed, timeout_seconds=5)) or bool(
                verify(pred_parsed, gold_parsed, timeout_seconds=5)
            )
        except Exception as e:
            _record_parse_fail(e)
            return False
    # Fallback: sympy-based grader on the extracted boxed answer.
    boxed = _extract_boxed(response) or response
    if _grade_answer is not None:
        try:
            return bool(_grade_answer(boxed, gold))
        except Exception as e:
            _record_parse_fail(e)
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
    fails_before = _PARSE_FAIL_COUNT
    empty_before = _EMPTY_PARSE_COUNT

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
    batch_fails = max(0, _PARSE_FAIL_COUNT - fails_before)
    batch_empty = max(0, _EMPTY_PARSE_COUNT - empty_before)
    extra_logs = {
        "reward_mean": rewards_tensor.mean(),
        "boxed_rate": torch.tensor(n_boxed / n, dtype=torch.float),
        # Logging only — reward tensors above are unchanged by this counter.
        "parse_fail_rate": torch.tensor(batch_fails / n, dtype=torch.float),
        "parse_fail_count": torch.tensor(float(batch_fails), dtype=torch.float),
        "empty_parse_rate": torch.tensor(batch_empty / n, dtype=torch.float),
        "empty_parse_count": torch.tensor(float(batch_empty), dtype=torch.float),
    }
    if REWARD_MODE == "gt":
        extra_logs["gt_accuracy"] = torch.tensor(n_correct / n, dtype=torch.float)

    return {"rewards": rewards_tensor, "scores": rewards_tensor, "extra_logs": extra_logs}
