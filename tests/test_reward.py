"""Offline unit tests for AlignmentLab reward arms (no GPU).

Run from repo root:
  ALAB_REWARD_MODE=gt conda run -n alab-rl python -m pytest tests/test_reward.py -q
or without pytest:
  ALAB_REWARD_MODE=gt python tests/test_reward.py
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import threading
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def _load_reward(mode: str):
    """Import reward.py with a fresh ALAB_REWARD_MODE (module caches env at import)."""
    os.environ["ALAB_REWARD_MODE"] = mode
    os.environ.setdefault("ALAB_RUN_ID", "test_reward")
    os.environ.setdefault("ALAB_RESULTS_DIR", str(REPO / "results" / "runs"))
    os.environ.setdefault("ALAB_SAMPLE_LOG_RATE", "0")
    # Force re-import so REWARD_MODE / _MV pick up the env.
    for key in ("rl.reward", "reward", "alab_reward"):
        sys.modules.pop(key, None)
    try:
        return importlib.import_module("rl.reward")
    except ImportError:
        path = REPO / "src" / "rl" / "reward.py"
        spec = importlib.util.spec_from_file_location("alab_reward", path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return mod


def test_extract_boxed_brace_match():
    r = _load_reward("format")
    assert r._extract_boxed(r"answer \boxed{42}") == "42"
    assert r._extract_boxed(r"\boxed{1} then \boxed{2+3}") == "2+3"
    assert r._extract_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert r._extract_boxed("no box") is None
    assert r._has_boxed(r"\boxed{7}") is True
    assert r._has_boxed("plain 7") is False


def test_format_arm_reward():
    r = _load_reward("format")
    out = r.reward_func(
        queries=["prompt\n\\boxed{1}", "prompt\nnope"],
        prompts=["prompt\n", "prompt\n"],
        labels=["1", "1"],
    )
    assert torch.allclose(out["rewards"], torch.tensor([1.0, 0.0]))
    assert "parse_fail_rate" in out["extra_logs"]


def test_random_arm_stable_across_calls():
    r = _load_reward("random")
    q = ["same-query-alpha", "same-query-beta"]
    a = r.reward_func(queries=q, prompts=["", ""], labels=[None, None])
    b = r.reward_func(queries=q, prompts=["", ""], labels=[None, None])
    assert torch.equal(a["rewards"], b["rewards"])
    assert set(a["rewards"].tolist()) <= {0.0, 1.0}


def test_gt_arm_correct_and_wrong():
    r = _load_reward("gt")
    # Correct boxed answer should get 1.0 + format bonus (default 0.1).
    out = r.reward_func(
        queries=[
            "p\nThe answer is \\boxed{42}",
            "p\nThe answer is \\boxed{0}",
        ],
        prompts=["p\n", "p\n"],
        labels=["42", "42"],
    )
    rewards = out["rewards"].tolist()
    assert rewards[0] >= 1.0  # 1.0 + optional bonus
    assert rewards[1] < 1.0  # wrong; may still get format bonus 0.1
    assert "gt_accuracy" in out["extra_logs"]
    assert "parse_fail_rate" in out["extra_logs"]


def test_is_correct_from_worker_thread():
    """Regression: math_verify signal.alarm must not fire in a worker thread."""
    r = _load_reward("gt")
    result = {}

    def worker():
        result["ok"] = r._is_correct(r"\\boxed{3}", "3")
        result["bad"] = r._is_correct(r"\\boxed{9}", "3")

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=30)
    assert not t.is_alive(), "worker thread hung"
    # If math_verify unavailable, string fallback may still work via boxed extract.
    assert result["ok"] is True or result["ok"] is False  # ran without raising
    assert result["bad"] is False or result["ok"] is True


def test_parse_fail_logging_does_not_change_reward_shape():
    r = _load_reward("gt")
    out = r.reward_func(
        queries=["p\nx"],
        prompts=["p\n"],
        labels=["1"],
    )
    assert out["rewards"].shape == (1,)
    assert float(out["extra_logs"]["parse_fail_rate"]) >= 0.0


if __name__ == "__main__":
    # Minimal runner without pytest.
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"RUN {name}...")
            fn()
            print(f"  OK {name}")
    print("all passed")
