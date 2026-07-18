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
    """Regression: math_verify signal.alarm must not fire in a worker thread.

    math-verify 0.7's parsing_timeout=None does NOT disable alarms — it makes
    parse return []. The reward module must patch timeout to a no-op so gt
    scoring works under OpenRLHF's remote-reward worker threads.
    """
    r = _load_reward("gt")
    assert r._MV is not None, "math-verify required for this regression test"
    result = {}
    err = {}

    def worker():
        try:
            result["ok"] = r._is_correct(r"The answer is \boxed{3}", "3")
            result["bad"] = r._is_correct(r"The answer is \boxed{9}", "3")
            out = r.reward_func(
                queries=[r"p\n\boxed{42}", r"p\n\boxed{0}"],
                prompts=["p\n", "p\n"],
                labels=["42", "42"],
            )
            result["rewards"] = out["rewards"].tolist()
            result["gt_acc"] = float(out["extra_logs"]["gt_accuracy"])
            result["empty"] = float(out["extra_logs"]["empty_parse_rate"])
        except Exception as e:  # noqa: BLE001 — surface to main thread
            err["exc"] = e

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=30)
    assert not t.is_alive(), "worker thread hung"
    assert not err, f"worker raised: {err.get('exc')}"
    assert result["ok"] is True
    assert result["bad"] is False
    assert result["rewards"][0] >= 1.0
    assert result["rewards"][1] < 1.0
    assert result["gt_acc"] == 0.5
    assert result["empty"] == 0.0


def test_parse_fail_logging_does_not_change_reward_shape():
    r = _load_reward("gt")
    out = r.reward_func(
        queries=["p\nx"],
        prompts=["p\n"],
        labels=["1"],
    )
    assert out["rewards"].shape == (1,)
    assert float(out["extra_logs"]["parse_fail_rate"]) >= 0.0
    assert "empty_parse_rate" in out["extra_logs"]


if __name__ == "__main__":
    # Minimal runner without pytest.
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"RUN {name}...")
            fn()
            print(f"  OK {name}")
    print("all passed")
