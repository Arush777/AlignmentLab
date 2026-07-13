#!/usr/bin/env python3
"""Collect per-run training metrics and eval outputs into results/summary.csv.

Scans:
  results/runs/<run_id>/metrics.json
  results/evals/<run_id>/lm_eval.json   (raw lm-evaluation-harness output)
  results/evals/<run_id>/passk.json     (list of {task, temperature, n_samples,
                                          k, pass_at_k, stderr})

Emits results/summary.csv with one row per run_id and these columns:
  run_id, model, method, reward_mode, gpu_hours,
  gsm8k, math500, gsm_plus,
  aime24_pass@1, aime24_pass@8, aime24_pass@64, aime24_pass@256

Blank cells wherever a metric is missing. Pure stdlib (csv, json, pathlib) --
no third-party dependencies. `model` / `method` / `reward_mode` are parsed from
the run_id per the frozen naming contract {model}_{method}_{data}_{MMDD}
(PLAN.md, Interface contracts), and overridden by metrics.json when present.

Run:  python scripts/collect_results.py
"""
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "results"
RUNS_DIR = RESULTS_DIR / "runs"
EVALS_DIR = RESULTS_DIR / "evals"
SUMMARY_CSV = RESULTS_DIR / "summary.csv"

COLUMNS = [
    "run_id", "model", "method", "reward_mode", "gpu_hours",
    "gsm8k", "math500", "gsm_plus",
    "aime24_pass@1", "aime24_pass@8", "aime24_pass@64", "aime24_pass@256",
]

MODEL_MAP = {
    "q3-0.6b": "Qwen/Qwen3-0.6B",
    "q3-8b":   "Qwen/Qwen3-8B",
    "q3-14b":  "Qwen/Qwen3-14B",
    "q3-32b":  "Qwen/Qwen3-32B",
    "llama31-8b":  "meta-llama/Llama-3.1-8B",
}

REWARD_FROM_METHOD = {
    "grpo-gt":   "gt",
    "grpo-rand": "random",
    "grpo-fmt":  "format",
}

# Preferred lm-eval-harness headline metrics, in priority order.
LM_EVAL_PRIMARY_KEYS = [
    "exact_match,strict-match",
    "exact_match,none",
    "exact_match,flex-extract",
    "acc,none",
    "acc_norm,none",
]

# For each output column: (list of candidate substring-patterns to try in order,
# substrings that disqualify a task key). A pattern [a, b] matches a task key iff
# both a and b occur in the lowercased key.
TASK_CANDIDATES = {
    "gsm8k":    ([["gsm8k"], ["gsm_8k"]], ("plus",)),
    "math500":  ([["math_500"], ["math500"], ["math", "500"],
                  ["minerva_math"], ["hendrycks_math"]], ()),
    "gsm_plus": ([["gsm_plus"], ["gsm-plus"], ["gsm8k_plus"],
                  ["gsm8k-plus"], ["gsm", "plus"]], ()),
}

PASSK_KS = [1, 8, 64, 256]


def warn(msg):
    print(f"[collect_results] WARN: {msg}", file=sys.stderr)


def read_json(path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError) as e:
        warn(f"could not read {path}: {e}")
        return None


def parse_run_id(run_id):
    """{model}_{method}_{data}_{MMDD} -> (model_full, method, reward_mode)."""
    parts = run_id.split("_")
    if len(parts) < 4:
        return ("", "", "")
    model_short, method = parts[0], parts[1]
    model_full = MODEL_MAP.get(model_short, model_short)
    reward_mode = REWARD_FROM_METHOD.get(method, "")
    return (model_full, method, reward_mode)


def _primary_metric(metrics_obj):
    """From a {metric_key: value} dict (or a bare number), pick headline acc."""
    if isinstance(metrics_obj, (int, float)):
        return float(metrics_obj)
    if not isinstance(metrics_obj, dict):
        return None
    for pref in LM_EVAL_PRIMARY_KEYS:
        v = metrics_obj.get(pref)
        if isinstance(v, (int, float)):
            return float(v)
    for k, v in metrics_obj.items():
        kl = k.lower()
        if any(s in kl for s in ("stderr", "alias", "samples", "_n", "total")):
            continue
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _find_task(results_obj, candidates, exclude):
    for patterns in candidates:
        for task_key, task_val in results_obj.items():
            kl = str(task_key).lower()
            if any(x in kl for x in exclude):
                continue
            if all(p in kl for p in patterns):
                return task_val
    return None


def extract_lm_eval(blob):
    """Return {gsm8k, math500, gsm_plus} from raw lm-eval-harness output."""
    out = {"gsm8k": "", "math500": "", "gsm_plus": ""}
    if not isinstance(blob, dict):
        return out
    inner = blob["results"] if isinstance(blob.get("results"), dict) else blob
    if not isinstance(inner, dict):
        return out
    for col, (candidates, exclude) in TASK_CANDIDATES.items():
        val = _primary_metric(_find_task(inner, candidates, exclude))
        if val is not None:
            out[col] = round(val, 6)
    return out


def extract_passk(blob):
    """Return {aime24_pass@k} from a passk.json list (unbiased estimator)."""
    out = {f"aime24_pass@{k}": "" for k in PASSK_KS}
    if not isinstance(blob, list):
        return out
    for entry in blob:
        if not isinstance(entry, dict):
            continue
        task = str(entry.get("task", "")).lower()
        if "aime" not in task or "24" not in task:
            continue
        try:
            k = int(entry.get("k"))
        except (TypeError, ValueError):
            continue
        if k in PASSK_KS and "pass_at_k" in entry:
            try:
                out[f"aime24_pass@{k}"] = round(float(entry["pass_at_k"]), 6)
            except (TypeError, ValueError):
                pass
    return out


def _blank_row(run_id):
    row = {c: "" for c in COLUMNS}
    row["run_id"] = run_id
    model_full, method, reward_mode = parse_run_id(run_id)
    row["model"], row["method"], row["reward_mode"] = model_full, method, reward_mode
    return row


def collect():
    rows = {}

    # 1. training-run metrics (primary key source).
    for mpath in sorted(RUNS_DIR.glob("*/metrics.json")):
        run_id = mpath.parent.name
        row = _blank_row(run_id)
        blob = read_json(mpath)
        if isinstance(blob, dict):
            if blob.get("model"):
                row["model"] = str(blob["model"])
            if blob.get("method"):
                row["method"] = str(blob["method"])
            if blob.get("reward_mode"):
                row["reward_mode"] = str(blob["reward_mode"])
            gh = blob.get("gpu_hours")
            if gh is not None:
                try:
                    row["gpu_hours"] = round(float(gh), 4)
                except (TypeError, ValueError):
                    row["gpu_hours"] = ""
        rows[run_id] = row

    # 2. eval outputs (merge into existing rows, else create a new row).
    edirs = sorted(EVALS_DIR.iterdir()) if EVALS_DIR.exists() else []
    for edir in edirs:
        if not edir.is_dir():
            continue
        run_id = edir.name
        row = rows.get(run_id) or _blank_row(run_id)
        rows[run_id] = row
        lm = read_json(edir / "lm_eval.json")
        if lm is not None:
            row.update(extract_lm_eval(lm))
        pk = read_json(edir / "passk.json")
        if pk is not None:
            row.update(extract_passk(pk))

    return rows


def _fmt(v):
    if v == "" or v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def print_table(rows):
    short_model = {v: k for k, v in MODEL_MAP.items()}
    cols = ["run_id", "model", "method", "reward_mode", "gpu_hours",
            "gsm8k", "math500", "gsm_plus", "aime24_pass@1"]
    table = []
    widths = {c: len(c) for c in cols}
    for run_id in sorted(rows):
        r = rows[run_id]
        line = {
            "run_id": r["run_id"],
            "model": short_model.get(r["model"], r["model"]),
            "method": r["method"],
            "reward_mode": r["reward_mode"],
            "gpu_hours": _fmt(r["gpu_hours"]),
            "gsm8k": _fmt(r["gsm8k"]),
            "math500": _fmt(r["math500"]),
            "gsm_plus": _fmt(r["gsm_plus"]),
            "aime24_pass@1": _fmt(r["aime24_pass@1"]),
        }
        for c in cols:
            widths[c] = max(widths[c], len(str(line[c])))
        table.append(line)
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for line in table:
        print("  ".join(str(line[c]).ljust(widths[c]) for c in cols))


def main():
    if not RUNS_DIR.exists() and not EVALS_DIR.exists():
        warn(f"no results dirs under {RESULTS_DIR}")
    rows = collect()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for run_id in sorted(rows):
            w.writerow(rows[run_id])
    print(f"[collect_results] wrote {SUMMARY_CSV} ({len(rows)} rows)\n")
    print_table(rows)


if __name__ == "__main__":
    main()
