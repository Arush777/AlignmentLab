#!/usr/bin/env python3
"""Render the RQ1 / RQ2 figures from results/summary.csv.

  docs/figs/rq1_arms.png    -- gsm8k / math500 per reward arm, grouped by model
  docs/figs/rq2_perturb.png -- gsm8k vs gsm-plus (with drop) per reward arm

Reads results/summary.csv (produced by scripts/collect_results.py). matplotlib
+ stdlib only; no pandas/numpy/seaborn. Colorblind-safe Okabe-Ito palette,
value labels on every bar, 150 dpi output. No CLI args.

Run:  python scripts/plot_arms.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
SUMMARY_CSV = REPO / "results" / "summary.csv"
FIGS_DIR = REPO / "docs" / "figs"

ARMS = ["gt", "random", "format"]

# Okabe-Ito colorblind-safe palette.
COLOR_GSM8K = "#0072B2"     # blue
COLOR_MATH500 = "#E69F00"   # orange
COLOR_GSM_PLUS = "#D55E00"  # vermillion
TITLE_COLOR = "#222222"

MODEL_ORDER = ["Qwen/Qwen3-8B", "meta-llama/Llama-3.1-8B"]
MODEL_LABEL = {
    "Qwen/Qwen3-0.6B": "Qwen3-0.6B",
    "Qwen/Qwen3-8B": "Qwen3-8B",
    "Qwen/Qwen3-14B": "Qwen3-14B",
    "Qwen/Qwen3-32B": "Qwen3-32B",
    "meta-llama/Llama-3.1-8B": "Llama-3.1-8B",
    "meta-llama/Llama-3.1-14B": "Llama-3.1-14B",
}


def load_rows():
    if not SUMMARY_CSV.exists():
        return []
    with SUMMARY_CSV.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _is_pct(vals):
    """If any value > 1.5, assume already on a 0-100 scale; else 0-1 fraction."""
    return any(v is not None and v > 1.5 for v in vals)


def _scale(v, already_pct):
    if v is None:
        return None
    return v if already_pct else v * 100.0


def _arm_rows(rows, model, arm):
    out = []
    for r in rows:
        if r.get("model") != model:
            continue
        if r.get("reward_mode", "").strip() != arm:
            continue
        out.append(r)
    return out


def _best(rows, col):
    """Mean of non-empty values for a column across rows (None if no data)."""
    vals = [_to_float(r.get(col)) for r in rows]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _label_real(ax, bars, raw):
    """Put a value label only on bars whose source value is not None."""
    for b, v in zip(bars, raw):
        if v is None:
            continue
        ax.annotate(f"{v:.1f}",
                    xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)


def _no_data_text(ax):
    ax.text(0.5, 0.5, "no eval data yet", transform=ax.transAxes,
            ha="center", va="center", fontsize=11, color="0.5")


def plot_rq1(rows):
    raw_all = []
    for model in MODEL_ORDER:
        for arm in ARMS:
            r = _arm_rows(rows, model, arm)
            raw_all += [_best(r, "gsm8k"), _best(r, "math500")]
    pct = _is_pct(raw_all)

    fig, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(11, 5), sharey=True)
    if len(MODEL_ORDER) == 1:
        axes = [axes]

    x = list(range(len(ARMS)))
    width = 0.38
    global_max, any_data = 0.0, False
    for ax, model in zip(axes, MODEL_ORDER):
        gsm_raw, math_raw = [], []
        for arm in ARMS:
            r = _arm_rows(rows, model, arm)
            g = _best(r, "gsm8k")
            m = _best(r, "math500")
            gsm_raw.append(_scale(g, pct))
            math_raw.append(_scale(m, pct))
            if g is not None or m is not None:
                any_data = True
        gsm_h = [v if v is not None else 0.0 for v in gsm_raw]
        math_h = [v if v is not None else 0.0 for v in math_raw]
        global_max = max(global_max, max(gsm_h + math_h)) if gsm_h else global_max
        b1 = ax.bar([i - width / 2 for i in x], gsm_h, width,
                    color=COLOR_GSM8K, label="GSM8K")
        b2 = ax.bar([i + width / 2 for i in x], math_h, width,
                    color=COLOR_MATH500, label="MATH-500")
        _label_real(ax, b1, gsm_raw)
        _label_real(ax, b2, math_raw)
        ax.set_title(MODEL_LABEL.get(model, model), fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(ARMS)
        ax.set_ylabel("Accuracy (%)" if not pct else "Accuracy")
        ax.grid(axis="y", alpha=0.3)
        if not any(gsm_raw) and not any(math_raw):
            _no_data_text(ax)

    ymax = (max(global_max, 1.0) * 1.18) if any_data else 100.0
    axes[0].set_ylim(0, ymax)
    axes[-1].legend(loc="upper right", frameon=False)
    fig.suptitle("RQ1: do gains survive the placebo arms?",
                 fontsize=14, fontweight="bold", color=TITLE_COLOR)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGS_DIR / "rq1_arms.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot_arms] wrote {out}")


def plot_rq2(rows):
    raw_all = []
    for model in MODEL_ORDER:
        for arm in ARMS:
            r = _arm_rows(rows, model, arm)
            raw_all += [_best(r, "gsm8k"), _best(r, "gsm_plus")]
    pct = _is_pct(raw_all)

    fig, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(11, 5), sharey=True)
    if len(MODEL_ORDER) == 1:
        axes = [axes]

    x = list(range(len(ARMS)))
    width = 0.38
    global_max, any_data = 0.0, False
    for ax, model in zip(axes, MODEL_ORDER):
        gsm_raw, plus_raw, drops = [], [], []
        for arm in ARMS:
            r = _arm_rows(rows, model, arm)
            g = _best(r, "gsm8k")
            p = _best(r, "gsm_plus")
            gsm_raw.append(_scale(g, pct))
            plus_raw.append(_scale(p, pct))
            if g is not None and p is not None:
                drops.append(g - p)  # raw-scale drop (matches g/p units)
                any_data = True
            else:
                drops.append(None)
        gsm_h = [v if v is not None else 0.0 for v in gsm_raw]
        plus_h = [v if v is not None else 0.0 for v in plus_raw]
        global_max = max(global_max, max(gsm_h + plus_h)) if gsm_h else global_max
        b1 = ax.bar([i - width / 2 for i in x], gsm_h, width,
                    color=COLOR_GSM8K, label="GSM8K")
        b2 = ax.bar([i + width / 2 for i in x], plus_h, width,
                    color=COLOR_GSM_PLUS, label="GSM-Plus")
        _label_real(ax, b1, gsm_raw)
        _label_real(ax, b2, plus_raw)
        for i, d in enumerate(drops):
            if d is None:
                continue
            dshow = d if pct else d * 100.0
            top = max(gsm_h[i], plus_h[i])
            ax.annotate(f"\u0394 {dshow:+.1f}", xy=(i, top), xytext=(0, 12),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=8, color="#444444")
        ax.set_title(MODEL_LABEL.get(model, model), fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(ARMS)
        ax.set_ylabel("Accuracy (%)" if not pct else "Accuracy")
        ax.grid(axis="y", alpha=0.3)
        if not any(gsm_raw) and not any(plus_raw):
            _no_data_text(ax)

    ymax = (max(global_max, 1.0) * 1.22) if any_data else 100.0
    axes[0].set_ylim(0, ymax)
    axes[-1].legend(loc="upper right", frameon=False)
    fig.suptitle("RQ2: perturbation robustness -- GSM8K vs GSM-Plus drop per arm",
                 fontsize=14, fontweight="bold", color=TITLE_COLOR)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGS_DIR / "rq2_perturb.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot_arms] wrote {out}")


def main():
    rows = load_rows()
    if not rows:
        print("[plot_arms] summary.csv missing/empty; writing placeholder figures.")
    plot_rq1(rows)
    plot_rq2(rows)


if __name__ == "__main__":
    main()
