# RQ1 — Interim findings (Qwen3-8B control battery)

**Status: 2 of 3 arms evaluated (format, random). gt still training — see caveat.**
Control battery: three GRPO arms differ *only* in the reward; identical data, steps,
KL, group size. Init from `Arushhh/alab-q3-8b_sft_tulu_0705` (SFT on tulu-3-mix).

## Training data volume
| stage | dataset | size | examples |
|---|---|---|---|
| SFT (init) | tulu-3-sft-mixture → `sft.jsonl` | 420 MB | 148,184 instructions |
| GRPO (all arms) | DeepScaleR + GSM8K-train → `rlvr_math.jsonl` | 14 MB | 45,467 math prompts |

~0.43 GB unique training text. Each GRPO arm does ~1 epoch (355 steps × 128 prompts)
and generates 8 responses/prompt ≈ 364k rollouts — that is where the compute lives.

## Eval accuracy (lm-eval exact_match)
| model | gsm8k | math500 | gsm_plus |
|---|---|---|---|
| base Qwen3-8B | 0.916 | 0.688 | 0.752 |
| **SFT init** (RL start point) | 0.861 | 0.430 | 0.701 |
| format arm (done) | **0.498** | **0.248** | **0.385** |
| random arm (done) | **0.563** | **0.252** | **0.455** |
| gt arm | _training (6-GPU non-colocated)_ | | |

Both correctness-blind controls **dropped sharply** from the SFT start (gsm8k 0.86 → 0.50–0.56;
math500 0.43 → 0.25). The base→SFT drop (0.92→0.86 gsm8k, 0.69→0.43 math500) is the tulu
SFT trading raw math for general instruction-following — expected and separate.

## Is this catastrophic forgetting? — No (collapse) / Yes (behavioral erosion)
KL stayed **tiny** the whole run — the policy never diverged far from the SFT model
(contrast the earlier k2/0.001 bug that let KL run to 0.43):

| arm | KL (start→mid→end) | reward_mean | resp_len (start→end) |
|---|---|---|---|
| random | 0.0004 → 0.0068 → 0.005 | ~0.5 (Bernoulli noise) | 904 → 725 |
| format | 0.0004 → 0.0020 → 0.0016 | 0.96 → 1.0 (saturated) | 811 → 832 |

So no knowledge wipe — but measured accuracy still fell, because a small *per-token* KL,
pushed consistently over long responses for 355 steps, still shifts *generation behavior*
enough to break task success.

## Direct sample evidence (AIME24, 7,680 generations/model)
| model | mean len | % with `\boxed` | % correct | % run-on / no answer |
|---|---|---|---|---|
| SFT baseline | 3,221 | 90.4% | 1.9% | 8.0% |
| format arm | 3,922 | 74.4% | 0.2% | 24.3% |
| random arm | 3,331 | 56.3% | 0.2% | 26.4% |

- **Answer-completion collapses**: controls reach a parseable `\boxed{}` far less often
  (90% → 74% → 56%); "runs on without concluding" **triples** (8% → ~25%).
- **Genuine degeneracy**, e.g. the format arm emitted
  `\boxed{{15/(89))}}` then looped `"efficient fashion securing results through efficiency
  fashion securing…"`; the random arm trails into `"please provide additional context"`.
  The SFT baseline reasons and closes cleanly (`\boxed{143}`).

**Verdict:** the correctness-blind RL did not erase math knowledge (KL-anchored, still
math-shaped) but caused real behavioral erosion — repetition, non-termination, and
failure to produce an extractable answer — which is exactly the control-arm signal that
a bad/no reward degrades the model. Notably the format arm, *rewarded* to emit `\boxed`,
produces it *less* on out-of-distribution AIME: the hack didn't generalize, the degradation did.

## Caveat — central RQ1 claim is still pending
The point of the battery is the **contrast with gt** (real correctness reward). gt is still
training (its KL is likewise tiny ~0.0005, gt_accuracy ~0.29–0.34). Whether correctness reward
*preserves/improves* eval accuracy vs. the degrading controls is the open question until gt
finishes and is evaluated.
