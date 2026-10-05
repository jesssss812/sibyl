# Sibyl — Backtest & Evaluation Strategy

> **For coding agents:** This is the authoritative spec for how Sibyl is evaluated. Any change that touches scoring, backtesting, experiment comparison, or result reporting must comply with the Rules below. Do not introduce alternative evaluation paths. If a request conflicts with a Rule, say so rather than silently working around it.

---

## Core stance

**Take the methodology, not the infrastructure.**

FutureSearch's BTF-3 is the gold-standard backtest: 1,907 resolved questions, each with a frozen crawl served through a private search engine with server-side date fencing, ~$1/forecast → **~$2,000 per experimental run**. We are not building that. It's a funded team's infrastructure.

What we *are* taking is free: their **statistical method** (paired comparisons, bootstrap CIs, cohort slicing, power analysis up front). That's a few hundred lines of numpy and it's where the differentiated value is.

The evaluation is not a supporting tool for this project. **It is the deliverable.** The bot exists to be evaluated.

---

## Non-goals — do not build these

- ❌ A frozen-corpus / date-fenced search environment (RetroSearch equivalent)
- ❌ An automated question generation + resolution pipeline
- ❌ A large-scale pastcasting corpus of our own
- ❌ Any leaderboard-rank-chasing feature that isn't independently justified by the eval
- ❌ Real-money market integration of any kind (see `Money` below)

---

## The Rules

### Rule 1 — Every comparison is paired. No exceptions.

Never compare two variants by their standalone scores. Run both on the **same question set**, compute the **per-question difference**, bootstrap over questions.

> A confidence interval on a single system's Brier says more about how hard the question set is than about the system.

Question difficulty is a nuisance variable; pairing cancels it. This is blocking / a paired test rather than a two-sample test.

**Report format:** `variant B beats A by 0.014 pooled score, 95% CI [0.006, 0.022], n=104 paired questions` — never `variant B scored 0.132`.

### Rule 2 — Power-check before you run, not after.

Compute the minimum detectable effect for your sample size *first*. If the change you're testing is plausibly smaller than the MDE, **don't run the experiment** — log the decision and move on.

Calibrated from BTF-3 (±0.003 at n=1,810, scaling as 1/√n):

| n (paired questions) | ~95% CI half-width | What's detectable |
|---|---|---|
| 60 (one MiniBench round) | ±0.016 | Only large architectural changes |
| 100 | ±0.013 | Large changes |
| 300 (a season) | ±0.007 | Moderate changes |
| 1,810 | ±0.003 | FutureSearch's production edge (0.010) |

**Practical rule: only test changes expected to move pooled score by >0.015.** For reference, at n=1,900 two frontier models differed by 0.0005 (p=0.85) — a tie. Prompt-wording tweaks are *not detectable at our scale*. Architectural swings (ensemble on/off, search on/off, capping on/off, model-family swap) are.

Reporting which experiments we *declined to run as underpowered* is itself a finding.

### Rule 3 — Never report an absolute backtest Brier as if it were real.

Our backtest will be leaky. We accept this. Models have seen outcomes; date-restricted search is unreliable; asking a model to "pretend it doesn't know" fails (measured 52% Brier gap — see `Simulated Ignorance Fails`).

**The key insight that makes a cheap backtest viable:** leakage inflates *both* arms of a paired comparison roughly equally, so it largely **cancels in the paired difference**. Absolute numbers from a leaky backtest are garbage; paired deltas are still informative.

Therefore:
- ✅ Report paired differences from backtest.
- ❌ Never publish/claim an absolute backtest Brier, and never compare it to anyone else's number.
- ⚠️ Caveat every backtest result with the residual risk that leakage *interacts* with the treatment.

### Rule 4 — Slice by cohort. The mean hides the finding.

Always record, and always be able to slice by:

- **forecast horizon** (days to resolution) — FutureSearch found two harnesses tied at 30 days (p=0.95) but diverged at 60 days (p=0.03)
- **question type** (binary / numeric / multiple-choice)
- **category/topic**
- **base rate / community prediction at open** (where available)

**Losses are tail-concentrated, not uniform.** FutureSearch's vendor-SDK arm was *ahead on 51% of individual questions* yet lost on the mean, purely because its bad misses were larger. This is the same mechanism behind Metaculus's finding that **capping predictions at min/max is the strongest within-winners differentiator** (r=+0.48, p=0.005). Two independent sources, one phenomenon. Always report the distribution of per-question losses, not just the mean.

### Rule 5 — Keep an uncontaminated holdout.

> A benchmark a team iterates against is a benchmark that team can overfit.

- **Iteration loop:** MiniBench + our own resolved-question backtest. We tune against these.
- **Holdout:** the live tournament record. We never tune against it; it is scored by someone else on questions unresolved at forecast time. This is our free, uncontaminated test of record.
- Additionally hold out a random slice of backtest questions that is never used for iteration.

### Rule 6 — Degraded runs fail loud and are excluded from both arms.

A paired comparison is only valid if both arms answered **the same questions**. Therefore:

- Any question where *either* arm errored, timed out, fell back, or returned an unparseable forecast must be **dropped from both arms**, and the drop count reported.
- A run must **refuse to report a summary** if the drop rate exceeds a threshold (default 5%).
- Never silently substitute a default/fallback forecast into a scored run. Log the engine/model actually used for every question.

*Rationale, from two directions:* nostream lost ~150 points to 10 missing questions and 80 points to a single units bug — **reliability bugs cost more than scaffolding gains**. And silent degradation is the failure mode that bit the Lumi project three separate ways. If a fallback exists, it must announce itself.

---

## Data sources

| Source | Use | Notes |
|---|---|---|
| **MiniBench** | Primary fast iteration loop | ~60 short-fuse auto-generated questions, low leakage, resolves fast. Metaculus built this for exactly this purpose; 54% of tournament participants use it for iteration. This is our poor-man's BTF — **do not rebuild it.** |
| **Resolved Metaculus questions (API)** | Secondary offline backtest | Free, large, but **leaky**. Governed by Rule 3. |
| **Live tournament (FutureEval / Summer season)** | Uncontaminated holdout | Never tune against. Source of truth. |

---

## Implementation spec

Target module: `sibyl/eval/`. Keep it dependency-light (`numpy` only for the stats).

### Scoring

```python
import numpy as np

def brier(probs, outcomes):
    """Per-question Brier scores. Lower is better. Returns an array, NOT a mean —
    per-question scores are required for paired comparison and cohort slicing."""
    return (np.asarray(probs, dtype=float) - np.asarray(outcomes, dtype=float)) ** 2
```

Numeric questions: use a normalized ranked probability score scaled to [0, 1] so binary and numeric can be pooled on one scale. Lower is better throughout.

### Paired bootstrap — the core primitive

```python
def paired_bootstrap(scores_a, scores_b, n_boot=10_000, seed=0):
    """
    scores_a, scores_b: per-question scores for two variants on the SAME questions,
                        aligned by question id. Lower = better.
    Returns mean paired difference (a - b), 95% CI, and two-sided bootstrap p.
    Negative mean_diff => A is better than B.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    assert a.shape == b.shape, "paired comparison requires identical question sets"

    d = a - b
    n = len(d)
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2 * min((boot >= 0).mean(), (boot <= 0).mean())

    return {"mean_diff": float(d.mean()), "ci95": (float(lo), float(hi)),
            "p": float(min(p, 1.0)), "n": n}
```

### Power / minimum detectable effect

```python
import math

def min_detectable_effect(n, ref_halfwidth=0.003, ref_n=1810):
    """Approx 95% CI half-width on a paired difference at sample size n.
    Extrapolated from FutureSearch BTF-3 (±0.003 at n=1810), scaling as 1/sqrt(n).
    Call this BEFORE running an experiment (Rule 2)."""
    return ref_halfwidth * math.sqrt(ref_n / n)
```

### Cohort slicing

```python
def by_cohort(scores_a, scores_b, cohort_labels):
    """Run paired_bootstrap within each cohort. Use for horizon buckets,
    question type, category. Report alongside the pooled result (Rule 4)."""
    out = {}
    labels = np.asarray(cohort_labels)
    for c in np.unique(labels):
        m = labels == c
        if m.sum() >= 20:  # below this, the interval is uninformative
            out[str(c)] = paired_bootstrap(np.asarray(scores_a)[m],
                                           np.asarray(scores_b)[m])
    return out
```

### Experiment log — JSONL, one row per question per run

Every scored run appends to `results/<run_id>.jsonl`. Required fields:

```json
{
  "run_id": "exp_014_ensemble_v2",
  "question_id": "38291",
  "question_type": "binary",
  "horizon_days": 47,
  "category": "geopolitics",
  "forecast": 0.23,
  "outcome": 1,
  "score": 0.5929,
  "model": "claude-sonnet-4-6",
  "variant": "ensemble_v2",
  "git_sha": "a1b2c3d",
  "prompt_version": "v3",
  "n_llm_calls": 26,
  "cost_usd": 1.32,
  "degraded": false,
  "degraded_reason": null,
  "timestamp": "2026-08-04T14:22:11Z"
}
```

`degraded` and `git_sha` are mandatory — the first enforces Rule 6, the second makes any result reproducible.

---

## Reporting template

Every experiment writeup states, in this order:

1. **Hypothesis** and the effect size expected.
2. **Power check** — MDE at this n; whether the expected effect clears it.
3. **Paired result** — mean diff, 95% CI, p, n.
4. **Drop count** — questions excluded and why.
5. **Cohort table** — result sliced by horizon and question type.
6. **Loss distribution** — are differences mean-driven or tail-driven?
7. **Caveats** — leakage exposure, multiple-comparison exposure, overfitting risk.

---

## Money

Real-money prediction markets are **out of scope**, and this is an evaluation decision, not just a risk one. From FutureSearch (currently ranked #1 on Metaculus):

> Real-money prediction markets are harder still to learn from, because profit and loss folds in everything besides forecast accuracy: when to enter a position, how to size it, when to exit, risk management across positions, liquidity and fees, and the chance that the counterparty is trading on information you do not have. **A profitable quarter is weak evidence about any one of those, let alone about calibration.**

P&L is a confounded metric. It cannot validate a forecaster. Any future money decision is gated on calibration evidence from the sources above, capped as tuition, and executed manually by the owner — **never programmatically by this codebase.**

---

## The open question we're positioned to answer

Metaculus lists as an open research direction: *how well do backtesting evaluations predict prospective performance?* FutureSearch answered this at industrial scale with strict controls (BTF rankings track live rankings).

**Still unanswered, and the thing Sibyl can uniquely contribute: does a *cheap, leaky, hobbyist-grade* backtest predict live performance?** That's the question every solo bot-maker actually faces. We will have both a backtest harness and a live forward-looking record on the same bot, which is exactly what's needed to measure it.

Design implication: **preserve the ability to compare backtest-predicted rank against live-realized rank.** Keep run IDs, variants, and question sets linkable across both regimes.

---

## References

- [Pitfalls in Evaluating Language Model Forecasters](https://arxiv.org/abs/2506.00723) — the leakage taxonomy. Canonical.
- [Iterating on a forecaster without waiting for the future](https://futuresearch.ai/blog/iterating-on-a-forecaster/) — paired comparisons, harness-as-variable, power reality.
- [AI Forecasting in 2026: What 11 Analyses Say](https://www.greaterwrong.com/posts/a82q6yd8zKpYk56cF/ai-forecasting-in-2026-what-11-analyses-say) — evidence-graded playbook.
- [Fall 2025 FutureEval Bot-Maker Survey](https://www.metaculus.com/notebooks/43337/fall-2025-futureeval-survey/) — capping, base rates, research breadth.
- [A Prediction Tournament Paradox](https://arxiv.org/pdf/1903.02131) — why a single tournament rank is mostly noise.
- [Simulated Ignorance Fails](https://arxiv.org/abs/2601.13717) — 52% Brier gap when models "pretend not to know".
