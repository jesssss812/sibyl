# Calibration Primer

> Reference for the scoring and recalibration concepts Sibyl depends on. Read alongside [`eval-strategy.md`](eval-strategy.md) (how we measure) and [`design-principles.md`](design-principles.md) (what we build).

---

## 1. Brier score

A probability isn't right or wrong on its own — "70% chance of rain" followed by rain tells you nothing. Probabilities can only be graded in batches.

```
Brier = average of (your probability − what happened)²
```

where *what happened* is 1 for yes, 0 for no. **Lower is better.**

### Worked example

Ten questions, five resolve YES and five resolve NO.

| Forecaster | Behavior | Calculation | Brier |
|---|---|---|---|
| **A** — hedger | 50% on everything | (0.5−1)² and (0.5−0)² = 0.25 throughout | **0.25** |
| **B** — informed | 90% on the YESes, 10% on the NOs | (0.9−1)² = 0.01, (0.1−0)² = 0.01 | **0.01** |
| **C** — confidently clueless | 90% on everything | 5×(0.9−1)² + 5×(0.9−0)² = 0.05 + 4.05 | **0.41** |

### The scale

| Brier | Meaning |
|---|---|
| 0.00 | Perfect |
| 0.08–0.11 | Superforecaster / top bot |
| **0.25** | **The "I know nothing" line — 50% on everything** |
| 0.41 | Confidently wrong |
| 1.00 | Certain and always wrong |

**0.25 is the threshold that matters.** Above it, a forecaster is actively harmful — confidence without knowledge is worse than silence.

---

## 2. Two different ways to be good

Brier blends two distinct skills. Separating them is the most useful diagnostic in this document.

**Doctor A** tells every patient "12% chance you have this." Across her practice, exactly 12% do. Her numbers are **honest** — and useless, because she can't distinguish patients.

**Doctor B** says 90% for some and 5% for others, and he's sorting correctly — but his 90% group only has the disease 70% of the time. **Informative but overconfident.**

- **Calibration** — are the numbers honest? When you say 30%, does it happen 30% of the time?
- **Discrimination** (resolution) — do you separate the yeses from the nos?

Murphy's decomposition:

```
Brier  =  calibration error  −  discrimination  +  difficulty of question set
           (want low)           (want high)        (not attributable to you)
```

### Why this matters operationally

**Calibration is cheap to fix; discrimination is not.**

Poor calibration is a post-processing problem — the entire recalibration toolkit below fixes it without the bot getting smarter. Poor discrimination means the bot genuinely doesn't know anything, and no math repairs that.

So decomposing and finding poor calibration is *good news*. Always report the decomposition, not just the scalar.

---

## 3. Reliability diagram

Bin the forecasts; in each bin, compare predicted to observed frequency.

| Forecast bin | n | Observed rate | |
|---|---|---|---|
| 0–10% | 40 | 6% | ✓ |
| 10–20% | 35 | 14% | ✓ |
| 40–50% | 28 | 46% | ✓ |
| 80–90% | 30 | 74% | ✗ overconfident |
| 90–100% | 25 | **72%** | ✗✗ badly overconfident |

Plot predicted (x) against observed (y); perfect calibration is the 45° diagonal.

That last row is the **standard LLM signature** — says 95%, right 72% of the time, with a mirrored failure at the low end (says 3%, happens 12%). This single chart determines whether and how much to recalibrate.

---

## 4. Log score, and why extremes are dangerous

Metaculus scores on **log score**, not Brier:

```
log penalty = −ln(probability you assigned to what actually happened)
```

Writing `p` for your forecast that the event happens, that splits into two cases:

```
event HAPPENS      →  penalty = −ln(p)
event DOESN'T      →  penalty = −ln(1 − p)
```

Penalties when the event does **not** happen (the `−ln(1 − p)` case):

| You said (p) | 1 − p | Brier penalty | Log penalty |
|---|---|---|---|
| 90% | 0.10 | 0.81 | −ln(0.10) = 2.30 |
| 95% | 0.05 | 0.90 | −ln(0.05) = 3.00 |
| 99% | 0.01 | 0.98 | −ln(0.01) = 4.61 |
| 99.9% | 0.001 | 0.998 | −ln(0.001) = 6.91 |
| 100% | 0 | 1.00 | −ln(0) = **∞** |

Brier caps damage at 1.0 per question. **Log score has no floor.** 99% → 99.9% barely moves Brier but nearly doubles the log penalty.

See §9 for why Metaculus chose this.

---

## 5. Capping

`cap(p) = min(max(p, 0.02), 0.98)` — three lines, and the strongest differentiator among tournament winners (r = +0.48, p = .005).

### The arithmetic

Upgrading a forecast from 95% → 99%:

- **Right:** log penalty 0.051 → 0.010. **Gain 0.041.**
- **Wrong:** log penalty 3.00 → 4.61. **Lose 1.609.**

You risk **1.609 to win 0.041** — a **39-to-1 bad bet**. Break-even requires being right **97.5%** of the time.

Now recall §3: that bot's 90–100% forecasts came true **72%** of the time. Every extreme forecast was a losing bet.

Capping isn't timidity — it's declining bad odds. It also bounds the worst day: one stale-article-driven 99% call can erase a season of gains.

### Setting the bound

Read it off the reliability diagram. If 98% forecasts resolve true 92% of the time, the honest ceiling is ~0.92.

---

## 6. Platt scaling

The learned version of capping. You know from the reliability diagram that you're overconfident; Platt learns *how much* from data rather than guessing.

### Log-odds

Probabilities are trapped in [0,1] and squished at the ends. Log-odds is an unbounded line, symmetric around 0 (= 50%), where confidence is distance from zero.

| Probability | Odds | Log-odds |
|---|---|---|
| 0.10 | 1:9 | −2.20 |
| 0.50 | 1:1 | **0** |
| 0.90 | 9:1 | +2.20 |
| 0.95 | 19:1 | +2.94 |
| 0.99 | 99:1 | +4.60 |

### The formula

```
calibrated = sigmoid( a × log-odds(raw forecast) + b )
```

- **`a` is a confidence dial.** `a < 1` turns confidence down (overconfident); `a = 1` no change; `a > 1` turns it up.
- **`b` is a thumb on the scale** — corrects systematic optimism/pessimism.

### Worked example, a = 0.7

| Raw | log-odds | × 0.7 | Calibrated |
|---|---|---|---|
| 0.99 | 4.60 | 3.22 | **0.96** |
| 0.95 | 2.94 | 2.06 | **0.89** |
| 0.80 | 1.39 | 0.97 | **0.73** |
| 0.50 | 0.00 | 0.00 | **0.50** ← fixed point |
| 0.20 | −1.39 | −0.97 | **0.27** |

Extremes get pulled in hard, moderates barely move, 50% never moves — a smooth, proportional version of capping.

### Fitting

Ordinary logistic regression: *find the a, b that best predicted actual outcomes on past resolved questions.*

1. **Fit on held-out questions only.** Fitting and evaluating on the same data invalidates the result.
2. **Needs ~100+ resolved questions** for two stable parameters. Below that, use capping alone.

**The fitted `a` is a measurement, not just a setting** — it quantifies the bot's overconfidence and is a reportable finding.

---

## 7. Other recalibration methods

| Method | Params | Questions needed | Notes |
|---|---|---|---|
| **Temperature scaling** | 1 | ~50+ | Platt with `b = 0`. Confidence dial only. Most data-efficient. **Start here.** |
| **Platt scaling** | 2 | ~100+ | Dial + shift. The workhorse. |
| **Beta calibration** | 3 | ~300+ | More flexible curve when Platt's S-shape underfits. |
| **Histogram binning** | per-bin | ~500+ | "When I said 80–90% it happened 62%, so output 62%." Chunky, data-hungry. |
| **Isotonic regression** | non-param | ~1000+ | Any monotone staircase. Overfits badly at small n — **a trap at our scale.** |

**For Sibyl:** temperature scaling → Platt at ~100 resolved questions. Ignore the rest.

---

## 8. Extremizing — the opposite direction

Three analysts research independently, read different sources, each concludes 70%. Averaging gives 70% — but that's too low. Each found *independent* evidence pointing the same way; collectively they've seen far more than any one of them. The right answer is higher, maybe 85%.

Averaging destroys that information; extremizing restores it. Same formula, `a > 1`.

### The tension in Sibyl

- Each individual LLM forecast is **overconfident** → wants `a < 1`
- The ensemble average is **underconfident** → wants `a > 1`

Which dominates is not knowable from theory — it depends on the models, the prompts, and how correlated the ensemble members are.

**This is the argument for fitting rather than hand-tuning.** Whichever way `a` lands is itself the result:

> *"After ensembling, Sibyl required a = 0.85 — individual-model overconfidence outweighed ensemble under-confidence."*

---

## 9. Why Metaculus uses log score instead of Brier

Four reasons, in rough order of importance.

### 9.1 Locality — only what happened counts

Log score depends **only** on the probability assigned to the outcome that actually occurred. Brier also depends on how you distributed probability among outcomes that *didn't* occur.

Multiple-choice question (A/B/C/D); truth is **A**:

| | A | B | C | D | Log score | Brier |
|---|---|---|---|---|---|---|
| Forecaster 1 | 0.5 | 0.5 | 0 | 0 | 0.693 | 0.500 |
| Forecaster 2 | 0.5 | 0.167 | 0.167 | 0.167 | 0.693 | 0.334 |

Both gave A the same 50%. Log score calls them identical. Brier prefers Forecaster 2 for spreading the losing mass evenly — rewarding a judgment about counterfactuals.

Log score is the **unique** local proper scoring rule (Shuford–Albert–Massengill). For a platform grading many multiple-choice and continuous questions, locality is the principled choice — and it extends naturally to continuous distributions, where you just take the log density at the realized value. Brier does not extend so cleanly.

### 9.2 Tail resolution — the decisive practical reason

Brier is nearly blind to differences among small probabilities. Suppose the event does **not** happen:

| You said | Brier penalty | Log penalty |
|---|---|---|
| 3% | 0.0009 | 0.0305 |
| 0.3% | 0.000009 | 0.0030 |
| **Difference** | **0.00089** | **0.0275** |

Those two forecasts differ **tenfold in odds** — a large, decision-relevant gap. Brier registers 0.00089, well under 1% of a typical score. Log registers a difference roughly 10× larger in relative terms.

Many of the most valuable forecasting questions are low-probability events, where the difference between 0.1% and 1% genuinely matters. Brier treats them as effectively the same; log score does not.

### 9.3 Information-theoretic meaning

Log score is **surprisal** — the negative log-likelihood of what happened under your model. Consequences:

- Log scores **add** across questions, so a season's total is the log-likelihood of the entire observed world under your forecasts.
- **Differences** of log scores are log-likelihood ratios — interpretable as strength of evidence, like Bayes factors.
- Minimizing log score is maximum-likelihood estimation.

Brier differences have no comparable interpretation.

### 9.4 It makes Metaculus's relative scores work

Peer and Baseline scores are built from *differences* between forecasters' log scores. Because of §9.3, "my log score minus yours" answers a real question: *how many times more likely was reality under my forecasts than yours?* That construction requires log.

### Why Brier survives elsewhere

Brier is **bounded**, so no single question or resolution error can dominate a season; it never blows up at p = 0; and it's intuitive (it's just MSE). ForecastBench and much of the academic literature use it. Both are proper scoring rules — both reward honesty. They differ in what they emphasize.

### ⚠️ Practical consequence for Sibyl

**We compete on log-based peer score but our backtest harness scores Brier.** These disagree precisely on tail behavior — which is exactly what capping and recalibration change.

An experiment evaluating capping under **Brier will understate its tournament benefit**, because Brier barely notices the extreme forecasts capping removes.

**Therefore: compute both scores on every backtest run.** It's the same data and two small functions. Report both, and flag any experiment where they disagree — that disagreement is itself informative.

---

## References

- [A Prediction Tournament Paradox](https://arxiv.org/pdf/1903.02131) — why single-tournament rank is mostly noise
- [Pitfalls in Evaluating Language Model Forecasters](https://arxiv.org/abs/2506.00723) — leakage taxonomy; notes Brier's base-rate sensitivity
- [Calibration adjustment analysis](https://www.metaculus.com/notebooks/43356/calibration-adjustment-analysis/) — Platt scaling results on Q1+Q2 bot forecasts
- Guo et al. 2017, *On Calibration of Modern Neural Networks* — temperature scaling
- Kull et al. 2017, *Beta calibration*
