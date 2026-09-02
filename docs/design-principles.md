# Sibyl — Design Principles (evidence-based)

> **For coding agents:** These are the architecture decisions for Sibyl, each tied to published evidence and graded by strength. Prefer a **Strong** finding over your own intuition. If you propose something contradicting a Strong finding, say so explicitly and give your reasoning — don't just do it.
>
> Companion doc: [`eval-strategy.md`](./eval-strategy.md) governs how anything here gets measured. This doc is *what to build*; that doc is *how to know if it worked*.

Source: [AI Forecasting in 2026: What 11 Analyses Say](https://www.greaterwrong.com/posts/a82q6yd8zKpYk56cF/ai-forecasting-in-2026-what-11-analyses-say) (Metaculus, Jul 2026) — a synthesis of 11 Metaculus analyses plus ~80 external papers, spanning 4 quarterly Pro-vs-bot tournaments and the Fall 2025 bot-maker survey (n=39).

**Evidence grades:** **Strong** = multiple sources, significant effect · **Moderate** = single source or marginal effect · **Suggestive** = anecdotal or small-N.

---

## Baseline reality

Pros (top human forecasters) have beaten the bot team in **every** head-to-head comparison to date: Q3'24 −11.3 (p=.036), Q4'24 −8.9 (p=.079), Q1'25 −17.7 (p=.001), Q2'25 −20.03 (p=.00001). Bots now beat the *median public* forecaster and rank roughly top 3–6% in open human tournaments, but trail superforecasters.

**Implication:** claims of "superhuman forecasting" are mostly backtest artifacts. Sibyl targets *competent and well-measured*, not *superhuman*.

---

## Build: what to do

### Strong — treat as settled

**1. Use a current frontier reasoning model.**
Model choice was the single largest differentiator across the first four quarters; simple one-shot bots on frontier models placed top-5. By Fall 2025, 34/39 surveyed bot-makers used one — it's now table stakes, not an edge. Two automated prompt-engineering studies confirm prompting **cannot** lift a sub-frontier model to frontier level.
→ *Don't try to be clever with a cheap model. Pay for the good one; economize elsewhere.*

**2. Invest in the harness/scaffolding.**
Top-5 scaffolded bots beat their own non-scaffolded baselines by 5–11 peer-score points per question. Base models improve ~0.9 pts/month, making good scaffolding worth **~9 months of base-model progress**.

**3. Agentic / iterative search, not one-shot retrieval.**
Removing search degrades Brier **3.6×**. Agentic search beats non-agentic in AIA Forecaster's ablation. Multiple independent sources converge.
→ *The research loop is the highest-leverage component after model choice.*

**4. Ensemble 3–7 runs across MODEL FAMILIES.**
86% of Fall 2025 winners aggregate multiple forecasts. Mantic's optimal ensemble: Gemini 3 Pro + GPT-5 + Grok 4 + a fine-tuned open-weight. nostreambot adds Grok specifically as a **low-correlation member**. Several respondents reported that cutting per-question forecast count to save money *backfired*.
→ **Diversity axis is the model, not the persona.** See Anti-pattern 4.
→ Default aggregation: **median** (robust to a single outlier arm).

**5. Apply Platt scaling (logistic recalibration) post-hoc.**
Metaculus Q1+Q2 analysis: Brier −0.016 binary (p=.00052), −0.005 multiple-choice (p=.000053). Phan et al.: 0.0999 → 0.0934. AIA Forecaster uses it as its final stage. Small absolute gains, but replicated in three independent places.
→ Implementation spec below.

### Moderate — do these, hold them loosely

**6. Wire up 2+ distinct research providers.** Number of sources correlates with score (r=0.42, n=39); winners averaged 1.75 providers, non-winners 1.00. Common winning stacks: AskNews + Perplexity + custom scraper, or AskNews + Exa + OpenAI web search.

**7. Cap predictions at a min/max.** 38% of winners cap. Strongest *within-winners* differentiator (r=+0.48, p=.005); 47% of top-15 cap vs 29% of the bottom half. Spec below.

**8. Explicitly compute base rates.** r=+0.38 (p=.032). 40% of top-15 winners do this vs 7% of the bottom half. Make it an explicit step in the reasoning prompt, not a hope.

**9. Look up similar previously-resolved questions.** 34% of winners vs 0% of non-winners (Fisher p=.04). Reference-class forecasting, mechanized.

**10. Budget ~28 LLM calls per question.** Winners averaged ~28 vs 7 for non-winners (p=.022); top-15 spent ~$1.40/question vs $0.50 for the bottom half. Split across research → forecast → critique → aggregation.

**11. Keep the model current.** Performance decays as pretraining data ages (Daily Oracle: 21.5% degradation on true/false, 11.3% on multiple-choice over 4 years). RAG narrows but does not close the gap. Pin the model in config; revisit each season.

---

## Anti-patterns — do not do these

**1. Don't chase a single "best" search provider.** Three seasons of sweeps found *no* provider with a significant or season-over-season consistent advantage; the winner rotates. **Breadth is the predictor, not brand.**

**2. Don't add scaffolding before reliability is right.** A single units bug cost nostream 80 points; missing 10 questions cost ~150. **Bugs cost more than cleverness gains.** Enforced by Rule 6 in `eval-strategy.md`.

**3. Don't let the bot confuse open questions with resolved ones.** Recurring failure: bots read a still-open question as settled and submit a near-certain forecast. **Add an explicit guard** that checks question state before allowing extreme confidence.

**4. Don't use multi-persona aggregation.** ⚠️ *This supersedes the original Sibyl project brief.* One Fall 2025 winner explicitly flagged multiple personas as a mistake, noting the final-stage LLM was bad at being genuinely critical. Metaculus ran two experiments (one informal, one n=30) finding little-to-no improvement.
→ Keep the ensemble; get diversity from **different models**, not different prompted "characters." A single explicit critique/red-team *step* is fine; a cast of personas is not.

**5. Don't emphasize "Bayesian" framing in prompts.** Prompts with "Bayesian" in the title significantly underperformed in both parts of the prompt-optimization series. Small sample, consistent direction. *(Compute base rates — Moderate #8 — without the incantation.)*

**6. Don't ship the template's default numeric pipeline unexamined.** Multiple winners flagged it as weak. Known fixes: elicit **(mean, std)** instead of percentiles; apply **PCHIP smoothing** for continuous questions (originated in Panshul42's bot, adopted by nostreambot).

---

## Recalibration & capping — implementation spec

Both are post-processing applied **after** aggregation, in this order: `aggregate → Platt → cap → submit`.

### Platt scaling

Fit a 1-D logistic regression mapping raw forecast logits to calibrated probabilities:

```
p_cal = σ(a · logit(p_raw) + b)
```

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

EPS = 1e-6

def fit_platt(p_raw, outcomes):
    """Fit on HELD-OUT resolved questions only. Never on the eval set."""
    z = np.log(np.clip(p_raw, EPS, 1 - EPS) / (1 - np.clip(p_raw, EPS, 1 - EPS)))
    lr = LogisticRegression(C=1e6)          # effectively unregularized, 2 params
    lr.fit(z.reshape(-1, 1), np.asarray(outcomes))
    return float(lr.coef_[0][0]), float(lr.intercept_[0])   # (a, b)

def apply_platt(p_raw, a, b):
    p = np.clip(p_raw, EPS, 1 - EPS)
    z = np.log(p / (1 - p))
    return 1.0 / (1.0 + np.exp(-(a * z + b)))
```

Requirements:
- **Fit on held-out resolved questions.** Fitting and evaluating on the same data invalidates the result (Rule 5, `eval-strategy.md`).
- Needs ~100+ resolved questions minimum for 2 stable parameters. Until we have that, **skip Platt and rely on capping.**
- Log the fitted `(a, b)` per run. `a < 1` means the raw forecasts were overconfident; `a > 1` means underconfident. This is a *finding*, not just a config value.

### Capping

```python
def cap(p, lo=0.02, hi=0.98):
    return float(min(max(p, lo), hi))
```

- Default `[0.02, 0.98]`. Make it configurable; log the cap in effect.
- **Set the bound empirically:** look at the top/bottom bins of the reliability diagram. If forecasts issued at 0.98 resolve true only ~92% of the time, the honest ceiling is ~0.92.
- Capping is a crude, training-free stand-in for what Platt learns. **Once Platt is fitted on enough data, capping becomes a safety rail, not the primary correction** — keep it as protection against catastrophic single misses.

---

## Candidate techniques — unvalidated, from the Summer 2026 bot-plans page

> **Read the grading difference before using anything here.** Everything above is graded
> from the [11-analyses synthesis](https://www.greaterwrong.com/posts/a82q6yd8zKpYk56cF/ai-forecasting-in-2026-what-11-analyses-say):
> measured effects, significance tests, multiple sources. This section comes from
> [Bot Plans](https://www.metaculus.com/notebooks/43497/bot-plans/), which is bot-makers
> describing what they **intend to try**. Almost none of it is validated, and popularity
> there is not evidence. These are hypotheses to test, and they are labelled `Candidate`
> rather than Strong/Moderate/Suggestive so nobody later mistakes a plan for a finding.

### C1 — Cost-tiered escalation: cheap tier first, escalate only on disagreement

Run a cheap (or free) forecaster tier, measure spread, and spend frontier tokens only on
the questions where the cheap tier disagrees with itself.

*Source:* amigon21-bot (MiniSim) routes on measured agent variance — low variance goes to a
single frontier call, high variance to a full 4-round deliberation — and reports **56% of
questions taking the cheap path**. Its whole development ran on local Ollama at $0 API cost.

**The mechanism already exists here.** `spread_metrics.compute_spread` dispatches per
question type and `stacking_route.route_after_forecasts` acts on it, with tuned thresholds
(binary 0.15, MC 0.20, numeric 0.15). What was measured as harmful and disabled is the
*action* on the high-spread branch — crux extraction plus a stacker LLM — not the routing.
This proposal replaces that action rather than building a router.

**The risk that decides whether this works at all: correlated error.** Cheap models share
architectures, training data and failure modes, so they can agree *confidently and wrongly*
— and plausibly do so precisely on the questions a frontier model would get right. If that
holds, the router sends easy questions to the expensive tier and keeps hard ones cheap,
which is exactly backwards and would look like a saving while costing accuracy.

That is falsifiable and must be tested before the routing is trusted:

> On the subset of questions where the cheap tier agrees, how often does the frontier tier
> materially disagree with it?

If cheap-tier agreement does not predict frontier-tier agreement, C1 is dead regardless of
how much it saves.

### C2 — Multiple-choice option shuffling and per-bin recalibration

Forecast the same MC question several times with the answer order shuffled and average, to
cancel position bias. Then fit a per-bin empirical recalibration map on held-out resolved
questions (if forecasts issued at 33% resolve ~20% of the time, shift that bin down).

*Source:* aroswift-bot and PressPause, independently. Both cite the same rationale: published
benchmarks show the **bot-vs-pro gap is substantially larger on multiple-choice** than on
binary, and almost nobody targets it directly.

**Be honest that this costs more, not less.** K shuffles is K× the forecaster calls on MC
questions. It is an accuracy play, not a budget play, and it only fits the budget because MC
is a minority of the question mix.

Both sources also warn about an objectivity check: a mechanically-determined question (dice
odds, a scheduled arithmetic outcome) must not be dragged toward a historical base rate by
the recalibration map.

### C3 — Extremization of the aggregate

Push the aggregated probability away from 0.5 in log-odds space, typically by a factor of
1.3–1.5, before capping.

*Source:* the most widely-proposed calibration technique on the page — hugo0-bot (1.3–1.5,
calls it "the highest-ROI single calibration intervention"), Ora (sqrt(3), then clipping to
[0.02, 0.98], which is our exact cap), pancratic-bot (scaled by inter-model agreement),
q_forc+bot, Chineme.

**We already have the analysis; what is missing is the fitting.** `calibration-primer.md` §8
sets out the tension precisely: individual LLM forecasts are overconfident (wanting `a < 1`)
while an ensemble average is underconfident (wanting `a > 1`), and which dominates is not
knowable from theory. **So extremization is not a constant to adopt — it is the same fitted
parameter as Platt, and adopting a literature default of 1.3–1.5 without fitting would be
exactly the "impressive-but-uncalibrated" failure `vision.md` names as non-goal #1.**

Two interactions to respect:
- Order is `aggregate → extremize → Platt → cap → A3 guard`. Extremizing after Platt would
  undo the fit; extremizing after the cap would push values back out of bounds.
- **Never extremize a cheap-tier aggregate on the strength of a frontier-tier fit.** The
  correction amplifies whatever calibration the input already has, so a badly-calibrated
  input gets a badly-calibrated output, more confidently. If C1 ships, extremization needs
  a per-tier parameter or it must be disabled on the cheap path.

### C4 — Shuffling as an escalation signal (composition of C1 and C2)

*Not from the page — a consequence of running C1 and C2 together, and untested anywhere.*

Option shuffling generates variance for free from a single model. On MC questions that makes
it both a debiasing step **and** a spread source: if K shuffled orders on the cheap tier
agree, you have a debiased answer *and* evidence that escalation is unnecessary; if they
disagree, that disagreement is the escalation trigger. It also gives the cheap tier a
disagreement signal that does not depend on cross-model diversity, which is the part most
vulnerable to correlated error.

Speculative, and worth exactly one cheap test before anyone builds on it.

## Effort envelope

**Reported dev time does not correlate with score (r=0.08).** Top-10 bots span from <40 hours to 4+ months of full-time work; hobbyists with no dedicated eval infrastructure routinely place top-5.

**But there is a floor:** non-winners cluster at the very low end of time invested, and every survey respondent reporting 80+ hours won prize money.

→ Target the 40–80 hour band. Spend it on reliability and the research loop, not on architecture novelty. Resist scope creep; this is an explicit guardrail, not a suggestion.

---

## Reference implementations

| Repo | Why |
|---|---|
| [nostreambot](https://github.com/No-Stream/metaculus-bot) | MIT. 9th in Fall 2025 (top open-source), 10th Spring 2026. Frontier ensemble via OpenRouter, Grok as low-correlation member, median aggregation, PCHIP smoothing, AskNews + Grok grounding. Author wrote a [retrospective](https://nostream.substack.com/p/on-writing-a-bitter-pilled-metaculus). **Primary reference.** |
| [Panshul42 Q2 winner](https://github.com/Panshul42/Forecasting_Bot_Q2) | 1st in Q2 2025. Multi-stage RAG, parallel agents, asyncio/aiohttp, origin of the PCHIP technique. |
| [Metaculus template](https://github.com/Metaculus/metac-bot-template) | What this repo forks. Easiest setup (<1 hour), was top-5 for a year, now lags as scaffolding has improved. Good plumbing, weak numerics. |

---

## What this means for Sibyl

The playbook above is well-trodden — executing it well produces a competent bot, not a differentiated project. Per `eval-strategy.md`, **Sibyl's contribution is the evaluation, not the architecture.**

Practical consequence: implement the Strong findings efficiently and without invention, spend the saved effort on measurement, and treat every Moderate/Suggestive item as a **hypothesis to test with a paired comparison** rather than a feature to assume. Several of them (capping, base rates, research breadth) are exactly the kind of large-effect changes that *are* detectable at our sample size.
