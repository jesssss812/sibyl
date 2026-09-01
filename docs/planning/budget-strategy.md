# Sibyl — Provider & Model Budget Strategy

> ### ⚠ Corrected 2026-08-31 — the free-search premise no longer holds
>
> This document's cost model assumes **Gemini grounded search is $0.00 marginal
> ("free tier, off OpenRouter entirely")**. That is no longer true, and it was the
> load-bearing assumption behind "AskNews + Gemini give you two free indexes".
>
> Google meters grounded search on a **monthly** free bucket (~5k prompts on the 3.x
> family), not a daily one. It ran out, and with no billing account attached every call
> returned `429 RESOURCE_EXHAUSTED`. The slot is now **Anthropic's server-side
> `web_search`** — a genuinely paid provider (tokens on `claude-sonnet-5`, plus ~$10 per
> 1,000 searches, capped at 15/question ≈ $0.15 worst case).
>
> Two further premises have also moved:
> * **The donated OpenRouter key is spent** for this season and unset, so the
>   "donated key covers OpenAI/Anthropic/Google" routing below is inactive — every
>   OpenRouter call bills the personal key.
> * **The Anthropic slots bill Anthropic directly** since 2026-08-31, and
>   `CREDIT_SPEND` does not see that spend at all, so every per-question figure in this
>   document is now a **lower bound**.
>
> The comparative reasoning — redundancy/overlap analysis, which providers earn their
> keep, where leverage sits — still stands. Only the price of the second search index
> and the key routing changed. See
> [anthropic-direct-swap.md](./anthropic-direct-swap.md).


> Brainstorm, 2026-08-24. Grounded in the repo's own **measured** cost data:
> `scratch/smoke_cost_breakdown_2026-07-17.md`, `scratch/bundle_token_audit_2026-07-17/`,
> and `scratch/ensemble_composition_2026-07-15/`. Numbers below are measured unless
> marked as estimates.
>
> Resolves the open decision in `vision.md`: *"Search-provider costs — confirm which
> research APIs have usable free tiers without tournament credits."*

---

## 1. The premise needs correcting: you have six free providers, not one

**Perplexity and Exa are not part of this bot's production research stack.** They sit in
the *primary-provider priority ladder* — AskNews → Exa → Perplexity → Perplexity-via-
OpenRouter → stub — and `research.md` is explicit: *"In production the AskNews
credentials are present, so Exa and the two Perplexity routes never run as the primary.
They are fallbacks, not peers."*

The Metaculus tutorial recommends them because the starter template has no AskNews-grade
primary. You do. They are redundant for you, not missing.

What actually runs in prod, and what each costs you:

| Provider | Your cost | Evidence |
|---|---|---|
| **AskNews** (primary, summarized) | **Free** — tournament access | — |
| **Gemini grounded search** | **$0.00 marginal** | Measured: *"free tier, off OpenRouter entirely."* Uses your `GOOGLE_API_KEY` via `google-genai`, **not** OpenRouter. A genuinely distinct first-party Google index. |
| **Prediction-market snapshot** | **Free venue APIs** + 2 LLM calls (~$0.02) | Kalshi/Polymarket/Manifold/PredictIt public endpoints |
| **Financial data** | **Free** — yfinance keyless, FRED free key | ~$0.01 classifier call |
| **Resolution-source fetcher** | **Free** | Tier-1 only: *"plain HTTP, no LLM calls, no retries"* |
| **Time-series anchor** | **Free** | *"No LLM, no model selection"* |

**Six providers. Design principle M6 asks for two.** Winners averaged 1.75. A1 says
breadth is the predictor, not brand — and breadth is what you have, for roughly $0.03.

The redundancy audit makes the case sharper. AskNews vs Gemini overlap **0.47 on
numbers** but **0.00 on source domains** — two genuinely independent retrieval paths
converging on the same facts through disjoint sources. That is exactly the corroboration
breadth is supposed to buy, and you get it free.

**Conclusion: the research problem is already solved. Don't spend money there.**

⚠️ **One thing to verify.** `research.md` notes AskNews returns a `403011 "subscription
is not currently active"` signature that the bot logs as `inactive` — *"an expected
off-season state."* AskNews access is tied to tournament seasons. **Confirm yours is
live for MiniBench before relying on it as primary.** If it isn't, the ladder falls
through Exa → Perplexity → **stub**, and with no keys for those you'd run add-ons only.
That still leaves Gemini + markets + financial + resolution + anchor — degraded but not
dead. Worth knowing which state you're in (and Rule 6 says it must announce itself).

---

## 2. The money is in reasoning tokens, not research

Measured decomposition of one full-config question (~$3.53 total):

- **Forecaster fan-out: ~$2.25 — 64%**
- Research stack: ~$1.30 — 36%, of which gap-fill v1+v2 ≈ $0.85

And from the token audit: bundle input is **~$0.39–0.60/question — only 10–13% of
question cost.** The audit's own verdict: *"the order-of-magnitude lever on cost remains
forecaster effort tiering."*

So the budget question is not "which providers can I afford." It is **"how do I spend
reasoning tokens."**

---

## 3. The key insight: "frontier" and "expensive" are different axes

Measured per-forecaster cost, same question, same research:

| Forecaster | Effort | Cost |
|---|---|---:|
| gpt-5.6-sol | xhigh | **$0.85** |
| gpt-5.5 | xhigh | $0.55 |
| claude-fable-5 | xhigh | $0.45 |
| claude-opus-4.8 | xhigh | $0.20 |
| gemini-3.1-pro-preview | default | **$0.11** |
| grok-4.5 | high | $0.08 |

**A 10× spread, and every one of them is a current frontier reasoning model.**

Design principle S1 says *use a current frontier reasoning model*. It does not say use
the priciest one at maximum reasoning effort. The $3.5–4.5/question figure is produced by
**xhigh effort on expensive OpenAI/Anthropic slots**, not by "frontier."

The author reached the same conclusion independently. `llm_configs.py:78-84`, dated
2026-07-20:

> *"effort xhigh → high. The reasoning-effort audit found default→high clearly worth it
> but **high→xhigh UNMEASURED**; sol is ~70% of forecaster reasoning spend, so we stop
> paying the unmeasured premium on the dominant-cost slot."*

**You are being asked to choose between S1 and your budget. You don't have to — that
tension is mostly an artifact of effort tiering, not of model tier.**

---

## 4. Recommended configuration

Three frontier families at the cheap end of the frontier tier. Each choice is backed by
the repo's measured ensemble screening:

| Slot | Why | Est. cost |
|---|---|---:|
| **gemini-3.1-pro-preview** | Measured **#2 individual member** (+48.4 mean log, n=42). Removing it hurts: −2.14 [−3.62, −0.62], binary Brier +0.0127 [+0.0039, +0.0238] — CI excludes zero. Already pinned to the **personal** key, so it works without a donated key. | ~$0.11 |
| **claude-opus-4.8** at `high` (not xhigh) | Measured **#1 individual member** (+50.5). Dropping to `high` follows the author's own sol precedent — the xhigh premium is unmeasured. | ~$0.10–0.15 |
| **One OpenAI 5.6 slot** at `medium`/`high` | The openai-gpt5 family is **the one clearly load-bearing family**: removal hurts in 6 of 8 eras, pooled −2.56 [−3.90, −1.25]. Keep the family, drop the effort tier. | ~$0.15–0.30 |

**Forecasters ≈ $0.35–0.55 · Research ≈ $0.35–0.55 · Total ≈ $0.70–1.10/question.**

**$50 buys roughly 45–70 questions — a full MiniBench round.**

Note what this is *not*: it isn't the "deliberately cheap tier" of `vision.md`
($0.05–0.20/question with cheap models). It's a **properly configured frontier bot**.
Three frontier families, six research providers, agentic gap-fill, median aggregation,
capping — nearly the full design-principles playbook, inside the cap.

**Explicitly skip `grok`.** Measured: worst individual member in *every* era it appears,
removal deltas straddle zero everywhere (pooled +0.24 [−1.13, +1.48]). *"Contributes
nothing measurable"* — so it's the one slot whose cost buys nothing. (Note this
contradicts `design-principles.md`, which cites Grok as nostreambot's low-correlation
member. The measurement supersedes the writeup; the current roster has already dropped it.)

---

## 5. On Groq and free models — right tool, wrong slot

**Do not use free models as forecasters.** Three reasons, in order of decisiveness:

**(a) The token math kills it.** Each forecaster reads the research bundle plus prompt
overhead: measured **~14.8k input tokens for a numeric question at the mean bundle**, up
to ~21k on a heavy one. Three forecasters ≈ 45–60k input tokens per question. Groq's
free tier binds on **tokens per minute** — and your own Lumi notes recorded exactly this:

> *"Groq's binding limit was tokens/min, and chat re-sends full history, so later turns
> cost more — sessions degrade in ways early testing never shows."*

A forecaster that can't get its tokens through inside `FORECASTER_SOFT_DEADLINE` gets
cancelled and counted as a drop. Enough drops and the **min-forecasters guard skips the
question entirely** — that's a Rule 6 degradation event, not a saving. You'd be trading
dollars for missing questions, and nostream lost ~150 points to 10 missing questions.

**(b) S1 forecloses it.** *"Two automated prompt-engineering studies confirm prompting
cannot lift a sub-frontier model to frontier level."* Strong evidence, and the forecaster
slot is precisely where it applies.

**(c) Plumbing friction.** OpenRouter `:free` slugs 404 on donated keys (their providers
aren't in `DONATED_KEY_PROVIDERS`), so free models must run as plain `GeneralLlm` on the
personal key — already documented in `ablation/forecaster_lineup.py`.

### Where free models genuinely belong: the dev/test lane

This is **study note #1's actual problem** — cost rules that *"depend on compliance, not
structure."* On a $50 cap, an accidental live run during development is your largest
single budget risk, and there is no structural guard on the live path (Finding 3 of the
improvement plan).

`ablation/forecaster_lineup.py::FREE_FORECASTER_MODELS` already exists for exactly this:
*"cheap free OpenRouter models so a 100-question medium backtest doesn't burn budget on
the forecaster stage."* **That is the right home for Groq and free models** — the lane
where their weaknesses (quality, TPM) don't matter and their strength ($0) is the whole
point. It's also the Lumi F7 fixture-mode pattern you already built and validated.

### ⚠️ The mixed-ensemble trap

If you're tempted by "one frontier + two free to keep the ensemble at three" — **don't**,
and the reason is mechanical rather than a matter of taste.

Aggregation is **MEDIAN**. With three members:

- **2 frontier + 1 free** is *safe*. Sort the three values: the median is always
  bracketed by the two frontier estimates. The free model can interpolate between them
  but can never pull the answer outside them. It's a free, harmless tie-break.
- **1 frontier + 2 free** is *unsafe*. Two free models that agree with each other
  outvote the frontier model completely — the expensive forecast is discarded.

**Rule: under MEDIAN, frontier must be a majority of the ensemble.** Nothing in the docs
states this; it falls straight out of the aggregation mechanism.

---

## 6. If you need to cut further — ranked by dollars per unit of information lost

From the repo's own audit cut list:

1. **Retire gap-fill v1 (~$0.55/q).** Ranked #1 in the audit's cut list (~$190/quarter).
   v1 and v2 *"do the same job"* — both find factual gaps, search, and append. In the
   Zambia smoke, v2 found the same core gaps in 66s that v1 spent four parallel web
   searches on. **Best single cut available.**
2. **Drop native search (~$0.20/q).** 0.45 fact-overlap with AskNews, 0.40 with Gemini,
   and you retain two independent news paths without it.
3. **Cap the AskNews summarizer's output (~$0.10/q).** Pure utility slot, so S1 doesn't
   apply — but note the cost is in *output* (33k chars at output pricing), because the
   prompt explicitly says *"Be COMPREHENSIVE."* **Cap the length, don't downgrade the
   model** — the summarizer's fidelity rules (never paraphrase numbers, `[PRE-WINDOW]`
   tagging, supersession) are load-bearing for calibration, and `[PRE-WINDOW]` is your
   only real mitigation for the open-vs-resolved gap.

**Do not cut:**
- **Prediction markets** — 191 mentions across **37 of 45** questions, the one section
  forecasters explicitly anchor on, at 5% of bundle cost. *"Highest value density in the
  bundle."* And the venues are free.
- **Financial data** — $0.01, heavily used on the 6% of questions where it fires.
- **Resolution source / TS anchor** — zero LLM calls, so cutting them saves nothing.

---

## 7. The strategic payoff: this gives you a better flagship experiment

`vision.md`'s flagship is **cheap model vs frontier model**. Two problems, one old and
one new:

- *(Improvement plan, Finding 2)* leakage **interacts** with that treatment — a frontier
  model has memorized more of the outcome space, so a leaky backtest can't separate
  capability from memorization. Rule 3's cancellation argument fails precisely here.
- *(This doc)* the more interesting axis turns out to be **reasoning effort and price
  within the frontier tier**, not tier-vs-tier.

**Proposed replacement flagship: *"What does reasoning effort actually buy?"***
Same models, same research, same questions — `effort=medium` vs `effort=high` (or
`high` vs `xhigh` on one slot).

Why it's strictly better:

- **Leakage cancels exactly.** Both arms are the *same model*, so memorization is
  identical on both sides of the paired difference. This is Rule 3's ideal case rather
  than its failure case.
- **It's a genuinely open question.** `FUTURE.md` carries *"Price the high→xhigh
  reasoning-effort premium"* as an open item, and the 2026-07-20 audit found
  default→high worth it but **high→xhigh unmeasured**. The author of the 10th-place bot
  doesn't know the answer either.
- **It's affordable now.** Both arms are cheap, so it doesn't wait on the Sept/Oct
  credit refresh — unlike the frontier arm of the original design.
- **It clears Rule 2.** Effort tiering is an architectural-scale change, not a
  prompt tweak.
- It is *directly actionable*: the answer changes your own config the day you have it.

That last point matters most. "What a hobbyist budget buys" was always the real research
question in `vision.md`. This is a sharper version of it — and one you can actually run.

---

## 8. Summary

| Question | Answer |
|---|---|
| Do I need Perplexity/Exa? | **No.** They're primary-ladder fallbacks AskNews preempts. |
| Do I have research breadth? | **Yes — six providers, ~$0.03/question.** M6 wants two. |
| Where does the money go? | **Forecaster reasoning tokens — 64%.** Research is 10–13%. |
| Must I choose cheap models over frontier? | **No.** The 10× spread is mostly *effort tiering*, not model tier. |
| What config fits $50? | 3 frontier families at reduced effort ≈ **$0.70–1.10/q → 45–70 questions.** |
| Where do Groq / free models go? | **The dev/test lane**, never the forecaster slot. |
| Best single cost cut? | **Retire gap-fill v1** (~$0.55/q, duplicates v2). |
| What should I verify first? | **That AskNews is live for your season.** |
