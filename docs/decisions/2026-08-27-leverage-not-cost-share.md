# Where to spend model capability — leverage, not cost share

> 2026-08-27. The reasoning behind the changes in
> `feat/agreement-telemetry-and-model-refresh`. Companion to
> [`cost-tiered-plan.md`](2026-08-27-cost-tiered-plan.md), which it partly supersedes.

---

## The frame

Cost share tells you where the money goes. It does not tell you where capability matters.
Three properties decide that:

| | |
|---|---|
| **Cost share** | what the slot spends |
| **Leverage** | how many downstream decisions its output touches |
| **Substitutability** | whether a weaker model does the job as well |

Economise where cost is high, leverage low, substitutability high. **Never economise where
leverage is high and cost is low** — that is free damage.

## Why this reframes the cost thesis

Measured decomposition: the forecaster fan-out is **64%** of per-question cost. So the
money sits in exactly the slot `design-principles.md` **S1** says not to economise on.

That is the whole tension. A naive "use cheap models" plan aims at the one place the
strongest evidence says don't, and every other slot combined is a third of the bill — half
of it already free.

The way out is not to cheapen the forecast. It is to notice that **the fan-out is
redundant and the funnel is not.**

> Three forecasters, median, outliers absorbed — that is a structure designed to survive
> one bad input. The briefing and the selection are singular: one call each, everything
> downstream conditioned on them, no redundancy to absorb a mistake.
>
> **Redundant work is where cheap is affordable. Singular work is where it is not.**

## The measured case for the summarizer

Not a hunch — this is the best-evidenced slot in the repo.

- 2026-07-17 content audit: the AskNews section is **44% of bundle tokens, 57% padding**,
  and **stale or directionally wrong in 5 of 10 judged questions** — while smaller sections
  had the right answer.
- 2026-07-18 follow-up attributing the failures: **SUMMARIZE implicated in 5/5, primary in
  3**. FETCH primary in 1, upstream primary in 1.
- The decisive line: *"in four of five cases the briefing **visibly contains** the fresh or
  correct material it then mishandles."* Raw articles were fresh — 48% of cited dates within
  30 days, <2% older than 180d.
- The measured replay shows the summarizer **expanding** its input: 17,858 chars of articles
  → 27,111 chars of briefing, **+52%**. A compression step that inflates.
- The first half of the briefing already holds **77%** of its distinct numbers.

The failure is **salience inversion** — stale on-topic facts outranking fresh adjacent ones.
Ranking salience under a length budget is a reasoning task, and reasoning effort is exactly
what that slot was not being given: it ran at **low**, the cheapest tier, on the one call
whose output all three frontier forecasters read.

Raised to **high**. Bounded downside: ~$0.25 and ~7% of per-question spend.

⚠️ **Two unmeasured changes now sit on one slot.** The 2026-07-18 prompt reform (R1–R5)
targeted these same failures and was never validated either. Measure effort and prompt
together, or the result attributes to neither.

## What we did NOT change, and why

**Raw-URL and boilerplate trimming — already done.** The 31.6% Gemini / 22.4% native-search
URL overhead is real in the 2026-07-17 audit, but commit `d8d4cc4` (2026-07-19, *"slim
research citations"*) replaced the opaque `grounding-api-redirect` blobs with source domain
labels in both providers. The audit figures are **two days stale**. The remaining AskNews
`Credibility:`/`Source:` apparatus (~6%) comes from AskNews's own article payloads and is
provenance the summarizer prompt's trust-ladder vocabulary actively uses.

**Gap-fill v1 — turned back ON.** Disabling it on 2026-08-26 was a mistake. That came from
the token audit (ranked it #1 to cut, ~$0.55/q, "both do the same job"). The content audit
one day later found the reverse: **v1 is the most load-bearing section per token — 59%
unique content**, carrying the decisive single-source fact in most sampled questions,
numeric novelty mean 37% (range 0–77%). bottega:FUTURE.md sets a gate — compare v1 vs v2 on ~20 prod
questions where both ran — and that comparison has never been run. Two audits a day apart
reached opposite conclusions; the content-level one is the stronger evidence.

**The `-pro` model variants — not adopted.** `sol-pro` / `terra-pro` / `luna-pro` are the
same underlying models served with `reasoning.mode=pro`, at the same list price. That is a
second reasoning-control mechanism alongside the `effort` ladder the repo already uses and
has measured. Running both would blur the config-era boundaries the residual analysis keys
on, for no clear gain.

**The forecaster count — left at 3.** 6→3 was measured non-inferior on binary (+0.01) and MC
(−0.14); numeric leans toward the full roster by +3.24 log pts/Q, 95% CI [−2.5, +9.1], and
that lean rests on **2 questions** (jackknifing them halves the mean). 3→2 has never been
measured, and the median only starts rejecting outliers at n=3 — at n=2 median equals mean.
S4 is also against it directly. The open question is not 3 vs 2; it is whether **numeric**
deserves a 4th.

## What the instrumentation is for

`compute_spread` has always run on every multi-forecaster question, but its value was logged
only when conditional stacking **triggered** — and stacking is disabled on all three types.
So the spread has been computed and discarded on every question this bot has ever forecast.

`FORECASTER_SPREAD` now records it unconditionally. It decides three open questions:

1. **Is 3 the right count?** High agreement ⇒ the ensemble is buying tail insurance. Low
   agreement ⇒ the median is doing real work and cutting would cost more than the
   frozen-triple analysis implies.
2. **Would disagreement-triggered escalation ever fire?** Any "spend more where they
   disagree" policy needs the base rate before it can be designed or costed.
3. **Do the thresholds still fit?** 0.15 / 0.20 / 0.15 were tuned on a 6-member roster and
   have never been rechecked against the triple.

**It measures and does not decide.** The distribution should be known before a policy is
written against it — and a measurement that quietly also acts cannot be trusted as one.

## On cost-tiered routing (C1)

Deprioritised, and the reason is worth recording. amigon21-bot's 56% cheap-dispatch figure
— the number that made the idea attractive — comes from a system running **entirely on Qwen
14B with no frontier tier at all**. It measures how often cheap agents agree *with each
other*, not whether that agreement was right, and their own plan lists requesting credits to
test frontier models while naming model choice "the single biggest accuracy differentiator".
Their headline Brier of 0.017 is on a **10-question** benchmark.

The trap is correlated error: cheap models share failure modes, so they can agree
confidently and wrongly — plausibly on exactly the questions a frontier model gets right,
which would route backwards while looking like a saving.

**What survives is the inverse.** Conditional stacking already measures spread among the
three *frontier* forecasters — models whose judgment we trust, whose forecasts are already
paid for, so the signal is free and carries no correlated-error problem. The useful policy
is not "economise where they agree" but **"spend more where they disagree"**, and the
natural first target is numeric high-spread questions, where the frozen-triple watch says
the only measured gap lives.

That policy needs the spread distribution first. Hence: instrument now, decide later.
