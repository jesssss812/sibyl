# Sibyl — Product Vision

> **For coding agents:** This is the top-level "why." When two decisions conflict and the other docs don't settle it, resolve it here. [`design-principles.md`](./design-principles.md) says what to build; [`eval-strategy.md`](./eval-strategy.md) says how to measure it; this doc says what matters more than what.

---

## What Sibyl is

A calibrated forecasting bot that competes in Metaculus tournaments, built to be **rigorously evaluated rather than to win**. Named for the classical prophetess who gave calibrated answers to high-stakes questions — and who, unlike Cassandra, was believed.

Third in a Venetian merchant-finance suite alongside **Rialto** (sell-planning) and **Florin** (budgeting). Sibyl is a standalone project; it does not integrate with either.

---

## Why it exists — the ranked purpose

**This ranking is the tie-breaker. When goals conflict, the higher one wins.**

1. **Build hands-on RAG / agent / eval skill** ← primary
2. **Produce an interview artifact and story**
3. **Genuinely forecast well** (money strictly deferred; see Non-goals)
4. **Be enjoyable to keep running long-term**

### What this ranking actually decides

**(1) over (3) — learning beats performance.** Where a choice would improve forecast accuracy but teach nothing, take the instructive path.

**Fork the bot; build the eval.** The owner directs coding agents rather than typing implementation, so skill here means *understanding architecture and making design decisions*, not authorship. That splits the work cleanly:

- **Fork** [nostreambot](https://github.com/No-Stream/metaculus-bot) (MIT, top open-source bot) for the forecasting engine. It's a solved problem with a known-good implementation; rebuilding it teaches architecture no faster than reading it does.
- **Build** the evaluation **inference layer** — *scope corrected 2026-08-26, see below.*

  > ⚠️ **This bullet originally claimed nostreambot had "no confidence intervals, no
  > paired comparison, and no significance testing anywhere" and scoped the contribution
  > at ~200 lines of net-new statistics. That was wrong**, and it was wrong because the
  > reading guide put `ablation/` on its skip list — which is exactly where the
  > statistics live. `ablation/scoring.py` already has a paired percentile bootstrap over
  > per-question deltas, a sign test, Wilcoxon signed-rank, and dual Brier + log scoring
  > (plus CRPS); `aggregate_paired` already slices by question type.

  What is **genuinely** missing, verified by repo-wide search: **MDE / power analysis**
  (zero occurrences), **horizon and category cohorts**, **loss-distribution / tail
  reporting**, and the fact that `backtest/analysis.py::aggregate_scores` is still
  means-only while the good statistics sit unused in a sibling package.

  So the contribution is not "build the inference layer". It is **generalise one that
  exists, wire it to the backtest path it was never connected to, and add the pieces
  that are genuinely absent** — and, more importantly, produce a *finding* with it. A
  harness is a tool; the deliverable is a measured answer. See
  [`improvement-plan.md`](decisions/2026-08-24-improvement-plan.md) Finding 1 and
  [`eval-strategy-gaps.md`](decisions/2026-08-24-eval-strategy-gaps.md) for the full audit.

**The comprehension checkpoint — non-negotiable.** Forking only serves purpose (1) if the understanding is real. Before modifying anything, produce **docs/architecture-notes.md** — not yet written, and tracked in [roadmap.md](roadmap.md) — a from-scratch account of how nostreambot works — the agent loop, where research enters, how forecasts aggregate, where the seams are, what you'd change. **Written by the owner, not generated.** If that document can't be written, the system hasn't been understood, and working code doesn't compensate.

This doubles as interview material: *"here's a production forecasting system and how it's wired"* is a stronger answer than *"I wrote a bot."*

> **Relationship to `design-principles.md`:** that doc's "implement without invention" applies to **technique selection** — use the evidence-backed approaches (model-family ensembling, agentic search, Platt scaling) rather than inventing alternatives. Forking a reference implementation is consistent with it.

**(2) over (3) — a clean negative result beats an unexplainable good one.** A well-measured finding you can walk an interviewer through is worth more than a better score you can't account for.

**(3) is real but subordinate.** Sibyl should honestly try to forecast well. It just shouldn't buy accuracy with money, complexity, or time that the top two goals need.

---

## Definition of success

**A capability, not a number:**

> *Three months from now, I can build and evaluate a multi-agent LLM system cold — and explain the evaluation with conviction.*

Explicitly **not** success criteria: tournament rank, prize money, beating the crowd median, P&L. Per the [Prediction Tournament Paradox](https://arxiv.org/pdf/1903.02131), a single season's rank is mostly noise anyway — the expected winner of a 300-contestant tournament is often genuinely mid-ranked and lucky.

---

## The strategic fork: eval artifact, not leaderboard rank

**Decided: the evaluation is the product. The bot is the thing being evaluated.**

Rationale:
- Rank is noise-dominated at achievable sample sizes and can't be defended in an interview.
- The playbook for rank is public and well-trodden — executing it produces a competent bot, not a differentiated project.
- Rigorous evaluation is the owner's actual differentiator and the market's scarcest LLM skill.
- The field has an **open question** we're positioned to answer: *does a cheap, leaky, hobbyist-grade backtest predict live performance?* FutureSearch answered this at industrial scale ($2k/run, 1,907 questions, frozen corpus). Nobody has answered it for the budget a solo builder actually has.

---

## Identity: the deliberately cheap, rigorously measured forecaster

The owner is indifferent between "mini-Preseen replica" and a distinct identity, so this is derived from the constraints rather than preference — **revisit freely if it stops fitting.**

Winning bots spend ~28 LLM calls and ~$1.40 per question. **Sibyl won't.** Budget is capped (below), so Sibyl will run leaner and will not be competitive for prizes. That's accepted, not regretted.

Making a virtue of it: **Sibyl is a cheap forecaster that knows exactly how good it is.** Every forecast ships its reasoning and its calibration track record. The interesting claim isn't "Sibyl is accurate," it's "Sibyl's confidence is honest, and here's the measurement" — plus "here's what a hobbyist budget can and can't buy," which is genuinely unanswered.

**Design implication — model tier is configuration, not architecture.** Start cheap, but keep the option to swap in frontier models via a **single config change and no code change**. nostreambot already routes through OpenRouter, so this is nearly free. Going all-in to compete later must never require a rewrite.

This turns the budget constraint into the flagship experiment. Because the swap is one line, **"cheap model vs frontier model, paired on identical questions"** becomes a first-class paired comparison — and it measures exactly *what a hobbyist budget buys*, the open question above. The cost instinct and the research question turn out to be the same thing.

---

## Scope

**Generalist.** Sibyl forecasts whatever Metaculus serves — geopolitics, economics, science, tech. No domain specialization.

Rationale: the tournament question mix isn't ours to choose, and a domain filter would shrink an already small sample, directly harming the eval work that is the point. *(Revisit only if a cohort analysis shows a large, consistent per-category effect.)*

---

## Audience and surface

**Now:** a private instrument. Audiences are the builder and, later, an interviewer hearing the story. No users, no public product.

**Later (explicitly deferred, deliberately not designed away):** a UI surfacing what the bot is doing — reasoning traces, forecast history, calibration curves. [FutureSearch's evals page](https://evals.futuresearch.ai) is the reference for the kind of thing worth building eventually.

**Design implication:** keep forecasts, reasoning, traces, and scores in **structured, queryable storage from day one** (the JSONL experiment log in `eval-strategy.md`). Don't build UI now; don't make UI impossible later. A surfacing layer should be readable off existing artifacts without re-plumbing.

---

## Load-bearing principles

Two, chosen deliberately. These are non-negotiable and coding agents should treat them as constraints.

### 1. Right tool for the right problem

No technique enters Sibyl because it's fashionable or because the owner wants practice with it. RAG, agents, ensembles, and recalibration are here because forecasting genuinely requires them:
- **Retrieval** — evidence is fresh, distributed, and must be cited; a static context window cannot answer these questions.
- **Agentic search** — iterative beats one-shot by a measured 3.6× in Brier.
- **Ensembling** — decorrelation across model families is a measured effect.
- **Recalibration** — the output is a probability, and probabilities are only meaningful when calibrated.

If a proposed component can't pass this test, don't build it.

### 2. Calibration before cleverness

Honest uncertainty outranks sophistication. A simple bot that knows how good it is beats a clever bot that doesn't. In practice:
- Recalibration and capping are **core**, not polish.
- Every forecast is scored and logged; unscored forecasts are not a feature.
- Never ship an accuracy claim without its interval, sample size, and caveats.

### Where the owner's other principles live

*No-silent-degradation* and *scope discipline* were not selected as vision-level principles, but both are already binding elsewhere and must not be dropped:
- **No silent degradation** is `eval-strategy.md` **Rule 6** — a *mathematical* requirement, since paired comparison is invalid unless both arms answered identical questions. Silent fallbacks corrupt the statistics, not just the hygiene.
- **Scope discipline** is enforced by the timebox and non-goals below.

---

## Non-goals

**1. An impressive-but-uncalibrated demo.**
The failure mode where Sibyl looks sophisticated and produces confident numbers nobody has validated. Guard: no accuracy claim without interval, n, and caveats; no forecast that isn't logged and scoreable.

**2. A perfectionism sink that eats interview prep.**
Interview preparation is the priority this project serves — the moment Sibyl competes with it, Sibyl loses. Guard: the timebox below is hard.

**3. Real-money betting.** (Standing, from the project brief.)
Out of scope as an *evaluation* decision, not merely a risk one: P&L confounds forecast skill with position sizing, timing, fees, and adverse selection, so it cannot validate a forecaster. Any future money decision is gated on calibration evidence, capped as tuition, and executed manually — **never programmatically by this codebase.**

**4. A sprawling pseudo-startup.** No users, no product surface, no roadmap beyond the milestones.

---

## Timebox

The owner's trip is **Aug 8–14** (fully unplugged), which removes the middle of Phase 1. Timeline adjusted to work around it rather than compress into it.

**Anchor: the next MiniBench runs Aug 15 – Aug 28** — opening the day after the trip ends. That window is the target.

| Sprint | Dates | Work | Effort |
|---|---|---|---|
| **0 — Read** | Aug 6–7 | Clone nostreambot and *read* it. No repo surgery before a week away. Begin `architecture-notes.md`. | ~5 hrs |
| **— Trip —** | Aug 8–14 | Nothing. Rest is the point; this is not a debt to repay. | 0 |
| **1 — Get live fast** | Aug 15–22 | Finish `architecture-notes.md`, restructure the repo onto nostreambot, configure the cheap model tier, **submit**. Improve for the rest of the window. | ~15 hrs |
| **2 — Eval harness** | Aug 23 – Sep 6 | Paired bootstrap, dual scoring, calibration fitting — over forecasts already submitted. Low intensity, alongside Phase 2 interview prep. | ~3–4 hrs/wk |
| **Hard stop** | **Sep 6** | Maintenance only, ~1 hr/week thereafter. | — |

| | |
|---|---|
| **Effort budget** | ~30 hours total, hard cap 35 |
| **Money budget** | **≤ $50, entirely out of pocket** — summer-tournament API credits are exhausted; they refresh Sept/Oct |
| **Milestone 1** | Live and submitting by **Aug 16–17**, i.e. early in the MiniBench window — not at the end of the sprint |

**Milestone 1 moves earlier on purpose.** MiniBench questions open across Aug 15–28, so every day offline is questions permanently missed. Forking makes early launch cheap — getting live is mostly credentials and config. **Launch minimally, then improve inside the window** rather than polishing toward a late launch.

**Timing works out well:** MiniBench questions are short-fuse, so they resolve around end of August — exactly when Sprint 2 builds the eval harness. The harness gets real scored data the moment it exists.

**Sprint 2 runs during Phase 2 by design.** The eval harness is offline analysis over forecasts the bot is already producing, so it doesn't block the track record and can absorb interruption. If it starts competing with interview prep, **it stops** — non-goal #2 governs.

### Funding sequence — the constraint schedules the experiment

Credits are gone until Sept/Oct, so the $50 cap genuinely binds. Rough per-forecast economics: a frontier model running full agentic research costs on the order of **$1–3**; a cheap model with limited search is closer to **$0.05–0.20**. Across ~60 MiniBench questions that's **$60–180 versus $3–12**. Only one of those fits.

This forces a sequence that happens to be the right one anyway:

1. **Aug (own money, cheap tier)** — establish the live track record and the baseline arm. Stays well inside $50, including the extra runs the eval harness needs.
2. **Sept/Oct (credits refreshed)** — run the **frontier arm** on the same questions. That completes the cheap-vs-frontier paired comparison, funded by credits rather than cash.

The flagship experiment is therefore paid for by the credit refresh, and the budget constraint sets its schedule rather than blocking it.

**Also verify:** research/search providers (AskNews, Perplexity, Exa) bill separately from the LLM. Free tiers may or may not survive the loss of tournament credits — confirm before configuring the research stack.

**Known consequence, accepted:** every surveyed bot-maker reporting 80+ hours won prize money; non-winners cluster at low effort. At ~30 hours Sibyl will not place. That is consistent with the ranked purpose and is not a failure.

---

## Open decisions

- [x] ~~Trip timing~~ — resolved: **Aug 8–14**, timeline restructured around it.
- [x] ~~Fork vs write from scratch~~ — resolved: **fork nostreambot**, build the eval harness, gated on the comprehension checkpoint.
- [x] ~~MiniBench window~~ — resolved: **Aug 15 – Aug 28**. Sprint 1 targets launch inside it.
- [x] ~~Tournament API credits~~ — resolved: **exhausted for summer; refresh Sept/Oct.** The $50 cap binds; funding sequence above.
- [ ] **Search-provider costs** — confirm which research APIs have usable free tiers without tournament credits.
- [ ] **Repo restructure approach** — how to move `sibyl` from the metac-bot-template fork onto nostreambot. Decide Aug 15, not before the trip.
- [ ] Money budget of $50 — confirm or adjust.
- [ ] Identity framing ("deliberately cheap, rigorously measured") — derived, not chosen. Adopt or replace.
