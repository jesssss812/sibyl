# Architecture overview

This is the map a new contributor should read first. It walks through what happens
end to end when the bot forecasts one question, names the entry points, and points
you at the deeper docs for each subsystem. Read this, then dive into the specific
page you need.

The bot is a fork of the Metaculus starter template built on the `forecasting-tools`
framework. The core idea: for each question, gather research from several providers,
run an ensemble of large language models to produce independent forecasts, then
combine them into one prediction and publish it as a comment on Metaculus.

## Entry points

Three files form the startup chain:

- `main.py` — a thin shim. It re-exports `SibylBot` (for anything that
  imports it) and, when run directly, calls `cli.main()`.
- `sibyl/cli.py` — the command-line entry point. It parses `--mode`
  (`tournament`, `minibench`, `metaculus_cup`, `quarterly_cup`, `test_questions`),
  builds the LLM roster dict from `llm_configs.py`, constructs a `SibylBot`
  with `aggregation_strategy=CONDITIONAL_STACKING`, and runs the mode-specific
  forecast loop. It also wires credit telemetry and decides the process exit code:
  the run exits non-zero when any degradation counter fired (`alertable_count` on
  `SibylBot` sums them: dropped forecasters, questions that failed to
  publish, stacker fallbacks, research-provider and summarizer failures, gap-fill
  v2 errors, and prediction-market degradation) or the donated OpenRouter key
  dropped below the refill floor. Credit-caused alerts are suppressed until
  2026-09-10 while the operator self-funds the season — see "Credit alerting is
  suppressed" in `docs/operations.md`. See `main` in `cli.py`.
- `sibyl/forecasting/bot.py` — the bot itself. `SibylBot` subclasses the
  framework's `ForecastBot` and owns the per-question pipeline. The method to read
  first is `_research_and_make_predictions`.

Publication happens inside the framework's forecast loop, not in `cli.py`. Every
question that clears the min-forecasters guard is already on Metaculus by the time
`cli.py` decides the exit code.

## The per-question pipeline

Everything below runs once per question inside `_research_and_make_predictions`,
under a shared per-question wall-clock budget (`PER_QUESTION_WALL_CLOCK_DEADLINE`,
sized to finish just inside the 60-minute Metaculus close window). Research,
forecaster fan-out, aggregation, and publish all draw from that one budget.

```
                        one Metaculus question
                                 │
                                 ▼
        ┌────────────────────────────────────────────────┐
        │  1. RESEARCH  (ResearchOrchestrator.run_research)│
        │  Providers fan out in parallel:                  │
        │    - AskNews (primary, summarized to a briefing) │
        │    - OpenAI native search                        │
        │    - Claude web search (Anthropic web_search)    │
        │    - financial data (yfinance / FRED)            │
        │    - prediction-market snapshot                  │
        │    - resolution-source fetcher                   │
        │    - reference class (resolved Metaculus Qs)     │
        │  Each is independently env-gated.                │
        └────────────────────────────────────────────────┘
                                 │  research bundle
                                 ▼
        ┌────────────────────────────────────────────────┐
        │  2. GAP-FILL  (two passes, run concurrently)     │
        │    v1: analyzer finds gaps → parallel searches   │
        │    v2: agentic tool loop (bounded, driver LLM)   │
        │  Each appends its own section; both soft-fail.   │
        └────────────────────────────────────────────────┘
                                 │  enriched bundle
                                 ▼
        ┌────────────────────────────────────────────────┐
        │  3. FORECASTER FAN-OUT                           │
        │  N forecaster LLMs run in parallel, each capped  │
        │  by FORECASTER_SOFT_DEADLINE. Type-specific      │
        │  runner per question (binary / MC / numeric).    │
        └────────────────────────────────────────────────┘
                                 │  N reasoned predictions
                                 ▼
        ┌────────────────────────────────────────────────┐
        │  4. MIN-FORECASTERS GUARD                        │
        │  Fewer than MIN_FORECASTERS_TO_PUBLISH valid     │
        │  → skip this question, keep the batch going.     │
        └────────────────────────────────────────────────┘
                                 │
                                 ▼
        ┌────────────────────────────────────────────────┐
        │  5. AGGREGATION  (CONDITIONAL_STACKING)          │
        │  Measure spread across forecasters.              │
        │  Low spread OR stacking disabled → MEDIAN.       │
        │  High spread + stacking on → crux + targeted     │
        │  search + stacker LLM rewrite.                   │
        └────────────────────────────────────────────────┘
                                 │  one aggregated prediction
                                 ▼
        ┌────────────────────────────────────────────────┐
        │  6. PUBLISHED COMMENT                            │
        │  Per-model bullets + full research + diagnostics.│
        └────────────────────────────────────────────────┘
```

### 1. Research fan-out

`run_research` (`forecaster.py`) delegates to a `ResearchOrchestrator`
(`research/orchestrator.py`). It picks one primary provider by priority (AskNews in
prod, then Exa, then Perplexity, then a stub) and runs several additional providers
alongside it in parallel, each behind its own env flag. AskNews returns raw article
text and gets summarized into an analyst briefing inline; the other providers write
their own prose and pass through as-is. See [research.md](research.md) for provider
selection, gating, and the shared-vs-personal API-key routing.

### 2. Gap-fill (two passes)

After the first-pass bundle is assembled, two gap-fill passes run concurrently in one
`asyncio.gather` inside `run_research` (`orchestrator.py`), so the research phase costs `max(v1, v2)`
in wall-clock, not the sum:

- **v1** (`research/targeted.py`): an analyzer LLM reads the bundle, names up to a few
  factual gaps, then resolves each with a parallel web search. Appends a
  `## Targeted Gap-Fill (second pass)` section.
- **v2** (`research/agentic/`): a bounded agentic tool loop. A driver LLM privately
  dry-runs the forecast to find what to verify, then iterates over search/fetch/read
  tools within a wall deadline and tool-call cap. Appends an
  `## Agentic Research Findings` section, led by a corrections block.

Both are wrapped so a failure in one never zeroes the other or kills the forecast.
Both are on in prod. See [agentic_gap_fill.md](agentic_gap_fill.md) for the v2 loop,
tools, and telemetry.

The orchestrator also builds a provider-diagnostics block that is deliberately
withheld from the forecaster-facing text (so it never pollutes prompts) but is
re-attached to the published comment later. This is the "diagnostics seam": the
orchestrator's `pop_provider_diagnostics`, which `_research_and_make_predictions` in
`forecaster.py` drains once the research phase is done.

### 3. Forecaster fan-out

Each forecaster LLM runs through `_forecaster_with_soft_deadline` (`forecaster.py`),
which caps a single model at `FORECASTER_SOFT_DEADLINE` so one stuck
model can't hold the whole question. `_make_prediction` dispatches to the
type-specific runner (`forecaster_runners.py`) for binary, multiple-choice, or numeric
questions. The N coroutines are gathered under the shared wall-clock budget by
`_gather_predictions_with_wall_clock` (`forecaster.py`), which cancels any
forecaster still pending at the deadline and counts the drop.

The ensemble is a handful of forecaster LLMs, one per vendor. The exact roster
rotates often, so **read `sibyl/config/models.py` for the current list** rather
than trusting any names written here. Support models (summarizer, parser, stacker,
disagreement analyzer) live in the same file.

Each forecaster emits its answer inside a fenced ```json STRUCTURED FORECAST block,
which is parsed by a deterministic extraction ladder (`value_extraction.py`). Numeric
questions produce the canonical percentile set (`STANDARD_PERCENTILES` in
`numeric/config.py`), turned into a PCHIP CDF on the `PCHIP_CDF_POINTS` grid. See
[numeric_pipeline.md](numeric_pipeline.md) for the percentile-to-CDF machinery and its
bound/step constraints.

### 4. Min-forecasters guard

If fewer than `MIN_FORECASTERS_TO_PUBLISH` (`constants.py`) forecasters
returned a valid prediction, the ensemble is too degraded to trust. The question is
skipped and a counter bumps for end-of-run alerting, but the rest of the batch and all
other publications continue. The guard lives in `_research_and_make_predictions`
(`forecaster.py`).

When the threshold is 1, a lone survivor publishes: the median of one forecast is
that forecast. Because the spread metrics in `spread_metrics.py` require at least two
predictions and raise otherwise, `route_after_forecasts` (`stacking_route.py`)
short-circuits the n == 1 case before spread computation and stacking and hands the
single prediction straight to the aggregator. Exception-driven drops still bump the
degradation counters, so a run thinned to one model reddens CI rather than silently
withholding the question.

### 5. Aggregation: CONDITIONAL_STACKING

The default strategy is `CONDITIONAL_STACKING` (set in `cli.py`'s `main`). Conceptually:

- Compute the spread across the N forecasts (`spread_metrics.compute_spread`).
- **Low spread**: return the MEDIAN of the raw per-model predictions.
- **High spread**: extract the disagreement crux with the analyzer LLM, run a targeted
  search on it, then hand the full base-model reasonings plus that research to a
  stacker LLM that rewrites the forecast. If the stacker fails, it falls back to a
  second stacker LLM, then to MEDIAN.

Spread thresholds live in `constants.py`, one per question type:
`CONDITIONAL_STACKING_BINARY_PROB_RANGE_THRESHOLD` (a probability range),
`CONDITIONAL_STACKING_MC_MAX_OPTION_THRESHOLD` (a max per-option spread), and
`CONDITIONAL_STACKING_NUMERIC_NORMALIZED_THRESHOLD` (a normalized percentile spread).

**Stacking is disabled in production.** All four workflow YAMLs set
`BINARY_STACKING_ENABLED`, `MC_STACKING_ENABLED`, and `NUMERIC_STACKING_ENABLED` to
`false`, so even when spread exceeds the threshold, the per-type gate in
`route_after_forecasts` (`stacking_route.py`) bypasses the stacker and forces the
MEDIAN path. In effect, **prod runs MEDIAN of the raw forecasts.** The stacker chain
stays fully wired and is exercised in
backtests and ablation runs. The aggregation dispatch, base-combine, and stacker
fallback ladder all live in `sibyl/aggregation/pipeline.py`
(`AggregationPipeline`). The conditional-stacking path runs the combined result
through a Platt-calibration hook (`_post_process_final` in
`aggregation_pipeline.py`), but that hook is gated by `PLATT_CALIBRATION_ENABLED`,
which is unset in every workflow, so in prod `apply_platt_calibration`
(`post_processing.py`) is a passthrough.

Two post-aggregation rails live in `post_processing.py` and both dispatch on question
type internally: `apply_platt_calibration` and `apply_resolution_state_guard`. Most
paths run both, in that order, via `_post_process_final` in `aggregation_pipeline.py`.

The **open-vs-resolved guard** (`resolution_state_guard.py`) is binary-only and pulls
an extreme probability back to 0.05/0.95 when the question is still open with at least
a quarter of its open→close window left — the case where near-certainty is more likely
a misread than earned. Firing is normal operation: it logs a `RESOLUTION_GUARD` marker
(spec'd in `sibyl/telemetry/markers.py`) and bumps no alertable counter.

**Two `_base_combine` returns take a different route.** The single-prediction branch
(the `MIN_FORECASTERS_TO_PUBLISH = 1` survivor path) and the plain-`STACKING` branch
deliberately skip Platt — see the module docstring of
`tests/test_main_platt_integration.py` — but they DO get the guard, via `_guard`. The
single-survivor case is the one A3 exists for: one model's raw probability with no
median to temper it.

**On the guard's size.** Every binary prediction is already clamped to
`[BINARY_PROB_MIN, BINARY_PROB_MAX]` = [0.02, 0.98] *upstream*, per forecaster, in
`forecaster_runners.py` and `stacking.py`. So the worst value the guard can see is
0.98 and its maximum correction is **0.03**. It is cheap insurance against a rare,
expensive misread, not a large intervention — and it cannot move a pooled mean at
tournament sample sizes, so measuring it needs a subgroup analysis restricted to the
questions where it fired.

### 6. Published comment

The framework assembles the comment: per-model forecast bullets (annotated with model
names so per-model attribution survives comment trimming), the full research bundle,
the targeted-research section if stacking fired, and the provider-diagnostics block
re-attached via the seam. These published comments are also the durable per-model
record the performance-analysis tooling later parses.

## Where the pieces live

| Concern | Module |
|---|---|
| Startup / CLI | `main.py`, `sibyl/cli.py` |
| Per-question orchestration | `sibyl/forecasting/bot.py` |
| Post-fan-out aggregation routing | `sibyl/aggregation/routing.py` |
| Drop attribution / degradation counters | `sibyl/telemetry/drops.py`, `degradation_counters.py` |
| Research fan-out | `sibyl/research/orchestrator.py`, `research/providers.py` |
| Gap-fill v1 / v2 | `research/targeted.py`, `research/agentic/` |
| Forecaster runners | `sibyl/forecasting/runners.py` |
| Value extraction | `sibyl/forecasting/value_extraction.py` |
| Numeric CDF | `sibyl/forecasting/numeric/` |
| Aggregation + stacking | `sibyl/aggregation/pipeline.py`, `stacking.py` |
| Model roster (source of truth) | `sibyl/config/models.py` |
| Prompts | `sibyl/forecasting/prompts.py` |
| Constants / thresholds / env flags | `sibyl/config/settings.py` |

## Related docs

- [research.md](research.md) — research providers, gating, API-key routing.
- [numeric_pipeline.md](numeric_pipeline.md) — percentiles to PCHIP CDF, bounds, steps.
- [agentic_gap_fill.md](agentic_gap_fill.md) — the v2 agentic research loop.
- [operations.md](operations.md) — running the bot, workflows, cost discipline, credits.

## A note on cost

Any command that hits live LLM or research APIs spends real money and, in live modes,
publishes comments to Metaculus. Do not launch one without the operator's approval.
The free, self-contained paths (`make test`, `make lint`, `make format`,
`make check_credits`) are safe to run anytime. Details in
[operations.md](operations.md) and the repo's `AGENTS.md`.


---

## Agent-facing detail (relocated from `AGENTS.md`, 2026-08-26)

The sections below were inline in `AGENTS.md`, which is loaded on every coding-agent
invocation. They are reference material rather than always-true rules, so they live here
now and are read on demand.

Some of this overlaps the prose above — it was written for a different audience and is
kept verbatim rather than merged, so nothing is silently lost. Reconciling the two is
worthwhile follow-up, but it is an editorial pass, not a mechanical one.
### Ensemble

`FORECASTER_LLMS` in `sibyl/config/models.py` is the authoritative roster. It rotates frequently, so read it rather than trusting any copy — this file deliberately does not name the current members outside the dated history below. The standing design is latest-per-vendor: the newest frontier reasoning model from each major vendor, one slot each. Do NOT hardcode model names outside `llm_configs.py`. Providers: **mixed transport since 2026-08-31** — the Anthropic slot calls the Anthropic API directly (`anthropic/...`, billed to `ANTHROPIC_API_KEY`), while the OpenAI and Google slots route via OpenRouter. The donated-key story below is now historical: `OAI_ANTH_OPENROUTER_KEY` is spent for the season and unset, so every OpenRouter slot bills the personal `OPENROUTER_API_KEY`. (Historically: dropping grok (x-ai) ended routine personal-key forecaster spend, leaving only the Google slot pinned by the `DONATED_KEY_BLOCKED_GOOGLE_MODELS` blocklist on the personal key while the rest rode the donated key.) Alongside the roster shrink, `MIN_FORECASTERS_TO_PUBLISH` (constants.py) was lowered 3→2→1 over 2026-07-20; the operator accepts publishing on a single surviving forecaster (median-of-1 = that forecast), with `route_after_forecasts` (`stacking_route.py`) short-circuiting the n==1 case before spread computation + stacking (the `spread_metrics` helpers require ≥2 predictions and raise otherwise). Exception-driven drops stay CI-visible via the alertable counter, so a degraded single-forecaster publish still reddens CI rather than silently withholding the question.

Roster-change history — **all dates below are AUTHORING dates on the july15 branch; every one of them reached prod together in merge `b4e9df0` at 2026-07-21T17:07:37Z, which is the single era boundary** (see the merge-date rule under era-bucketing): 2026-07-15 — Fable-5 joined the forecaster roster (was stacker-only; stacking disabled in prod made it idle) and opus-4.6 retired, keeping n=6 at a 2 Anthropic / 2 OpenAI / 1 Google / 1 xAI balance. 2026-07-20 (first change) — Fable-5 PULLED from the forecaster roster and the stacker after it returned `message.content=None` on 4/4 attempts for Q14333's numeric forecast + a truncated no-JSON-block output on Q578 in the 2026-07-19 test_bot run (suspected content classifiers refusing certain question content — fast deterministic empty completions, not timeouts); opus-4.7 took the slot, keeping n=6. Reconsidering fable-5 is a tracked follow-up (bottega:FUTURE.md). 2026-07-20 (second change, current) — dropped from 6 to the 3-member latest-per-vendor triple, removing gpt-5.5, opus-4.7, and grok-4.5. Two adversarially-verified analyses (`bottega:scratch/ensemble_3member_audit_2026-07-20/` + `bottega:scratch/ensemble_power_model_2026-07-20/`) found the triple non-inferior on binary/MC and only a fragile numeric lean toward the full roster (+3.24, 95% CI [-2.5, +9.1], P(loss>1pt/Q)=0.80, driven by 2 questions); accepted as a ship-and-watch bet — see bottega:FUTURE.md "Frozen-triple numeric watch". This second change supersedes the first as the roster in effect, but both landed in the same merge, so residual analysis sees ONE boundary at 2026-07-21T17:07:37Z — no prod run ever used the intermediate opus-4.7 roster.

Support models (also in `llm_configs.py`):

- **Stacker**: `STACKER_LLM` — the Anthropic slot, since 2026-07-20 when fable-5 was pulled from both roles (see the 2026-07-20 roster-change note above; it uses effort-based adaptive thinking rather than a max_tokens budget; the stacker chain remains configured but prod-disabled. Was `claude-fable-5` from 2026-07-07 to 2026-07-20). Falls back to `STACKER_FALLBACK_LLM`, deliberately a different vendor so an Anthropic stall doesn't take both attempts down; it stays one effort tier below the primary because it fires late on the critical path under `STACKER_FALLBACK_SOFT_DEADLINE`. Both are `allowed_tries=1` — on stall, we fall back rather than burn budget retrying the same provider. OpenRouter's full effort enum (live-verified 2026-07-15): none/minimal/low/medium/high/xhigh/max; the xhigh slots are the Anthropic forecaster and the stacker. The OpenAI forecaster dropped xhigh→high on 2026-07-20 evening (it is ~70% of forecaster reasoning spend and the high→xhigh premium is unmeasured — the Anthropic slot keeps xhigh as the remaining premium bet; see bottega:FUTURE.md "Price the high→xhigh reasoning-effort premium"). The Google forecaster has no xhigh tier and runs at provider defaults. "max" is Anthropic-only (one tier above xhigh; OpenAI's ceiling is xhigh and rejects max upstream) — held back on the Anthropic slots for latency.
- **Disagreement analyzer**: `DISAGREEMENT_ANALYZER_LLM` (low-effort crux extractor; quality drives targeted-search query, running under `CRUX_SOFT_DEADLINE`; sol→terra 2026-07-17 per the blind role audit — terra 2nd, sol 3rd, at −49% cost, and the role fires rarely with stacking disabled in prod).
- **Summarizer / researcher**: `SUMMARIZER_LLM` (aliased as `RESEARCHER_LLM`; low effort, deterministic; sol→terra 2026-07-18 operator decision — AskNews is an auxiliary source per the content audit (16% unique content), the role audit had sol over terra only at "MARGINAL EDGE", and 4/5 audited briefing failures were prompt-era not model-tier; terra −43% cost, ~50s vs ~118s wall). The summarizer prompt carries the 2026-07-18 AskNews-audit rules: a hard per-article relevance gate (off-topic articles DROPPED to a one-line "Screened out as not decision-relevant" list), recency-first ordering (lead with the newest resolution-relevant facts, don't mirror the raw Historical/Recent input structure), supersession + quote-the-deadline-inputs arithmetic transparency, an evidence-age disclosure opening the briefing ("Newest directly-relevant article: ..."), and a proportionality rule (length tracks decision-relevant content, not article count).
- **Parser**: `PARSER_LLM` (low effort, deterministic; a capability-saturated extraction task, so it sits on the cheapest tier that saturates it — the per-token comparison behind that choice is in the `llm_configs.py` comment).

### CONDITIONAL_STACKING (default)

`AggregationStrategy.CONDITIONAL_STACKING` (set in `main`, `sibyl/cli.py`). Note: this describes the code default — stacking is currently DISABLED in prod, since all four prod workflow yamls set `BINARY_STACKING_ENABLED`/`MC_STACKING_ENABLED`/`NUMERIC_STACKING_ENABLED` to `'false'` (the chain stays live in backtests/ablation). Behavior:

- Compute spread across the N forecasters via `spread_metrics.compute_spread`.
- If spread ≤ threshold → return **MEDIAN** of raw per-model predictions (base-combine via `_base_combine`, `aggregation_pipeline.py`).
- If spread > threshold → extract the **disagreement crux**, run **targeted search** (OpenAI native search on the same `NATIVE_SEARCH_*` model / effort / verbosity / timeout settings as the native-search provider, described under Research providers below), then invoke the **stacker LLM** with the full base-model reasonings + targeted research (`stacking.run_stacking_{binary,mc,numeric}`).
- Stacker fallback chain: primary `STACKER_LLM` under `STACKER_SOFT_DEADLINE` → `STACKER_FALLBACK_LLM` under `STACKER_FALLBACK_SOFT_DEADLINE` → MEDIAN. (`_stacking_aggregate`, `aggregation_pipeline.py`.)

Thresholds (all in `sibyl/config/settings.py`):

- Binary: probability range (max − min) ≥ `CONDITIONAL_STACKING_BINARY_PROB_RANGE_THRESHOLD`.
- MC: max per-option spread ≥ `CONDITIONAL_STACKING_MC_MAX_OPTION_THRESHOLD`.
- Numeric: normalized percentile spread ≥ `CONDITIONAL_STACKING_NUMERIC_NORMALIZED_THRESHOLD`.

