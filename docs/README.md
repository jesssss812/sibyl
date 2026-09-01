# Documentation

Two families live here, and the distinction matters when you are deciding whether a
document describes reality.

**[How the bot works today](#how-the-bot-works-today)** describes the code as it is.
If one of these disagrees with the code, the doc is wrong and should be fixed.

**[Planning and evaluation strategy](#planning-and-evaluation-strategy)** is intent:
goals, evidence reviews, and specs for work that is partly or wholly unbuilt. These
describe what *should* be true, so a mismatch with the code is expected rather than a
bug — check the dates before treating any of it as current.

---

## How the bot works today

| Doc | What it covers |
| --- | --- |
| [architecture.md](architecture.md) | How a question flows end to end: research fan-out, the forecaster ensemble, the min-forecasters guard, aggregation (MEDIAN in prod, stacking disabled), and the two post-aggregation rails (Platt, open-vs-resolved guard). |
| [research.md](research.md) | The providers that build each question's briefing: AskNews (primary), OpenAI native search, Claude web search, financial data, prediction markets, resolution-source fetcher, time-series anchor, reference-class retrieval, plus both gap-fill passes. All fan out in parallel, each independently env-gated. |
| [agentic_gap_fill.md](agentic_gap_fill.md) | Gap-fill v2: the bounded agentic research loop — dry-run, target selection, tool iteration, citation-only findings artifact. |
| [numeric_pipeline.md](numeric_pipeline.md) | Percentiles to PCHIP CDF: extraction, sanitizing, tail widening, step enforcement, bound pinning, discrete snapping, the unit-mismatch guard. |
| [performance_analysis.md](performance_analysis.md) | Evaluating the live bot's calibration against real resolutions, and recovering per-model forecasts from published Metaculus comments. Both pulls are read-only and free. |
| [probabilistic_tools.md](probabilistic_tools.md) | The `probabilistic_tools/` package and the tool-runner that exposes it to forecaster models. |
| [operations.md](operations.md) | Running it: `uv` setup, the shared-vs-personal key model, workflows and their env flags, credit telemetry, backtesting, and the cost gate (any live or paid run needs operator approval). |

## Planning and evaluation strategy

> These are the owner's documents for the Sibyl project, which forked this bot. They
> state intent and record evidence reviews. Several describe work that does not exist
> yet; two are dated point-in-time audits. Read the date before acting on any of them.

| Doc | What it covers |
| --- | --- |
| [planning/vision.md](planning/vision.md) | The ranked purpose and what wins when goals conflict. The tie-breaker when other docs don't settle a question. |
| [planning/design-principles.md](planning/design-principles.md) | Evidence-graded architecture decisions (Strong / Moderate / Suggestive) drawn from the Metaculus analyses and the Fall-2025 bot-maker survey. |
| [planning/eval-strategy.md](planning/eval-strategy.md) | The spec for how anything is measured: paired comparison, power checks, leakage handling, cohort slicing, holdout, no-silent-degradation. |
| [planning/calibration-primer.md](planning/calibration-primer.md) | Brier vs log score, reliability diagrams, capping, Platt scaling, extremizing — and why Metaculus grades on log. |
| [planning/budget-strategy.md](planning/budget-strategy.md) | Measured per-question costs and which providers and model tiers fit a fixed budget. |
| [planning/nostreambot-reading-guide.md](planning/nostreambot-reading-guide.md) | Reading order for this codebase, written when it was still an external reference. *Partly superseded — see its header.* |
| [planning/improvement-plan.md](planning/improvement-plan.md) | *Audit, 2026-08-24.* What the fork already had vs. what was genuinely missing. |
| [planning/leverage-not-cost-share.md](planning/leverage-not-cost-share.md) | *2026-08-27.* Where to spend model capability: leverage over cost share, why the summarizer is the highest-leverage slot, and what was deliberately not changed. |
| [planning/cost-tiered-plan.md](planning/cost-tiered-plan.md) | *Plan, 2026-08-27.* Cost-tiered escalation, MC shuffling and extremization — sequenced so the assumption each rests on is tested before it is built on. |
| [planning/eval-strategy-gaps.md](planning/eval-strategy-gaps.md) | *Audit, 2026-08-24.* `design-principles.md` and `eval-strategy.md` checked rule by rule against the code. |
- [Moving Anthropic off OpenRouter, and Gemini search off Google](planning/anthropic-direct-swap.md) — why the Anthropic slots route direct, and what silently breaks if the OpenRouter kwargs come along.

---

## Repo-root references

- [README.md](../README.md) — install, configure, run.
- [AGENTS.md](../AGENTS.md) — always-true rules and hard prohibitions for coding agents.
  Detail lives in the guides above rather than inline.
- [FUTURE.md](../FUTURE.md) — the branch design log: shipped, planned, and rejected
  ideas. Read it for the "why", never for the current state. Expect dead paths in it.

Subsystem drivers live inside their packages: `python -m metaculus_bot.backtest`,
`python -m metaculus_bot.ensemble_analysis`, `python -m metaculus_bot.performance_analysis`,
and `metaculus_bot/ablation/cli.py`. The first two moved off the repo root on
2026-08-26 to match the other three.

For the authoritative model roster, read `metaculus_bot/llm_configs.py`. It rotates
often, so the guides describe it rather than pin names.
