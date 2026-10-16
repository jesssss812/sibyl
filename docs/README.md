# Documentation

Two families live here, and the distinction matters when you are deciding whether a
document describes reality.

**Reference** — [how the bot works](#how-the-bot-works) and [running it](#running-it) —
describes the code as it is. If one of these disagrees with the code, the doc is wrong
and should be fixed.

**Intent** — [strategy](#planning-and-evaluation-strategy), [decisions](#decisions) and
[plans](#plans) — records goals, evidence reviews, the reasons behind choices, and specs
for work that is partly or wholly unbuilt. These describe what *should* be true, or what
was true when they were written, so a mismatch with the code is expected rather than a
bug. Check the date before treating any of it as current.

The folders follow the families: `architecture/` and `runbooks/` are reference;
`strategy/`, `decisions/` and `plans/` are intent. What is not done yet is in
[roadmap.md](roadmap.md), one line per item.

Every document under `docs/` is listed below; `tests/unit/test_docs_index.py` fails
when one is not.

---

## How the bot works

| Doc | What it covers |
| --- | --- |
| [architecture/overview.md](architecture/overview.md) | How a question flows end to end: research fan-out, the forecaster ensemble, the min-forecasters guard, aggregation (MEDIAN in prod, stacking disabled), and the two post-aggregation rails (Platt, open-vs-resolved guard). |
| [architecture/research.md](architecture/research.md) | The providers that build each question's briefing: AskNews (primary), OpenAI native search, Claude web search, financial data, prediction markets, resolution-source fetcher, time-series anchor, reference-class retrieval, plus both gap-fill passes. All fan out in parallel, each independently env-gated. |
| [architecture/agentic-gap-fill.md](architecture/agentic-gap-fill.md) | Gap-fill v2: the bounded agentic research loop — dry-run, target selection, tool iteration, citation-only findings artifact. |
| [architecture/numeric-pipeline.md](architecture/numeric-pipeline.md) | Percentiles to PCHIP CDF: extraction, sanitizing, tail widening, step enforcement, bound pinning, discrete snapping, the unit-mismatch guard. |
| [architecture/probabilistic-tools.md](architecture/probabilistic-tools.md) | The `probabilistic_tools/` package and the tool-runner that exposes it to forecaster models. |

## Running it

| Doc | What it covers |
| --- | --- |
| [runbooks/operations.md](runbooks/operations.md) | Running it: `uv` setup, the shared-vs-fallback key model, workflows and their env flags, credit telemetry, backtesting, and the cost gate (any live or paid run needs operator approval). |
| [runbooks/experiment-funding.md](runbooks/experiment-funding.md) | The isolated grant-funded MiniBench runner, local spending ledger, and private forecast archive. |
| [runbooks/minibench-readiness.md](runbooks/minibench-readiness.md) | October 2026 launch checks and the statistical limits of the first MiniBench round. |
| [runbooks/performance-analysis.md](runbooks/performance-analysis.md) | Evaluating the live bot's calibration against real resolutions, and recovering per-model forecasts from published Metaculus comments. Both pulls are read-only and free. |

## Planning and evaluation strategy

> These are the owner's documents for the Sibyl project, which forked this bot. They
> state intent and record evidence reviews. Several describe work that does not exist
> yet. Read the date before acting on any of them.

| Doc | What it covers |
| --- | --- |
| [strategy/vision.md](strategy/vision.md) | The ranked purpose and what wins when goals conflict. The tie-breaker when other docs don't settle a question. |
| [strategy/design-principles.md](strategy/design-principles.md) | Evidence-graded architecture decisions (Strong / Moderate / Suggestive) drawn from the Metaculus analyses and the Fall-2025 bot-maker survey. |
| [strategy/eval-strategy.md](strategy/eval-strategy.md) | The spec for how anything is measured: paired comparison, power checks, leakage handling, cohort slicing, holdout, no-silent-degradation. |
| [strategy/calibration-primer.md](strategy/calibration-primer.md) | Brier vs log score, reliability diagrams, capping, Platt scaling, extremizing — and why Metaculus grades on log. |
| [strategy/budget-strategy.md](strategy/budget-strategy.md) | Measured per-question costs and which providers and model tiers fit a fixed budget. |
| [strategy/study-notes.md](strategy/study-notes.md) | *Raw log, pre-fork.* Numbered notes from reading the reference implementation. Its header says it belongs in bottega; see the roadmap. |
| `bottega:docs/planning/nostreambot-reading-guide.md` | Reading order for this codebase, written when it was still an external reference. Not carried into this repo; *partly superseded* — see its header, in bottega. |

## Decisions

Numbered records are the clean fork's decisions, each with its status. Dated records are
point-in-time audits and design arguments from before the fork.

| Doc | What it covers |
| --- | --- |
| [decisions/0001-parity-contract.md](decisions/0001-parity-contract.md) | The parity contract: record the pipeline's output before a restructure and compare after, via the golden scenarios in `tests/e2e/test_golden.py`. |
| [decisions/0002-clean-fork-scope.md](decisions/0002-clean-fork-scope.md) | What the clean fork deliberately did not rewrite, and why a restructure carries no behaviour changes. |
| [decisions/0003-model-tiers.md](decisions/0003-model-tiers.md) | The free and prod model tiers, and the spend guard that refuses scored tournaments on the free tier. |
| [decisions/0004-key-fallback.md](decisions/0004-key-fallback.md) | Naming the two OpenRouter keys for their role (primary, fallback) rather than for who paid. |
| [decisions/0005-eval-consolidation.md](decisions/0005-eval-consolidation.md) | One `python -m sibyl.eval` entry point for the offline eval tools, and shared paired statistics. |
| [decisions/2026-08-24-improvement-plan.md](decisions/2026-08-24-improvement-plan.md) | *Audit, 2026-08-24.* What the fork already had vs. what was genuinely missing. |
| [decisions/2026-08-24-eval-strategy-gaps.md](decisions/2026-08-24-eval-strategy-gaps.md) | *Audit, 2026-08-24.* `design-principles.md` and `eval-strategy.md` checked rule by rule against the code. |
| [decisions/2026-08-27-leverage-not-cost-share.md](decisions/2026-08-27-leverage-not-cost-share.md) | *2026-08-27.* Where to spend model capability: leverage over cost share, why the summarizer is the highest-leverage slot, and what was deliberately not changed. |
| [decisions/2026-08-27-cost-tiered-plan.md](decisions/2026-08-27-cost-tiered-plan.md) | *Plan, 2026-08-27.* Cost-tiered escalation, MC shuffling and extremization — sequenced so the assumption each rests on is tested before it is built on. |
| [decisions/2026-08-31-anthropic-direct-swap.md](decisions/2026-08-31-anthropic-direct-swap.md) | *2026-08-31.* Moving Anthropic off OpenRouter and Gemini search off Google: why the Anthropic slots route direct, and what silently breaks if the OpenRouter kwargs come along. |

## Plans

Handoffs written for an implementing agent. They record what was planned, not what was
built, so `tests/unit/test_docs_paths.py` does not check the paths they name.

| Doc | What it covers |
| --- | --- |
| [plans/2026-09-clean-fork.md](plans/2026-09-clean-fork.md) | *2026-09-01.* The clean-fork implementation plan: the move from bottega's layout into purpose-named packages. |
| [plans/2026-09-25-evaluation-integrity.md](plans/2026-09-25-evaluation-integrity.md) | *2026-09-25.* Fixture spend isolation, paired eligibility, uncertainty, cache identity, historical clocks, provenance, and prospective power checks. |
| [plans/2026-10-01-experiment-funding.md](plans/2026-10-01-experiment-funding.md) | *2026-10-01.* Isolated funding for the OpenRouter experiment grant, kept separate from personal spending. |
| [plans/2026-10-04-minibench-readiness.md](plans/2026-10-04-minibench-readiness.md) | *2026-10-04.* MiniBench kickoff assessment: what stood between the workflow and a grant-only launch. |
| [plans/2026-10-05-grant-billing-handoff.md](plans/2026-10-05-grant-billing-handoff.md) | *2026-10-05.* Grant billing review: what was already fixed, and what new evidence answered. |

---

## Repo-root references

- [README.md](../README.md) — install, configure, run.
- [AGENTS.md](../AGENTS.md) — always-true rules and hard prohibitions for coding agents.
  Detail lives in the guides above rather than inline.
- `bottega:FUTURE.md` — the branch design log: shipped, planned, and rejected
  ideas. Read it for the "why", never for the current state. Expect dead paths in it.

The offline eval tools share one entry point, `python -m sibyl.eval <command>`; run it
with no command to list them. Each tool lives in its own package under `sibyl/eval/`,
and the older per-package invocations (`python -m sibyl.eval.backtest` and the rest)
still work as aliases.

For the authoritative model roster, read `sibyl/config/models.py`. It rotates
often, so the guides describe it rather than pin names.
