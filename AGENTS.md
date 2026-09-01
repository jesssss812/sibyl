# AGENTS.md

Interim file for the clean-fork work. Phase 5 of
[`docs/plans/2026-09-clean-fork.md`](docs/plans/2026-09-clean-fork.md) replaces it with the
real one. Read the plan before changing anything; the rules below are its section 0.2 and
they are not negotiable.

## Hard rules

1. **No paid calls.** Never invoke a paid model or a paid research API — OpenRouter
   non-free slugs, Anthropic direct, OpenAI, AskNews, Exa, Perplexity, Google grounded
   search. The network-egress guard in `tests/conftest.py` stays autouse for every test
   outside `tests/live/`.
2. **Free-model calls only in test mode, against bot-testing-area.** The only live LLM
   calls permitted are OpenRouter `:free` slugs, only through `python -m sibyl test --tier
   free`, only publishing to <https://www.metaculus.com/tournament/bot-testing-area/>, and
   only when the operator says go for that specific run. Each run needs `METACULUS_TOKEN`
   and `OPENROUTER_API_KEY` in `.env`; the operator supplies them.
3. **Functional parity.** Every entry point, env var name, flag default, workflow secret
   name, comment format, telemetry marker, aggregation default and exit-code rule behaves as
   it does in bottega, *proven* by `tests/e2e/test_golden.py` rather than asserted.
4. **Test-driven.** Each phase writes or moves its tests before restructuring code. Before
   every commit: `make test`, `make lint`, `make typecheck` and the golden tests all green.
5. **Clean fork, not rewrite.** No logic rewrites and no comment-prose rewrites beyond the
   ones the plan names. If a module looks wrong, leave it and add a line to
   `docs/roadmap.md`.
6. **Never** re-enable a `schedule:` block, dispatch a workflow, commit a secret, or commit
   on `main`. Work on `clean-fork`, a feature branch per phase.
7. **Keep commits small.** One reviewable change each — a rename, a test, an extraction —
   not a whole phase in one blob. A phase is a branch and a PR; it is not a commit. Small
   commits are what make `git bisect` and `git log --follow` worth having after a
   restructure this large, and what let a reviewer see a rename separately from a
   behaviour change.

## Naming

The plan says `sibyl-ouro` throughout, but a hyphen is illegal in a Python module name. By
operator decision (2026-09-01) the package is **`sibyl`**: `sibyl/` on disk, `python -m
sibyl`, class `SibylBot`, env `SIBYL_MODEL_TIER`. Read every `sibyl-ouro` as `sibyl`.

## Layout

| Package | Holds |
| --- | --- |
| `sibyl/common/` | Types and helpers with no dependency on the rest of the bot. |
| `sibyl/config/` | Environment loading, tunable settings, the model roster. |
| `sibyl/platform/` | Adapters to third parties: keys, retries, forecasting-tools patches. |
| `sibyl/research/` | Research providers and the orchestrator that selects them. |
| `sibyl/forecasting/` | Prompts, per-type runners, extraction, and `SibylBot` itself. |
| `sibyl/aggregation/` | Combining forecasters into one published forecast. |
| `sibyl/publishing/` | Rendering and posting the comment. |
| `sibyl/telemetry/` | Counters, markers, and the run-log archive. |
| `sibyl/eval/` | Backtest, ablation, ensemble and performance analysis. |
| `sibyl/ops/` | Operator scripts. Not imported by the bot. |

Tests are laned: `tests/unit/`, `tests/integration/`, `tests/e2e/`, `tests/live/`, with
shared fakes in `tests/helpers/` and fixtures in `tests/data/`.

## Commands

`make test` (live deselected), `make lint`, `make typecheck` (must be 0 errors),
`uv run pytest tests/e2e/test_golden.py` (the parity records).

Citations of the form `bottega:path` point at the bottega repo, which holds the study notes
and design log this fork deliberately does not carry.
