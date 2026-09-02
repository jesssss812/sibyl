# 0001 — The parity contract

*2026-09-01. Status: in force.*

## Context

The clean fork moved 42 modules and 11 subpackages out of `metaculus_bot/` into ten
purpose-named packages. A move that large does not usually fail loudly — `make test`
catches that. It changes a number by 1e-6, reorders an aggregation, or drops a counter,
in a way no existing test happens to assert, because those tests were written to pin
*intent* rather than *output*.

## Decision

Record the output before moving anything, and compare after. `tests/e2e/test_golden.py`
runs six scenarios through real production code and writes `tests/e2e/golden/*.json`;
floats match to an absolute 1e-9, strings exactly.

What is covered:

1. **Pipeline, MEDIAN** — five question types (binary, multiple choice, numeric closed,
   numeric open-upper, discrete) through `forecast_questions`. Records every prediction,
   the full explanation text, sixteen degradation and stacking counters, and
   `alertable_count`.
2. **Pipeline, CONDITIONAL_STACKING** — the same questions with all three per-type gates
   open, plus the per-question stacker outcome read back out of the published comment.
3. **Min-forecasters guard** — a raising forecaster below the threshold: nothing
   publishes, and the drop is attributed to model, question and cause.
4. **CLI exit codes** — clean, alertable, and a credit fallback inside and outside the
   suppression window, with the date injected rather than read from the clock.
5. **Eval** — `aggregate_scores`, `aggregate_paired`, every comment parser over the
   twelve-record mini fixture, `trim_comment` on an inflated comment, and three
   percentile ladders through the numeric pipeline across three bound shapes.
6. **Surface inventory** — env var names, Makefile targets, `python -m` entry points and
   telemetry marker names.

Zero API calls, by construction rather than convention: every LLM slot is a seeded
`FixtureLlm` and research is the canned fixture provider, so the suite's autouse
egress guard is a backstop rather than the plan.

Recording is a separate, explicit action (`--record-golden`). A harness that auto-heals
on mismatch records the regression instead of catching it.

## Consequences

- Scenarios 1–5 are **byte-identical to the records made in bottega** and pass against
  the restructured code. That is the fork's parity evidence, and it is evidence rather
  than assertion.
- Scenario 6 is layout-dependent by construction, so it is re-recorded when the surface
  legitimately changes and its diff is reviewed by hand. Across the whole fork it moved
  by six entries: `python -m sibyl` replacing `python main.py`, the removal of
  `scripts.research_sync.verify_completeness`, and the four additions the plan
  allowlisted (`SIBYL_MODEL_TIER`, `OPENROUTER_PRIMARY_KEY_PROVIDERS`,
  `OPENROUTER_PRIMARY_KEY_BLOCKED_MODELS`, `make e2e_free`). All 36 env var names and all
  29 telemetry markers are unchanged.
- The pipeline scenarios freeze the clock. The forecaster prompt embeds `Today: <date>`
  and `FixtureLlm` seeds on the prompt, so unfrozen records were reproducible only within
  a single UTC day — they went red at midnight mid-restructure and read as a parity break
  in an unrelated change.
- **`make e2e_free` has not yet been run**, so no live free-tier run is recorded here.
  When one is, its summary lines belong in this file.
