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
## Verification (Phase 6, 2026-09-02)

**Surface inventory**, with the move table applied to the bottega side so a rename is not
counted as a change:

| | bottega | sibyl | change |
| --- | --- | --- | --- |
| env var names | 36 | 39 | +3, all allowlisted |
| Makefile targets | 46 | 47 | +1 (`e2e_free`) |
| `python -m` entry points | 8 | 9 | +`sibyl`, +`sibyl.eval`, −`scripts.research_sync.verify_completeness` |
| telemetry markers | 29 | 29 | identical |

Every change is on the plan's allowlist. All 29 telemetry marker names are unchanged,
which is the one list where a rename would silently orphan archived data.

**Coverage**, per module against `tests/e2e/golden/coverage_baseline.txt`: 169 of 171
modules at or above their bottega figure. Two are below, and neither is lost test
coverage:

- `sibyl/config/settings.py`, 98% → 94%. The uncovered lines are the bodies of
  `_int_env` and `_float_env`, which execute at import only when the env vars they read
  are set. bottega's local `.env` is fully populated and sibyl's holds two keys, so this
  measures the two `.env` files rather than the two test suites.
- `sibyl/forecasting/value_extraction.py`, 95% → 94%. Two lines of one error branch in
  the JSON-repair ladder. Statement count is identical, no test was lost (both
  `test_template_forecaster.py` and `test_ft_pin_mc_high_cardinality.py` were renamed,
  not deleted), and running the three relevant test files in each repo gives *identical*
  coverage — 166 statements, 8 missed, the same lines, 95%. The difference appears only
  when the whole suite is combined, so it is a test-ordering artifact. Tracked in
  [roadmap.md](../roadmap.md) rather than papered over.

The whole-repo total reads 87% → 86%, but the two figures are not comparable: `--cov=sibyl`
now includes `sibyl/ops/` and three `sibyl/telemetry/` modules that were `scripts/` before
and therefore outside `--cov=metaculus_bot`. That is ~1,900 statements of operator
scripts added to the denominator.

Note that the baseline file's *paths* were rewritten by the Phase 1 import rewrite, which
ran over every tracked file. The rename applied was exactly the move table, so the
comparison stays valid, but the file no longer reads as a verbatim bottega artifact.

## The first free end-to-end run (2026-09-02)

`make e2e_free`, with operator go for that run. ~4 minutes, **$0 API spend**, publishing
to `bot-testing-area` under the operator's second bot account.

```
MODEL TIER: free — forecasters=[gemma-4-31b-it:free, nemotron-3-super-120b-a12b:free,
                               qwen3-next-80b-a3b-instruct:free]
Using research providers: ['none']
FREE TIER: primary research provider disabled

✅ .../questions/43327 | Prediction: 5.0%
✅ .../questions/43326 | Prediction: Democrats 58.0% / Republicans 38.0% / Other 4.0%
✅ .../questions/43325 | Prediction: 22.0%

FORECASTERS_SURVIVED: survived=1/3   (on each of the three questions)
Degradation counters: forecasters_dropped=6, questions_failed_to_publish=0, ...
CREDIT_SPEND: key=fallback run_delta_usd=0.00
Run completed with 6 alertable degradation event(s); exiting non-zero.
```

**The guards held.** The primary research ladder was vetoed, so the run used no research
provider at all rather than billing on Perplexity; the primary key was never set; spend
was zero. The only paid model slugs appearing in the log are litellm's import-time
cost-tracking warnings from `config/models.py` building the prod singletons, which open
no socket.

**The pipeline worked.** All three questions were forecast, published, and commented,
through the real tournament-fetch path — including the resolution guard adjusting a 2.0%
binary up to 5.0% on an open window, and the single-survivor stacking short-circuit.

**Two of three forecasters failed on every question**, which is the finding this run
exists to produce:

1. `qwen/qwen3-next-80b-a3b-instruct:free` returns **404 — the free variant no longer
   exists**. Confirmed against OpenRouter's public model list: of the roster's three
   slugs, that one is gone. The free roster is carrying a dead model.
2. `google/gemma-4-31b-it:free` was **429 rate-limited upstream** on the shared free pool
   every time. It is both a forecaster *and* all five utility slots, so one run asks the
   same shared quota for summarize, parse, stack, analyze *and* forecast.

The non-zero exit is correct behaviour, not a failure of the lane: six dropped
forecasters is six alertable events, and the run still published every question it could.
But it does mean `make e2e_free` stays red until the roster is repaired. Both items are
in [roadmap.md](../roadmap.md).
