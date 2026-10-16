# sibyl-ouro clean fork: implementation plan

Written 2026-09-01 for the implementing agent. Decisions in section 0.3 are final; do not reopen them.

## 0. Read this first

### 0.1 The two repositories

| Repo       | Path                                                          | Role                                                                                                                   |
| ---------- | ------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| bottega    | `/Users/Jess/Documents/Claude/Projects/Consulting/bottega`    | Source. A fork of nostreambot (MIT) with the operator's additions. Treated as read-only except for the Phase 0 branch. |
| sibyl-ouro | `/Users/Jess/Documents/Claude/Projects/Consulting/sibyl-ouro` | Target. Today it's empty.                                                                                              |

No remote on git yet for sibyl-ouro.

### 0.2 Hard rules

1. **No paid calls.** Never invoke a paid model or a paid research API (OpenRouter non-free slugs, Anthropic direct, OpenAI, AskNews, Exa, Perplexity, Google grounded search). The test suite's network-egress guard in `tests/conftest.py` stays autouse for every test outside `tests/live/`.
2. **Free-model calls only in test mode against bot-testing-area.** The only live LLM calls permitted in this work are OpenRouter `:free` slugs, only through `python -m sibyl-ouro test --tier free`, only publishing to <https://www.metaculus.com/tournament/bot-testing-area/>, and only when the operator says go for that specific run. Each run needs `METACULUS_TOKEN` and `OPENROUTER_API_KEY` in `sibyl-ouro/.env`; the operator supplies them.
3. **Functional parity.** Every entry point, env var name, flag default, workflow secret name, published-comment format, telemetry marker, aggregation default and exit-code rule behaves as it does in bottega today. Parity is proven by the golden harness from Phase 0, not asserted. The two deliberate behaviour changes are listed in section 0.4.
4. **Test-driven.** Each phase writes or moves its tests before restructuring code. Before every commit: `make test`, `make lint`, `make typecheck` all green and the golden tests green. One PR per phase; Phase 1 may be split into two.
5. **Clean fork, not rewrite.** No logic rewrites and no comment-prose rewrites except the ones this plan names. If a module looks wrong, leave it and add a line to `docs/roadmap.md`.
6. **Never** re-enable a `schedule:` block, dispatch a workflow, commit a secret, or commit on `main`. Work on branch `clean-fork` in sibyl-ouro with a feature branch per phase.

### 0.3 Decisions already made by the operator

1. Copy files fresh into sibyl-ouro. Do not merge bottega's git history.
2. Fold `ablation`, `backtest`, `ensemble_analysis`, `performance_analysis`, `benchmark` and the calibration CLI under one `sibyl-ouro/eval/` package.
3. Prediction-market retrieval stays as a read-only research provider and stays enabled in the prod workflows. Nothing places bets.
4. The donated-key routing is generalised to a primary-key / fallback-key policy. Env var names do not change.
5. `AGENTS.md` is the canonical agent file, at most 200 lines. `CLAUDE.md` contains only `@AGENTS.md`.

### 0.4 The only intended behaviour changes

1. The default model tier for a local run becomes `free`. The prod roster is used only when `sibyl-ouro_MODEL_TIER=prod` is set explicitly (the prod workflows set it). See Phase 3.
2. `python -m sibyl-ouro test` forecasts the open questions of the bot-testing-area tournament instead of the four hard-coded evergreen question URLs. `TEST_QUESTIONS_OVERRIDE` keeps working as a URL override.

### 0.5 Baseline measured in bottega at d15eac5

| Item        | Value                                                                    |
| ----------- | ------------------------------------------------------------------------ |
| Source      | 55,003 lines in `sibyl/`, 4,561 in `scripts/`                    |
| Tests       | 105,624 lines, 205 files, 5,205 collected                                |
| Suite       | 5,179 passed, 21 skipped, 5 deselected (`live`), 112 s                   |
| Lint, types | ruff clean; basedpyright 0 errors                                        |
| Docs        | `README.md` 277, `AGENTS.md` 284, `FUTURE.md` 1,676, `docs/` 5,191 lines |

## 1. Target layout

```
sibyl-ouro/
  __init__.py  __main__.py  cli.py  exceptions.py
  common/        simple_types.py  question_types.py  time_utils.py  prob_math.py
  config/        env.py  settings.py  models.py
  platform/      keys.py  key_fallback.py  llm_retry.py  metaculus_preflight.py
                 fetch_hardening.py  publish_hardening.py  question_patches.py
                 scoring_patches.py  aiohttp_cleanup.py
  research/      (moved wholesale, internal names unchanged)
  forecasting/   bot.py  runners.py  llm_setup.py  logging_utils.py  prompts.py
                 structured_output_schema.py  structured_parse.py  value_extraction.py
                 multiple_choice.py  post_processing.py  tool_runner.py
                 numeric/  probabilistic_tools/
  aggregation/   pipeline.py  strategies.py  stacking.py  routing.py  spread.py
                 agreement.py  resolution_guard.py  calibration/
  publishing/    comment/  close_margin.py
  telemetry/     degradation.py  drops.py  credits.py  check_credits.py
                 markers.py  archive.py  jsonl.py
  eval/          __main__.py  stats.py  scoring.py  backtest/  ablation/  ensemble/
                 performance/  benchmark/
  ops/           (former scripts/*.py)
tests/
  conftest.py  helpers/  data/  unit/  integration/  e2e/  live/
scripts/hooks/   (pre-commit shell hooks only)
docs/            (see Phase 5)
```

### 1.1 Move table (Phase 1)

Every path is relative to the repo root. "Same name" means the file keeps its basename.

| From (bottega)                                                                                                                                  | To (sibyl-ouro)                                                                     |
| ----------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `sibyl/__init__.py`                                                                                                                     | `sibyl-ouro/__init__.py` (preserve its import-time side effects exactly)            |
| `sibyl/cli.py`                                                                                                                          | `sibyl-ouro/cli.py`; add `sibyl-ouro/__main__.py` calling `cli.main()`              |
| `sibyl/exceptions.py`                                                                                                                   | `sibyl-ouro/exceptions.py`                                                          |
| `sibyl/config/env.py`                                                                                                                       | `sibyl-ouro/config/env.py`                                                          |
| `sibyl/config/settings.py`                                                                                                                    | `sibyl-ouro/config/settings.py` (no split in this plan)                             |
| `sibyl/config/models.py`                                                                                                                  | `sibyl-ouro/config/models.py`                                                       |
| `sibyl/common/simple_types.py`, `question_types.py`, `time_utils.py`                                                                           | `sibyl-ouro/common/` same names                                                     |
| `sibyl/common/prob_math.py`                                                                                                              | `sibyl-ouro/common/prob_math.py`                                                    |
| `sibyl/platform/keys.py`                                                                                                                | `sibyl-ouro/platform/keys.py`                                                       |
| `sibyl/platform/key_fallback.py`                                                                                                          | `sibyl-ouro/platform/key_fallback.py`                                               |
| `sibyl/platform/llm_retry.py`                                                                                                                    | `sibyl-ouro/platform/llm_retry.py`                                                  |
| `sibyl/platform/metaculus_preflight.py`                                                                                                                | `sibyl-ouro/platform/metaculus_preflight.py`                                        |
| `sibyl/platform/fetch_hardening.py`, `publish_hardening.py`, `question_patches.py`, `scoring_patches.py`, `aiohttp_cleanup.py`                   | `sibyl-ouro/platform/` same names                                                   |
| `sibyl/research/**`                                                                                                                     | `sibyl-ouro/research/**` unchanged inside                                           |
| `sibyl/forecasting/bot.py`                                                                                                                   | `sibyl-ouro/forecasting/bot.py`; class `TemplateForecaster` becomes `sibyl-ouroBot` |
| `sibyl/forecasting/runners.py`                                                                                                           | `sibyl-ouro/forecasting/runners.py`                                                 |
| `sibyl/forecasting/llm_setup.py`                                                                                                                    | `sibyl-ouro/forecasting/llm_setup.py`                                               |
| `sibyl/forecasting/logging_utils.py`                                                                                                          | `sibyl-ouro/forecasting/logging_utils.py` (drop the `utils/` package)               |
| `sibyl/forecasting/prompts.py`, `structured_output_schema.py`, `structured_parse.py`, `value_extraction.py`, `post_processing.py`, `tool_runner.py` | `sibyl-ouro/forecasting/` same names                                                |
| `sibyl/forecasting/multiple_choice.py`                                                                                                                | `sibyl-ouro/forecasting/multiple_choice.py`                                         |
| `sibyl/forecasting/numeric/`, `sibyl/forecasting/probabilistic_tools/`                                                                                  | `sibyl-ouro/forecasting/numeric/`, `sibyl-ouro/forecasting/probabilistic_tools/`    |
| `sibyl/aggregation/pipeline.py`                                                                                                         | `sibyl-ouro/aggregation/pipeline.py`                                                |
| `sibyl/aggregation/strategies.py`                                                                                                       | `sibyl-ouro/aggregation/strategies.py`                                              |
| `sibyl/aggregation/stacking.py`                                                                                                                     | `sibyl-ouro/aggregation/stacking.py`                                                |
| `sibyl/aggregation/routing.py`                                                                                                               | `sibyl-ouro/aggregation/routing.py`                                                 |
| `sibyl/aggregation/spread.py`                                                                                                               | `sibyl-ouro/aggregation/spread.py`                                                  |
| `sibyl/aggregation/agreement.py`                                                                                                         | `sibyl-ouro/aggregation/agreement.py`                                               |
| `sibyl/aggregation/resolution_guard.py`                                                                                                       | `sibyl-ouro/aggregation/resolution_guard.py`                                        |
| `sibyl/aggregation/calibration/`                                                                                                                    | `sibyl-ouro/aggregation/calibration/`                                               |
| `sibyl/publishing/comment/`                                                                                                                        | `sibyl-ouro/publishing/comment/`                                                    |
| `sibyl/publishing/close_margin.py`                                                                                                                 | `sibyl-ouro/publishing/close_margin.py`                                             |
| `sibyl/telemetry/degradation.py`                                                                                                         | `sibyl-ouro/telemetry/degradation.py`                                               |
| `sibyl/telemetry/drops.py`                                                                                                               | `sibyl-ouro/telemetry/drops.py`                                                     |
| `sibyl/telemetry/credits.py`                                                                                                             | `sibyl-ouro/telemetry/credits.py`                                                   |
| `sibyl/telemetry/check_credits.py`                                                                                                     | `sibyl-ouro/telemetry/check_credits.py`                                             |
| `sibyl/telemetry/{markers,archive,jsonl}.py`                                                                                                  | `sibyl-ouro/telemetry/` same names                                                  |
| `sibyl/eval/scoring.py`                                                                                                               | `sibyl-ouro/eval/scoring.py`                                                        |
| `sibyl/eval/backtest/`                                                                                                                       | `sibyl-ouro/eval/backtest/`                                                         |
| `sibyl/eval/ablation/`                                                                                                                       | `sibyl-ouro/eval/ablation/`                                                         |
| `sibyl/eval/ensemble/`                                                                                                              | `sibyl-ouro/eval/ensemble/`                                                         |
| `sibyl/eval/performance/`                                                                                                           | `sibyl-ouro/eval/performance/`                                                      |
| `sibyl/eval/benchmark/`                                                                                                                      | `sibyl-ouro/eval/benchmark/`                                                        |
| `scripts/*.py` (except `__init__.py`)                                                                                                           | `sibyl-ouro/ops/` same names                                                        |
| `scripts/hooks/*.sh`                                                                                                                            | `scripts/hooks/` unchanged                                                          |
| `main.py`                                                                                                                                       | deleted; `python -m sibyl-ouro` replaces `python main.py` everywhere                |

Known cross-package dependency to leave alone: `forecasting/bot.py` imports `eval/performance/parsing.py` for comment annotation. Note it in `docs/roadmap.md`, do not fix it here.

### 1.2 Not copied from bottega

`scratch/`, `FUTURE.md`, `AGENTS.md` (rewritten in Phase 5), `.github/commands/`, `.github/workflows/gemini-*.yml`, `.github/workflows/claude.yml`, `scripts/research_sync/` (a launchd job for another operator's machine) and its test `tests/test_research_sync_job.py`, `tests/test_template_forecaster.py` (its tests move into `tests/unit/test_bot.py`), `REFERENCE_COPY_OF_*` gitignore entries.

`tests/test_ft_*.py` (six files): read each docstring. Keep a test only if it asserts behaviour of the currently installed forecasting-tools version that the code relies on; move those to `tests/integration/test_framework_*.py`. Delete tests whose purpose is comparing the 0.2.54 and 0.2.92 versions.

## 2. Phase 0: parity harness (in bottega, branch `parity-harness`)

Goal: record what the unmodified code does, so every later phase can prove it still does it. Zero API calls; use `sibyl/eval/benchmark/fixture_llm.py`, which is seeded and opens no socket.

Build `tests/test_golden.py` plus a `tests/golden/` directory and a `--record-golden` pytest option that rewrites the JSON files. Each scenario writes one JSON file; the test compares with an absolute tolerance of 1e-9 on floats and exact match on strings.

Scenarios:

1. **Pipeline, MEDIAN.** Five synthetic questions (binary, multiple choice, numeric closed bounds, numeric open upper bound, discrete numeric) through `forecast_questions` with three fixture forecasters, fixture summarizer and parser, a canned research string, every add-on provider flag off, gap-fill off, `publish_reports_to_metaculus=False`. Record per question: prediction (binary float, MC option probabilities, numeric CDF), the full explanation text, and after the batch the degradation counters and `alertable_count`.
2. **Pipeline, CONDITIONAL_STACKING.** Same questions with the three per-type stacking flags on, a fixture stacker, and forecaster seeds chosen so the spread exceeds the thresholds. Record the same fields plus the stacker outcome per question.
3. **Min-forecasters guard.** One forecaster raising; record that the question is skipped and the drop telemetry.
4. **CLI exit codes.** Call `cli.main()` with `forecast_on_tournament` patched to return canned reports and counters set by hand. Record the exit code for: clean run; alertable count above zero; credit fallback inside and outside the suppression window (inject the date). `tests/test_cli.py` already shows the patching pattern.
5. **Eval.** `backtest.analysis.aggregate_scores` on a fixed list of scores; `ablation.scoring.aggregate_paired` on fixed paired scores with `seed=0`; `performance_analysis.parsing` over `tests/data/performance_comments_mini.jsonl`; `comment.trimming` on a fixed over-length comment; the numeric pipeline on three fixed percentile sets. Record outputs.
6. **Surface inventory.** Sorted list of every env var name ending in `_ENV` in `constants.py`, every Makefile target, every `python -m` module, every marker name in `sibyl/telemetry/markers.py`. Later phases assert this list is unchanged except for an explicit allowlist of additions.

Gate: the golden test passes twice in a row in bottega. Commit on the branch. These files travel to sibyl-ouro in Phase 1 as `tests/e2e/test_golden.py` and `tests/e2e/golden/`.

## 3. Phase 1: import and skeleton (sibyl-ouro, branch `clean-fork`)

1. Delete the template: `main.py`, `main_with_no_framework.py`, `bot_helpers.py`, `integrations/`, `pyproject.toml`, `poetry.lock`, `.github/workflows/*`, `README.md`, `.env.template`, `.gitignore`. Remove the `upstream` remote.
2. Copy from bottega: `sibyl/`, `tests/`, `scripts/`, `pyproject.toml`, `uv.lock`, `Makefile`, `.pre-commit-config.yaml`, `.python-version`, `.markdownlint.jsonc`, `osv-scanner.toml`, `ty.toml`, `.env.template`, `.gitignore`, `LICENSE`, `.github/workflows/{ci.yaml,test_bot.yaml,test_bot_basic.yaml,run_bot_on_tournament.yaml,run_bot_on_minibench.yaml,run_bot_on_metaculus_cup.yaml}`, and `docs/` per Phase 5 step 1. Apply section 1.2.
3. Apply the move table with `git mv`. Add `__init__.py` to every new package.
4. Rewrite imports mechanically. Build the mapping from the move table as a sed script, apply to `sibyl-ouro/`, `tests/`, `Makefile`, workflows, docs, then `uv run ruff check --fix` for import order. Update `[tool.uv.build-backend] module-name`, `[tool.coverage.run] source`, `[tool.basedpyright] include`, the `pytest-full-suite` hook command, and every Makefile recipe (`python -m sibyl.eval.backtest` becomes `python -m sibyl-ouro.eval backtest` only after Phase 4; in Phase 1 use `python -m sibyl-ouro.eval.backtest`).
5. Rename identifiers: `TemplateForecaster` to `sibyl-ouroBot` everywhere including tests and docs; `metaculus-bot` project name to `sibyl-ouro`; the argparse description string in `cli.py`; log lines that say "TemplateForecaster". Fix by hand the 25 comment lines that mention nostreambot or "the fork" (`git grep -n -iE 'nostreambot|no-stream'`).
6. Strip harness markers. Remove every `HARNESS-SCAN-EXEMPT-[a-z-]+:?` token, then normalise leftovers: `# noqa: PLC0415,` at end of line becomes `# noqa: PLC0415`; a comment line that is now empty is deleted. Confirm with `uv run ruff check` that every remaining `noqa` is still needed (`RUF100` if enabled, otherwise leave).
7. Redirect study citations: in surviving code and docs, `scratch/` becomes `bottega:scratch/` and `FUTURE.md` becomes `bottega:FUTURE.md`. Do not delete the citations.
8. Reorganise tests. Move `agentic_fakes.py`, `market_retrieval_fakes.py`, `ts_anchor_fakes.py`, `pipeline_test_helpers.py` into `tests/helpers/`. Classify each `test_*.py`: `pytest.mark.live` present goes to `tests/live/`; `pytest.mark.e2e` present or `_e2e` in the name goes to `tests/e2e/`; `pytest.mark.integration` present or `_integration` or `_wiring` in the name goes to `tests/integration/`; everything else to `tests/unit/`. Add `__init__.py` to each test directory so `from tests.helpers...` imports resolve. Fix the two tests that import from `tests.test_market_retrieval_generation` by moving the shared code into `tests/helpers/`. `tests/data/` stays where it is. Add markers `unit` and `integration` to `pyproject.toml` only if pytest warns.
9. Bring in the golden harness from Phase 0 as `tests/e2e/test_golden.py` and `tests/e2e/golden/`.
10. `LICENSE`: keep the MIT text and the existing copyright line; add `Copyright (c) 2026 Jess Sun` on the line after it.
11. Write an interim `AGENTS.md` (under 60 lines) carrying section 0.2 of this plan and a pointer to this file. `CLAUDE.md` becomes the single line `@AGENTS.md`.
12. CI: `ci.yaml` unchanged in substance, paths updated. All bot workflows stay `workflow_dispatch` only. Delete the commented-out `schedule:` blocks entirely; the operator will add a schedule deliberately when there is a season to run.

Gate: `uv sync --dev`, `make test` (same pass count as baseline minus the deleted files, no new failures), `make lint`, `make typecheck` at 0 errors, golden green, `git grep -c sibyl` returns nothing outside `docs/decisions/`.

## 4. Phase 2: generalise the key fallback

Files: `sibyl-ouro/platform/key_fallback.py` (758 lines), `sibyl-ouro/telemetry/credits.py`, `sibyl-ouro/cli.py`, `tests/unit/test_fallback_openrouter.py` (rename to `test_key_fallback.py`), `tests/unit/test_cli.py`, `tests/unit/test_credit_telemetry.py`.

Vocabulary: "donated" becomes "primary", "personal" and "general" become "fallback", in identifiers, log messages and docstrings. Env var names stay: `OAI_ANTH_OPENROUTER_KEY` is the primary key, `OPENROUTER_API_KEY` the fallback. Keep `build_llm_with_openrouter_fallback` as the public builder name. Counters keep their semantics; rename `donated_404` to `primary_route_404`.

Config instead of code: `DONATED_KEY_PROVIDERS` becomes `OPENROUTER_PRIMARY_KEY_PROVIDERS` read from env with default `openai,anthropic,google`; `DONATED_KEY_BLOCKED_GOOGLE_MODELS` becomes `OPENROUTER_PRIMARY_KEY_BLOCKED_MODELS` with default `gemini-3.1-pro`. `CREDIT_ALERT_RESUME_DATE` is already env-driven (`OPENROUTER_CREDIT_ALERT_RESUME_DATE`); keep its default so behaviour is identical, and document it in `.env.template` as the operator's suppression window. Remove the `TODO(gemini-3.1-pro-donated)` prose; the blocklist is now a documented setting.

Tests first: parametrised tests for provider routing and blocklist parsing from env, then the mechanical rename. Golden scenario 4 must still pass unchanged.

## 5. Phase 3: model tiers, the spend guard, and the free end-to-end lane

### 5.1 Roster object

In `sibyl-ouro/config/models.py` add:

```python
class ModelTier(StrEnum): FREE = "free"; PROD = "prod"
@dataclass(frozen=True) class Roster: ...  # one field per slot below
def active_tier() -> ModelTier   # sibyl-ouro_MODEL_TIER env, default FREE
def get_roster(tier: ModelTier) -> Roster
```

Slots, with today's prod value as the prod entry (verbatim from the current code): `forecasters`, `summarizer`, `parser`, `researcher`, `stacker`, `stacker_fallback`, `analyzer`, `market_ranker_config`, `market_query_author_config`, `native_search_model` (`constants.NATIVE_SEARCH_DEFAULT_MODEL`), `gemini_search_model`, `anthropic_search_model`, `gap_fill_resolver_model`, `gap_fill_v2_driver_model`, `gap_fill_v2_reader_model`, `leakage_detector_model` (`backtest/leakage.py`), the financial-data classifier (`research/financial_data.py:589`), the crux analyzer (`research/targeted.py:200`), the structured-parse fallback (`structured_parse.py:75`), and the AskNews-fallback prose model (`research/providers.py:375`). Enumerate the construction sites with `git grep -n -E 'GeneralLlm\(|build_llm_with_openrouter_fallback\(' sibyl-ouro` and make every one read its model from the active roster. Existing per-slot env overrides (`NATIVE_SEARCH_MODEL`, `GAP_FILL_V2_DRIVER_MODEL`, and so on) keep winning in the prod tier only.

Free tier values: forecasters `openrouter/google/gemma-4-31b-it:free`, `openrouter/nvidia/nemotron-3-super-120b-a12b:free`, `openrouter/qwen/qwen3-next-80b-a3b-instruct:free`, built as plain `GeneralLlm` on `OPENROUTER_API_KEY` without the key-fallback wrapper (the reason is in `eval/ablation/forecaster_lineup.py`'s docstring: free slugs 404 on the primary key's provider list). Every utility slot is `openrouter/google/gemma-4-31b-it:free`. `MIN_FORECASTERS_TO_PUBLISH` is 1 today, so three free forecasters clear the guard. Reuse `eval/ablation/forecaster_lineup.py` rather than duplicating it: it becomes a thin re-export of the free roster.

### 5.2 Provider gating in the free tier

Providers whose value is a paid search tool cannot be made free by swapping the model. In the free tier the orchestrator force-disables them regardless of env flags and logs one line per disabled provider: AskNews, OpenAI native search, Claude web search, Gemini grounded search, gap-fill v1, gap-fill v2. Providers that stay on in the free tier when their flag is set: financial data, prediction markets, resolution source, reference class, time-series anchor, with their LLM stages on the free utility model. Verify each of those five has no paid call path by reading it (`FRED_API_KEY` is a free key; venue APIs are public).

### 5.3 Selection and refusal

- `python -m sibyl-ouro <mode> [--tier free|prod]`. The flag overrides `sibyl-ouro_MODEL_TIER`. Default is free.
- Modes: `tournament`, `minibench`, `cup` (alias `metaculus_cup`, keep `quarterly_cup` as a hidden alias), `test`. `test` calls `forecast_on_tournament("bot-testing-area")` with `skip_previously_forecasted_questions=False`, honours `TEST_QUESTIONS_OVERRIDE`, and adds `--max-questions` (default 3 in the free tier, existing default otherwise) mapped to `sibyl-ouroBot(max_questions_per_run=...)`.
- Refusals, raised before any client is built: free tier with mode `tournament`, `minibench` or `cup` exits non-zero with a one-line reason (free models never publish to a scored tournament). Prod tier logs a banner naming the tier and the roster.
- The three `run_bot_on_*.yaml` and `test_bot_basic.yaml` set `sibyl-ouro_MODEL_TIER: 'prod'` in `env:`. `test_bot.yaml` becomes the free lane: `sibyl-ouro_MODEL_TIER: 'free'`, `python -m sibyl-ouro test --tier free`, still `workflow_dispatch` only, and only `METACULUS_TOKEN`, `OPENROUTER_API_KEY` and `FRED_API_KEY` in its env.

### 5.4 Tests

Unit (`tests/unit/test_models.py`, `tests/unit/test_cli_tier.py`, `tests/unit/test_free_tier_gating.py`): every slot populated in both tiers; every free-tier slug ends in `:free`; prod roster equals the values recorded in golden scenario 6 (add the prod roster to the surface inventory); default tier is free when the env var is unset; the refusal matrix; each of the six paid providers is disabled in the free tier even with its flag on; the five free providers select the free utility model.

Live (`tests/live/test_free_tier.py`, marker `live`, skipped without `OPENROUTER_API_KEY`): the five existing tests from `test_smoke_real_llm.py` plus one that builds `get_roster(FREE)` and runs `forecast_questions` on one synthetic binary question without publishing.

End to end: `make e2e_free` runs `sibyl-ouro_MODEL_TIER=free uv run python -m sibyl-ouro test --tier free --max-questions 3`. Publishes to bot-testing-area. Run once in this phase, only after the operator says go for that run, and paste the summary lines into the PR. OpenRouter free slugs allow roughly 50 requests a day without a balance, so keep the cap at 3.

## 6. Phase 4: eval consolidation

1. `sibyl-ouro/eval/__main__.py` with subcommands `backtest`, `ablation`, `ensemble`, `performance`, `fit-platt`, `compare`, each delegating to the existing module's `main`. Existing flags unchanged. Makefile targets point at the new commands. `python -m sibyl-ouro.eval.backtest` keeps working too.
2. Extract into `sibyl-ouro/eval/stats.py`, as pure moves: `bootstrap_mean_ci`, `bootstrap_median_ci`, `sign_test`, `wilcoxon_signed_rank` from `eval/ablation/scoring.py`, which re-exports them so ablation and its tests are untouched. Add `minimum_detectable_effect(sd, n, power=0.8, alpha=0.05)` and `paired_summary(deltas, n_bootstrap=5000, seed=0)` returning n, mean and median delta, both bootstrap CIs, sign-test and Wilcoxon p-values, and the MDE at n.
3. `compare`: `python -m sibyl-ouro.eval compare A.json B.json [--by question_type]` over two `backtest_data_*.json` files written by `eval/backtest/analysis.py:save_backtest_data`. Match questions by id; a question missing or failed in either file is dropped from both and counted; refuse to print a summary when the drop rate exceeds 5%; report Brier and log score deltas with `paired_summary`, overall and per question type; print the caveats block (leaky backtest, no absolute Brier). This implements `docs/strategy/eval-strategy.md` rules 1, 2, 3 and 5.
4. `eval/backtest/analysis.py:aggregate_scores` gains a bootstrap CI for `bot_minus_community` using `stats.py`. Output keys already present keep their values (golden scenario 5 must pass); the CI is an added key.
5. Tests first: `tests/unit/eval/test_stats.py` with known-answer cases (deltas all zero gives CI containing 0 and sign p of 1; a constant positive delta gives a CI excluding 0); `tests/integration/test_compare.py` on two small synthetic backtest files; `tests/e2e/test_fixture_aa.py` running two identical fixture-LLM backtest arms through `compare` and asserting a zero delta, a CI containing zero, and a drop count of zero.

## 7. Phase 5: docs

1. `docs/` in sibyl-ouro after Phase 1: bottega's `docs/{architecture,research,agentic_gap_fill,numeric_pipeline,operations,performance_analysis,probabilistic_tools}.md` and `docs/README.md`; bottega's `docs/planning/{vision,design-principles,eval-strategy,calibration-primer,budget-strategy}.md` moved to `docs/` (bottega's copies are newer than sibyl-ouro's; sibyl-ouro's `study-notes.md` is kept as is); bottega's `docs/planning/{improvement-plan,eval-strategy-gaps,leverage-not-cost-share,cost-tiered-plan,anthropic-direct-swap}.md` moved to `docs/decisions/` with a date prefix. `nostreambot-reading-guide.md` is dropped.
2. New `docs/decisions/`: `0001-parity-contract.md` (what golden covers), `0002-clean-fork-scope.md` (what this plan deliberately did not rewrite and why), `0003-model-tiers.md`, `0004-key-fallback.md`, `0005-eval-consolidation.md`. Each under 60 lines: context, decision, consequences.
3. `docs/roadmap.md`: one line per `##`/`###` heading of bottega's `FUTURE.md` "High-priority" section plus the items this plan deferred, each pointing at `bottega:FUTURE.md`. No prose copied.
4. `README.md` (under 150 lines): what sibyl-ouro is; inherited from nostreambot under MIT with attribution; the operator's contributions (reference-class retrieval, open-vs-resolved guard, zero-cost fixture backtest, ensemble agreement telemetry, Anthropic direct routing and Claude web search, model tiers and the spend guard, the free test lane, eval consolidation with paired comparison, this restructure); quickstart with `uv`; the cost rules in three lines; a link to `docs/README.md`.
5. `AGENTS.md` (at most 200 lines): always-true rules, the paid-call prohibition and what "operator go" means, the tier guard, the parity contract, layout map with one line per package, commands, where detail lives. No history, no narrative. `CLAUDE.md` is `@AGENTS.md`.
6. Pass over every doc: replace old paths and names, `python main.py` with `python -m sibyl-ouro`, `TemplateForecaster` with `sibyl-ouroBot`. Add `tests/unit/test_docs_paths.py`: every backticked path under `sibyl-ouro/`, `tests/`, `scripts/` or `docs/` mentioned in `README.md`, `AGENTS.md` and `docs/**/*.md` must exist.

## 8. Phase 6: verification and hand-off

1. Walk the surface inventory from golden scenario 6 against the allowlist of additions (`sibyl-ouro_MODEL_TIER`, `OPENROUTER_PRIMARY_KEY_PROVIDERS`, `OPENROUTER_PRIMARY_KEY_BLOCKED_MODELS`, the `compare` and `e2e_free` targets).
2. `make test`, `make lint`, `make typecheck`, `make cov`; per-module coverage not below the bottega baseline (record the baseline report in Phase 0).
3. CI green on the `clean-fork` branch; merge to `main` by PR.
4. One more `make e2e_free` run with operator go; record the run summary in `docs/decisions/0001-parity-contract.md`.

## 9. Budget

Estimated at Claude Opus 5 list prices ($5 in, $25 out per million tokens).

| Phase                 | Estimate     |
| --------------------- | ------------ |
| 0 parity harness      | $20 to $40   |
| 1 import and skeleton | $60 to $100  |
| 2 key fallback        | $20 to $40   |
| 3 tiers and free lane | $40 to $80   |
| 4 eval consolidation  | $40 to $80   |
| 5 docs                | $30 to $50   |
| 6 verification        | $10 to $20   |
| Total                 | $220 to $410 |

Free-tier live runs cost $0 in API spend.

## 10. Appendix: recipes

Import rewrite skeleton (extend from the move table):

```bash
sed -i '' -E \
  -e 's/\bmetaculus_bot\.forecaster\b/sibyl-ouro.forecasting.bot/g' \
  -e 's/\bmetaculus_bot\.constants\b/sibyl-ouro.config.settings/g' \
  -e 's/\bmetaculus_bot\.llm_configs\b/sibyl-ouro.config.models/g' \
  -e 's/\bmetaculus_bot\.fallback_openrouter\b/sibyl-ouro.platform.key_fallback/g' \
  -e 's/\bmetaculus_bot\.research\b/sibyl-ouro.research/g' \
  $(git ls-files 'sibyl-ouro/**/*.py' 'tests/**/*.py' Makefile '.github/**' 'docs/**')
```

Harness marker strip:

```bash
sed -i '' -E -e 's/ *HARNESS-SCAN-EXEMPT-[a-z-]+:?//g' -e 's/# noqa: ([A-Z0-9, ]+),\s*$/# noqa: \1/' $(git ls-files '*.py')
git grep -n -E '^\s*#\s*$' -- '*.py'   # delete comment lines left empty
```

Classification of tests:

```bash
for f in tests/test_*.py; do
  if grep -q 'pytest.mark.live' "$f"; then d=live
  elif grep -qE 'pytest.mark.e2e|_e2e' <<<"$f$(cat "$f")"; then d=e2e
  elif grep -qE 'pytest.mark.integration|_integration|_wiring' <<<"$f$(cat "$f")"; then d=integration
  else d=unit; fi
  git mv "$f" "tests/$d/"
done
```

Free-tier check before wiring a slug (operator go required, $0): `uv run pytest -m live tests/live/test_free_tier.py -k binary`.
