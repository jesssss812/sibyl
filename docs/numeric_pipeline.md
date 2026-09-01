# Numeric forecasting and aggregation

This is the reference for how the bot turns a numeric question into a probability
distribution. It covers the full path: each forecaster declares the standard percentile
set in plain text, the bot extracts and cleans them, builds a 201-point CDF that
satisfies Metaculus' server-side constraints, aggregates across the ensemble in CDF
space, and decides whether the ensemble disagrees enough to trigger conditional
stacking. It also documents the time-series anchor research provider, which grounds
numeric forecasts in the resolution series' own history.

Every model name, flag, and default here is verified against the code. Where a value
lives in a constants file this doc names the constant rather than restating its value,
so read the constant for the current magnitude.

## What a numeric question looks like

A Metaculus `NumericQuestion` has a lower and upper bound, and each bound is either
open or closed. A closed bound is a hard limit (the outcome cannot fall past it); an
open bound is just the edge of the displayed range, and the true outcome can land
beyond it. Some questions also carry a `zero_point` (log-scaled axis) and a `cdf_size`
that differs from 201 (a discrete question). All of these change how the CDF is built,
so they thread through every stage below.

## Step 1: forecasters declare the standard percentiles

Each forecaster is prompted to emit the standard percentile set for its distribution.
`STANDARD_PERCENTILES` in `numeric/config.py` owns the set, and
`EXPECTED_PERCENTILE_COUNT` is derived from it with `len()` so the count can never
disagree with the list. As shipped today the set is:

```
1, 2.5, 5, 10, 20, 40, 50, 60, 80, 90, 95, 97.5, 99
```

Stored internally as decimals in `[0, 1]` rather than as 1-100 labels. The tail anchors
P1 and P99 exist so a forecaster can express probability mass beyond an open bound: if
you believe the outcome very likely exceeds an open upper bound, you place P50 and above
outside the displayed range. The prompt (see `prompts.numeric_prompt` and the bound
helper text from `bound_messages` in `numeric/utils.py`) tells forecasters exactly this
and warns them not to pile percentiles against an open edge.

The forecaster writes its percentiles inside a fenced ```json STRUCTURED FORECAST
block, which the prompt requires to be the last thing in the response.

## Step 2: value extraction ladder

`extract_numeric` in `value_extraction.py` pulls the declared percentiles out of the
model's free-text rationale. It runs a four-rung ladder, most deterministic first, and
stops at the first rung that produces a valid result:

1. **block** — parse the fenced ```json block directly (`json.loads` plus Pydantic
   validation against `NumericStructured`). When the block carries a
   `declared_percentiles` dict, it is lifted straight into `Percentile` objects
   (`_numeric_from_block` in `value_extraction.py`). This is the normal path.
2. **repair** — if the block was malformed, run deterministic JSON repair
   (`json_repair`), or scan the rationale's last `_TAIL_SCAN_CHARS` characters for
   balanced braces. Skipped when rung 1 already produced a schema-valid model (repairing
   valid JSON is a no-op).
3. **llm** — as a last resort, call the parser LLM (`parse_structured`) over the full
   rationale. Logged loudly as a salvage.
4. **raise** — if every rung fails, raise `ValueExtractionError`; the caller drops or
   soft-fails that forecaster.

Post-rung validation is strict regardless of which rung produced the value
(`_validate_numeric` in `value_extraction.py`): it requires every label in
`STANDARD_PERCENTILES` and returns exactly that set, never padded. So even the LLM
salvage rung cannot smuggle in a fabricated or partial set.

Every successful extraction logs one line:

```
EXTRACTION_RUNG: question=... model=... qtype=numeric rung=... block_present=...
```

Watch for `rung=llm` and `block_present=False` in prod logs. Those are the two signals
that a forecaster stopped emitting a clean block.

The same call also decides discrete vs. continuous. `run_numeric_forecast`
(`forecaster_runners.py`) reads `NumericStructured.outcome_type` from the block first,
and only falls back to a parser LLM call (`OutcomeTypeResult`) when the block does not
declare it. That vote is carried forward to the discrete-snap decision in Step 7.

## Step 3: sanitize_percentiles

`sanitize_percentiles` (`numeric/pipeline.py`) turns the raw declared percentiles into
a clean, strictly-increasing, in-bounds set. In order:

1. `filter_to_standard_percentiles` — keep only the standard labels, drop extras and
   duplicates.
2. `validate_percentile_count_and_values` — assert the count and label set match
   `EXPECTED_PERCENTILE_COUNT` / `STANDARD_PERCENTILES` exactly
   (`numeric/validation.py`).
3. `sort_percentiles_by_value` — order by percentile.
4. `_apply_jitter_and_clamp` (`numeric/pipeline.py`) — detect count-like (integer-
   adjacent) clusters and spread them, jitter exact duplicates, clamp values into the
   question bounds with a safety buffer, then enforce strictly increasing values.
5. `_maybe_widen_tails` — optional tail widening (Step 4).

It also decides whether to force `zero_point=None`: discrete questions and questions
whose `zero_point` equals the lower bound fall back to a linear axis
(`check_discrete_question_properties` in `numeric/validation.py`).

## Step 4: tail widening

`widen_declared_percentiles` (`numeric/tail_widening.py`) can fatten the tails by
scaling each percentile's distance from the median in a transformed space (bounded
logit for closed-closed questions, log transforms for one-sided questions, identity for
open-open). The stretch ramps from zero near the center to a maximum `k_tail` at the
deepest tails.

In production this is an identity pass: `TAIL_WIDEN_K_TAIL` and
`TAIL_WIDEN_SPAN_FLOOR_GAMMA` (`numeric/config.py`) both default to their no-op
settings, so nothing is widened and no span floor is enforced. That is why: an empirical
calibration on 43 resolved numerics found `k_tail=1.0` gave the best-calibrated tails,
and `k_tail=1.25` moved away from ideal in every segment (see
`scratch_docs_and_planning/tail_widening_empirical_calibration.md`). Both knobs are
still per-call configurable, and the function raises `ValueError` if asked to narrow
rather than widen (narrowing is not implemented) or on a negative `span_floor_gamma`.

## Step 5: PCHIP 201-point CDF

`build_numeric_distribution` (`numeric/pipeline.py`) hands the sanitized percentiles
to `generate_pchip_cdf_with_smoothing`, which calls `generate_pchip_cdf`
(`numeric/pchip_cdf.py`). This produces the 201-point CDF the bot actually submits.

The construction:

- Build a value grid from lower to upper bound: linear normally, geometric when
  `zero_point` is set (`build_cdf_value_grid` in `numeric/pchip_cdf.py`, matches the
  Metaculus backend's non-linear spacing).
- Fit a monotone PCHIP interpolator through the declared percentiles (log-space for
  strictly-positive series), evaluate it on the grid, and clamp to `[0, 1]`.
- Blend in a uniform mixture so the minimum step is satisfied before any repair tier is
  reached. This is the primary min-step mechanism, inline in `generate_pchip_cdf`
  (`numeric/pchip_cdf.py`).
- Enforce the min-step and max-step constraints, then re-pin the bounds.

### Server-side constraints

Metaculus validates `continuous_cdf` submissions. The server formulas below are the
upstream contract; the constants in `constants.py` mirror them, and those are what to
read for the value we actually submit. `numeric/config.py` carries grid-scoped aliases
(`MIN_CDF_PROB_STEP`, `MAX_CDF_PROB_STEP`) — note that only the max-step alias imports
from `constants.py`; the min-step one restates the literal, so the two min-step
constants have to be changed together.

- **Length** = `cdf_size`, whose standard-continuous default is `PCHIP_CDF_POINTS`.
- **Min step** per bin `NUM_MIN_PROB_STEP` — no flat segments allowed. The
  server formula is `round(0.01 / N, 9)` where `N = cdf_size - 1`.
- **Max step** per bin `NUM_MAX_STEP` — a spikiness cap. Server formula
  `0.2 * 200 / N`.
- **Strictly increasing**, implied by min step > 0.

`grid_step_constraints` (`numeric/config.py`) is what applies those formulas to a
non-standard grid: the min step is floored at `MIN_CDF_PROB_STEP` so a fine grid never
demands a step below the historical constant, and the max step is clamped at `1.0`
(a probability step larger than that is vacuous). On the standard continuous grid it
returns exactly `(MIN_CDF_PROB_STEP, MAX_CDF_PROB_STEP)`, so continuous questions are
unaffected; a coarse discrete grid relaxes the max step upward, which is what lets a
small-count distribution keep its mass concentrated on the low integers.

`safe_cdf_bounds` (`numeric/pchip_cdf.py`) enforces the max-step rule by
redistributing excess mass while preserving the total, then re-enforces min-step after
the pin-and-cummax pass. `enforce_min_steps` (`numeric/pchip_cdf.py`) does a
forward-then-backward sweep to guarantee every adjacent pair is at least one min-step
apart. `_apply_ramp_smoothing` (`pchip_processing.py`) is a final tilt that adds a
tiny linear ramp when the raw CDF still has a sub-min-step bin.

### Open vs. closed bounds: a one-sided constraint, not a box

This is the subtle part. Bound pinning is **one-sided per tail**, not a clamp on
out-of-bound mass:

- **Closed lower bound** → `cdf[0]` is pinned to exactly `0.0`.
- **Closed upper bound** → `cdf[-1]` is pinned to exactly `1.0`.
- **Open lower bound** → `cdf[0]` is floored at a *minimum* of `0.001`. This is a
  required minimum positive mass, not a cap. The CDF can start well above 0.001.
- **Open upper bound** → `cdf[-1]` is ceilinged at a *maximum* of `0.999`. Again a
  required headroom, not a floor on how much mass sits below the ceiling.

There is no cap on out-of-bound mass. A distribution can legitimately place, say, 78%
of its mass below an open lower bound. That mass is expressed by placing declared
percentile *values* beyond the displayed range (values are not clamped on open bounds),
so that `F(bound)` interpolates to the intended fraction. The only ceiling on
out-of-bound mass comes from min/max-step feasibility (roughly 0.99, since 200 bins each
need at least a min-step). The `_pin_endpoints` helper (`numeric/utils.py`) and the
validation in `_validate_pchip_cdf` (`pchip_processing.py`) both apply this one-sided
logic.

If PCHIP construction fails outright, `create_fallback_numeric_distribution`
(`pchip_processing.py`) delegates the CDF build to forecasting-tools, but still
re-pins open-bound endpoints through `safe_cdf_bounds` (the native builder would anchor
an open lower bound at 0% once the standard set includes P1, which Metaculus rejects).

### The repair-tier signals are dead code on real forecasts

`generate_pchip_cdf` logs several repair signals (`pchip_aggressive`, clamp fractions,
violated-step fractions, ramp-smoothing deltas). On real model output these never fire,
because the uniform-mixture construction pre-enforces the min-step before any repair
tier is reached. Keep the guards (they defend against pathological inputs), but their
absence in logs carries no information, and they must not be used as model-quality
features.

## Step 6: PchipNumericDistribution

The result is wrapped in a `PchipNumericDistribution` (`pchip_processing.py`), a
subclass of forecasting-tools' `NumericDistribution` whose `get_cdf()` override returns
the pre-computed 201-point CDF instead of rebuilding it. `get_cdf()` is the real
override — the `.cdf` property is a deprecated shim that delegates to it, so overriding
`.cdf` alone would miss the publish and aggregate paths. The `_pchip_cdf_values` attribute
also acts as the marker that CDF validation should be skipped (the constraints were
already enforced) and that discrete snapping can read the CDF back out.

## Step 7: discrete integer snapping

Some questions are labeled continuous (`cdf_size=201`) but resolve on integers ("how
many X will happen?"). A smooth CDF wastes mass between integers. If a strict majority
of forecasters voted DISCRETE (`majority_votes_discrete`), `maybe_snap_to_integers`
(`post_processing.py`) snaps the ensemble distribution to a step function
concentrated on integer values.

`snap_cdf_to_integers` (`numeric/discrete_snap.py`) extracts an integer PMF by
half-integer interpolation, rebuilds a step CDF, mixes in a uniform component for
min-step compliance, then runs `safe_cdf_bounds`. It is skipped when the range holds
more than `DISCRETE_SNAP_MAX_INTEGERS` integers (`constants.py`), when there
are no integers in bounds, when bounds are non-finite, or when the question is already
natively discrete (`cdf_size != 201`). Snapping is decided at the ensemble level, after
aggregation.

## Step 8: unit-mismatch guard

Before a per-model prediction is accepted, `detect_unit_mismatch`
(`numeric/validation.py`) checks whether the declared values look off by orders of
magnitude relative to the question range. It flags a mismatch when any of three ratios
falls below its threshold — each threshold is a keyword argument on
`detect_unit_mismatch`, so the signature is where the values live:

- span between lowest and highest declared value, over the range
  (`span_ratio_threshold`);
- minimum adjacent step, over the range (`min_step_ratio_threshold`);
- maximum absolute value, over the range (`max_magnitude_ratio_threshold`).

On a flagged mismatch, `run_numeric_forecast` (`forecaster_runners.py`) raises
`UnitMismatchError` and withholds that forecaster's prediction rather than submitting a
distribution in the wrong units. No network or community stats are needed; it is a pure
sanity check on the numbers.

## Step 9: ensemble aggregation in CDF space

`aggregate_numeric` (`numeric/utils.py`) combines the per-model distributions
**pointwise in CDF space**, not by averaging percentiles:

1. Concatenate every model's 201-point CDF into one long frame of `(value, percentile)`
   pairs.
2. Group by `value` and take the mean or median of the cumulative probabilities at each
   value.
3. `_postprocess_ensemble_cdf` (`numeric/utils.py`) re-pins the endpoints (one-sided
   open/closed logic), enforces monotonicity, applies ramp smoothing if any bin is below
   min-step, and — for discrete questions whose `cdf_size != 201` — resamples the CDF to
   the target grid via PCHIP with the grid-scaled min-step.

Percentile-space averaging would blur multi-modal disagreement; CDF-space averaging
preserves it. In production the base-combine path uses **MEDIAN** of the raw per-model
CDFs (`_base_combine` in `aggregation_pipeline.py`, because the default strategy is
`CONDITIONAL_STACKING` and stacking is disabled in prod). Backtests and the mean arm use
MEAN.

## Step 10: the numeric spread metric

`numeric_percentile_spread` (`spread_metrics.py`) is what decides whether the
ensemble disagrees enough to trigger conditional stacking. It reads each model's P10,
P50, and P90 values (by percentile label, so growing the standard set cannot silently
shift them), takes the max-minus-min spread at each of those three percentiles, and
normalizes:

- **Closed-bound questions** → divide by the question range (`upper - lower`).
- **Open-ended questions** → divide by the ensemble interquartile range (median P90
  minus median P10), since the range is unbounded.

The largest of the three normalized spreads is the reported value. If it exceeds
`CONDITIONAL_STACKING_NUMERIC_NORMALIZED_THRESHOLD` (`constants.py`), the
aggregator extracts the disagreement crux, runs a targeted search, and invokes the
stacker LLM; otherwise it returns the MEDIAN. (Stacking is disabled in all four prod
workflows, so in prod this metric is computed but the stacker branch does not fire; the
chain stays live in backtests and ablation.)

For discrete questions, the per-model `declared_percentiles` have already been
overwritten with a resampled CDF grid whose labels are cumulative probabilities, so
`_key_percentile_values` (`spread_metrics.py`) reads P10/P50/P90 by interpolating the
empirical CDF instead of looking up label nodes.

## The time-series anchor provider

`research/timeseries_anchor.py` is a research provider (not part of the CDF pipeline)
that grounds numeric forecasts whose resolution series is a fetchable FRED or yfinance
series. It renders a deterministic empirical anchor with no LLM: the latest value, a
multi-resolution history, a 52-week range, and a horizon-matched empirical band. Its
section header in the briefing is `## Time Series Anchor` (`TS_ANCHOR_SECTION_HEADER`,
rendered by `_provider_header` in `research/orchestrator.py`). Gated by
`TS_ANCHOR_ENABLED` (`_select_research_providers` in `research/orchestrator.py`); on in
all four workflows.

### Routing (deterministic, no LLM)

`route_question` (`timeseries_anchor.py`) maps a question to a series two ways, URL
first:

1. **URL extraction** from resolution criteria and fine print — a cited FRED series or
   Yahoo ticker is the ground-truth resolving source and wins.
2. **A conservative curated keyword registry** (`_TEMPLATE_REGISTRY` in
   `timeseries_anchor.py`) — 10-year Treasury, VIX, CPI, unemployment, nonfarm
   payrolls, S&P 500, gold, and so on. Deliberately small and unambiguous.

Anything ambiguous (more than one series that is not a two-ticker spread, or more than
one keyword match) returns `""` and logs. Two Yahoo tickers become a relative-return
spread block.

### The empirical band

The band is the naive, model-free choice, deliberately. The Phase-A offline replay
(`scratch/ts_anchor_replay_2026-07-16/synthesis.md`) found that CV-gated statistical
model picks beat the naive out-of-sample only 43% of the time, while the naive empirical
h-step-change band was both sharper and better tail-calibrated. So the provider just
computes empirical quantiles (P10 / P50 / P90) of every overlapping h-step change in the
series' own history and applies them to the latest value (`_empirical_change_band` in
`timeseries_anchor.py`). Log-multiplicative for strictly-positive series, additive
otherwise. "Highest / peak / maximum" questions use a forward-window-max band instead
(`_empirical_max_band`). The horizon `h` is matched to the question's actual
forecast window, converted to native series steps by frequency (`horizon_steps`).

Derived-quantity questions (month-over-month change, MoM % inflation, monthly averages)
fit the band on the derived series, not the raw level (`_apply_derivation` in
`timeseries_anchor.py`).

### Point-in-time leakage safety

This is the only backtest-safe research provider that runs during benchmarks. Others
hard-disable under `is_benchmarking`; this one instead pins `as_of` to
`question.open_time` in benchmarks (live: `datetime.now(UTC)`) and fetches the series
point-in-time up to that date, so data known at forecast time is fair game without
leaking the resolution (`timeseries_anchor_provider` in `timeseries_anchor.py`).

The fetch layer (`research/ts_fetch.py`) enforces the invariant. Revising macro series
(CPI, payrolls, GDP) would leak if fetched from today's FRED, because today's data
contains revised historical values not known at forecast time. So those go through
**ALFRED point-in-time vintages** instead of plain FRED CSV. `fetch_series`
(`ts_fetch.py`) defaults every FRED series to ALFRED vintages *except* a small
non-revising allowlist (`FRED_NON_REVISING_SERIES` in `ts_fetch.py`: market prices and
survey levels like DGS10, Brent, gasoline). That default is fail-safe — an over-inclusive
ALFRED guess costs nothing for a non-revising series, but a revising series routed to
plain FRED would silently leak. A belt-and-suspenders check, `_assert_no_leakage`
(`ts_fetch.py`), raises `LeakageError` if any observation postdates the ceiling.

### Text anchor on, chart off

The provider always returns a text section. It can also render a small chart image
(matplotlib ribbon), stashed in a per-session side-channel for the forecaster's vision
message, but only for plain single-level questions. The chart is gated separately by
`TS_ANCHOR_CHART_ENABLED` (`_maybe_stash_single_chart` in `timeseries_anchor.py`), which
is **off** in all four workflows while the text anchor is **on**. Chart render failures
are swallowed so a plotting hiccup never breaks the text section.


---

## Agent-facing detail (relocated from `AGENTS.md`, 2026-08-26)

The sections below were inline in `AGENTS.md`, which is loaded on every coding-agent
invocation. They are reference material rather than always-true rules, so they live here
now and are read on demand.

Some of this overlaps the prose above — it was written for a different audience and is
kept verbatim rather than merged, so nothing is silently lost. Reconciling the two is
worthwhile follow-up, but it is an editorial pass, not a mechanical one.
### Clamps / bounds

- **Binary**: `[BINARY_PROB_MIN, BINARY_PROB_MAX]` (`constants.py`). Applied per-model in `forecaster_runners.py` and on stacker output in `stacking.py`. Median/mean of already-clamped values stays in-bounds, so no post-aggregation clip needed.
- **MC**: `[MC_PROB_MIN, MC_PROB_MAX]` (`constants.py`), set to match ft 0.2.92's `PredictedOptionList` validator, which clamps every option on construction — matching bounds makes it a no-op and removes publish-time `ValueError` risk on many-option ballots. Drift-free clamp-then-renormalize via `clamp_and_renormalize_probs` (`mc_processing.py`), applied BEFORE every `PredictedOptionList` construction and re-applied idempotently by `clamp_and_renormalize_mc` (`numeric/utils.py`); the repair pass keeps floored options from dividing back below the floor.
- **Numeric CDF**: a `PCHIP_CDF_POINTS`-point grid with every bin's step held between `MIN_CDF_PROB_STEP` and `MAX_CDF_PROB_STEP` (`numeric/config.py`; `grid_step_constraints` rescales both for off-standard grid sizes, relaxing the max on coarse discrete grids). Open bounds are a **one-sided** constraint per tail, not a box: open lower bound → `cdf[0]` is floored at a *minimum* of 0.001 (a required minimum positive mass, NOT a cap on below-bound mass); open upper bound → `cdf[-1]` is ceilinged at 0.999. There is no cap on out-of-bound mass — a distribution can legitimately place e.g. 78% of its mass below an open lower bound. Out-of-bound mass is expressed by placing percentile values beyond the displayed range (values are not clamped on open bounds) so `F(bound)` interpolates to the intended fraction; it's bounded only by min/max-step feasibility (~0.99). Closed bounds are pinned to `{0.0, 1.0}`. Enforced in `numeric/pchip_cdf.py` `generate_pchip_cdf` with aggressive repair + `safe_cdf_bounds` redistribution on max-step violations.
  - **Repair-tier WARN signals are effectively dead code on real forecasts.** The five logged numeric repair signals — `pchip_aggressive`/aggressive enforcement, `clamp_frac`/`clamp_dist` (bounds-clamp corrections), `violated_steps_frac`, and `ramp_smoothing_delta` — never fire on real model output: `generate_pchip_cdf`'s uniform-mixture construction pre-enforces the min-step before any of those repair tiers is reached (0 of 1182 archived numeric forecasts fired any of them; the one degenerate case raises `pchip_failed` instead). Keep the guards — they defend against pathological inputs — but their absence in logs carries no information, and they must NOT be used as model-quality features (verified 2026-07-15; receipts in `scratch/coherence_2026-07-15/synthesis.md`).

### Numeric pipeline (percentiles → PCHIP CDF)

Each forecaster emits the `STANDARD_PERCENTILES` set as plain text (`numeric/config.py` owns the list and derives both `EXPECTED_PERCENTILE_COUNT` and the `STANDARD_PERCENTILES_CSV` label string the prompts interpolate — never restate the percentiles elsewhere; prompt example in `prompts.numeric_prompt`). Per-model stages:

1. Value extraction ladder (`value_extraction.py` `extract_numeric`): deterministic block-first parse of the fenced ```json STRUCTURED FORECAST block into `list[Percentile]`, with json-repair salvage and the parser LLM (`parse_structured`) only as rung-3 salvage. Returns exactly the canonical percentile set (or raises `ValueExtractionError`).
2. `sanitize_percentiles` (`numeric/pipeline.py`): filter to the canonical set, validate, sort, spread count-like clusters, jitter duplicates, clamp to bounds, ensure strictly increasing, optionally widen tails.
3. `widen_declared_percentiles` (`numeric/tail_widening.py`): bound-aware stretch of distance-from-median by `TAIL_WIDEN_K_TAIL`, whose default is the identity value — widening only kicks in when a caller raises it. `TAIL_WIDEN_SPAN_FLOOR_GAMMA` gates span-floor enforcement on a positive value and defaults off. Both knobs are configurable per-call.
4. `build_numeric_distribution` → `generate_pchip_cdf_with_smoothing` produces the `PCHIP_CDF_POINTS`-point PCHIP CDF → ramp smoothing for min-step → validation. On failure: `create_fallback_numeric_distribution` delegates CDF build to forecasting-tools.
5. **Discrete integer snapping**: if a majority of forecasters vote DISCRETE, snap the distribution to integers.
6. **Unit-mismatch guard** (`numeric/validation.py` `detect_unit_mismatch`): withholds the prediction if values look off by orders of magnitude.

### Value extraction ladder (`value_extraction.py`)

Since 2026-07-10, forecast values (binary probability, MC option probabilities, numeric percentiles) come out of the fenced ```json STRUCTURED FORECAST block via a deterministic four-rung ladder — the block is the authoritative source, and the parser LLM survives only as a salvage rung. `extract_binary`, `extract_mc`, and `extract_numeric` share the same driver and are called by both `forecaster_runners.py` (per-model forecasts) and `stacking.py` (stacker output), so both flows share one extraction path:

1. **block** — deterministic parse of the fenced ```json block (`parse_structured_block`: `json.loads` + Pydantic validation against `BinaryStructured` / `MultipleChoiceStructured` / `NumericStructured`). This absorbs the old numeric_format_router F5 block-lift: when the schema-valid numeric block carries a `declared_percentiles` dict, it is lifted directly into `Percentile(percentile=..., value=...)` objects.
2. **repair** — deterministic JSON repair (`json_repair`) of a malformed fenced block, or a balanced-braces scan of the rationale's trailing `_TAIL_SCAN_CHARS` when no fence survived.
3. **llm** — the existing LLM parser (`parse_structured`) run over the full rationale as a last-resort salvage. Post-rung validation stays strict (bounds, the full canonical percentile set for numeric, option-set match for MC) so the parser can never smuggle in a fabricated value.
4. Raise `ValueExtractionError` — the numeric forecaster (or the stacker) has no usable value source; the caller drops or soft-fails the forecaster exactly as parser failures propagated before the ladder.

Every successful extraction emits one `EXTRACTION_RUNG: question=... model=... qtype=... rung=... block_present=...` INFO log line — this is the drift-signal telemetry that superseded the deleted shadow-divergence comparison. Watch `rung=llm` salvages and `block_present=false` in prod logs; those are the two symptoms that indicate a forecaster stopped emitting a well-formed block. A sibling WARN, `OPEN_BOUND_PILING: question=... model=... bound=... bin_mass=... declared_edge=... bound_value=...` (`numeric/diagnostics.py`, threshold `OPEN_BOUND_PILING_THRESHOLD` in `numeric/config.py`), fires when a model piles at least that fraction of mass on the terminal displayed bin of an *open*-bound numeric question without declaring any percentile beyond the edge — the "crammed the open ceiling" failure mode fixed 2026-07-12 by rendering nominal/displayed bounds in the numeric prompts (`nominal_bounds` in `numeric/utils.py`). It takes the pre-resample model-declared percentiles explicitly (the discrete resample overwrites `prediction.declared_percentiles` with a grid pinned to the raw bounds, which would defeat the above-edge exemption). Since 2026-07-12 every bot-run workflow tees stdout+stderr to a `run_logs/` artifact, so this telemetry is durably grep-able after the fact; as of 2026-08-03 that artifact is named `research-<run_id>` in all of them (the two test workflows used `logs-<run_id>` before that, and `RUN_LOG_ARTIFACT_PREFIXES` still harvests those). Runs also emit OpenRouter credit telemetry (`metaculus_bot/credit_telemetry.py`, wired in `cli.py`): `CREDIT_BALANCE: key=... phase=start|end remaining=... usage=...` per key, `CREDIT_SPEND: key=... run_delta_usd=... remaining=...` at end of run, and a `CREDIT_FLOOR_BREACH` WARN when the donated key drops below `OPENROUTER_CREDIT_FLOOR_USD` (constants.py) — a breach lets the run complete/publish normally, then exits non-zero as a refill reminder ONLY when credit alerting is active (see below).

**Credit alerting is suppressed until 2026-09-10.** The operator is funding the rest of the season out of pocket, so a drained donated key is expected rather than broken, and until `CREDIT_ALERT_RESUME_DATE` (constants.py, `2026-09-10`, a few days past `TOURNAMENT_END_DATE`) a credit shortfall must not redden CI. Both paths that turned a shortfall into `sys.exit(1)` are gated on `credit_alerts_active()` (reads the system clock at CALL time, accepts an injected `today` for tests, so the resume needs no redeploy): (1) the floor breach in `cli.py` skips the exit and logs an INFO line naming the resume date; (2) the credit-caused donated→personal key fallbacks are tracked in `_credit_key_fallback_count`, a strict subset of the all-causes `_generic_key_fallback_count` (same relationship `_donated_404_fallback_count` has), and `cli.py` subtracts that subset from `alertable` while suppressed. All three counters are incremented in one place, `record_donated_key_fallback` (`fallback_openrouter.py`), and each event is counted exactly once: generic adds it, at most one subset subtracts it. The accounting block there must stay free of any `await` after the threaded probe — `+=` on a module global is interruptible between bytecodes, so an await would let N forecasters failing on one dry key race the increment and take a degraded run green. **Every other fallback cause stays fully alertable** — 401 invalid/disabled key, 404 "no allowed providers", 429 rate limit, guardrail/data-policy — as does all bot-side degradation. No log line is silenced: all `CREDIT_*` markers and every `PAID PERSONAL-KEY FALLBACK` WARN still fire, and only the exit status plus the alertable arithmetic change. Credit classification lives in one place (`_is_credit_failure` / its public form `is_credit_caused_error` in `fallback_openrouter.py`) so the retry decision and the counter classification can't drift apart. After the resume date, behavior is exactly what it was before the suppression.

**A dry donated key returns 403, not 402, and only a genuinely DRAINED one is suppressed (2026-07-26).** OpenRouter reports a breached per-key spend cap as HTTP 403 with the message `Key limit exceeded (total limit)`; litellm has no 403 branch for OpenRouter, so it always arrives as a bare `APIError` carrying `"code":403` in the body. The old negative rule vetoed any message containing "403" (written for content moderation), so a drained donated key never fell back — a tournament run lost 2 of 3 forecasters, native search, the AskNews summarizer, the financial classifier, prediction-market keyword extraction, and both gap-fill passes while the operator's funded personal key sat idle. The classifier now matches `key limit exceeded`; the **full phrase is load-bearing** because the shorter `limit exceeded` is a substring of `rate limit exceeded: free-models-per-day` and would exempt every 429 from alerting. Because a key Metaculus revoked or re-capped to zero produces the identical 403 text, `credit_telemetry.classify_donated_key_state` probes the free read-only `/auth/key` endpoint once per run (cached; bounded by `DONATED_KEY_PROBE_TIMEOUT_S`, which httpx applies PER NETWORK OPERATION rather than as a cap on elapsed time — the hard total cap is the `asyncio.wait_for` that `record_donated_key_fallback` wraps around the threaded probe, which unblocks the caller on schedule but leaves a trickling probe running orphaned) and `is_suppressible_credit_error` exempts ONLY the `drained` verdict — `zeroed` (cap set to 0), `revoked` (401/404), `funded` (money remains, so the failure wasn't about credit), and `unknown` (probe failed) all keep CI red, and the verdict rides the end-of-run summary as `donated_key=<state>`. That probe belongs to the AMBIGUOUS spend-cap 403; the unambiguous 402 / insufficient-credit family is suppressed without probing at all (deliberate — it predates the discriminator, so an unreachable `/auth/key` can't change its long-standing behavior). Fallback ROUTING reads the status the provider reported (`llm_retry.llm_status_code`, an int already on the exception) and never a live balance: a reported 403 falls back only on the spend-cap phrase or route-scoped wording, a reported 402 always falls back, and a statusless exception falls back on text alone. The probe is consulted for ALERTING only, so a stale `funded` read can never strand the ensemble on a dry key. Related hardening: a bare `402` substring is now trusted only when the body doesn't look like a moderation refusal, since OpenRouter replays up to ~100 chars of our own prompt as `flagged_input` and forecasting prompts routinely contain dollar figures. Full detail in `docs/operations.md` "What a dry donated key actually returns".

Discrete-integer snapping is decided separately by the C3 block-read in `forecaster_runners.run_numeric_forecast` — it reads `NumericStructured.outcome_type` directly before falling back to a parser LLM call for `OutcomeTypeResult`, and the ladder does not touch that path.

**Ensemble aggregation** (`aggregate_numeric` in `numeric/utils.py`): pointwise **in CDF space** — concatenate each model's `PCHIP_CDF_POINTS`-point CDF, groupby value, mean or median the probabilities, then `_postprocess_ensemble_cdf` re-pins endpoints, enforces monotonic + min-step, resamples via PCHIP for discrete questions. Not percentile-space averaging.

