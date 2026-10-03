# Evaluation integrity: implementation handoff

Date: 2026-09-25
Status: engineering fixes implemented; scientific registration pending
Audience: maintainers and coding agents reviewing evaluation invariants

## Objective

Make Sibyl's evaluation machinery enforce its stated experimental contract: frozen
paired inputs, explicit degradation accounting, honest uncertainty, prospective power
checks, and enough provenance to compare historical and forward-looking experiments.
Fix the fixture-mode spend escape first. Preserve production forecasting behavior
unless a task below explicitly calls for a change.

Read [AGENTS.md](../../AGENTS.md), [evaluation strategy](../eval-strategy.md),
[architecture](../architecture.md), and the
[cost-tiered study plan](../decisions/2026-08-27-cost-tiered-plan.md) before implementation.
The strategy contains intended behavior that is not all implemented. Do not describe
completion of this plan as completion of either scientific study.

## Boundaries and working rules

- No paid API calls, workflow dispatches, publishing, `make e2e_free`, or live tests
  are authorized by this plan. Each such run needs its own operator authorization.
- Keep the test network-egress guard intact. Implement and validate with fixtures,
  mocks, synthetic outcomes, and temporary directories. Do not read or print secrets.
- Work on a feature branch, never commit on `main`. Inspect the working tree first
  and preserve unrelated work. Use small commits; separate extraction from behavior
  changes. Write reproducing tests before changing behavior or restructuring.
- Before every commit run `make test`, `make lint`, and `make typecheck`. Do not hide
  existing failures, weaken assertions, or re-record goldens to obtain a green build.
- Keep `forecasting-tools==0.2.92` unless a separately justified dependency change is
  necessary. Preserve historical telemetry names and existing CLI aliases.
- Never rewrite historical experiment artifacts or caches in place to make them
  appear to have provenance they did not record. Use versioned readers and explicit
  migration or descriptive-only handling.
- Do not build a live router, tune prompts, fit new calibration parameters, or run a
  scientific experiment as part of these fixes. Record unrelated discoveries in the
  roadmap rather than expanding this workstream.

## Review evidence and baseline

The review ran `.venv/bin/pytest -q`, `.venv/bin/ruff check .`, and
`.venv/bin/basedpyright` without changing repository files.

- Tests: **5,432 passed, 2 failed, 21 skipped, 7 deselected**.
- Ruff: passed. Basedpyright: zero errors, warnings, or notes.
- Existing failures:
  - `tests/unit/test_tournament_dates.py::TestTournamentConfigFreshness::test_tournament_not_expired`:
    configured `summer-futureeval-2026` ended September 6; its September 20 hard stop
    had passed on the review date.
  - `tests/unit/test_anthropic_direct_routing.py::test_effort_xhigh_is_actually_supported_by_the_pinned_model[forecaster-]`:
    `claude-opus-5` was absent from the installed LiteLLM model map. This proves a
    local validation gap, not that the remote API rejects the model.

Offline reproductions confirmed:

1. `run_backtest(--fixture)` reaches `screen_research_for_leakage` before fixture
   construction. Mock fetching and make screening raise a sentinel to reproduce
   without any network or model invocation.
2. `paired_summary([-0.1])` yields a zero-width 95% interval, MDE zero, and a
   comparison verdict of “B is better than A.”
3. Two inputs listing 100 ground truths, 99 failures per arm, and one shared scored
   question produce “1 seen, 0 dropped.” A `degraded` field on the score is ignored.
4. A cached median is returned even after replacing its forecaster inputs.
5. A historical prompt under `patched_window_for_question` simultaneously says
   `Today: 2025-01-30` and `open and unresolved as of 2026-09-25`.

Reproduce against the current checkout; these observations are a dated baseline.

## Order of implementation

| Step | Deliverable | Depends on |
| --- | --- | --- |
| 1 | Fixture and evaluation spend isolation | None |
| 2 | Honest small-sample uncertainty | None |
| 3 | Versioned experiment manifest and paired eligibility | None; needed by later integration |
| 4 | Content-valid experimental caches | Step 3 identity conventions |
| 5 | One explicit forecast timestamp | Step 3 forecast context |
| 6 | Fail-closed research preparation and frozen replay | Steps 3–5 |
| 7 | Production-faithful baseline and real fixture A/A | Steps 1, 3, 5, 6 |
| 8 | Prospective power and endpoint contract | Steps 2–3; operator decisions for pooled endpoint |
| 9 | Operational study preregistration and forward provenance | Steps 3, 7–8; operator scientific choices |
| 10 | Resolve existing configuration checks; final validation | Independent diagnosis; finish before handoff |

These are work packages, not single commits. Implement each in reviewable increments.
Continue mechanical fixes while scientific choices remain unresolved; do not invent
those choices or treat elapsed time as approval.

## 1. Close fixture-mode and evaluation spend escapes

Primary files: `sibyl/eval/backtest/cli.py`, `sibyl/eval/backtest/leakage.py`,
`sibyl/eval/benchmark/fixture_llm.py`, `sibyl/eval/benchmark/bot_factory.py`, `Makefile`.

- Branch into fixture research immediately after obtaining the question snapshot,
  before provider selection or leakage-detector setup. Use canned fixture research
  and fixture LLMs for every reachable model slot. Do not screen canned research with
  a real detector. Mark fixture provenance explicitly.
- Audit reachable evaluation model and research construction against the free-tier
  contract. The ordinary backtest currently uses hard-coded paid forecasters and a
  paid detector; merely setting the process tier to free must not leave those paths
  paid. Resolve the complete evaluation configuration before any call, refuse an
  inconsistent tier, and require explicit prod configuration for paid execution.
- Keep fixture and real-research preparation separate enough that a later refactor
  cannot accidentally put paid preparation above the fixture branch.
- CLI fixture mode may retain its documented read-only Metaculus fetch. Tests must
  mock that fetch; provide a fully offline snapshot path for repeatable validation.

Acceptance tests:

- Invoke `run_backtest`/CLI with fixture arguments, mocked question fetching, and
  tripwires on real screening, research selection, and paid invocation. Complete the
  pipeline without hitting any tripwire, including with paid-provider flags present.
- Missing fixture research cannot trigger live research. All helper slots are fixtures.
- Unset/free tier cannot invoke a paid evaluation slot. Prod configurations can be
  constructed and inspected offline without making calls.
- Add coverage beyond `_create_fixture_bots`: constructor-only tests missed this bug.

## 2. Represent insufficient uncertainty honestly

Primary files: `sibyl/eval/stats.py`, `sibyl/eval/compare.py`,
`sibyl/eval/ablation/scoring.py`, `sibyl/eval/ablation/scoring_report.py`.

- For fewer than `BOOTSTRAP_MIN_N` usable pairs, retain descriptive point estimates
  but represent intervals as unavailable. Use a consistent internal representation
  and JSON `null` externally; never print a zero-width interval as a 95% CI.
- For fewer than two observations, SD and MDE are unavailable, not zero. Do not add
  a made-up variance floor. Label zero empirical variance at larger n accurately;
  it does not establish universal zero uncertainty.
- Add an explicit inference status used by every renderer. Suppress better/worse
  inferential verdicts when the interval is unavailable or the run is ineligible.
- Validate finite deltas and numerical parameters. Non-finite observations must
  become explicit invalid-score/drop records upstream, not vanish from the denominator.
- Preserve seeded reproducibility for supported bootstrap samples. Keep sign and
  Wilcoxon results distinct from mean-difference inference; they test different things.
- Remove the verdict logic that infers what effects were ruled out merely from
  `abs(mean_delta) < MDE`. An equivalence claim needs an explicit margin and interval.

Acceptance tests: n=0, 1, 2, 4, and 5; positive and negative one-pair deltas; all ties;
constant nonzero deltas; NaN/infinity; invalid alpha/power/counts; deterministic seeds.
Check both compare and ablation reports. Update the test that currently requires a
collapsed interval, documenting why that behavior was incorrect.

## 3. Establish experiment identity and count every scheduled question

Primary files: `sibyl/eval/backtest/{cli,analysis,question_prep,scoring}.py`,
`sibyl/eval/compare.py`, `sibyl/eval/ablation/{cli,scoring,stage_payload}.py`.

Introduce a small versioned schema shared by the existing evaluation paths. Avoid a
second parallel framework. It needs three linked records:

| Record | Required information |
| --- | --- |
| Prepared input manifest | Schema version, manifest/content ID, full question snapshots and ordered IDs, resolutions or linked outcome snapshot, per-question forecast timestamp, research hash/provenance/status, cohort fields, selection seed and eligibility exclusions |
| Experiment/arm specification | Experiment ID, arm/variant ID, code SHA plus dirty-state information, prompt/config hashes, model and helper configurations, provider policy, aggregation/post-processing settings, declared treatment, endpoint/scoring version, manifest ID, power-plan ID |
| Per-question arm result | Question and arm IDs, status, degradation reasons, requested/actual models and survivor counts, forecast or artifact link, scores, timing, available cost/call accounting, input identity |

Never hash or serialize secrets. Unknown cost is unknown, not zero. A dirty checkout
must be distinguishable from the clean commit; either capture a safe code fingerprint
or refuse to label its output a reproducible registered result.

- Record every scheduled question even when forecasting raises or returns no report.
- Define a shared pair-eligibility function: an error, timeout, model substitution,
  disallowed fallback, missing required metric, invalid score, or required unknown
  provenance in either arm excludes that question from the confirmatory pair.
- Distinguish model substitution from key/transport retry to the same model. Preserve
  retry telemetry without automatically changing the scientific arm's identity.
- Normal specified post-processing is not degradation. Make this distinction explicit
  so clipping or the resolution guard is not accidentally counted as a failed forecast.
- Separate preparation exclusions from failures after the experimental manifest is
  frozen. Report both. Compute experiment drop rates against the fixed scheduled set.
- Enforce the default `>5%` refusal per declared comparison, including ablation.
  Exactly 5% follows the existing rule. Diagnostic artifacts may still be written,
  but a refused run cannot print an inferential success summary or exit as one.
- Validate unique IDs/metric keys, matching input identities and resolutions, and
  compatible endpoint definitions. Do not silently overwrite duplicate scores.
- Legacy files without enough evidence may remain readable for descriptive analysis;
  label provenance unavailable and refuse registered/inferential certification.

Acceptance tests:

- Both arms fail the same 99/100 questions: denominator 100, 99 drops, refused.
- One arm fails; both fail; degraded-but-scored fallback; missing secondary metric;
  duplicated score; mismatched ground truth; mismatched research or snapshot IDs.
- 5% and just-over-5% boundaries; per-comparison eligibility with more than two arms.
- Missing metadata is never silently interpreted as a clean run.
- Warn that complete-case results describe survivors; a 5% gate alone does not prove
  missingness is ignorable. Report arm-specific failure counts and reasons.

## 4. Make cache reuse depend on the actual inputs

Primary files: `sibyl/eval/ablation/{cache,cli,forecasters,run_simple_agg,run_pdf,run_stacker}.py`.

- Fingerprint canonical stage inputs: question/forecast context, research bytes,
  prompt and model configuration, parser configuration, algorithm/scoring version,
  and upstream predictions as applicable.
- Aggregation identity must include the exact surviving forecaster inputs. A shared
  `arm_median.json` filename is not evidence that the same computation was requested.
- Store and validate fingerprints on read. Either content-address artifacts or use
  explicit experiment namespaces plus validated input hashes. Schema version alone
  is insufficient.
- Treat old unverified entries as legacy/stale. Never silently reuse them as fresh.
  Avoid automatic paid recomputation outside an explicitly authorized run.
- Retain force-stage cascading, but do not rely on the operator to remember it.
- Bind research metadata to blob bytes with a hash so interrupted two-file writes
  cannot validate mismatched content. Handle failure caches with an explicit retry
  policy; claiming to retry must not simply reload the same error payload.

Acceptance tests: unchanged inputs hit; changing research, prompt, effort, parser,
lineup, individual forecast, survivor set, forecast date, or relevant algorithm
version misses. Swapping free to prod in the same directory cannot reuse a free
median. Test cache-only resume and interrupted research-pair writes using temp dirs.

## 5. Inject one forecast clock through historical execution

Primary files: `sibyl/forecasting/prompts.py`, `sibyl/aggregation/pipeline.py`,
`sibyl/eval/ablation/window_patch.py`, and backtest/ablation runner call sites.

- Introduce an explicit forecast context carrying a timezone-aware UTC `as_of` time.
  Pass it through prompt builders and time-dependent post-processing. Keep production
  defaults equivalent to current behavior when no historical context is supplied.
- Replace the partial module-global date monkey patches with explicit context, or a
  task-local mechanism if passing context requires disproportionate API changes.
  Historical contexts must not leak across concurrent questions.
- Audit `_forecasting_window_str`, `_today_str`, gap-fill year examples, research
  date logic, and the resolution guard. Capture one timestamp per forecast event.
- Make the historical timestamp policy explicit in the manifest. Mid-window is an
  existing ablation convention, not a validated proxy for the live forecast horizon.
- Distinguish simulated forecast time from actual execution/research retrieval time.
  Changing prompt dates does not make modern search or model knowledge non-leaky.

Acceptance tests: all dates agree in binary/MC/numeric and stacker prompts; concurrent
questions retain distinct contexts; historical guard firing matches the injected
time; timezone boundaries work. Production golden scenarios remain byte-identical
under their fixed test inputs; investigate any change rather than re-recording.

## 6. Prepare research once, fail closed, and replay exactly

Primary files: `sibyl/eval/backtest/{leakage,cli,question_prep}.py`,
`sibyl/research/orchestrator.py`, and research persistence/archive readers.

- Use typed screening outcomes: clean, leaked, unknown/failed. An exception or
  malformed detector response is unknown, never clean.
- Retain screening evidence and preparation exclusion reasons. Freeze eligibility
  only after preparation; changing it later creates a new manifest.
- Add manifest export/load to the existing CLI. Run both arms on the same question
  snapshots and research bytes instead of re-fetching and re-sampling independently.
- Cache-only replay must fail or mark ineligible on missing/empty/invalid research;
  it must not fall through to live research. If an explicit live-fill preparation
  mode exists, screen and snapshot its output before either arm runs.
- Validate archive record identity, retrieval timestamp, source, and applicable
  `as_of` cutoff. Later research must not be labeled historical non-leaky evidence.
- Record research provenance categories, including live-after-resolution, historical
  artifact, reconstructed comment, and fixture. Preserve residual model-leakage caveats.
- Only changes to research itself may vary research between arms, and then the declared
  treatment must identify that difference. All other shared inputs remain frozen.

Acceptance tests: detector timeout/malformed output; research failure followed by
attempted fallback; missing archive; partial archive; empty blob; wrong identity;
post-cutoff archive; exact replay after the mock API changes; reproducible selection.

## 7. Evaluate the intended bot and exercise a real offline A/A path

Primary files: `sibyl/eval/benchmark/bot_factory.py`, `sibyl/config/models.py`,
`sibyl/eval/backtest/cli.py`, `sibyl/eval/ablation/forecaster_lineup.py`,
`tests/e2e/test_fixture_aa.py`, `Makefile`.

- Build a production-baseline ensemble from the authoritative roster/configuration
  source, with the production aggregation and post-processing configuration. Do not
  copy current model names into another static catalog.
- Keep individual-model evaluation as an explicit named mode for compatibility.
  Do not describe three separate single-model bots as a three-member ensemble.
- Label the historical ablation “prod-ish” lineup accurately; preserve old results'
  identities rather than rebranding them as current production.
- Make research policy explicit. The ordinary screener currently caches a single
  provider result and bypasses the orchestrator's full fan-out. A historical-safe
  provider subset is valid but is a declared deviation from full live production.
- Make the fixture A/A exercise runner → forecasts → aggregation → persisted results
  → comparison, twice against one frozen snapshot. The present A/A test compares
  manually constructed score rows and cannot certify the runner.
- Adapt the fixture A/A Make target to prepare once and compare the produced arms.
  Avoid timestamp filename collisions and require distinct arm output identities.

Acceptance tests: offline construction matches production roster/settings without
calls; individual-mode compatibility; shared research hash for both arms; A/A gives
exactly zero deltas and zero drops; a deterministic perturbed fixture gives known
nonzero deltas. Verify the ensemble is actually aggregating multiple predictions.

## 8. Add a prospective power plan and define score semantics

Primary files: `sibyl/eval/stats.py`, evaluation CLI/manifest code,
`sibyl/eval/backtest/scoring.py`, `docs/eval-strategy.md`.

- Correct the documentation's distinction between CI half-width and MDE. Under the
  existing normal approximation, 95% half-width is `1.96 * sd / sqrt(n)`; two-sided
  80%-power MDE is `(1.96 + 0.842) * sd / sqrt(n)`.
- The current reference assumption gives half-width about 0.01648 and MDE about
  0.02355 at n=60. Label this as an extrapolation, not measured power for Sibyl.
- Add a no-network preflight that consumes a versioned power plan: primary endpoint,
  expected effect, planned usable pairs, alpha, power, variance estimate and its
  independent source, attrition allowance, and decision. Persist it before model calls.
- Refuse a registered run whose plan is missing, incompatible, or underpowered under
  the stated rule. Write a decline record with arithmetic. Structural fixture smoke
  tests are exempt from scientific power requirements and clearly labeled as such.
- Post-run empirical MDE may remain as a diagnostic, distinctly labeled; it cannot
  substitute for a prospective plan or establish equivalence.
- Validate basic formula behavior and add seeded synthetic coverage/type-I-error
  checks with broad statistical tolerances. A/A alone is not a coverage test.

Endpoint decisions requiring operator input:

- Binary Brier and log scores already exist. Numeric and MC backtest outputs do not
  currently provide the strategy's common normalized pooled loss.
- Propose an explicit normalized numeric RPS definition, MC loss definition, treatment
  of open numeric tails, type weighting, and primary/secondary endpoint hierarchy.
  Show examples and scoring tests. Do not silently pool incompatible units or use
  raw dimensional CRPS as normalized RPS.
- Until the pooled endpoint is agreed and versioned, report per-type endpoints and
  power plans only. Do not attach the pooled 0.016/0.024 figure to them.
- Resolve multiplicity for confirmatory contrasts/endpoints; mark other slices and
  arm comparisons exploratory. Correlated question families may require grouped
  sampling/resampling; document the independence assumption and available group IDs.

## 9. Turn both study ideas into executable, auditable specifications

### Cheap agreement and routing validity

Update the study plan with an explicit unapproved draft specification for the operator
to resolve. Required fields: eligible population, cheap/frontier configurations,
per-type spread definitions, fixed bucket boundaries or prespecified construction rule,
material cross-tier disagreement threshold, primary statistical contrast, score-loss
margin, tail-loss measure, missingness rule, multiplicity policy, and proceed/stop/
inconclusive criteria. Include power for the actual bucket/routing endpoint; pooled
mean power does not establish power for concentration across small buckets.

Measure both tiers on every question. Distinguish cross-tier disagreement from
outcome-based error: disagreement alone cannot identify which tier is wrong or
confirm correlated error. Evaluate a prespecified routing policy's paired score loss
against always-frontier. Any threshold selected from this dataset is exploratory and
needs held-out evaluation. Do not build or enable the router in this workstream.

### Historical results predicting forward performance

Primary additional files: `sibyl/eval/performance/{collector,parsing,analysis}.py`
and forecast/research persistence call sites.

- Link forecast events to immutable variant/configuration IDs, question IDs, `as_of`
  times, predictions, model survivors, and research identity. Distinguish actual
  published predictions from comment-derived reconstructions.
- Match comments to forecast events through explicit identity or justified timestamp
  rules. Independently taking the latest forecast and latest comment is insufficient;
  ambiguous joins must remain unknown rather than being guessed.
- Keep forecast histories needed for a prespecified horizon. Preserve retrospective
  and forward outcome snapshots and evaluation versions.
- Draft the study around multiple frozen variants or prespecified paired changes,
  observed on comparable forward questions/horizons. One bot's aggregate backtest and
  live scores cannot establish rank prediction across variants.
- Define held-out forward collection, leakage exposure, ranking/association endpoint,
  uncertainty, and stopping/sample-size rules before outcomes are inspected. Do not
  treat a tournament used for tuning as an untouched holdout.

Implement provenance and validators now; leave empirical thresholds and final study
registration visibly pending operator decisions. Legacy artifacts may lack enough
information to reconstruct exact events; document that limitation without fabrication.

## 10. Resolve existing failures without weakening safeguards

- Tournament configuration: verify the intended current competition using official
  read-only information and the operator's target. Do not guess a season or extend
  the hard-stop date merely to pass a test. Preserve expiration enforcement. If the
  bot is intentionally dormant, document that and keep deployment blocked explicitly.
- Anthropic compatibility: inspect the installed/locked LiteLLM version, bundled
  model metadata, and official model/parameter documentation. Separate a missing
  cost-map entry from request-transform support and remote model availability.
  Validate request construction offline. Make the smallest justified dependency or
  configuration correction; do not silently downgrade the model or reasoning effort.
- No paid smoke call is authorized. Report any capability that remains unverified.

## Final acceptance and implementation handoff

The coding agent's completion report must include:

- A checklist against steps 1–10, with implemented changes, remaining decisions,
  legacy compatibility limits, and anything still blocked.
- Reproduction and regression-test results for all five original offline failures.
- Full `make test`, `make lint`, and `make typecheck` results, including any remaining
  baseline failures. Do not claim a green build while either baseline failure remains.
- Confirmation that golden parity was checked and no snapshots were re-recorded
  without investigation; production behavior changes, if any, called out separately.
- A fully offline fixture A/A result using frozen inputs, plus an intentionally
  degraded run that refuses inference and an underpowered preflight decline artifact.
- Safe example commands using temporary/synthetic data for prepare, preflight, run,
  and compare. Clearly distinguish these from commands capable of spending or publishing.
- Updated documentation/index/roadmap links and migration guidance. Historical results
  affected by uncertainty, drop accounting, cache identity, or clock bugs should be
  flagged for re-analysis where sufficient original data exists; do not silently
  invalidate or relabel every historical result without checking its execution path.

Completion of the engineering work means the harness can enforce a valid experiment.
It does not mean routing works, leakage cancels, or retrospective rankings predict
forward results. Those remain empirical questions for separately authorized studies.

## Implementation validation note (2026-10-01)

The production pipeline golden scenarios 1–4 remained byte-identical. Scenario 5
`eval_aggregate_scores` moved for the intended small-sample correction: its two-pair
confidence bounds had been recorded as point estimates, although an interval is
unavailable below five pairs. The corrected JSON uses `null` bounds. That one
fixture was deliberately re-recorded after inspecting the exact old and new values.
The surface-inventory check also detected the plan's quoted Python/pytest module invocation as a new CLI surface; that prose was corrected to the equivalent
`.venv/bin/pytest` invocation without changing runtime behavior or its golden file.
