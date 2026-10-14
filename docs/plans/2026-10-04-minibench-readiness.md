# MiniBench kickoff assessment, 2026-10-04

> **Superseded.** The separate grant runner, its flags and `grant_route_for` described
> here were removed; grant runs now use `python -m sibyl --grant-budget-usd`. See
> [one forecast route](../decisions/2026-10-14-one-forecast-route.md) and
> [experiment-funding.md](../runbooks/experiment-funding.md).

The current **Forecast on new Minibench questions** workflow is not ready for the
requested grant-only launch. Manual GitHub Actions execution is a reasonable
eventual operating method, but the checked-in workflow still runs the older
funding configuration. No forecast workflow was dispatched during this review.

## Competition target

The [active MiniBench page](https://www.metaculus.com/tournament/minibench/) lists
October 5, 2026 as its start, October 23 as its end, and 60 questions. The pinned
framework resolves `CURRENT_MINIBENCH_ID` to `minibench`, which matches the
[official resources](https://www.metaculus.com/notebooks/38928/docs.adj.news).
Those resources require the participant form and private forecast comments.
Form completion was not verified in this review. Check actual open questions and
their submission deadlines at execution time; a start date is not an opening time.

## What the workflow currently does

- Runs `python -m sibyl --mode minibench` with the production tier. This reaches the
  correct current tournament and skips questions already forecasted.
- Uses the ordinary production roster, including direct Anthropic, and enables
  AskNews, other paid research, and multiple enrichers. It does not invoke
  `sibyl.ops.experiment_grant` or supply the grant credential.
- Has no dispatch inputs for question count or budget. The CLI currently defaults
  to ten questions, so a single dispatch is not full coverage of a 60-question round.
- Is manual-only and serializes its own runs. It will not start tomorrow by itself
  or collect later questions unless someone runs it again.
- Streams output through `tee` with unbuffered Python and preserves the exit code.
  It uploads research and logs, but the ordinary CLI sets its report directory to
  `None`; it does not produce the grant runner's manifest and full report archive.
- Has no repository Actions secrets configured (`gh secret list`, October 4).
  A local `.env` is not available to a GitHub runner.

## Recommended first participation

Use the grant runner locally for the first round, after an independently authorized
unscored test succeeds. This keeps the existing private SQLite budget ledger and
question/report archives together. Open a visible terminal, stream and retain the
log, inspect the current question list, then run an explicitly budgeted batch with
a unique run ID. Further batches skip already forecasted questions. Preserve one
configuration for the measurement cohort and use the archived run collection and
per-type scoring tools; do not treat a partial or degraded run as a clean benchmark.

Moving that path into Actions requires the grant and Metaculus secrets, explicit
batch/budget inputs, persistent private accounting shared across runs, and private
manifest/report storage. A fresh SQLite file in every ephemeral runner would reset
the apparent allocation. Public Actions artifacts are not appropriate storage for
the private grant ledger and reports. Simply replacing the CLI command is insufficient.

## Live test findings

Run `grant-test-2026-10-04-q43325` selected only question 43325 in
`bot-testing-area`. Its report list is empty, and the subsequent read-only API
check showed `already_forecasted=False`. No publication occurred.

- Research and two successful model responses could not be reconciled to provider
  billing. The original code swallowed the billing lookup failure and discarded
  any captured generation ID on the unknown-cost path. Existing artifacts cannot
  distinguish missing callbacks from failed generation lookups. A longer callback
  wait alone is not a demonstrated root-cause fix.
- Gemini failed with `RateLimitError`. Its original response detail was discarded,
  so quota exhaustion versus transient upstream rate limiting remains unknown.
- The outer retry loop treated the grant wrapper error as retryable, producing four
  Gemini attempts. The provider research error was swallowed by normal multi-provider
  fail-soft handling, so frontier calls ran despite unconfirmed research charges.

Offline regression tests now cover preserving generation IDs, sanitized billing
HTTP diagnostics, stopping uncertain grant retries, and refusing to forecast after
unconfirmed research. Real LiteLLM with mocked HTTP passes for all three frontier
routes. The next paid test remains necessary to verify live provider behavior;
none was performed for this investigation.
