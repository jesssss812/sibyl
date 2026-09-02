# 0002 — What the clean fork deliberately did not rewrite

*2026-09-01. Status: in force.*

## Context

The fork's purpose was to move working code into a layout that explains itself, not to
improve the code. Those two jobs look similar from inside a diff and are very different
to review: a restructure whose diff also contains behaviour changes cannot be checked
against a parity record, because any mismatch has two possible causes.

## Decision

No logic rewrites and no comment-prose rewrites beyond the ones the plan names. A module
that looks wrong is left alone and gets a line in [roadmap.md](../roadmap.md).

Exactly two behaviour changes were intended, and both are in Phase 3:

1. The default model tier for a local run is **free**. The prod roster is used only when
   `SIBYL_MODEL_TIER=prod` is set, which the prod workflows do.
2. `python -m sibyl test` forecasts the open questions of the bot-testing-area tournament
   instead of four hard-coded evergreen URLs. `TEST_QUESTIONS_OVERRIDE` still works.

## Consequences

Things that were found and left alone, each in the roadmap: the ops scripts' `--repo`
default pointing at upstream rather than the operator's fork; a pre-existing cross-file
test isolation leak; the forecasting path's import of the eval layer.

Four changes were made that the plan did not name, because leaving them would have left
the repo broken rather than merely imperfect:

- **A latent test-isolation bug, fixed because laning the tests exposed it.** The
  base-URL seam test reloads `metaculus_preflight`, which rebinds every module-level name
  to a new object — including the exception class `test_api_preflight.py` binds at import.
  Reloading a second time to "restore" it does not restore identity, so twelve tests
  failed with the exception escaping their `pytest.raises`. It reproduces in bottega;
  alphabetical ordering had hidden it.
- **Two test classes deleted**, not moved: they exercised `bottega:scripts/research_sync/`, a
  launchd job for another operator's machine that section 1.2 excludes. With no subject in
  this repo they could only have been kept by vendoring the thing the plan excluded.
- **The primary research ladder is disabled in the free tier.** See
  [0003](0003-model-tiers.md).
- **The golden harness's clock was frozen.** See [0001](0001-parity-contract.md).

Two plan instructions were followed even though their consequences were not obvious, and
both are recorded where they bite: `eval/ablation/forecaster_lineup.py` became a
re-export of the config roster, shrinking the ablation lineup from four free models to
three; and the vocabulary rename stopped short of two parsed log lines, because renaming
them would orphan every archived telemetry record.

One plan instruction was **not** followed as written: the six `tests/test_ft_*.py` files
were all kept. The plan expected some to be 0.2.54-vs-0.2.92 comparison tests, but with
`forecasting-tools==0.2.92` pinned such a test is not constructible. All 22 classes
assert live behaviour of the installed version on production seams, and all 66 pass.
