# One forecast route; funding picks the key, not the method

**Date:** 2026-10-14 · **Status:** shipped

## Decision

Grant-funded runs use the ordinary pipeline. The OpenRouter grant is one more key that
can pay for a request, offered before the ordinary keys, and nothing else about the run
depends on it. Every run records every paid call in the spend ledger.

## Why

The separate grant runner had become a second forecasting method. It used one research
provider instead of the production fan-out, its own question selection and publication
gates, and grant checks spread through research and aggregation code. Results from it
could not be compared with ordinary runs, and every new provider needed its own grant
branch. Its accounting also covered grant calls only, so total spend was never in one place.

## What it means

- Funding decides which credential carries a request at one place: forecasting-tools'
  per-attempt call (`sibyl/platform/funding.py`), plus gap-fill v2, which calls LiteLLM
  directly. Models, research, retries and publication are the same code whoever pays.
- A grant refusal another key can fix moves that request to the ordinary key. Any other
  failure is raised as usual.
- Billing state never blocks publication. A charge that is not yet confirmed is
  reconciled after the run.
- Runs from the old grant runner are their own configuration era, which ends when this
  change merged (the era rule uses merge dates, not authoring dates).

See [experiment-funding.md](../runbooks/experiment-funding.md) for how to run and inspect it.
