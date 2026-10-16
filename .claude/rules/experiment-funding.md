---
paths:
  - "sibyl/platform/experiment_*.py"
  - "sibyl/platform/personal_fallback_spend.py"
  - "sibyl/ops/experiment_grant.py"
  - "tests/unit/test_experiment_*.py"
---

# Experiment grant funding

The grant runner spends from a separately provisioned OpenRouter key. Its job is to keep
that money separate from personal spend and to never under-report what was spent.

- `SIBYL_EXPERIMENT_OPENROUTER_API_KEY` must stay distinct from `OPENROUTER_API_KEY` and
  `OAI_ANTH_OPENROUTER_KEY`; the run refuses to start when it matches either. A grant
  call has no personal-key fallback.
- `preview`, `report` and `reconcile` make no paid model call. `run-test` and
  `run-minibench` spend and publish: each needs the operator's go for that run.
- Unconfirmed cost holds its whole reservation until reconciled. Only response usage
  that states the BYOK split, or a provider generation record, confirms a charge; a
  LiteLLM estimate or a bare `usage.cost` never does.
- Gates are on missing answers, not on cost. Each kind of failed call has one
  consequence, tabled in `docs/runbooks/experiment-funding.md`. A change to that
  handling changes the table in the same commit.
- Ledgers, manifests and reports live under `backtests/`, which is private and ignored.
  Never log a credential or an upstream error body.
