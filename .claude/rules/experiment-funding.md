---
paths:
  - "sibyl/platform/experiment_*.py"
  - "sibyl/platform/key_fallback.py"
  - "sibyl/platform/keys.py"
  - "sibyl/platform/llm_retry.py"
  - "sibyl/config/models.py"
  - "sibyl/aggregation/pipeline.py"
  - "sibyl/research/orchestrator.py"
  - "sibyl/research/providers.py"
  - "sibyl/ops/experiment_grant.py"
  - "tests/unit/test_experiment_*.py"
  - "tests/integration/test_grant_*.py"
---

# Experiment grant funding

The grant runner spends from a separately provisioned OpenRouter key. Its job is to keep
that money separate from personal spend and to never under-report what was spent.

- `SIBYL_EXPERIMENT_OPENROUTER_API_KEY` must stay distinct from `OPENROUTER_API_KEY` and
  `OAI_ANTH_OPENROUTER_KEY`; the run refuses to start when it matches either. A grant
  call has no personal-key fallback.
- Shared code branches on `active_grant()` to route a call to the grant key
  (`key_fallback.py`, `keys.py`, `config/models.py`, `aggregation/pipeline.py`,
  `research/orchestrator.py`, `research/providers.py`). A new call site that builds an
  LLM or picks a key needs the same branch, or it bills a non-grant key.
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
