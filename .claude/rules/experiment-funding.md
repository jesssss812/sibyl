---
paths:
  - "sibyl/platform/funding.py"
  - "sibyl/platform/experiment_*.py"
  - "sibyl/platform/llm_stage.py"
  - "sibyl/platform/key_fallback.py"
  - "sibyl/platform/keys.py"
  - "sibyl/platform/llm_retry.py"
  - "sibyl/config/models.py"
  - "sibyl/research/agentic/llm.py"
  - "sibyl/research/agentic/tools.py"
  - "sibyl/research/providers.py"
  - "sibyl/research/anthropic_search.py"
  - "sibyl/research/gemini_search.py"
  - "sibyl/cli.py"
  - "sibyl/ops/experiment_grant.py"
  - "tests/unit/test_funding.py"
  - "tests/unit/test_spend_report.py"
  - "tests/unit/test_experiment_*.py"
  - "tests/integration/test_grant_*.py"
---

# Funding and spend accounting

There is one forecasting pipeline. Funding decides which credential carries a request,
and records it; it never changes models, research, retries or publication.

- The key is chosen in one place: `apply_funding_seam` in `sibyl/platform/funding.py`
  patches forecasting-tools' per-attempt call, and gap-fill v2 (`research/agentic/llm.py`)
  calls `funded_openrouter_call` itself because it uses LiteLLM directly. Code outside
  these must not branch on who pays. A new direct LiteLLM caller needs the same call.
- A call to a service billed outside OpenRouter goes through `record_direct_call`, so it
  is in the ledger.
- `SIBYL_EXPERIMENT_OPENROUTER_API_KEY` must stay distinct from `OPENROUTER_API_KEY` and
  `OAI_ANTH_OPENROUTER_KEY`. A grant run (`--grant-budget-usd`) is refused on the free
  tier and when the grant key's OpenRouter cap does not bound it.
- A grant refusal another key can fix moves that request to the ordinary key; any other
  failure is raised as it would be without the grant. The payer never adds a retry.
- Only response usage that states the BYOK split, or reconciliation, confirms a charge,
  and the confirmed total must equal the fee (plus the upstream cost under BYOK). An
  unconfirmed grant charge holds its whole reservation until reconciled or written off.
- `report`, `reconcile`, `write-off` and `record-prior-spend` make no paid model call.
  A run with a grant budget spends: it needs the operator's go for that run.
- Ledgers and run archives live under `backtests/`, which is private and ignored. Never
  log a credential or an upstream error body.
