---
paths:
  - ".github/**"
---

# GitHub Actions workflows

- **No `schedule:` block** in a bot workflow. Scheduled runs were disabled on 2026-08-27
  and `tests/unit/test_workflow_tier_config.py` asserts none is back. Re-enabling one is
  the operator's call, never part of another change.
- **Never dispatch a workflow.** A bot workflow forecasts and publishes to Metaculus.
- **The tier is explicit.** Every prod workflow sets `SIBYL_MODEL_TIER: 'prod'`; the
  free workflow sets `'free'`. The same test checks each one. An unset tier means free.
- **Never commit a secret.** Workflows read Actions secrets; the `GEMINI_API_KEY` secret
  is surfaced as `GOOGLE_API_KEY` for the `google-genai` SDK.
- Workflow env flags, the artifact retention that `make sync_all` races, and how to
  re-enable a schedule safely are in `docs/runbooks/operations.md`,
  "GitHub Actions workflows". Update it in the same change.
