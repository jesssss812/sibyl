# OpenRouter experiment grant

This isolated entry point spends from a separately provisioned OpenRouter key and writes local request accounting. It supports reviewed MiniBench runs and single-question runs in the unscored bot-testing area. The ordinary bot CLI is unchanged. The grant path runs the production model identities through explicit OpenRouter routes, including the two Anthropic slots that normally call Anthropic directly.

Rotate the key previously shared in conversation. Put the replacement in the local secret store as `SIBYL_EXPERIMENT_OPENROUTER_API_KEY`; never commit it or paste it into a task. Keep it distinct from `OPENROUTER_API_KEY` and `OAI_ANTH_OPENROUTER_KEY`. A configured key alone starts no run.

## Inspect before use

```bash
python -m sibyl.ops.experiment_grant preview
python -m sibyl.ops.experiment_grant report --ledger backtests/experiment_grant_spend.sqlite3
```

`preview` requires no grant key and makes no paid provider call. The package's existing import path may try to refresh LiteLLM's public model price metadata; it falls back to the installed map when offline. The preview prints the funding route and selected research path. Compare it with the desired experiment manifest before authorizing execution.

The grant MiniBench runner selects OpenRouter native web search as its sole research provider. It refuses enabled native-search add-ons, Gemini/Anthropic search, gap-fill, financial data, time-series anchors, prediction markets, resolution sources, and reference-class enrichers until each path has been audited for grant-only billing. This changes the research fan-out from a normal production run; treat it as a distinct configuration era and do not combine its accuracy results with earlier runs without accounting for that change. The three-slot grant roster retains the OpenRouter OpenAI and Google slots and explicitly maps the direct Anthropic slot to OpenRouter. Inspect the public model catalogue and supported reasoning parameters before approving a live run; offline tests prove routing, not current provider availability.

## Run only after approval for that specific run

For a single unscored test, select the exact post and question IDs from the open `bot-testing-area` tournament. The command refuses an ID outside that tournament or a question that is not open, then publishes the forecast and comment to the selected test question:

```bash
python -m sibyl.ops.experiment_grant run-test \
  --run-id reviewed-test-001 \
  --post-id 43323 --question-id 43325 \
  --max-cost-usd 15 --reserve-per-call-usd 3 \
  --ledger backtests/experiment_grant_spend.sqlite3
```

This is a command template, not standing authorization. Each live test needs operator approval for that specific run. Its manifest, question snapshot, and reports are written under `backtests/grant_test_runs/<run-id>/` by default. It uses the same grant-only routes, research preflight, per-call reservations, and publication gate as MiniBench.

For publishing runs, open a visible terminal and use `python -u` with `2>&1 | tee <private-log-path>` so the operator can watch progress. Enable `set -o pipefail` in the shell so `tee` cannot hide a failed run. The runner emits question starts and completions, model call starts and settlements, and sanitized failure status codes at INFO level. It never logs credentials or upstream error bodies. A question is stopped before its forecasters start only when a research call yields **no cost evidence at all**; a cost that is known but not yet provider-confirmed proceeds and is reconciled afterwards. Grant accounting errors are not retried by the outer LLM retry loop.

The following is a command template, **not authorization to execute it**. It publishes to scored MiniBench and spends credits. Set a unique run ID and a per-call reservation large enough for the chosen model's maximum output and research charges.

```bash
python -m sibyl.ops.experiment_grant run-minibench \
  --run-id reviewed-run-001 \
  --max-questions 3 \
  --max-cost-usd 20 \
  --reserve-per-call-usd 2 \
  --ledger backtests/experiment_grant_spend.sqlite3
```

The run fails before calling a model if the grant key is missing or matches either existing key. Every supported model call receives the grant credential explicitly and has no personal-key fallback. The grant ledger reserves funds before each request. Calls with missing reported cost, failures, and interrupted calls keep their reservation as uncertain liability; review those before another run. The local allocation is $100 with $5 held back. A per-run budget is required.

The runner writes `backtests/minibench_runs/<run-id>/manifest.json` before its first request, recording the code identity, routed models, research profile, budget, and start time without credentials. It records the exact selected question IDs and snapshots in `scheduled_questions.json` before forecasting. The forecasting framework saves successful reports in that run's `forecasts/` directory when the batch returns. These paths are private and excluded from Git. An interrupted batch can leave a manifest and ledger entries without a report file; reconcile those against Metaculus before scoring. A run ID with an existing manifest is refused so a second execution cannot overwrite its provenance. Use `--reports-dir` only when the alternate directory is private.

The reservation is a local stop-loss, **not a provider-enforced hard cap**. A single provider charge can exceed its reservation, and spending from other machines or applications is invisible until recorded. The runner records the overage and stops before publishing that question. Use OpenRouter's own account/key controls where available, and reconcile against provider billing. The pinned framework does not report a reliable cost for every route or every web-search surcharge. Cost evidence is ranked rather than demanded from one source:

1. the [generation billing record](https://openrouter.ai/docs/api/api-reference/generations/get-generation) — the only source carrying the BYOK split, so the only one that can confirm a billable figure for a BYOK route;
2. the completion response's `usage.cost` — OpenRouter's own account charge, already in hand. A **positive** value confirms;
3. the LiteLLM framework estimate — recorded, never confirming.

A **zero** OpenRouter charge is preserved but does not confirm: it is either a genuinely free route or a BYOK route billing upstream, and the response alone cannot tell them apart. Such a call publishes, settles as an estimate, and keeps its whole reservation until reconciled — so an unseen BYOK expense cannot make a run read as free.

**Billable vs. components.** `charged_usd` is the figure the budget holds. Under BYOK that is the upstream provider's charge, not OpenRouter's — which is often `0`. The components (`openrouter_charge_usd`, `upstream_cost_usd`, `is_byok`) are stored separately, because collapsing them is what let a run report `$0` spent while `$0.1873175` moved upstream.

**Reconciliation.** `python -m sibyl.ops.experiment_grant reconcile --run-id <id>` confirms outstanding estimates from the provider's records. Read-only, bounded retry, and it never repeats a paid request to learn what one cost; the generation record is eventually consistent, and has been observed resolving minutes after the call. It exits non-zero while anything is still unaccounted.

If grant credits were used elsewhere, record the prior amount with a stable unique ID:

```bash
python -m sibyl.ops.experiment_grant record-prior-spend \
  --id outside-001 --amount-usd 4.25 --note 'previous grant use'
```

`report` separates confirmed charges, local estimates, reserved calls, uncertain calls, and prior-spend adjustments. A charge is confirmed by a provider generation record, a positive provider-reported response cost, or later reconciliation. LiteLLM/framework estimates and zero-valued response costs never confirm; their full pre-call reservation remains budget liability until reconciled. The model/stage breakdown never contains prompts or credentials. Use one shared ledger path for every local process using the grant. Keep the ledger as a private artifact; it is excluded from Git under `backtests/`.

Ordinary production routing is unchanged. To record *actual primary-to-personal fallback invocations* in the same report, set `SIBYL_PERSONAL_FALLBACK_LEDGER=backtests/experiment_grant_spend.sqlite3` for that process. Optionally set `SIBYL_PERSONAL_SPEND_RUN_ID` to a non-secret run label; otherwise a process-local ID is generated. The observer writes a separate `personal` table only when the secondary call is reached. A callback cost is an estimate, and a failed call or missing cost stays unknown. SDK-internal retries may create additional charges the observer cannot distinguish. Direct calls that use `OPENROUTER_API_KEY` without entering the primary-to-personal fallback wrapper are outside this table; do not interpret its total as all personal spending. Ledger failures are logged but never block the fallback call or change its route. This accounting is a separate change from the grant-only route and does not rewrite historical spend.

This first implementation does not route the legacy backtest and ablation CLIs through the grant. Their frozen input and provenance checks are separate from grant funding. They continue to use their existing credentials until an explicit integration is implemented and tested. The ordinary MiniBench and tournament commands likewise keep their existing behavior.
