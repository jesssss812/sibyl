# OpenRouter experiment grant

This isolated entry point spends from a separately provisioned OpenRouter key and writes local request accounting. It is intended for reviewed MiniBench runs. The ordinary bot CLI is unchanged. The grant path runs the production model identities through explicit OpenRouter routes, including the two Anthropic slots that normally call Anthropic directly.

Rotate the key previously shared in conversation. Put the replacement in the local secret store as `SIBYL_EXPERIMENT_OPENROUTER_API_KEY`; never commit it or paste it into a task. Keep it distinct from `OPENROUTER_API_KEY` and `OAI_ANTH_OPENROUTER_KEY`. A configured key alone starts no run.

## Inspect before use

```bash
python -m sibyl.ops.experiment_grant preview
python -m sibyl.ops.experiment_grant report --ledger backtests/experiment_grant_spend.sqlite3
```

`preview` requires no grant key and makes no paid provider call. The package's existing import path may try to refresh LiteLLM's public model price metadata; it falls back to the installed map when offline. The preview prints the funding route and selected research path. Compare it with the desired experiment manifest before authorizing execution.

The grant MiniBench runner selects OpenRouter native web search as its sole research provider. It refuses enabled native-search add-ons, Gemini/Anthropic search, gap-fill, financial data, time-series anchors, prediction markets, resolution sources, and reference-class enrichers until each path has been audited for grant-only billing. This changes the research fan-out from a normal production run; treat it as a distinct configuration era and do not combine its accuracy results with earlier runs without accounting for that change. The three-slot grant roster retains the OpenRouter OpenAI and Google slots and explicitly maps the direct Anthropic slot to OpenRouter. Inspect the public model catalogue and supported reasoning parameters before approving a live run; offline tests prove routing, not current provider availability.

## Run only after approval for that specific run

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

The reservation is a local stop-loss, **not a provider-enforced hard cap**. A single provider charge can exceed its reservation, and spending from other machines or applications is invisible until recorded. The runner records the overage and stops. Use OpenRouter's own account/key controls where available, and reconcile against provider billing. The pinned framework does not report a reliable cost for every route or every web-search surcharge. The runner attempts a read-only lookup of each [generation's `total_cost`](https://openrouter.ai/docs/api/api-reference/generations/get-generation) to settle it. If no verifiable cost is available, the call remains uncertain and the runner stops rather than reporting it as free.

If grant credits were used elsewhere, record the prior amount with a stable unique ID:

```bash
python -m sibyl.ops.experiment_grant record-prior-spend \
  --id outside-001 --amount-usd 4.25 --note 'previous grant use'
```

`report` separates confirmed charges, local estimates, reserved calls, uncertain calls, and prior-spend adjustments. Only a provider generation billing record or later provider reconciliation confirms a charge. LiteLLM/framework costs and callback costs are estimates even when token usage is known; their full pre-call reservation remains budget liability until reconciled. The model/stage breakdown never contains prompts or credentials. Use one shared ledger path for every local process using the grant. Keep the ledger as a private artifact; it is excluded from Git under `backtests/`.

Ordinary production routing is unchanged. To record *actual primary-to-personal fallback invocations* in the same report, set `SIBYL_PERSONAL_FALLBACK_LEDGER=backtests/experiment_grant_spend.sqlite3` for that process. Optionally set `SIBYL_PERSONAL_SPEND_RUN_ID` to a non-secret run label; otherwise a process-local ID is generated. The observer writes a separate `personal` table only when the secondary call is reached. A callback cost is an estimate, and a failed call or missing cost stays unknown. SDK-internal retries may create additional charges the observer cannot distinguish. Direct calls that use `OPENROUTER_API_KEY` without entering the primary-to-personal fallback wrapper are outside this table; do not interpret its total as all personal spending. Ledger failures are logged but never block the fallback call or change its route. This accounting is a separate change from the grant-only route and does not rewrite historical spend.

This first implementation does not route the legacy backtest and ablation CLIs through the grant. Their frozen input and provenance work is changing independently. They continue to use their existing credentials until an explicit integration is implemented and tested. The ordinary MiniBench and tournament commands likewise keep their existing behavior.
