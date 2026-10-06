# Isolated OpenRouter experiment funding

## Handoff to Sol

Implement a separate funding path for a $100 OpenRouter grant intended for MiniBench and experiments, with reliable separation from personal spending. This plan authorizes implementation and offline validation only. It does not authorize paid requests, publishing, workflow dispatch, or changing tournament enablement.

Read `AGENTS.md` and inspect the current working tree before starting. Evaluation-integrity work may be in progress: preserve it, integrate with its final interfaces, and do not revert, stage, or commit unrelated changes. Work on a feature branch, never `main`. Follow the repository's small-commit and pre-commit verification requirements.

The credential previously supplied in conversation must be considered exposed. Never reproduce it, test it, or add it to any file. The operator must rotate it and provision the replacement privately. Implementation and offline tests must work without the real credential.

## Required behavior

- Add `SIBYL_EXPERIMENT_OPENROUTER_API_KEY` as a dedicated secret. Document it using an empty placeholder only, in the existing secret-configuration convention.
- Add explicit funding selection, proposed CLI `--funding existing|experiment-grant`, defaulting to `existing`. Funding and model tier are independent: a configured grant secret must not change the default free tier or authorize a run.
- Preserve current routing for ordinary runs. Grant-funded runs use only the grant credential for eligible OpenRouter requests, including retries, extraction, parsing, stackers, leakage detection, and OpenRouter-backed research if enabled.
- Never fall back from the grant to personal or existing primary credentials, or from ordinary runs to the grant. Missing/empty grant credentials fail before creating clients or making requests. Supplying a grant credential on a free-tier run must not enable paid models or bypass publishing restrictions.
- Expose grant funding only for MiniBench and evaluation entry points that can enforce these rules end to end. Unsupported modes and paid paths fail clearly before network activity. Keep tournament execution disabled as currently requested; do not enable schedules or dispatch workflows.
- Track grant, personal, existing primary, and unknown funding separately. Do not assume the existing primary key is personal or that all costs reported by a legacy run belong to one payer.

## Current integration points to verify

The repository currently routes credentials through `sibyl/platform/keys.py`, `sibyl/platform/key_fallback.py`, and `sibyl/config/models.py`. `build_llm_with_openrouter_fallback` can construct a primary-to-personal fallback wrapper; other branches rely on ambient `OPENROUTER_API_KEY`. Grant mode must bypass both behaviors and supply its credential explicitly.

Inspect every client construction and provider invocation, not just this helper. Direct `GeneralLlm` construction exists in the ablation CLI; backtests use `MonetaryCostManager`. Inspect model factories, lazy/default parser models, research providers, SDK retries, and framework-owned calls. Trace the actual MiniBench path through `sibyl/cli.py`.

Preserve the existing `PAID PERSONAL-KEY FALLBACK` text and the `personal_key_fallback`, `donated_404`, and `donated_key` telemetry identifiers. Add new structured funding fields without changing those compatibility contracts. Keep `forecasting-tools==0.2.92` pinned.

## Implementation sequence

### 1. Write routing regression tests, then introduce funding configuration

Create a small immutable funding configuration carrying a source identifier, non-secret credential label, and budget policy. Resolve secrets privately at the client boundary; exclude them from repr, serialized settings, manifests, logs, and exceptions. Prefer explicit propagation through factories. Do not temporarily replace process-wide environment variables or mutate shared model objects to select a payer.

Make grant-client construction a single enforced path. It must pass the grant credential explicitly and disable credential fallback, including SDK/environment defaults. Preserve same-key retries, subject to budget limits. Test concurrent calls with different funding configurations for isolation.

### 2. Validate the full run before starting paid work

Produce a local dry-run funding summary listing model slots, routes, research providers, funding sources, and limits. Dry-run must make no network calls and require no real credentials; execution validation must require credentials.

The first supported live target is a grant-funded MiniBench run on the production model tier. Define a separate, explicit grant roster: retain the OpenRouter OpenAI and Google forecaster slots; propose the OpenRouter equivalent of the current direct Anthropic forecaster as the third slot; and route every enabled parser, analyzer, stacker, fallback stacker, and extraction model through the grant key. Verify each proposed slug and its reasoning/settings support against the public model catalogue before claiming the roster is usable. Do not silently rewrite the ordinary roster or silently drop a required slot. Record the provider/parameter differences as a configuration change that may affect paired-study comparability. If an equivalent route cannot be validated, report MiniBench as unsupported rather than claiming this deliverable is complete.

Use an explicit grant research profile for that MiniBench path: OpenRouter native search on the grant key, with cached research usable for replays. Bypass the default AskNews → Exa → direct Perplexity research ladder; disable direct Anthropic/Google search and every other paid non-OpenRouter provider or gap-fill path unless it is explicitly routed and budgeted to the grant. OpenRouter-backed search plugins and retries must also pass the request ledger and budget gate; if their price cannot be bounded, fail preflight. Do not rely on absent vendor credentials or ordinary provider flags to enforce this profile. Free providers may remain available under existing policies, with funding recorded accurately.

Reject direct Anthropic/OpenAI and other paid research APIs in grant mode unless separately funded execution has been explicitly implemented and authorized. The initial scope rejects mixed paid funding rather than silently charging personal accounts. Do not silently translate direct model slugs to OpenRouter outside the explicit grant roster. Validate the entire effective roster and research profile before network work, and fail unsupported slots or providers before starting the run.

### 3. Add durable request-level accounting

Use the existing accounting hook if it provides sufficient detail; otherwise add a narrow adapter around billable request boundaries. A small local SQLite ledger is suitable for transactional updates and concurrent processes. Keep generated databases and exports outside tracked source files using existing artifact conventions.

Capture run ID, experiment/arm ID, question ID where known, stage/purpose, funding source, non-secret credential label, requested model and observed route where available, attempt ID, provider request ID where available, timestamps, status, token usage, cost in USD, and cost certainty (`confirmed`, `estimated`, or `unknown`). Use decimal or integer monetary units consistently. `confirmed` requires provider-issued billing evidence tied to the attempt or a reconciled provider charge; local token counts multiplied by a price table, including `MonetaryCostManager` estimates, are `estimated` even when usage is observed. Missing cost data, unmatched provider charges, and requests whose billing outcome is uncertain remain `unknown` with a reserved liability until reconciled. Store the evidence source and reconciliation timestamp without storing credentials or request content.

Count preparation, parsers, retries, and failed attempts that may have been billed. Model request events and later cost settlement as updates to one attempt, not duplicate charges. Provider-internal retries that cannot be observed must be documented as an accounting limitation. Cache hits create no new API charge; reports may separately attribute historical preparation costs without charging the ledger twice.

Retain unknown charges explicitly. Never turn missing cost metadata or an interrupted request into a zero-cost success. Do not store prompts, research text, authorization headers, or credentials in the spend ledger.

Implement ordinary-run personal fallback accounting as a separate, reviewable commit after the grant-only routing and ledger work, with routing-parity and frozen-telemetry tests. Attribute a fallback attempt to personal funding only when that credential was actually used; preserve existing routing and markers. Historical uninstrumented costs remain unknown; do not fabricate a backfill. If this independent commit is not ready, report the ordinary-run attribution gap explicitly rather than changing the grant-mode accounting claim.

### 4. Enforce local budgets and reconcile costs

Require a positive explicit per-run budget for grant execution, proposed `--max-cost-usd`. Configure the grant allocation as $100, with a documented reserve; label this a local allocation, not a verified account balance. Allow the operator to record prior/out-of-band spend and adjustments without rewriting request history.

Before each request, atomically reserve a conservative cost estimate against both the run and grant limits, including concurrent in-flight reservations. Bound output tokens and retries. If pricing or an adequate bound is unavailable, fail before sending the request rather than pretending to enforce a hard cap. Settle reservations against returned cost; retain uncertain liabilities after timeouts/crashes until reconciled. Do not release possibly billed reservations merely because a process restarted.

Use a shared ledger location for processes spending the same grant. Document that application budgets cannot guarantee a provider-side hard cap or see spending from other machines automatically. Verify current official OpenRouter documentation before implementing optional provider reconciliation or key limits; do not assume account credits are isolated per key. Do not call authenticated endpoints during implementation. Mock their responses in tests.

Provide an offline spend-report command with totals by funding source, run, experiment, model, and stage, separating settled, estimated, unknown, and reserved amounts. Include local remaining allocation and reconciliation status. Provider reconciliation may be an explicit operator-triggered command, with read-only behavior and no forecast calls.

### 5. Connect accounting to evaluation artifacts and operator docs

Store non-secret funding identity, budget, ledger run ID, and actual model/provider configuration in run artifacts. Use existing experiment manifest/run metadata interfaces; do not build a competing provenance system. Keep paid-call accounting distinct from statistical eligibility: invalid forecasts and excluded questions still cost money.

For paired studies, keep funding and provider configuration consistent across arms unless they are the preregistered treatment. Record per-arm spend without treating payer identity as evidence of comparable model behavior. Reconcile cached/shared preparation costs explicitly.

Document private secret setup, funding selection, offline dry-run/report examples, limits, reconciliation, error recovery, and removal of the grant configuration. Mark execution examples as examples requiring approval for that specific run. A replacement key and its existence never count as operator approval.

## Offline acceptance tests

Use dummy keys, mocked providers, a temporary ledger, and the existing autouse egress guard.

1. Unset funding preserves ordinary routing and free-tier defaults; the grant secret alone changes nothing.
2. Missing/empty grant secret fails execution before client construction; offline dry-run still works.
3. Every supported grant model path receives only the dummy grant key, including parsers, retries, blocked-primary models, stackers, and eligible research. An offline end-to-end MiniBench run with mocked model, research, and publishing boundaries completes with the explicit three-slot grant roster and grant research profile; a missing or unsupported slot/provider fails before network work. Direct unsupported routes fail in preflight.
4. Grant errors (401/402/429/5xx and timeout) never invoke personal/primary credentials. Ordinary requests never use grant credentials. Concurrent funding contexts do not contaminate each other.
5. The separately reviewable existing-mode fallback change attributes actual personal fallback attempts to personal funding while routing and frozen telemetry markers remain compatible.
6. Ledger settlement is idempotent; cache replay adds no charge; retries are counted once per attempt. Provider-issued or reconciled cost can be confirmed, locally calculated cost remains estimated, and missing cost data or crash recovery preserves unknown liability.
7. Concurrent processes cannot reserve beyond the configured local limits. Test exact-limit boundaries, exhausted allocation, unknown pricing, and restarting with pending reservations.
8. Logs, error messages, artifacts, ledger exports, and object repr contain none of the dummy secrets. No secret is committed.
9. MiniBench and eval integration tests validate funding before paid work. Free scored-publishing restrictions and paused tournament behavior remain intact.
10. Golden scenarios 1–5 remain byte-identical. Investigate changes; never automatically rerecord them.

Run `make test`, `make lint`, and `make typecheck` before every commit as required by `AGENTS.md`. No live tests, paid calls, `make e2e_free`, publishing, or workflow dispatch. If unrelated work prevents a clean baseline, report it precisely and do not commit a falsely described passing result.

## Delivery

Deliver the implementation, offline tests, and concise operator documentation in reviewable changes. Report supported entry points, any deliberately rejected billing paths, verification results, accounting limitations, and the operator's remaining steps: rotate/provision the key and authorize a particular run with an explicit budget. Do not launch that run as part of this handoff.
