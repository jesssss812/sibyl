# Grant billing — handoff to Astra

*2026-10-05. Independent review of the funding break, after your
`investigation.md`. Everything below is offline-verified; no paid calls.*

Your diagnosis was right and I reproduced it independently. This covers only the
**delta**: what is already fixed, and the two things your report left open that
new evidence now answers.

## Already fixed — don't redo it

Branch `fix-grant-billing` (4 commits, off `main`), suite green at 5539 /
ruff clean / basedpyright 0:

- `charged_usd` is now the **billable** figure, with `openrouter_charge_usd`,
  `upstream_cost_usd`, `is_byok` stored beside it.
- Cost evidence is ranked, not demanded from one source. Positive `usage.cost`
  **confirms**; zero is preserved but does not confirm; framework estimates never
  confirm. A call raises only when there is no evidence at all.
- `reconcile_pending()` + `experiment_grant reconcile --run-id <id>`: bounded
  backoff, read-only, never re-calls a model, exits non-zero while anything is
  unaccounted.
- `tests/unit/test_experiment_usage.py` rewritten — the old file *asserted* the
  defect, which is why a passing suite never caught it.

This implements your recommendations 1, 3, 4 and 5.

## New evidence: `GET /api/v1/key` (free, read-only)

```
"limit": 100,          "limit_remaining": 99.203972,
"include_byok_in_limit": true,
"usage": 0,            "byok_usage": 0.796028,
"byok_usage_daily": 0.1873175
```

`byok_usage` is exactly `limit − limit_remaining`; `byok_usage_daily` is exactly
the record you recovered.

## Finding 1 — answers your recommendations 2 and 6

**BYOK spend IS metered against the $100 limit** (`include_byok_in_limit: true`),
and OpenRouter enforces the cap. So:

- The grant allocation and the upstream expense are the **same budget**, not two
  currencies. Billable = upstream cost under BYOK is correct, and matches
  OpenRouter's own accounting.
- No workspace BYOK change is needed before the next run. The routing is
  deliberate: the grantor bounded their upstream OpenAI spend with the $100 cap.
- The key belongs to a workspace the operator did not create
  (`workspace_id: 3c192787-5312-571a-9ffe-a1414ea17a14`). The metadata still does
  not name the upstream payer — your caveat stands.

## Finding 2 — the ledger is wrong in the *other* direction

| | local ledger | OpenRouter |
| --- | --- | --- |
| committed | **$24.00** | **$0.796028** |

Eight attempts, all `uncertain`, all `charged_usd = NULL`, each holding its full
$3 reservation (your "$21 + $3"). So the old code failed both ways depending on
path: over-reserve ~30× with no evidence, under-report to $0 once the record
resolved. At $3 held per call a $100 allocation reads as exhausted after ~33
calls having actually spent ~$2.65.

**`reconcile` cannot clear these** — their `provider_request_id` is `NULL`, so
there is nothing to look up. It will report `no_generation_id: 8` and exit 1,
correctly.

## Remaining work

1. **Preflight against `limit_remaining`**, not the local ledger. It is
   authoritative and includes spend from other machines — which
   `docs/experiment-funding.md` currently calls out as invisible.
2. **True up the ledger** against `byok_usage` (clears the phantom $24 to
   $0.796028). Needs an operator decision on the 8 stale rows: write off against
   OpenRouter's figure, or leave held.
3. **Re-check `--reserve-per-call-usd` sizing.** `BudgetExceeded` still stops a
   run when one charge exceeds the reserve, and the billable figure is now
   upstream cost rather than the old `$0` — so size it against upstream prices.

## Untested, unchanged by this work

`grant_route_for` maps the roster's Anthropic-direct slots onto OpenRouter under
a grant (`anthropic/claude-opus-4-8` → `openrouter/anthropic/claude-opus-4.8`,
note the dot). That path has never run against a live grant key.

## Addendum, 2026-10-06 — review follow-up

A review of this branch before merge changed three points above:

- **A positive `usage.cost` no longer confirms by itself.** Under BYOK it is only
  OpenRouter's fee. The response confirms when its usage accounting states
  `is_byok` (and, under BYOK, `cost_details.upstream_inference_cost`); otherwise the
  generation record is consulted, and failing that the call settles as an estimate.
- **Billable under BYOK is upstream cost plus OpenRouter's fee**, since the key's
  limit meters both. With the fee at `0`, as in the record above, nothing changes.
- **Estimated charges publish.** Withholding a forecast that is already paid for
  recovers nothing; the estimate's held reservation guards the limit, and
  `reconcile` confirms it later. Failed calls and over-reserve charges still block.
- **Remaining work item 1 is done:** a run preflights against the key's
  `limit_remaining` and refuses when it cannot cover the budget plus the reserve.
