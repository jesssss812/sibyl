"""Inspect and reconcile the spend ledger. Makes no model or research request.

Forecasting runs, including grant-funded ones, go through ``python -m sibyl`` (see
``--grant-budget-usd``). This tool reads what those runs recorded:

- ``report``: totals per paying source, and calls grouped by source, model and stage.
- ``reconcile``: confirm estimated OpenRouter charges from OpenRouter's own billing
  records, using the key that made each request. Read-only; it never repeats a request.
- ``record-prior-spend``: count grant spending made outside the ledger against the
  local grant allocation.
- ``write-off``: replace grant attempts that can never be reconciled (no generation
  ID) with the total OpenRouter metered for them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
from decimal import Decimal, InvalidOperation
from pathlib import Path

from sibyl.platform.experiment_funding import FundingSource, SpendLedger
from sibyl.platform.funding import DEFAULT_SPEND_LEDGER, OPENROUTER_KEY_ENVS, SPEND_LEDGER_ENV


def _positive(raw: str, label: str) -> Decimal:
    try:
        value = Decimal(raw)
    except InvalidOperation as exc:
        raise ValueError(f"{label} must be a positive decimal") from exc
    if not value.is_finite() or value <= 0:
        raise ValueError(f"{label} must be a positive decimal")
    return value


def _legacy_personal_fallback(path: Path, run_id: str | None) -> dict[str, str | int] | None:
    """Totals from the separate fallback table older versions wrote, when it exists."""
    with sqlite3.connect(path) as db:
        if db.execute("SELECT 1 FROM sqlite_master WHERE name = 'personal_fallback_attempts'").fetchone() is None:
            return None
        rows = db.execute(
            "SELECT certainty, estimated_usd FROM personal_fallback_attempts"
            + (" WHERE run_id = ?" if run_id is not None else ""),
            (run_id,) if run_id is not None else (),
        ).fetchall()
    return {
        "attempts": len(rows),
        "estimated_usd": str(sum((Decimal(cost) for certainty, cost in rows if certainty == "estimated"), Decimal(0))),
        "unknown_attempts": sum(certainty == "unknown" for certainty, _ in rows),
    }


def _report(ledger: SpendLedger, run_id: str | None) -> dict[str, object]:
    sources = ledger.sources(run_id)
    if FundingSource.EXPERIMENT_GRANT not in sources:
        sources.insert(0, FundingSource.EXPERIMENT_GRANT)  # always show the allocation
    report: dict[str, object] = {
        "totals": {source.value: ledger.summary(run_id, source=source) for source in sources},
        "by_source_model_stage": ledger.breakdown(run_id),
    }
    legacy = _legacy_personal_fallback(ledger.path, run_id)
    if legacy is not None:
        report["legacy_personal_fallback"] = legacy
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m sibyl.ops.experiment_grant")
    sub = parser.add_subparsers(dest="command", required=True)
    report = sub.add_parser("report", help="Show recorded spending by source; makes no API call")
    report.add_argument("--ledger", type=Path, default=None)
    report.add_argument("--run-id")
    rec = sub.add_parser(
        "reconcile",
        help="Confirm estimated OpenRouter charges from OpenRouter's billing records; read-only, makes no paid call",
    )
    rec.add_argument("--ledger", type=Path, default=None)
    rec.add_argument("--run-id", required=True)
    adjust = sub.add_parser("record-prior-spend", help="Record previous grant spending in the local allocation")
    adjust.add_argument("--ledger", type=Path, default=None)
    adjust.add_argument("--amount-usd", required=True)
    adjust.add_argument("--note", required=True)
    adjust.add_argument("--id", required=True, help="Stable unique adjustment ID for audit and duplicate protection")
    write_off = sub.add_parser(
        "write-off",
        help="Replace unconfirmed attempts with no generation ID by OpenRouter's own total; makes no API call",
    )
    write_off.add_argument("--ledger", type=Path, default=None)
    write_off.add_argument("--provider-total-usd", required=True, help="What OpenRouter metered for those attempts")
    write_off.add_argument("--expect-count", type=int, required=True, help="Refuse unless exactly this many match")
    write_off.add_argument("--note", required=True)
    write_off.add_argument("--id", required=True, help="Stable unique adjustment ID for audit and duplicate protection")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from sibyl.config.env import load_environment

    load_environment()
    try:
        # The same file the CLI writes: --ledger, else SIBYL_SPEND_LEDGER, else the default.
        path = args.ledger or Path(os.getenv(SPEND_LEDGER_ENV, "").strip() or DEFAULT_SPEND_LEDGER)
        # Opening a ledger creates it; a mistyped path must not read as "nothing
        # outstanding", or record prior spend in a file no run reads.
        if not path.is_file():
            raise ValueError(f"No spend ledger at {path}")
        ledger = SpendLedger(path)
        if args.command == "report":
            print(json.dumps(_report(ledger, args.run_id), indent=2))
            return 0
        if args.command == "reconcile":
            from sibyl.platform.experiment_usage import reconcile_pending

            if not ledger.has_run(args.run_id):
                raise ValueError(f"Run {args.run_id} is not in the ledger at {ledger.path}")

            outcome: dict[str, object] = {}
            missing_keys: list[str] = []
            for source, key_env in OPENROUTER_KEY_ENVS.items():
                if not ledger.unreconciled(args.run_id, source=source):
                    continue
                # Read-only: the generation GET reads a billing record, it never creates
                # one. A record is only visible to the key that made the request.
                key = os.getenv(key_env, "").strip()
                if not key:
                    outcome[source.value] = f"skipped: {key_env} is not set"
                    missing_keys.append(key_env)
                    continue
                outcome[source.value] = asyncio.run(reconcile_pending(ledger, args.run_id, key, source=source))
            print(json.dumps(outcome, indent=2))
            if missing_keys:
                return 2
            # Non-zero while anything could still be confirmed or written off, so a wrapper
            # notices. An ordinary-key attempt with no generation ID can do neither: it holds
            # no budget and stays in the report as unpriced, so it does not fail the run.
            unresolved = 0
            for source_name, counts in outcome.items():
                if isinstance(counts, dict):
                    unresolved += counts["still_unknown"]
                    if source_name == FundingSource.EXPERIMENT_GRANT.value:
                        unresolved += counts["no_generation_id"]
            return 0 if unresolved == 0 else 1
        if args.command == "write-off":
            written = ledger.write_off(
                args.id,
                _positive(args.provider_total_usd, "--provider-total-usd"),
                args.note,
                expected_count=args.expect_count,
            )
            print(json.dumps({"written_off": written, "totals": ledger.summary()}, indent=2))
            return 0
        ledger.adjust(args.id, _positive(args.amount_usd, "--amount-usd"), args.note)
        print(json.dumps(ledger.summary(), indent=2))
        return 0
    except (ValueError, RuntimeError, sqlite3.Error) as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
