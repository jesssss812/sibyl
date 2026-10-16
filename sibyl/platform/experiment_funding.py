"""Local, transactional accounting for every paid call, whichever key pays.

Each attempt is recorded under the source that paid for it. Only the OpenRouter
grant has a spending limit here: its $100 allocation is a local envelope, not a
claim about the provider account balance, and pending or uncertain grant requests
retain their full reservation after a crash. The other sources are recorded for the
audit and never refused.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from typing import Iterator

GRANT_KEY_ENV = "SIBYL_EXPERIMENT_OPENROUTER_API_KEY"
# Evidence that can establish a real charge, as opposed to a local guess. Each source
# confirms only when it carries every billing component (see provider_billable_cost):
# a response's `usage.cost` with the BYOK route unknown is still an estimate.
# `framework_estimate` (LiteLLM token maths) is never here: a guess must not discharge
# a reservation.
CONFIRMING_EVIDENCE: frozenset[str] = frozenset({"provider_response", "provider_reconciliation"})

GRANT_ALLOCATION_USD = Decimal("100")
GRANT_RESERVE_USD = Decimal("5")


# ``runs.source`` for a run with no grant budget: it may draw only on the ordinary keys.
# Not a FundingSource, because no attempt is ever paid by "ordinary keys" as such.
RUN_WITHOUT_GRANT = "ordinary-keys"


class FundingSource(StrEnum):
    """Who paid for an attempt. The values are stored in the ledger; do not rename them."""

    EXPERIMENT_GRANT = "experiment-grant"
    METACULUS_KEY = "metaculus-key"
    PERSONAL_KEY = "personal-key"
    DIRECT_SERVICE = "direct-service"
    # Retained so rows written under the earlier two-source scheme still parse.
    EXISTING = "existing"


class BudgetExceeded(RuntimeError):
    """A local spend limit has insufficient unreserved capacity.

    ``in_flight`` is True when the limit would hold if the requests still in flight
    were left out: capacity may come back as they settle, so only this attempt is turned
    away. False means the budget is spent whatever happens to them.
    """

    def __init__(self, message: str, *, in_flight: bool = False) -> None:
        super().__init__(message)
        self.in_flight = in_flight


def _cost(value: Decimal) -> Decimal:
    try:
        amount = Decimal(value)
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError("Cost must be a finite nonnegative decimal") from exc
    if not amount.is_finite() or amount < 0:
        raise ValueError("Cost must be a finite nonnegative decimal")
    return amount


def provider_billable_cost(
    openrouter_charge: Decimal | None, upstream_cost: Decimal | None, is_byok: bool | None
) -> Decimal | None:
    """The whole charge for one call, or None while any part of it is unknown.

    A BYOK call bills twice: OpenRouter's fee from the key, and the upstream provider
    on the operator's own account. Counting only one of them is how a BYOK run once
    read as free. With the route unknown, neither component alone is the total.
    """
    charge = _cost(openrouter_charge) if openrouter_charge is not None else None
    upstream = _cost(upstream_cost) if upstream_cost is not None else None
    if is_byok is True and charge is not None and upstream is not None:
        return charge + upstream
    if is_byok is False:
        return charge
    return None


def _amount(value: Decimal) -> Decimal:
    try:
        amount = Decimal(value)
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError("USD amount must be a finite positive decimal") from exc
    if not amount.is_finite() or amount <= 0:
        raise ValueError("USD amount must be a finite positive decimal")
    return amount


class SpendLedger:
    """SQLite reservations shared safely by local processes using the same file."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    limit_usd TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS attempts (
                    attempt_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES runs(run_id),
                    source TEXT NOT NULL,
                    model TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    question_id TEXT,
                    arm_id TEXT,
                    credential_label TEXT NOT NULL,
                    reserved_usd TEXT NOT NULL,
                    charged_usd TEXT,
                    certainty TEXT NOT NULL,
                    status TEXT NOT NULL,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    provider_request_id TEXT,
                    evidence_source TEXT,
                    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    completed_at TEXT,
                    reconciled_at TEXT
                );
                CREATE TABLE IF NOT EXISTS adjustments (
                    adjustment_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    amount_usd TEXT NOT NULL,
                    note TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            columns = {row[1] for row in db.execute("PRAGMA table_info(attempts)")}
            if "evidence_source" not in columns:
                db.execute("ALTER TABLE attempts ADD COLUMN evidence_source TEXT")
            if "reconciled_at" not in columns:
                db.execute("ALTER TABLE attempts ADD COLUMN reconciled_at TEXT")
            # charged_usd holds the BILLABLE figure -- the money the budget must hold.
            # Under BYOK that is the upstream provider's charge, not OpenRouter's, so the
            # two components are stored separately: a run can show a $0 OpenRouter charge
            # and a real upstream expense, and collapsing them to one number is what let
            # a BYOK run report $0 spent while money moved.
            if "openrouter_charge_usd" not in columns:
                db.execute("ALTER TABLE attempts ADD COLUMN openrouter_charge_usd TEXT")
            if "upstream_cost_usd" not in columns:
                db.execute("ALTER TABLE attempts ADD COLUMN upstream_cost_usd TEXT")
            if "is_byok" not in columns:
                db.execute("ALTER TABLE attempts ADD COLUMN is_byok INTEGER")

            # Earlier versions confirmed a charge from incomplete provider evidence: a
            # response fee with the BYOK route unknown, or a BYOK upstream cost without
            # OpenRouter's fee. Reopen those rows so their reservation is held again and
            # reconciliation can confirm them; the original evidence is kept. Rows closed
            # without provider evidence -- a rejected request, an operator write-off --
            # carry no billing components by design and stay closed.
            db.execute("BEGIN IMMEDIATE")
            for attempt, charged, charge, upstream, byok in db.execute(
                """SELECT attempt_id, charged_usd, openrouter_charge_usd, upstream_cost_usd, is_byok
                FROM attempts WHERE certainty = 'confirmed'
                AND evidence_source IN ('provider_response', 'provider_generation', 'provider_reconciliation')"""
            ).fetchall():
                expected = provider_billable_cost(
                    Decimal(charge) if charge is not None else None,
                    Decimal(upstream) if upstream is not None else None,
                    bool(byok) if byok in (0, 1) else None,
                )
                if expected is None or charged is None or Decimal(charged) != expected:
                    db.execute("UPDATE attempts SET certainty = 'estimated' WHERE attempt_id = ?", (attempt,))
            db.commit()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        try:
            db.execute("PRAGMA foreign_keys = ON")
            db.execute("PRAGMA busy_timeout = 30000")
            yield db
        finally:
            db.close()

    def start_run(self, run_id: str, limit_usd: Decimal = Decimal(0)) -> None:
        """Register a run and its grant limit. Zero means the run may not use the grant."""
        limit = _cost(limit_usd)
        if not run_id.strip():
            raise ValueError("run_id is required")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute("SELECT limit_usd FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if prior is not None and Decimal(prior[0]) != limit:
                db.rollback()
                raise ValueError("Existing run has a different budget")
            db.execute(
                "INSERT OR IGNORE INTO runs (run_id, source, limit_usd) VALUES (?, ?, ?)",
                # What the run may draw on: the grant when it has a budget, else only the ordinary keys.
                (run_id, FundingSource.EXPERIMENT_GRANT if limit > 0 else RUN_WITHOUT_GRANT, str(limit)),
            )
            db.commit()

    @staticmethod
    def _liability(
        db: sqlite3.Connection,
        *,
        run_id: str | None = None,
        source: FundingSource = FundingSource.EXPERIMENT_GRANT,
        settled_only: bool = False,
    ) -> Decimal:
        query = "SELECT reserved_usd, charged_usd, certainty FROM attempts WHERE source = ?"
        args: tuple[str, ...] = (source,)
        if settled_only:
            query += " AND status != 'reserved'"
        if run_id is not None:
            query += " AND run_id = ?"
            args += (run_id,)
        total = Decimal(0)
        for reserve, charge, certainty in db.execute(query, args):
            # Estimates do not discharge the conservative pre-call liability.
            total += (
                Decimal(charge)
                if certainty == "confirmed" and charge is not None
                else max(Decimal(reserve), Decimal(charge) if charge is not None else Decimal(0))
            )
        if run_id is None and source is FundingSource.EXPERIMENT_GRANT:
            for (amount,) in db.execute(
                "SELECT amount_usd FROM adjustments WHERE source = ?", (FundingSource.EXPERIMENT_GRANT,)
            ):
                total += Decimal(amount)
        return total

    def reserve(
        self,
        attempt_id: str,
        run_id: str,
        model: str,
        stage: str,
        reserve_usd: Decimal,
        *,
        question_id: str | None = None,
        arm_id: str | None = None,
        source: FundingSource = FundingSource.EXPERIMENT_GRANT,
    ) -> None:
        """Open an attempt before the request is sent.

        A grant attempt holds ``reserve_usd`` against the run limit and the allocation,
        and raises BudgetExceeded when either cannot cover it. Every other source is
        recorded without a limit, so accounting can never refuse a call it pays for.
        """
        grant = source is FundingSource.EXPERIMENT_GRANT
        reserve = _amount(reserve_usd) if grant else _cost(reserve_usd)
        if not attempt_id or not model or not stage:
            raise ValueError("Model or service, stage, and attempt ID are required")
        if grant and not model.startswith("openrouter/"):
            raise ValueError("Grant attempts require an OpenRouter model")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute("SELECT run_id FROM attempts WHERE attempt_id = ?", (attempt_id,)).fetchone()
            if prior is not None:
                db.rollback()
                raise ValueError("Attempt ID already used")
            row = db.execute("SELECT limit_usd FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if row is None:
                db.rollback()
                raise ValueError("Run has not been started")
            run_limit = Decimal(row[0])
            allocation = GRANT_ALLOCATION_USD - GRANT_RESERVE_USD
            if grant and self._liability(db, run_id=run_id) + reserve > run_limit:
                in_flight = self._liability(db, run_id=run_id, settled_only=True) + reserve <= run_limit
                db.rollback()
                raise BudgetExceeded("Run budget exhausted", in_flight=in_flight)
            if grant and self._liability(db) + reserve > allocation:
                in_flight = self._liability(db, settled_only=True) + reserve <= allocation
                db.rollback()
                raise BudgetExceeded("Local grant allocation exhausted", in_flight=in_flight)
            db.execute(
                """INSERT INTO attempts
                (attempt_id, run_id, source, model, stage, question_id, arm_id,
                 credential_label, reserved_usd, certainty, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'unknown', 'reserved')""",
                (
                    attempt_id,
                    run_id,
                    source,
                    model,
                    stage,
                    question_id,
                    arm_id,
                    source,
                    str(reserve),
                ),
            )
            db.commit()

    def settle(
        self,
        attempt_id: str,
        charged_usd: Decimal,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        provider_request_id: str | None = None,
        *,
        certainty: str,
        evidence_source: str,
        openrouter_charge_usd: Decimal | None = None,
        upstream_cost_usd: Decimal | None = None,
        is_byok: bool | None = None,
    ) -> None:
        charge = _cost(charged_usd)
        if certainty not in {"confirmed", "estimated"} or not evidence_source:
            raise ValueError("Settlement requires a valid certainty and evidence source")
        if certainty == "confirmed" and evidence_source not in CONFIRMING_EVIDENCE:
            raise ValueError("Confirmed cost requires provider billing evidence")
        if certainty == "confirmed" and charge != provider_billable_cost(
            openrouter_charge_usd, upstream_cost_usd, is_byok
        ):
            raise ValueError("Confirmed total requires complete, matching provider billing components")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT charged_usd, certainty FROM attempts WHERE attempt_id = ?", (attempt_id,)
            ).fetchone()
            if row is None:
                db.rollback()
                raise ValueError("Unknown attempt ID")
            if row[0] is not None:
                db.rollback()
                if Decimal(row[0]) != charge or row[1] != certainty:
                    raise ValueError("Attempt has already settled at a different cost")
                return
            db.execute(
                """UPDATE attempts SET charged_usd = ?, certainty = ?, evidence_source = ?,
                status = 'succeeded', prompt_tokens = ?, completion_tokens = ?,
                provider_request_id = ?, openrouter_charge_usd = ?, upstream_cost_usd = ?,
                is_byok = ?, completed_at = CURRENT_TIMESTAMP
                WHERE attempt_id = ?""",
                (
                    str(charge),
                    certainty,
                    evidence_source,
                    prompt_tokens,
                    completion_tokens,
                    provider_request_id,
                    None if openrouter_charge_usd is None else str(Decimal(openrouter_charge_usd)),
                    None if upstream_cost_usd is None else str(Decimal(upstream_cost_usd)),
                    None if is_byok is None else int(is_byok),
                    attempt_id,
                ),
            )
            db.commit()

    def mark_uncertain(
        self, attempt_id: str, *, provider_request_id: str | None = None, evidence_source: str | None = None
    ) -> None:
        with self._connect() as db:
            db.execute(
                """UPDATE attempts SET status = 'uncertain', completed_at = CURRENT_TIMESTAMP,
                provider_request_id = COALESCE(?, provider_request_id), evidence_source = COALESCE(?, evidence_source)
                WHERE attempt_id = ? AND charged_usd IS NULL""",
                (provider_request_id, evidence_source, attempt_id),
            )

    def mark_rejected(self, attempt_id: str, *, evidence_source: str) -> None:
        """Close a request the provider refused before processing; it is not billed."""
        with self._connect() as db:
            db.execute(
                """UPDATE attempts SET status = 'rejected', charged_usd = '0', certainty = 'confirmed',
                evidence_source = ?, completed_at = CURRENT_TIMESTAMP
                WHERE attempt_id = ? AND charged_usd IS NULL""",
                (evidence_source, attempt_id),
            )

    def reconcile(
        self,
        attempt_id: str,
        billable_usd: Decimal,
        *,
        openrouter_charge_usd: Decimal | None = None,
        upstream_cost_usd: Decimal | None = None,
        is_byok: bool | None = None,
        provider_request_id: str | None = None,
    ) -> None:
        """Replace an estimated settlement with the provider's billing record.

        Separate from ``settle`` because reconciliation happens AFTER the run, off the
        hot path, and reads the provider's record without a second paid request. Until
        then the estimate holds its reservation.
        """
        charge = _cost(billable_usd)
        if charge != provider_billable_cost(openrouter_charge_usd, upstream_cost_usd, is_byok):
            raise ValueError("Reconciliation requires complete, matching provider billing components")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT 1 FROM attempts WHERE attempt_id = ?", (attempt_id,)).fetchone()
            if row is None:
                db.rollback()
                raise ValueError("Unknown attempt ID")
            db.execute(
                """UPDATE attempts SET charged_usd = ?, certainty = 'confirmed',
                evidence_source = 'provider_reconciliation', status = 'succeeded',
                provider_request_id = COALESCE(?, provider_request_id),
                openrouter_charge_usd = COALESCE(?, openrouter_charge_usd),
                upstream_cost_usd = COALESCE(?, upstream_cost_usd),
                is_byok = COALESCE(?, is_byok),
                reconciled_at = CURRENT_TIMESTAMP WHERE attempt_id = ?""",
                (
                    str(charge),
                    provider_request_id,
                    None if openrouter_charge_usd is None else str(Decimal(openrouter_charge_usd)),
                    None if upstream_cost_usd is None else str(Decimal(upstream_cost_usd)),
                    None if is_byok is None else int(is_byok),
                    attempt_id,
                ),
            )
            db.commit()

    def has_run(self, run_id: str) -> bool:
        with self._connect() as db:
            return db.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone() is not None

    def unreconciled(
        self, run_id: str, *, source: FundingSource = FundingSource.EXPERIMENT_GRANT
    ) -> list[tuple[str, str | None]]:
        """Attempts whose real cost is still unknown, as (attempt_id, provider_request_id).

        These hold their full reservation against the budget until reconciled, which is
        what keeps a BYOK run from reading as free.
        """
        with self._connect() as db:
            return [
                (row[0], row[1])
                for row in db.execute(
                    """SELECT attempt_id, provider_request_id FROM attempts
                    WHERE run_id = ? AND source = ? AND certainty != 'confirmed'""",
                    (run_id, source),
                )
            ]

    def write_off(self, adjustment_id: str, provider_total_usd: Decimal, note: str, *, expected_count: int) -> int:
        """Replace attempts that can never be reconciled with the provider's own total.

        Covers unconfirmed attempts with no generation ID, so there is no billing record
        to look up. Each is closed at $0 as ``operator_write_off`` and one adjustment of
        ``provider_total_usd`` takes their place, in a single transaction. The count must
        match what the operator expects, so a wrong ledger or a second application
        changes nothing.
        """
        if not adjustment_id or not note:
            raise ValueError("Adjustment ID and note are required")
        amount = _amount(provider_total_usd)
        selector = """FROM attempts WHERE source = ? AND certainty != 'confirmed'
            AND provider_request_id IS NULL AND status != 'reserved'"""
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            (count,) = db.execute(f"SELECT COUNT(*) {selector}", (FundingSource.EXPERIMENT_GRANT,)).fetchone()
            if count != expected_count:
                db.rollback()
                raise ValueError(f"Found {count} unreconcilable attempts, expected {expected_count}; nothing changed")
            if db.execute("SELECT 1 FROM adjustments WHERE adjustment_id = ?", (adjustment_id,)).fetchone():
                db.rollback()
                raise ValueError("Adjustment ID already used")
            db.execute(
                f"""UPDATE attempts SET charged_usd = '0', certainty = 'confirmed',
                evidence_source = 'operator_write_off', reconciled_at = CURRENT_TIMESTAMP
                WHERE attempt_id IN (SELECT attempt_id {selector})""",
                (FundingSource.EXPERIMENT_GRANT,),
            )
            db.execute(
                "INSERT INTO adjustments (adjustment_id, source, amount_usd, note) VALUES (?, ?, ?, ?)",
                (adjustment_id, FundingSource.EXPERIMENT_GRANT, str(amount), note),
            )
            db.commit()
        return count

    def adjust(self, adjustment_id: str, amount_usd: Decimal, note: str) -> None:
        if not adjustment_id or not note:
            raise ValueError("Adjustment ID and note are required")
        amount = _amount(amount_usd)
        with self._connect() as db:
            db.execute(
                "INSERT INTO adjustments (adjustment_id, source, amount_usd, note) VALUES (?, ?, ?, ?)",
                (adjustment_id, FundingSource.EXPERIMENT_GRANT, str(amount), note),
            )

    def summary(
        self, run_id: str | None = None, *, source: FundingSource = FundingSource.EXPERIMENT_GRANT
    ) -> dict[str, str | int]:
        """Totals for one source. ``remaining_local_usd`` is reported for the grant only."""
        grant = source is FundingSource.EXPERIMENT_GRANT
        with self._connect() as db:
            query = "SELECT status, reserved_usd, charged_usd, certainty FROM attempts WHERE source = ?"
            args: tuple[str, ...] = (source,)
            if run_id is not None:
                query += " AND run_id = ?"
                args += (run_id,)
            rows = db.execute(query, args).fetchall()
            charged = sum((Decimal(c) for _, _, c, _ in rows if c is not None), Decimal(0))
            confirmed = sum(
                (Decimal(c) for _, _, c, certainty in rows if c is not None and certainty == "confirmed"), Decimal(0)
            )
            estimated = sum(
                (Decimal(c) for _, _, c, certainty in rows if c is not None and certainty == "estimated"), Decimal(0)
            )
            pending = sum((Decimal(r) for status, r, c, _ in rows if status == "reserved" and c is None), Decimal(0))
            uncertain = sum((Decimal(r) for status, r, c, _ in rows if status != "reserved" and c is None), Decimal(0))
            adjustments = Decimal(0)
            if run_id is None and grant:
                adjustments = sum(
                    (
                        Decimal(row[0])
                        for row in db.execute(
                            "SELECT amount_usd FROM adjustments WHERE source = ?", (FundingSource.EXPERIMENT_GRANT,)
                        )
                    ),
                    Decimal(0),
                )
            totals: dict[str, str | int] = {
                "attempts": len(rows),
                "unconfirmed_attempts": sum(certainty != "confirmed" for _, _, _, certainty in rows),
                "charged_usd": str(charged),
                "confirmed_usd": str(confirmed),
                "estimated_usd": str(estimated),
                "reserved_usd": str(pending),
                "uncertain_usd": str(uncertain),
                "adjustments_usd": str(adjustments),
            }
            if grant:
                totals["remaining_local_usd"] = str(GRANT_ALLOCATION_USD - GRANT_RESERVE_USD - self._liability(db))
            return totals

    def sources(self, run_id: str | None = None) -> list[FundingSource]:
        """Every source with at least one attempt, in a stable order."""
        with self._connect() as db:
            query = "SELECT DISTINCT source FROM attempts" + (" WHERE run_id = ?" if run_id is not None else "")
            found = {row[0] for row in db.execute(query, (run_id,) if run_id is not None else ())}
        return [source for source in FundingSource if source in found]

    def breakdown(self, run_id: str | None = None) -> list[dict[str, str | int]]:
        """Accounted calls grouped by run, source, model and stage, without request data."""
        with self._connect() as db:
            query = "SELECT run_id, source, model, stage, status, reserved_usd, charged_usd, certainty FROM attempts"
            args: tuple[str, ...] = ()
            if run_id is not None:
                query += " WHERE run_id = ?"
                args += (run_id,)
            groups: dict[tuple[str, str, str, str], dict[str, str | int]] = {}
            for item_run, source, model, stage, status, reserved, charged, certainty in db.execute(query, args):
                key = (item_run, source, model, stage)
                group = groups.setdefault(
                    key,
                    {
                        "run_id": item_run,
                        "source": source,
                        "model": model,
                        "stage": stage,
                        "attempts": 0,
                        "charged_usd": "0",
                        "confirmed_usd": "0",
                        "estimated_usd": "0",
                        "reserved_usd": "0",
                        "uncertain_usd": "0",
                    },
                )
                group["attempts"] = int(group["attempts"]) + 1
                field = (
                    "charged_usd"
                    if charged is not None
                    else ("reserved_usd" if status == "reserved" else "uncertain_usd")
                )
                group[field] = str(Decimal(str(group[field])) + Decimal(charged if charged is not None else reserved))
                if charged is not None and certainty in {"confirmed", "estimated"}:
                    group[certainty + "_usd"] = str(Decimal(str(group[certainty + "_usd"])) + Decimal(charged))
            return [groups[key] for key in sorted(groups)]
