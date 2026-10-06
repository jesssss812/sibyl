"""Local, transactional accounting for a separately funded OpenRouter grant.

The $100 allocation is a local envelope, not a claim about provider account balance.
Pending and uncertain requests retain their full reservation after a crash.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from typing import Iterator

GRANT_KEY_ENV = "SIBYL_EXPERIMENT_OPENROUTER_API_KEY"
# Evidence that establishes a real charge, as opposed to a local guess.
#
# `provider_response` is OpenRouter's usage accounting on the completion response, used
# only when it states `is_byok` and so carries the full BYOK split. Accepting it means a
# confirmed charge does not wait on the eventually-consistent /generation GET.
# `provider_response_partial` (a bare `usage.cost`, possibly just the BYOK fee) and
# `framework_estimate` (LiteLLM token maths) are NOT here: they must not discharge a
# reservation.
CONFIRMING_EVIDENCE: frozenset[str] = frozenset({"provider_response", "provider_generation", "provider_reconciliation"})

GRANT_ALLOCATION_USD = Decimal("100")
GRANT_RESERVE_USD = Decimal("5")


class FundingSource(StrEnum):
    EXISTING = "existing"
    EXPERIMENT_GRANT = "experiment-grant"


class GrantExecutionError(RuntimeError):
    """Grant accounting requires a stop; another paid attempt cannot repair it."""


class BudgetExceeded(GrantExecutionError):
    """A local spend limit has insufficient unreserved capacity."""


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

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        try:
            db.execute("PRAGMA foreign_keys = ON")
            db.execute("PRAGMA busy_timeout = 30000")
            yield db
        finally:
            db.close()

    def start_run(self, run_id: str, limit_usd: Decimal) -> None:
        limit = _amount(limit_usd)
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
                (run_id, FundingSource.EXPERIMENT_GRANT, str(limit)),
            )
            db.commit()

    @staticmethod
    def _liability(db: sqlite3.Connection, *, run_id: str | None = None) -> Decimal:
        query = "SELECT reserved_usd, charged_usd, certainty FROM attempts WHERE source = ?"
        args: tuple[str, ...] = (FundingSource.EXPERIMENT_GRANT,)
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
        if run_id is None:
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
    ) -> None:
        reserve = _amount(reserve_usd)
        if not attempt_id or not model or not model.startswith("openrouter/") or not stage:
            raise ValueError("OpenRouter model, stage, and attempt ID are required")
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
            if self._liability(db, run_id=run_id) + reserve > Decimal(row[0]):
                db.rollback()
                raise BudgetExceeded("Run budget exhausted")
            if self._liability(db) + reserve > GRANT_ALLOCATION_USD - GRANT_RESERVE_USD:
                db.rollback()
                raise BudgetExceeded("Local grant allocation exhausted")
            db.execute(
                """INSERT INTO attempts
                (attempt_id, run_id, source, model, stage, question_id, arm_id,
                 credential_label, reserved_usd, certainty, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'unknown', 'reserved')""",
                (
                    attempt_id,
                    run_id,
                    FundingSource.EXPERIMENT_GRANT,
                    model,
                    stage,
                    question_id,
                    arm_id,
                    "experiment-grant",
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
        charge = Decimal(charged_usd)
        if not charge.is_finite() or charge < 0:
            raise ValueError("Settled cost must be a finite nonnegative decimal")
        if certainty not in {"confirmed", "estimated"} or not evidence_source:
            raise ValueError("Settlement requires a valid certainty and evidence source")
        if certainty == "confirmed" and evidence_source not in CONFIRMING_EVIDENCE:
            raise ValueError("Confirmed cost requires provider billing evidence")
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
        then the estimate holds its reservation and blocks its question's publication.
        """
        charge = Decimal(billable_usd)
        if not charge.is_finite() or charge < 0:
            raise ValueError("Reconciled cost must be a finite nonnegative decimal")
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

    def unreconciled(self, run_id: str) -> list[tuple[str, str | None]]:
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
                    (run_id, FundingSource.EXPERIMENT_GRANT),
                )
            ]

    def has_blocking_attempts(self, run_id: str, question_id: str) -> bool:
        """Block for a call that did not succeed, or a per-call overage.

        An estimated charge does not block: the money is spent either way, and its held
        reservation already guards the limit. It is confirmed later by reconciliation.
        """
        with self._connect() as db:
            rows = db.execute(
                """SELECT 1 FROM attempts
                WHERE run_id = ? AND (question_id = ? OR question_id IS NULL) AND source = ?
                AND status != 'succeeded' LIMIT 1""",
                (run_id, question_id, FundingSource.EXPERIMENT_GRANT),
            ).fetchone()
            if rows is not None:
                return True
            return any(
                Decimal(charged) > Decimal(reserved)
                for reserved, charged in db.execute(
                    """SELECT reserved_usd, charged_usd FROM attempts
                    WHERE run_id = ? AND (question_id = ? OR question_id IS NULL)
                    AND source = ? AND charged_usd IS NOT NULL""",
                    (run_id, question_id, FundingSource.EXPERIMENT_GRANT),
                )
            )

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

    def summary(self, run_id: str | None = None) -> dict[str, str | int]:
        with self._connect() as db:
            query = "SELECT status, reserved_usd, charged_usd, certainty FROM attempts WHERE source = ?"
            args: tuple[str, ...] = (FundingSource.EXPERIMENT_GRANT,)
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
            uncertain = sum((Decimal(r) for status, r, c, _ in rows if status == "uncertain" and c is None), Decimal(0))
            adjustments = Decimal(0)
            if run_id is None:
                adjustments = sum(
                    (
                        Decimal(row[0])
                        for row in db.execute(
                            "SELECT amount_usd FROM adjustments WHERE source = ?", (FundingSource.EXPERIMENT_GRANT,)
                        )
                    ),
                    Decimal(0),
                )
            return {
                "attempts": len(rows),
                "charged_usd": str(charged),
                "confirmed_usd": str(confirmed),
                "estimated_usd": str(estimated),
                "reserved_usd": str(pending),
                "uncertain_usd": str(uncertain),
                "adjustments_usd": str(adjustments),
                "remaining_local_usd": str(GRANT_ALLOCATION_USD - GRANT_RESERVE_USD - self._liability(db)),
            }

    def breakdown(self, run_id: str | None = None) -> list[dict[str, str | int]]:
        """Accounted calls grouped by run, model and stage, without request data."""
        with self._connect() as db:
            query = (
                "SELECT run_id, model, stage, status, reserved_usd, charged_usd, certainty "
                "FROM attempts WHERE source = ?"
            )
            args: tuple[str, ...] = (FundingSource.EXPERIMENT_GRANT,)
            if run_id is not None:
                query += " AND run_id = ?"
                args += (run_id,)
            groups: dict[tuple[str, str, str], dict[str, str | int]] = {}
            for item_run, model, stage, status, reserved, charged, certainty in db.execute(query, args):
                key = (item_run, model, stage)
                group = groups.setdefault(
                    key,
                    {
                        "run_id": item_run,
                        "source": FundingSource.EXPERIMENT_GRANT,
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
                    else ("uncertain_usd" if status == "uncertain" else "reserved_usd")
                )
                group[field] = str(Decimal(str(group[field])) + Decimal(charged if charged is not None else reserved))
                if charged is not None and certainty in {"confirmed", "estimated"}:
                    group[certainty + "_usd"] = str(Decimal(str(group[certainty + "_usd"])) + Decimal(charged))
            return [groups[key] for key in sorted(groups)]


@dataclass(frozen=True)
class GrantSession:
    run_id: str
    ledger: SpendLedger
    per_call_reserve_usd: Decimal
    key: str

    def __repr__(self) -> str:
        return f"GrantSession(run_id={self.run_id!r}, key=<redacted>)"


_ACTIVE_GRANT: ContextVar[GrantSession | None] = ContextVar("sibyl_active_grant", default=None)
_ACTIVE_QUESTION: ContextVar[str | None] = ContextVar("sibyl_grant_question", default=None)


def active_grant() -> GrantSession | None:
    return _ACTIVE_GRANT.get()


def active_grant_question() -> str | None:
    return _ACTIVE_QUESTION.get()


@contextmanager
def grant_question(question_id: int | None) -> Iterator[None]:
    token = _ACTIVE_QUESTION.set(str(question_id) if question_id is not None else None)
    try:
        yield
    finally:
        _ACTIVE_QUESTION.reset(token)


@contextmanager
def grant_context(session: GrantSession) -> Iterator[None]:
    token: Token[GrantSession | None] = _ACTIVE_GRANT.set(session)
    try:
        yield
    finally:
        _ACTIVE_GRANT.reset(token)
