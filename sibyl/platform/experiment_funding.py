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
    ) -> None:
        charge = Decimal(charged_usd)
        if not charge.is_finite() or charge < 0:
            raise ValueError("Settled cost must be a finite nonnegative decimal")
        if certainty not in {"confirmed", "estimated"} or not evidence_source:
            raise ValueError("Settlement requires a valid certainty and evidence source")
        if certainty == "confirmed" and evidence_source not in {"provider_generation", "provider_reconciliation"}:
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
                if row[1] == "estimated" and certainty == "confirmed" and evidence_source == "provider_reconciliation":
                    db.execute(
                        """UPDATE attempts SET charged_usd = ?, certainty = 'confirmed', evidence_source = ?,
                        provider_request_id = COALESCE(?, provider_request_id), reconciled_at = CURRENT_TIMESTAMP
                        WHERE attempt_id = ?""",
                        (str(charge), evidence_source, provider_request_id, attempt_id),
                    )
                    db.commit()
                    return
                db.rollback()
                if Decimal(row[0]) != charge or row[1] != certainty:
                    raise ValueError("Attempt has already settled at a different cost")
                return
            db.execute(
                """UPDATE attempts SET charged_usd = ?, certainty = ?, evidence_source = ?,
                status = 'succeeded', prompt_tokens = ?, completion_tokens = ?,
                provider_request_id = ?, completed_at = CURRENT_TIMESTAMP
                WHERE attempt_id = ?""",
                (
                    str(charge),
                    certainty,
                    evidence_source,
                    prompt_tokens,
                    completion_tokens,
                    provider_request_id,
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

    def has_blocking_attempts(self, run_id: str, question_id: str) -> bool:
        """Block publication for unconfirmed charges or a per-call overage."""
        with self._connect() as db:
            rows = db.execute(
                """SELECT 1 FROM attempts
                WHERE run_id = ? AND (question_id = ? OR question_id IS NULL) AND source = ?
                AND (status != 'succeeded' OR certainty != 'confirmed') LIMIT 1""",
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
