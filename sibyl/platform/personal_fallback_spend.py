"""Optional, local accounting for ordinary primary-to-personal OpenRouter fallback.

This observes the secondary call only when SIBYL_PERSONAL_FALLBACK_LEDGER is set.
It never changes key selection, retry policy, or the frozen telemetry markers.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
from collections.abc import Awaitable, Callable
from decimal import Decimal
from pathlib import Path
from typing import TypeVar
from uuid import uuid4

from sibyl.platform.experiment_usage import capture_grant_usage

LEDGER_ENV = "SIBYL_PERSONAL_FALLBACK_LEDGER"
RUN_ID_ENV = "SIBYL_PERSONAL_SPEND_RUN_ID"
_PROCESS_RUN_ID = uuid4().hex
_T = TypeVar("_T")
logger = logging.getLogger(__name__)


class PersonalFallbackLedger:
    """Separate source table; grant budget calculations never include these rows."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path, timeout=30) as db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS personal_fallback_attempts (
                    attempt_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, model TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'personal', credential_label TEXT NOT NULL DEFAULT 'OPENROUTER_API_KEY',
                    status TEXT NOT NULL, certainty TEXT NOT NULL, estimated_usd TEXT,
                    provider_request_id TEXT, started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    completed_at TEXT
                )"""
            )

    def start(self, attempt_id: str, run_id: str, model: str) -> None:
        with sqlite3.connect(self.path, timeout=30) as db:
            db.execute(
                """INSERT INTO personal_fallback_attempts
                (attempt_id, run_id, model, status, certainty) VALUES (?, ?, ?, 'started', 'unknown')""",
                (attempt_id, run_id, model),
            )

    def finish(
        self, attempt_id: str, *, status: str, estimated_usd: Decimal | None, provider_request_id: str | None
    ) -> None:
        if status not in {"succeeded", "failed"}:
            raise ValueError("Invalid fallback attempt status")
        with sqlite3.connect(self.path, timeout=30) as db:
            db.execute(
                """UPDATE personal_fallback_attempts SET status = ?, certainty = ?, estimated_usd = ?,
                provider_request_id = ?, completed_at = CURRENT_TIMESTAMP WHERE attempt_id = ?""",
                (
                    status,
                    "estimated" if estimated_usd is not None else "unknown",
                    str(estimated_usd) if estimated_usd is not None else None,
                    provider_request_id,
                    attempt_id,
                ),
            )

    def summary(self) -> dict[str, str | int]:
        with sqlite3.connect(self.path, timeout=30) as db:
            rows = db.execute("SELECT certainty, estimated_usd FROM personal_fallback_attempts").fetchall()
        return {
            "source": "personal",
            "attempts": len(rows),
            "estimated_usd": str(sum((Decimal(cost) for certainty, cost in rows if certainty == "estimated"), Decimal(0))),
            "unknown_attempts": sum(certainty == "unknown" for certainty, _ in rows),
        }


async def invoke_personal_fallback(call: Callable[[], Awaitable[_T]], *, model: str) -> _T:
    """Observe a secondary invocation; accounting failures must not block recovery."""
    path = os.getenv(LEDGER_ENV, "").strip()
    if not path:
        return await call()
    ledger: PersonalFallbackLedger | None = None
    attempt_id = uuid4().hex
    try:
        ledger = await asyncio.to_thread(PersonalFallbackLedger, Path(path))
        await asyncio.to_thread(ledger.start, attempt_id, os.getenv(RUN_ID_ENV) or _PROCESS_RUN_ID, model)
    except (OSError, sqlite3.Error) as exc:
        logger.warning("Personal fallback accounting unavailable (%s); call proceeds", type(exc).__name__)
        ledger = None
    if ledger is None:
        return await call()

    with capture_grant_usage() as capture:
        try:
            result = await call()
        except BaseException:
            try:
                await asyncio.to_thread(ledger.finish, attempt_id, status="failed", estimated_usd=None,
                                        provider_request_id=capture.generation_id)
            except (OSError, sqlite3.Error):
                logger.warning("Could not finish personal fallback ledger event")
            raise
        # LiteLLM may schedule its success callback just after invoke returns.
        for _ in range(10):
            if capture.generation_id is not None or capture.reported_cost is not None:
                break
            await asyncio.sleep(0.01)
    estimate = capture.reported_cost if capture.reported_cost is not None and capture.reported_cost > 0 else None
    try:
        await asyncio.to_thread(ledger.finish, attempt_id, status="succeeded", estimated_usd=estimate,
                                provider_request_id=capture.generation_id)
    except (OSError, sqlite3.Error):
        logger.warning("Could not finish personal fallback ledger event")
    return result
