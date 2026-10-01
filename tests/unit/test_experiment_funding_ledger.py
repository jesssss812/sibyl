"""Grant accounting uses dummy data and never invokes a provider."""

from decimal import Decimal
from pathlib import Path

import pytest

from sibyl.platform.experiment_funding import BudgetExceeded, SpendLedger


def test_reservations_are_atomic_across_ledger_instances(tmp_path: Path) -> None:
    path = tmp_path / "spend.sqlite3"
    first = SpendLedger(path)
    second = SpendLedger(path)
    first.start_run("run-a", Decimal("1.00"))
    first.reserve("a", "run-a", "openrouter/test", "forecast", Decimal("0.60"))
    with pytest.raises(BudgetExceeded):
        second.reserve("b", "run-a", "openrouter/test", "forecast", Decimal("0.50"))
    first.settle("a", Decimal("0.40"), 10, 5)
    second.reserve("b", "run-a", "openrouter/test", "forecast", Decimal("0.50"))
    assert second.summary("run-a")["charged_usd"] == "0.40"
    assert second.summary("run-a")["reserved_usd"] == "0.50"


def test_uncertain_attempt_retains_liability_and_settlement_is_idempotent(tmp_path: Path) -> None:
    ledger = SpendLedger(tmp_path / "spend.sqlite3")
    ledger.start_run("run-a", Decimal("1"))
    ledger.reserve("a", "run-a", "openrouter/test", "parser", Decimal("0.8"))
    ledger.mark_uncertain("a")
    with pytest.raises(BudgetExceeded):
        ledger.reserve("b", "run-a", "openrouter/test", "parser", Decimal("0.3"))
    ledger.settle("a", Decimal("0.4"), 10, 5)
    ledger.settle("a", Decimal("0.4"), 10, 5)
    assert ledger.summary("run-a")["charged_usd"] == "0.4"
    assert ledger.summary("run-a")["uncertain_usd"] == "0"


def test_secret_is_not_part_of_ledger_schema(tmp_path: Path) -> None:
    ledger = SpendLedger(tmp_path / "spend.sqlite3")
    ledger.start_run("run-a", Decimal("1"))
    ledger.reserve("a", "run-a", "openrouter/test", "forecast", Decimal("0.2"))
    assert b"secret" not in (tmp_path / "spend.sqlite3").read_bytes()
