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
    first.settle("a", Decimal("0.40"), 10, 5, certainty="confirmed", evidence_source="provider_generation")
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
    ledger.settle("a", Decimal("0.4"), 10, 5, certainty="confirmed", evidence_source="provider_generation")
    ledger.settle("a", Decimal("0.4"), 10, 5, certainty="confirmed", evidence_source="provider_generation")
    assert ledger.summary("run-a")["charged_usd"] == "0.4"
    assert ledger.summary("run-a")["uncertain_usd"] == "0"


def test_estimate_keeps_reservation_until_provider_reconciliation(tmp_path: Path) -> None:
    ledger = SpendLedger(tmp_path / "spend.sqlite3")
    ledger.start_run("run-a", Decimal("1"))
    ledger.reserve("a", "run-a", "openrouter/test", "forecast", Decimal("0.8"))
    ledger.settle("a", Decimal("0.4"), 10, 5, certainty="estimated", evidence_source="framework_estimate")
    assert ledger.summary("run-a")["estimated_usd"] == "0.4"
    with pytest.raises(BudgetExceeded):
        ledger.reserve("b", "run-a", "openrouter/test", "forecast", Decimal("0.3"))
    ledger.settle("a", Decimal("0.4"), 10, 5, certainty="confirmed", evidence_source="provider_reconciliation")
    ledger.reserve("b", "run-a", "openrouter/test", "forecast", Decimal("0.3"))
    assert ledger.summary("run-a")["confirmed_usd"] == "0.4"


def test_local_estimate_cannot_be_called_confirmed(tmp_path: Path) -> None:
    ledger = SpendLedger(tmp_path / "spend.sqlite3")
    ledger.start_run("run-a", Decimal("1"))
    ledger.reserve("a", "run-a", "openrouter/test", "forecast", Decimal("0.8"))
    with pytest.raises(ValueError, match="provider billing evidence"):
        ledger.settle("a", Decimal("0.4"), 10, 5, certainty="confirmed", evidence_source="framework_estimate")


def test_secret_is_not_part_of_ledger_schema(tmp_path: Path) -> None:
    ledger = SpendLedger(tmp_path / "spend.sqlite3")
    ledger.start_run("run-a", Decimal("1"))
    ledger.reserve("a", "run-a", "openrouter/test", "forecast", Decimal("0.2"))
    assert b"secret" not in (tmp_path / "spend.sqlite3").read_bytes()
