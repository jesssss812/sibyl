"""Grant accounting uses dummy data and never invokes a provider."""

import sqlite3
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
    ledger.reconcile("a", Decimal("0.4"))
    ledger.reserve("b", "run-a", "openrouter/test", "forecast", Decimal("0.3"))
    assert ledger.summary("run-a")["confirmed_usd"] == "0.4"


def test_unconfirmed_attempts_are_scoped_to_question(tmp_path: Path) -> None:
    ledger = SpendLedger(tmp_path / "spend.sqlite3")
    ledger.start_run("run-a", Decimal("3"))
    ledger.reserve("a", "run-a", "openrouter/test", "forecast", Decimal("0.8"), question_id="101")
    ledger.reserve("b", "run-a", "openrouter/test", "forecast", Decimal("0.8"), question_id="102")
    ledger.settle("a", Decimal("0.4"), 10, 5, certainty="estimated", evidence_source="framework_estimate")
    ledger.settle("b", Decimal("0.4"), 10, 5, certainty="confirmed", evidence_source="provider_generation")
    assert ledger.has_blocking_attempts("run-a", "101")
    assert not ledger.has_blocking_attempts("run-a", "102")
    ledger.reconcile("a", Decimal("0.4"))
    assert not ledger.has_blocking_attempts("run-a", "101")


def test_provider_charge_above_reservation_blocks_publication(tmp_path: Path) -> None:
    ledger = SpendLedger(tmp_path / "spend.sqlite3")
    ledger.start_run("run-a", Decimal("3"))
    ledger.reserve("a", "run-a", "openrouter/test", "forecast", Decimal("0.8"), question_id="101")
    ledger.settle("a", Decimal("1.2"), 10, 5, certainty="confirmed", evidence_source="provider_generation")
    assert ledger.has_blocking_attempts("run-a", "101")


def test_unattributed_call_blocks_every_question(tmp_path: Path) -> None:
    ledger = SpendLedger(tmp_path / "spend.sqlite3")
    ledger.start_run("run-a", Decimal("3"))
    ledger.reserve("a", "run-a", "openrouter/test", "forecast", Decimal("0.8"))
    assert ledger.has_blocking_attempts("run-a", "101")


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


def _liability(ledger: SpendLedger, run_id: str) -> Decimal:
    """The figure the budget actually enforces, read the way the ledger computes it."""
    with sqlite3.connect(ledger.path) as db:
        return SpendLedger._liability(db, run_id=run_id)


class TestByokCostComponents:
    """OpenRouter's charge and the upstream provider's charge are different numbers.

    Collapsing them is what let a BYOK run settle a confirmed $0 while $0.187 of real
    money moved upstream. `charged_usd` is now the BILLABLE figure the budget holds; the
    components are kept beside it so a run can be audited rather than believed.
    """

    def _ledger(self, tmp_path: Path) -> SpendLedger:
        ledger = SpendLedger(tmp_path / "spend.sqlite3")
        ledger.start_run("run", Decimal("50"))
        ledger.reserve("a1", "run", "openrouter/openai/t", "forecaster", Decimal("1"))
        return ledger

    def test_a_provider_response_cost_can_confirm(self, tmp_path: Path) -> None:
        """The behavioural fix: OpenRouter's own usage.cost is account evidence.

        Before, only a /generation GET could confirm, so publication depended on an
        eventually-consistent endpoint being readable within two seconds.
        """
        ledger = self._ledger(tmp_path)
        ledger.settle(
            "a1",
            Decimal("0.42"),
            10,
            5,
            "gen-1",
            certainty="confirmed",
            evidence_source="provider_response",
            openrouter_charge_usd=Decimal("0.42"),
            upstream_cost_usd=None,
            is_byok=False,
        )
        assert _liability(ledger, "run") == Decimal("0.42")

    def test_a_framework_estimate_still_cannot_confirm(self, tmp_path: Path) -> None:
        # LiteLLM token maths is a guess, and a guess must not discharge a reservation.
        ledger = self._ledger(tmp_path)
        with pytest.raises(ValueError, match="provider billing evidence"):
            ledger.settle(
                "a1",
                Decimal("0.42"),
                10,
                5,
                None,
                certainty="confirmed",
                evidence_source="framework_estimate",
            )

    def test_byok_components_are_stored_separately(self, tmp_path: Path) -> None:
        ledger = self._ledger(tmp_path)
        ledger.settle(
            "a1",
            Decimal("0.1873175"),
            718,
            2014,
            "gen-byok",
            certainty="confirmed",
            evidence_source="provider_generation",
            openrouter_charge_usd=Decimal("0"),
            upstream_cost_usd=Decimal("0.1873175"),
            is_byok=True,
        )
        with sqlite3.connect(ledger.path) as db:
            row = db.execute(
                "SELECT charged_usd, openrouter_charge_usd, upstream_cost_usd, is_byok FROM attempts"
            ).fetchone()
        # Billable tracks the money that actually moved, not OpenRouter's $0.
        assert row == ("0.1873175", "0", "0.1873175", 1)
        assert _liability(ledger, "run") == Decimal("0.1873175")

    def test_an_unconfirmed_attempt_holds_its_whole_reservation(self, tmp_path: Path) -> None:
        """The anti-under-accounting invariant.

        A $0 OpenRouter charge with unknown upstream cost must NOT read as free: until
        reconciliation the reservation stands, so the budget stays conservative.
        """
        ledger = self._ledger(tmp_path)
        ledger.settle(
            "a1",
            Decimal("0"),
            718,
            2014,
            "gen-pending",
            certainty="estimated",
            evidence_source="provider_response",
            openrouter_charge_usd=Decimal("0"),
            upstream_cost_usd=None,
            is_byok=None,
        )
        assert _liability(ledger, "run") == Decimal("1"), "a $0 estimate must not discharge the reserve"
        assert ledger.unreconciled("run") == [("a1", "gen-pending")]

    def test_reconciliation_replaces_the_estimate_and_confirms(self, tmp_path: Path) -> None:
        ledger = self._ledger(tmp_path)
        ledger.settle(
            "a1",
            Decimal("0"),
            718,
            2014,
            "gen-pending",
            certainty="estimated",
            evidence_source="provider_response",
            openrouter_charge_usd=Decimal("0"),
        )
        ledger.reconcile(
            "a1",
            Decimal("0.1873175"),
            openrouter_charge_usd=Decimal("0"),
            upstream_cost_usd=Decimal("0.1873175"),
            is_byok=True,
        )
        assert _liability(ledger, "run") == Decimal("0.1873175")
        assert ledger.unreconciled("run") == []

    def test_a_zero_provider_record_reconciles(self, tmp_path: Path) -> None:
        # A free route bills $0; that is a provider fact, not an invalid amount.
        ledger = self._ledger(tmp_path)
        ledger.settle(
            "a1", Decimal("0"), 1, 1, "gen-free", certainty="estimated", evidence_source="provider_response_partial"
        )
        ledger.reconcile("a1", Decimal("0"), openrouter_charge_usd=Decimal("0"), is_byok=False)
        assert _liability(ledger, "run") == Decimal("0")
        assert ledger.unreconciled("run") == []

    def test_a_negative_record_is_refused(self, tmp_path: Path) -> None:
        ledger = self._ledger(tmp_path)
        with pytest.raises(ValueError, match="nonnegative"):
            ledger.reconcile("a1", Decimal("-0.1"))

    def test_settle_cannot_overwrite_an_estimate(self, tmp_path: Path) -> None:
        # Confirming an estimate goes through reconcile(), which records the BYOK split.
        ledger = self._ledger(tmp_path)
        ledger.settle("a1", Decimal("0.4"), 1, 1, certainty="estimated", evidence_source="framework_estimate")
        with pytest.raises(ValueError, match="already settled"):
            ledger.settle("a1", Decimal("0.4"), 1, 1, certainty="confirmed", evidence_source="provider_reconciliation")

    def test_reconciling_an_unknown_attempt_is_an_error(self, tmp_path: Path) -> None:
        ledger = self._ledger(tmp_path)
        with pytest.raises(ValueError, match="Unknown attempt"):
            ledger.reconcile("nope", Decimal("1"))
