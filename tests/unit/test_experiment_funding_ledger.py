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


def test_failed_attempts_are_scoped_to_question(tmp_path: Path) -> None:
    ledger = SpendLedger(tmp_path / "spend.sqlite3")
    ledger.start_run("run-a", Decimal("5"))
    ledger.reserve("a", "run-a", "openrouter/test", "research", Decimal("0.8"), question_id="101")
    ledger.reserve("b", "run-a", "openrouter/test", "research", Decimal("0.8"), question_id="101")
    ledger.reserve("c", "run-a", "openrouter/test", "research", Decimal("0.8"), question_id="102")
    ledger.mark_uncertain("a")
    ledger.mark_rejected("b", evidence_source="provider_rejected status=429")
    ledger.settle("c", Decimal("0.4"), 10, 5, certainty="confirmed", evidence_source="provider_generation")
    assert ledger.failed_attempts("run-a", "101") == 2
    assert ledger.failed_attempts("run-a", "102") == 0


def test_estimated_unknown_and_over_reserve_answers_are_not_failures(tmp_path: Path) -> None:
    # Each returned an answer; their costs are handled by the reservation, not the gate.
    ledger = SpendLedger(tmp_path / "spend.sqlite3")
    ledger.start_run("run-a", Decimal("5"))
    for attempt in ("est", "unknown", "over"):
        ledger.reserve(attempt, "run-a", "openrouter/test", "research", Decimal("0.8"), question_id="101")
    ledger.settle("est", Decimal("0.4"), 10, 5, certainty="estimated", evidence_source="framework_estimate")
    ledger.mark_cost_unknown("unknown", 10, 5, provider_request_id=None, evidence_source="unknown")
    ledger.settle("over", Decimal("1.5"), 10, 5, certainty="confirmed", evidence_source="provider_generation")
    assert ledger.failed_attempts("run-a", "101") == 0
    summary = ledger.summary("run-a")
    assert summary["uncertain_usd"] == "0.8", "the unknown cost holds its reservation"
    assert summary["estimated_usd"] == "0.4"


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


class TestWriteOff:
    """Truing up attempts that can never be reconciled against OpenRouter's own total."""

    def _ledger(self, tmp_path: Path) -> SpendLedger:
        ledger = SpendLedger(tmp_path / "spend.sqlite3")
        ledger.start_run("old", Decimal("30"))
        for index in range(8):
            ledger.reserve(f"x{index}", "old", "openrouter/test", "forecaster", Decimal("3"))
            ledger.mark_uncertain(f"x{index}")
        return ledger

    def test_stale_rows_are_replaced_by_the_provider_total(self, tmp_path: Path) -> None:
        ledger = self._ledger(tmp_path)
        assert ledger.summary()["remaining_local_usd"] == "71"
        written = ledger.write_off("trueup-1", Decimal("0.796028"), "OpenRouter byok_usage", expected_count=8)
        assert written == 8
        summary = ledger.summary()
        assert summary["remaining_local_usd"] == str(Decimal("95") - Decimal("0.796028"))
        assert summary["uncertain_usd"] == "0"
        assert ledger.unreconciled("old") == []

    def test_a_count_mismatch_changes_nothing(self, tmp_path: Path) -> None:
        ledger = self._ledger(tmp_path)
        with pytest.raises(ValueError, match="expected 7"):
            ledger.write_off("trueup-1", Decimal("0.796028"), "note", expected_count=7)
        assert ledger.summary()["remaining_local_usd"] == "71"

    def test_reconcilable_and_confirmed_rows_are_left_alone(self, tmp_path: Path) -> None:
        ledger = self._ledger(tmp_path)
        ledger.reserve("g", "old", "openrouter/test", "forecaster", Decimal("1"))
        ledger.settle("g", Decimal("0.1"), 1, 1, "gen-g", certainty="estimated", evidence_source="framework_estimate")
        ledger.reserve("c", "old", "openrouter/test", "forecaster", Decimal("1"))
        ledger.settle("c", Decimal("0.1"), 1, 1, certainty="confirmed", evidence_source="provider_generation")
        assert ledger.write_off("trueup-1", Decimal("0.5"), "note", expected_count=8) == 8
        assert ledger.unreconciled("old") == [("g", "gen-g")]

    def test_it_cannot_be_applied_twice(self, tmp_path: Path) -> None:
        ledger = self._ledger(tmp_path)
        ledger.write_off("trueup-1", Decimal("0.796028"), "note", expected_count=8)
        with pytest.raises(ValueError, match="expected 8"):
            ledger.write_off("trueup-2", Decimal("0.796028"), "note", expected_count=8)
