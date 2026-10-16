"""The ledger tool: report by source, and read-only reconciliation per key."""

from __future__ import annotations

import json
import sqlite3
from decimal import Decimal
from pathlib import Path

import pytest

from sibyl.config import env as env_module
from sibyl.ops import experiment_grant
from sibyl.platform import experiment_usage
from sibyl.platform.experiment_funding import FundingSource, SpendLedger


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    # main() loads .env, which would put real keys back.
    monkeypatch.setattr(env_module, "load_environment", lambda: None)


def _ledger_with_estimates(path: Path) -> SpendLedger:
    ledger = SpendLedger(path)
    ledger.start_run("r", Decimal("50"))
    ledger.reserve("g", "r", "openrouter/openai/t", "forecaster", Decimal("1"))
    ledger.settle("g", Decimal("0.2"), 1, 1, "gen-g", certainty="estimated", evidence_source="framework_estimate")
    ledger.reserve("p", "r", "openrouter/openai/t", "parser", Decimal(0), source=FundingSource.PERSONAL_KEY)
    ledger.settle("p", Decimal("0.01"), 1, 1, "gen-p", certainty="estimated", evidence_source="framework_estimate")
    ledger.reserve("d", "r", "exa/search", "gap_fill_v2", Decimal(0), source=FundingSource.DIRECT_SERVICE)
    return ledger


def test_report_shows_every_source_that_paid(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "l.sqlite3"
    _ledger_with_estimates(path)
    assert experiment_grant.main(["report", "--ledger", str(path), "--run-id", "r"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert list(payload["totals"]) == ["experiment-grant", "personal-key", "direct-service"]
    assert payload["totals"]["personal-key"]["estimated_usd"] == "0.01"
    assert "legacy_personal_fallback" not in payload


def test_report_includes_the_legacy_fallback_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "l.sqlite3"
    SpendLedger(path)
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE personal_fallback_attempts (attempt_id TEXT, run_id TEXT, certainty TEXT, estimated_usd TEXT)"
        )
        db.execute("INSERT INTO personal_fallback_attempts VALUES ('a', 'old', 'estimated', '0.5')")
    assert experiment_grant.main(["report", "--ledger", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["legacy_personal_fallback"]["estimated_usd"] == "0.5"


def test_reconcile_reads_each_source_with_its_own_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "l.sqlite3"
    _ledger_with_estimates(path)
    monkeypatch.setenv("SIBYL_EXPERIMENT_OPENROUTER_API_KEY", "grant-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "personal-key")
    looked_up: list[tuple[str, str]] = []

    async def record(generation_id: str, api_key: str) -> experiment_usage.GenerationRecord:
        looked_up.append((generation_id, api_key))
        return experiment_usage.GenerationRecord(Decimal("0.3"), None, False)

    monkeypatch.setattr(experiment_usage, "generation_record", record)
    assert experiment_grant.main(["reconcile", "--ledger", str(path), "--run-id", "r"]) == 0
    assert sorted(looked_up) == [("gen-g", "grant-key"), ("gen-p", "personal-key")]
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"experiment-grant", "personal-key"}, "direct services have no OpenRouter record"


def test_reconcile_skips_only_the_source_whose_key_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "l.sqlite3"
    _ledger_with_estimates(path)
    monkeypatch.setenv("SIBYL_EXPERIMENT_OPENROUTER_API_KEY", "")
    monkeypatch.setenv("OPENROUTER_API_KEY", "personal-key")

    async def record(generation_id: str, api_key: str) -> experiment_usage.GenerationRecord:
        return experiment_usage.GenerationRecord(Decimal("0.3"), None, False)

    monkeypatch.setattr(experiment_usage, "generation_record", record)
    assert experiment_grant.main(["reconcile", "--ledger", str(path), "--run-id", "r"]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["experiment-grant"] == "skipped: SIBYL_EXPERIMENT_OPENROUTER_API_KEY is not set"
    assert payload["personal-key"]["reconciled"] == 1, "a missing key must not stop the other sources"


def test_an_unreconcilable_attempt_exits_non_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An attempt with no generation ID must not look like a reconciled run."""
    path = tmp_path / "l.sqlite3"
    ledger = SpendLedger(path)
    ledger.start_run("r", Decimal("50"))
    ledger.reserve("a1", "r", "openrouter/openai/t", "forecaster", Decimal("1"))
    ledger.settle("a1", Decimal("0"), 1, 1, None, certainty="estimated", evidence_source="provider_response_partial")
    monkeypatch.setenv("SIBYL_EXPERIMENT_OPENROUTER_API_KEY", "grant-key")
    assert experiment_grant.main(["reconcile", "--ledger", str(path), "--run-id", "r"]) == 1
    assert json.loads(capsys.readouterr().out)["experiment-grant"]["no_generation_id"] == 1


def test_reconcile_refuses_a_missing_ledger_instead_of_creating_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SIBYL_EXPERIMENT_OPENROUTER_API_KEY", "grant-key")
    typo = tmp_path / "typo.sqlite3"
    assert experiment_grant.main(["reconcile", "--ledger", str(typo), "--run-id", "r"]) == 2
    assert "No spend ledger" in capsys.readouterr().out
    assert not typo.exists()


def test_reconcile_refuses_an_unknown_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SIBYL_EXPERIMENT_OPENROUTER_API_KEY", "grant-key")
    path = tmp_path / "l.sqlite3"
    SpendLedger(path).start_run("real-run", Decimal("5"))
    assert experiment_grant.main(["reconcile", "--ledger", str(path), "--run-id", "real-rnu"]) == 2
    assert "not in the ledger" in capsys.readouterr().out


def test_write_off_trues_up_the_ledger(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "l.sqlite3"
    ledger = SpendLedger(path)
    ledger.start_run("old", Decimal("30"))
    for index in range(8):
        ledger.reserve(f"x{index}", "old", "openrouter/test", "forecaster", Decimal("3"))
        ledger.mark_uncertain(f"x{index}")
    argv = ["write-off", "--ledger", str(path), "--provider-total-usd", "0.796028"]
    argv += ["--expect-count", "8", "--note", "OpenRouter byok_usage 2026-10-05", "--id", "trueup-2026-10-05"]
    assert experiment_grant.main(argv) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["written_off"] == 8
    assert payload["totals"]["remaining_local_usd"] == "94.203972"


def test_the_tool_reads_the_ledger_the_cli_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "configured.sqlite3"
    _ledger_with_estimates(path)
    monkeypatch.setenv("SIBYL_SPEND_LEDGER", str(path))
    assert experiment_grant.main(["report", "--run-id", "r"]) == 0
    assert "personal-key" in json.loads(capsys.readouterr().out)["totals"]


@pytest.mark.parametrize(
    "command", [["report"], ["record-prior-spend", "--amount-usd", "40", "--note", "n", "--id", "x"]]
)
def test_no_command_creates_a_ledger_at_a_mistyped_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], command: list[str]
) -> None:
    typo = tmp_path / "typo.sqlite3"
    assert experiment_grant.main([command[0], "--ledger", str(typo), *command[1:]]) == 2
    assert "No spend ledger" in capsys.readouterr().out
    assert not typo.exists()


def test_an_ordinary_key_attempt_without_a_generation_id_does_not_fail_reconcile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """It holds no budget and nothing can price it; it stays visible in the report instead."""
    path = tmp_path / "l.sqlite3"
    ledger = SpendLedger(path)
    ledger.start_run("r")
    ledger.reserve("p", "r", "openrouter/openai/t", "parser", Decimal(0), source=FundingSource.PERSONAL_KEY)
    ledger.mark_uncertain("p", evidence_source="TimeoutError status=None")
    monkeypatch.setenv("OPENROUTER_API_KEY", "personal-key")
    assert experiment_grant.main(["reconcile", "--ledger", str(path), "--run-id", "r"]) == 0
    assert json.loads(capsys.readouterr().out)["personal-key"]["no_generation_id"] == 1
