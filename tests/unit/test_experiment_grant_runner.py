"""Offline preflight for the isolated grant-funded MiniBench entry point."""

import argparse
from decimal import Decimal
from pathlib import Path

import pytest

from sibyl.ops.experiment_grant import _run_minibench, checked_session, preflight_research
from sibyl.platform.experiment_funding import GRANT_KEY_ENV, GrantSession, SpendLedger, active_grant
from sibyl.platform.experiment_llm import FundedGeneralLlm


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        run_id="test-run",
        ledger=tmp_path / "spend.sqlite3",
        max_cost_usd="3",
        reserve_per_call_usd="1",
    )


def test_key_is_required_only_for_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(GRANT_KEY_ENV, raising=False)
    with pytest.raises(ValueError, match=GRANT_KEY_ENV):
        checked_session(_args(tmp_path))


def test_grant_key_must_differ_from_existing_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(GRANT_KEY_ENV, "dummy-grant")
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-grant")
    with pytest.raises(ValueError, match="distinct"):
        checked_session(_args(tmp_path))


def test_direct_paid_research_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_SEARCH_ENABLED", "true")
    with pytest.raises(ValueError, match="ANTHROPIC_SEARCH_ENABLED"):
        preflight_research()


@pytest.mark.parametrize("flag", [
    "FINANCIAL_DATA_ENABLED", "TS_ANCHOR_ENABLED", "PREDICTION_MARKETS_ENABLED",
    "RESOLUTION_SOURCE_ENABLED", "REFERENCE_CLASS_ENABLED",
])
def test_initial_grant_profile_rejects_unaudited_enrichers(flag: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(flag, "true")
    with pytest.raises(ValueError, match=flag):
        preflight_research()


def test_per_call_reservation_must_fit_run_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(GRANT_KEY_ENV, "dummy-grant")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OAI_ANTH_OPENROUTER_KEY", raising=False)
    args = _args(tmp_path)
    args.max_cost_usd = "0.5"
    with pytest.raises(ValueError, match="per-call"):
        checked_session(args)


@pytest.mark.asyncio
async def test_mocked_minibench_uses_three_grant_slots_and_only_grant_research(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sibyl.forecasting import bot as bot_module
    from sibyl.platform import metaculus_preflight
    from sibyl.research.orchestrator import ResearchOrchestrator

    for flag in (
        "NATIVE_SEARCH_ENABLED", "GEMINI_SEARCH_ENABLED", "ANTHROPIC_SEARCH_ENABLED",
        "GAP_FILL_ENABLED", "GAP_FILL_V2_ENABLED", "FINANCIAL_DATA_ENABLED",
        "TS_ANCHOR_ENABLED", "PREDICTION_MARKETS_ENABLED", "RESOLUTION_SOURCE_ENABLED",
        "REFERENCE_CLASS_ENABLED",
    ):
        monkeypatch.delenv(flag, raising=False)
    assert preflight_research()["primary_research"].startswith("openrouter/")

    called: list[str] = []

    class FakeBot:
        alertable_count = 0

        def __init__(self, **kwargs):
            assert active_grant() is not None
            llms = kwargs["llms"]
            assert len(llms["forecasters"]) == 3
            assert all(isinstance(llm, FundedGeneralLlm) for llm in llms["forecasters"])
            assert all(llm.litellm_kwargs["api_key"] == "dummy-grant" for llm in llms["forecasters"])
            assert llms["forecasters"][1].model == "openrouter/anthropic/claude-opus-5"
            assert llms["stacker"].model == "openrouter/anthropic/claude-opus-4.8"
            research = ResearchOrchestrator(default_llm=llms["researcher"], summarizer_llm=llms["summarizer"])
            assert [name for _, name in research._select_research_providers()] == ["native_search_grant"]
            called.append("constructed")

        async def forecast_on_tournament(self, tournament_id, *, return_exceptions):
            assert return_exceptions is True
            called.append("mocked_forecast")
            return []

        @staticmethod
        def log_report_summary(reports):
            assert reports == []

    def fake_verify():
        assert called == []  # no model or research call has happened

    monkeypatch.setattr(bot_module, "SibylBot", FakeBot)
    monkeypatch.setattr(metaculus_preflight, "verify_metaculus_api_identity", fake_verify)
    ledger = SpendLedger(tmp_path / "spend.sqlite3")
    ledger.start_run("mock-run", Decimal("5"))
    session = GrantSession("mock-run", ledger, Decimal("1"), "dummy-grant")
    await _run_minibench(argparse.Namespace(max_questions=1), session)
    assert called == ["constructed", "mocked_forecast"]
    assert ledger.summary("mock-run")["attempts"] == 0
