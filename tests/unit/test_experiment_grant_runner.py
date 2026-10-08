"""Offline preflight for the isolated grant-funded MiniBench entry point."""

import argparse
import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

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


@pytest.mark.parametrize(
    "flag",
    [
        "FINANCIAL_DATA_ENABLED",
        "TS_ANCHOR_ENABLED",
        "PREDICTION_MARKETS_ENABLED",
        "RESOLUTION_SOURCE_ENABLED",
        "REFERENCE_CLASS_ENABLED",
    ],
)
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


def _stub_key_lookup(monkeypatch: pytest.MonkeyPatch, outcome: object) -> list[str]:
    from sibyl.telemetry import check_credits

    seen: list[str] = []

    def fake_fetch(api_key: str, **kwargs: object) -> object:
        seen.append(api_key)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(check_credits, "fetch_auth_key", fake_fetch)
    return seen


class TestProviderLimitPreflight:
    """OpenRouter's limit_remaining, not the local ledger, decides whether a run fits."""

    def test_a_run_that_fits_the_key_limit_starts(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(GRANT_KEY_ENV, "dummy-grant")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("OAI_ANTH_OPENROUTER_KEY", raising=False)
        seen = _stub_key_lookup(monkeypatch, {"limit": 100, "limit_remaining": 99.203972})
        session = checked_session(_args(tmp_path))
        assert seen == ["dummy-grant"], "the grant key, and only it, is checked"
        assert session.run_id == "test-run"

    @pytest.mark.parametrize("remaining", [7.99, 0])
    def test_a_run_the_key_cannot_cover_is_refused(
        self, remaining: float, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from sibyl.platform.experiment_funding import BudgetExceeded

        monkeypatch.setenv(GRANT_KEY_ENV, "dummy-grant")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("OAI_ANTH_OPENROUTER_KEY", raising=False)
        _stub_key_lookup(monkeypatch, {"limit": 100, "limit_remaining": remaining})
        with pytest.raises(BudgetExceeded, match="remaining"):
            checked_session(_args(tmp_path))
        assert not (tmp_path / "spend.sqlite3").exists(), "refused before the run is recorded"

    def test_a_key_without_a_limit_is_refused(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(GRANT_KEY_ENV, "dummy-grant")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("OAI_ANTH_OPENROUTER_KEY", raising=False)
        _stub_key_lookup(monkeypatch, {"limit": None, "limit_remaining": None})
        with pytest.raises(ValueError, match="no spending limit"):
            checked_session(_args(tmp_path))

    def test_an_unreachable_lookup_is_refused(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import httpx

        monkeypatch.setenv(GRANT_KEY_ENV, "dummy-grant")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("OAI_ANTH_OPENROUTER_KEY", raising=False)
        _stub_key_lookup(monkeypatch, httpx.ConnectError("down"))
        with pytest.raises(ValueError, match="lookup failed"):
            checked_session(_args(tmp_path))


@pytest.mark.asyncio
async def test_grant_test_run_selects_one_exact_sandbox_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from forecasting_tools import BinaryQuestion, QuestionState

    from sibyl.forecasting import bot as bot_module
    from sibyl.ops import experiment_grant
    from sibyl.platform import metaculus_preflight

    def question(post_id: int, question_id: int) -> BinaryQuestion:
        return BinaryQuestion(
            question_text="Synthetic sandbox question",
            id_of_question=question_id,
            id_of_post=post_id,
            page_url=f"https://www.metaculus.com/questions/{post_id}",
            background_info="",
            resolution_criteria="Resolves yes if it occurs.",
            fine_print="",
            open_time=datetime(2026, 10, 4),
            scheduled_resolution_time=datetime(2026, 10, 20),
            tournament_slugs=["bot-testing-area"],
            state=QuestionState.OPEN,
        )

    selected = question(43323, 43325)
    other = question(43327, 43332)
    forecasted: list[int] = []

    class FakeClient:
        def get_all_open_questions_from_tournament(self, tournament_id):
            assert tournament_id == "bot-testing-area"
            return [other, selected]

    class FakeBot:
        alertable_count = 0

        def __init__(self, **kwargs):
            assert kwargs["publish_reports_to_metaculus"] is True
            assert kwargs["skip_previously_forecasted_questions"] is False
            assert kwargs["max_questions_per_run"] == 1
            assert all(isinstance(llm, FundedGeneralLlm) for llm in kwargs["llms"]["forecasters"])
            self.skip_previously_forecasted_questions = kwargs["skip_previously_forecasted_questions"]
            self.max_questions_per_run = kwargs["max_questions_per_run"]
            self.metaculus_client = FakeClient()

        async def forecast_questions(self, questions, return_exceptions=False):
            assert return_exceptions is True
            forecasted.extend(q.id_of_question for q in questions)
            return []

        @staticmethod
        def log_report_summary(reports):
            assert reports == []

    monkeypatch.setattr(bot_module, "SibylBot", FakeBot)
    monkeypatch.setattr(metaculus_preflight, "verify_metaculus_api_identity", lambda: None)
    ledger = SpendLedger(tmp_path / "spend.sqlite3")
    ledger.start_run("test-sandbox", Decimal("15"))
    session = GrantSession("test-sandbox", ledger, Decimal("3"), "dummy-grant")
    args = argparse.Namespace(
        max_questions=1, max_cost_usd="15", reports_dir=tmp_path / "runs", post_id=43323, question_id=43325
    )
    await experiment_grant._run_test_question(args, session)
    assert forecasted == [43325]
    manifest = json.loads((tmp_path / "runs" / "test-sandbox" / "manifest.json").read_text())
    assert manifest["configuration"]["mode"] == "test"

    args.question_id = 99999
    args.reports_dir = tmp_path / "wrong-run"
    with pytest.raises(ValueError, match="not an open question in bot-testing-area"):
        await experiment_grant._run_test_question(args, session)
    assert forecasted == [43325]


@pytest.mark.asyncio
async def test_mocked_minibench_uses_three_grant_slots_and_only_grant_research(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sibyl.forecasting import bot as bot_module
    from sibyl.platform import metaculus_preflight
    from sibyl.research.orchestrator import ResearchOrchestrator

    for flag in (
        "NATIVE_SEARCH_ENABLED",
        "GEMINI_SEARCH_ENABLED",
        "ANTHROPIC_SEARCH_ENABLED",
        "GAP_FILL_ENABLED",
        "GAP_FILL_V2_ENABLED",
        "FINANCIAL_DATA_ENABLED",
        "TS_ANCHOR_ENABLED",
        "PREDICTION_MARKETS_ENABLED",
        "RESOLUTION_SOURCE_ENABLED",
        "REFERENCE_CLASS_ENABLED",
    ):
        monkeypatch.delenv(flag, raising=False)
    assert preflight_research()["primary_research"].startswith("openrouter/")

    called: list[str] = []

    class FakeBot:
        alertable_count = 0

        def __init__(self, **kwargs):
            self.skip_previously_forecasted_questions = kwargs["skip_previously_forecasted_questions"]
            self.max_questions_per_run = kwargs["max_questions_per_run"]
            assert active_grant() is not None
            llms = kwargs["llms"]
            assert len(llms["forecasters"]) == 3
            assert all(isinstance(llm, FundedGeneralLlm) for llm in llms["forecasters"])
            assert all(llm.litellm_kwargs["api_key"] == "dummy-grant" for llm in llms["forecasters"])
            assert llms["forecasters"][1].model == "openrouter/anthropic/claude-opus-5.5"
            assert llms["stacker"].model == "openrouter/anthropic/claude-opus-5.5"
            assert kwargs["folder_to_save_reports_to"] == str(tmp_path / "reports" / "mock-run" / "forecasts")
            research = ResearchOrchestrator(default_llm=llms["researcher"], summarizer_llm=llms["summarizer"])
            assert [name for _, name in research._select_research_providers()] == ["native_search_grant"]
            called.append("constructed")

        async def forecast_on_tournament(self, tournament_id, *, return_exceptions):
            from forecasting_tools import BinaryQuestion

            assert return_exceptions is True
            called.append("mocked_forecast")
            question = BinaryQuestion(
                question_text="Will the synthetic event occur?",
                id_of_question=101,
                id_of_post=201,
                page_url="https://www.metaculus.com/questions/201/",
                background_info="",
                resolution_criteria="Resolves yes if it occurs.",
                fine_print="",
                open_time=datetime(2026, 10, 5),
                scheduled_resolution_time=datetime(2026, 10, 19),
            )
            second = question.model_copy(update={"id_of_question": 102, "id_of_post": 202})
            return await self.forecast_questions([question, second], return_exceptions=return_exceptions)

        async def forecast_questions(self, questions, return_exceptions=False):
            assert len(questions) == 2
            called.append("scheduled")
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
    await _run_minibench(
        argparse.Namespace(max_questions=1, max_cost_usd="5", reports_dir=tmp_path / "reports"), session
    )
    assert called == ["constructed", "mocked_forecast", "scheduled"]
    scheduled = json.loads((tmp_path / "reports" / "mock-run" / "scheduled_questions.json").read_text())
    assert scheduled["question_ids"] == [101]
    assert scheduled["questions"][0]["question_type"] == "binary"
    assert ledger.summary("mock-run")["attempts"] == 0
    manifest_path = tmp_path / "reports" / "mock-run" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["run_id"] == "mock-run"
    assert manifest["code"]["git_sha"]
    assert manifest["configuration"]["forecasters"][1] == "openrouter/anthropic/claude-opus-5.5"
    assert "dummy-grant" not in manifest_path.read_text()
    with pytest.raises(ValueError, match="already has a forecast manifest"):
        await _run_minibench(
            argparse.Namespace(max_questions=1, max_cost_usd="5", reports_dir=tmp_path / "reports"), session
        )
    assert called == ["constructed", "mocked_forecast", "scheduled"]


@pytest.mark.asyncio
async def test_estimated_charge_still_publishes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Withholding a paid-for forecast recovers nothing; the reservation stays held."""
    from sibyl.forecasting import bot as bot_module
    from sibyl.platform import metaculus_preflight

    ledger = SpendLedger(tmp_path / "spend.sqlite3")
    ledger.start_run("mock-run", Decimal("5"))
    session = GrantSession("mock-run", ledger, Decimal("1"), "dummy-grant")
    published: list[int] = []

    class Base:
        alertable_count = 0

        def __init__(self, **kwargs):
            assert kwargs["publish_reports_to_metaculus"] is True

        async def forecast_on_tournament(self, tournament_id, *, return_exceptions):
            ledger.reserve("a", "mock-run", "openrouter/test", "forecast", Decimal("1"), question_id="101")
            ledger.settle("a", Decimal("0.2"), 10, 5, certainty="estimated", evidence_source="framework_estimate")
            self._create_unified_explanation(SimpleNamespace(id_of_question=101))
            return []

        def _create_unified_explanation(self, question, *args, **kwargs):
            published.append(question.id_of_question)
            return "explanation"

        @classmethod
        def log_report_summary(cls, *args, **kwargs):
            pass

    monkeypatch.setattr(bot_module, "SibylBot", Base)
    monkeypatch.setattr(metaculus_preflight, "verify_metaculus_api_identity", lambda: None)
    await _run_minibench(
        argparse.Namespace(max_questions=1, max_cost_usd="5", reports_dir=tmp_path / "reports"), session
    )
    assert published == [101]
    assert ledger.unreconciled("mock-run") == [("a", None)], "still owed a reconciliation"


@pytest.mark.asyncio
async def test_a_halt_on_the_last_call_still_fails_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from sibyl.forecasting import bot as bot_module
    from sibyl.platform import metaculus_preflight

    ledger = SpendLedger(tmp_path / "spend.sqlite3")
    ledger.start_run("mock-run", Decimal("5"))
    session = GrantSession("mock-run", ledger, Decimal("1"), "dummy-grant")

    class Base:
        alertable_count = 0

        def __init__(self, **kwargs):
            pass

        async def forecast_on_tournament(self, tournament_id, *, return_exceptions):
            session.halt.stop("a charge exceeded the per-call reservation")
            return []

        @classmethod
        def log_report_summary(cls, *args, **kwargs):
            pass

    monkeypatch.setattr(bot_module, "SibylBot", Base)
    monkeypatch.setattr(metaculus_preflight, "verify_metaculus_api_identity", lambda: None)
    with pytest.raises(RuntimeError, match="halted: a charge exceeded"):
        await _run_minibench(
            argparse.Namespace(max_questions=1, max_cost_usd="5", reports_dir=tmp_path / "reports"), session
        )


@pytest.mark.asyncio
async def test_a_failed_forecaster_call_still_publishes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A dropped forecaster publishes and alerts, as in prod; the run then exits non-zero."""
    from sibyl.forecasting import bot as bot_module
    from sibyl.platform import metaculus_preflight

    ledger = SpendLedger(tmp_path / "spend.sqlite3")
    ledger.start_run("mock-run", Decimal("5"))
    session = GrantSession("mock-run", ledger, Decimal("1"), "dummy-grant")
    published: list[int] = []

    class Base:
        alertable_count = 1

        def __init__(self, **kwargs):
            pass

        async def forecast_on_tournament(self, tournament_id, *, return_exceptions):
            ledger.reserve("a", "mock-run", "openrouter/test", "forecaster", Decimal("1"), question_id="101")
            ledger.mark_uncertain("a")
            self._create_unified_explanation(SimpleNamespace(id_of_question=101))
            return []

        def _create_unified_explanation(self, question, *args, **kwargs):
            published.append(question.id_of_question)
            return "explanation"

        @classmethod
        def log_report_summary(cls, *args, **kwargs):
            pass

    monkeypatch.setattr(bot_module, "SibylBot", Base)
    monkeypatch.setattr(metaculus_preflight, "verify_metaculus_api_identity", lambda: None)
    with pytest.raises(RuntimeError, match="finished with failed forecasts"):
        await _run_minibench(
            argparse.Namespace(max_questions=1, max_cost_usd="5", reports_dir=tmp_path / "reports"), session
        )
    assert published == [101]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["uncertain", "rejected"])
async def test_failed_research_stops_before_forecasting(failure, tmp_path, monkeypatch):
    from sibyl.forecasting import bot as bot_module
    from sibyl.platform import metaculus_preflight
    from sibyl.platform.experiment_funding import GrantExecutionError

    ledger = SpendLedger(tmp_path / "spend.sqlite3")
    ledger.start_run("research-stop", Decimal("5"))
    session = GrantSession("research-stop", ledger, Decimal("1"), "dummy-grant")

    class FakeBot:
        def __init__(self, **kwargs):
            pass

        async def forecast_on_tournament(self, *args, **kwargs):
            await self.run_research(SimpleNamespace(id_of_question=101))
            pytest.fail("Failed research reached the forecasters")

        async def run_research(self, question):
            ledger.reserve("research", session.run_id, "openrouter/test", "research", Decimal("1"), question_id="101")
            if failure == "uncertain":
                ledger.mark_uncertain("research")
            else:
                ledger.mark_rejected("research", evidence_source="provider_rejected status=429")
            return "Research unavailable"

    monkeypatch.setattr(bot_module, "SibylBot", FakeBot)
    monkeypatch.setattr(metaculus_preflight, "verify_metaculus_api_identity", lambda: None)
    with pytest.raises(GrantExecutionError, match="research call failed"):
        await _run_minibench(
            argparse.Namespace(max_questions=1, max_cost_usd="5", reports_dir=tmp_path / "reports"), session
        )


class TestReconcileSubcommand:
    """`reconcile` is read-only: it needs the grant key but spends nothing."""

    def test_it_refuses_without_the_grant_key(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from sibyl.config import env as env_module
        from sibyl.ops import experiment_grant

        # main() imports load_environment from sibyl.config.env inside the function and
        # would put a real key back from .env, so patch it at the source. main() also
        # converts errors to an exit code and prints them, which is the right CLI contract.
        monkeypatch.setattr(env_module, "load_environment", lambda: None)
        monkeypatch.setenv("SIBYL_EXPERIMENT_OPENROUTER_API_KEY", "")
        rc = experiment_grant.main(["reconcile", "--ledger", str(tmp_path / "l.sqlite3"), "--run-id", "r"])
        assert rc != 0
        assert "required to read the grant" in capsys.readouterr().out

    def test_it_reports_outstanding_attempts_and_exits_non_zero(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An unreconciled attempt must not look like success.

        A $0 OpenRouter charge whose upstream cost is still unknown is the shape that
        previously read as a free run; the exit status now says otherwise.
        """
        import json

        from sibyl.ops.experiment_grant import main
        from sibyl.platform import experiment_usage
        from sibyl.platform.experiment_funding import SpendLedger

        ledger_path = tmp_path / "l.sqlite3"
        ledger = SpendLedger(ledger_path)
        ledger.start_run("r", Decimal("50"))
        ledger.reserve("a1", "r", "openrouter/openai/t", "forecaster", Decimal("1"))
        ledger.settle(
            "a1",
            Decimal("0"),
            1,
            1,
            None,
            certainty="estimated",
            evidence_source="provider_response_partial",
            openrouter_charge_usd=Decimal("0"),
        )
        monkeypatch.setenv("SIBYL_EXPERIMENT_OPENROUTER_API_KEY", "grant-key")

        async def never_found(generation_id: str, api_key: str):
            raise AssertionError("no generation id, so no lookup should happen")

        monkeypatch.setattr(experiment_usage, "generation_record", never_found)
        rc = main(["reconcile", "--ledger", str(ledger_path), "--run-id", "r"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["no_generation_id"] == 1
        assert rc == 1, "outstanding spend must not exit zero"

    def test_a_missing_ledger_is_refused_not_created(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from sibyl.config import env as env_module
        from sibyl.ops import experiment_grant

        monkeypatch.setattr(env_module, "load_environment", lambda: None)
        monkeypatch.setenv("SIBYL_EXPERIMENT_OPENROUTER_API_KEY", "grant-key")
        typo = tmp_path / "typo.sqlite3"
        rc = experiment_grant.main(["reconcile", "--ledger", str(typo), "--run-id", "r"])
        assert rc != 0, "a wrong path must not read as everything accounted"
        assert "No grant ledger" in capsys.readouterr().out
        assert not typo.exists()

    def test_an_unknown_run_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from sibyl.config import env as env_module
        from sibyl.ops import experiment_grant

        monkeypatch.setattr(env_module, "load_environment", lambda: None)
        monkeypatch.setenv("SIBYL_EXPERIMENT_OPENROUTER_API_KEY", "grant-key")
        ledger_path = tmp_path / "l.sqlite3"
        SpendLedger(ledger_path).start_run("real-run", Decimal("5"))
        rc = experiment_grant.main(["reconcile", "--ledger", str(ledger_path), "--run-id", "real-rnu"])
        assert rc != 0
        assert "not in the ledger" in capsys.readouterr().out


def test_write_off_command_trues_up_the_ledger(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from sibyl.ops.experiment_grant import main

    ledger_path = tmp_path / "l.sqlite3"
    ledger = SpendLedger(ledger_path)
    ledger.start_run("old", Decimal("30"))
    for index in range(8):
        ledger.reserve(f"x{index}", "old", "openrouter/test", "forecaster", Decimal("3"))
        ledger.mark_uncertain(f"x{index}")
    argv = ["write-off", "--ledger", str(ledger_path), "--provider-total-usd", "0.796028"]
    argv += ["--expect-count", "8", "--note", "OpenRouter byok_usage 2026-10-05", "--id", "trueup-2026-10-05"]
    assert main(argv) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["written_off"] == 8
    assert payload["totals"]["remaining_local_usd"] == "94.203972"
