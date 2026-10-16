import sqlite3
from decimal import Decimal
from pathlib import Path

import pytest
from forecasting_tools import GeneralLlm
from forecasting_tools.ai_models.ai_utils.response_types import TextTokenCostResponse

from sibyl.ops.experiment_grant import _funded_roster
from sibyl.platform.experiment_funding import GrantSession, SpendLedger, grant_context
from sibyl.platform.experiment_llm import FundedGeneralLlm, build_grant_llm
from sibyl.platform.key_fallback import build_llm_with_openrouter_fallback


def _session(tmp_path: Path) -> GrantSession:
    ledger = SpendLedger(tmp_path / "spend.sqlite3")
    ledger.start_run("test-run", Decimal("5"))
    return GrantSession("test-run", ledger, Decimal("1"), "dummy-grant-key")


def test_factory_uses_only_explicit_grant_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-personal-key")
    monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "dummy-primary-key")
    with grant_context(_session(tmp_path)):
        llm = build_llm_with_openrouter_fallback("openrouter/openai/gpt-test", allowed_tries=1)
    assert isinstance(llm, FundedGeneralLlm)
    assert llm.litellm_kwargs["api_key"] == "dummy-grant-key"
    assert "dummy-grant-key" not in repr(llm)


def test_grant_rejects_direct_provider(tmp_path: Path) -> None:
    with grant_context(_session(tmp_path)), pytest.raises(ValueError, match="OpenRouter"):
        build_llm_with_openrouter_fallback("anthropic/claude-opus-5")


def test_explicit_anthropic_route_and_effort(tmp_path: Path) -> None:
    with grant_context(_session(tmp_path)):
        llm = build_grant_llm("openrouter/anthropic/claude-opus-5", reasoning={"effort": "xhigh"})
    assert llm.model == "openrouter/anthropic/claude-opus-5"
    assert llm.litellm_kwargs["reasoning"] == {"effort": "xhigh"}


def test_every_roster_slot_uses_grant_key(tmp_path: Path) -> None:
    from sibyl.config.models import ModelTier, get_roster

    source = get_roster(ModelTier.PROD)
    with grant_context(_session(tmp_path)):
        roster = _funded_roster(source)
    slots = [
        *roster.forecasters,
        roster.summarizer,
        roster.parser,
        roster.researcher,
        roster.stacker,
        roster.stacker_fallback,
        roster.analyzer,
    ]
    assert all(isinstance(llm, FundedGeneralLlm) for llm in slots)
    assert all(llm.litellm_kwargs["api_key"] == "dummy-grant-key" for llm in slots)
    assert roster.forecasters[1].model == "openrouter/anthropic/claude-opus-5"
    assert roster.stacker.model == "openrouter/anthropic/claude-opus-4.8"


@pytest.mark.asyncio
async def test_each_mocked_model_attempt_settles_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = _session(tmp_path)

    async def _fake_call(self: GeneralLlm, prompt: object) -> TextTokenCostResponse:
        return TextTokenCostResponse(
            data="answer",
            prompt_tokens_used=10,
            completion_tokens_used=5,
            total_tokens_used=15,
            model=self.model,
            cost=0.2,
        )

    monkeypatch.setattr(GeneralLlm, "_mockable_direct_call_to_model", _fake_call)
    with grant_context(session):
        llm = build_grant_llm("openrouter/openai/test", allowed_tries=1)
    # A framework-only estimate no longer aborts the call: the money is already spent, so
    # discarding the result buys nothing. It settles ONCE, as an estimate, and the
    # reservation stays held until the provider record reconciles it.
    await llm._mockable_direct_call_to_model("prompt")
    summary = session.ledger.summary(session.run_id)
    assert summary["charged_usd"] == "0.2"
    assert summary["estimated_usd"] == "0.2"
    assert summary["confirmed_usd"] == "0"
    assert session.ledger.unreconciled(session.run_id), "an estimate must stay reconcilable"
    with sqlite3.connect(session.ledger.path) as db:
        # The row names the evidence it rests on, not a confirming source it lacks.
        assert db.execute("SELECT evidence_source FROM attempts").fetchone() == ("framework_estimate",)


@pytest.mark.asyncio
async def test_missing_cost_retains_full_liability(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = _session(tmp_path)

    async def _fake_call(self: GeneralLlm, prompt: object) -> TextTokenCostResponse:
        return TextTokenCostResponse(
            data="answer",
            prompt_tokens_used=10,
            completion_tokens_used=5,
            total_tokens_used=15,
            model=self.model,
            cost=0,
        )

    monkeypatch.setattr(GeneralLlm, "_mockable_direct_call_to_model", _fake_call)
    with grant_context(session):
        llm = build_grant_llm("openrouter/openai/test", allowed_tries=1)
    # The answer is paid for, so it is kept; the cost is held at the full reservation.
    response = await llm._mockable_direct_call_to_model("prompt")
    assert response.data == "answer"
    assert session.ledger.summary(session.run_id)["uncertain_usd"] == "1"


@pytest.mark.asyncio
async def test_unknown_cost_keeps_generation_id_for_reconciliation(tmp_path, monkeypatch):
    from sibyl.platform import experiment_llm, experiment_usage

    session = _session(tmp_path)

    async def fake_call(self, prompt):
        return TextTokenCostResponse(
            data="answer",
            prompt_tokens_used=10,
            completion_tokens_used=5,
            total_tokens_used=15,
            model=self.model,
            cost=0,
        )

    async def unknown_cost(*args):
        return experiment_usage.CostResolution(None, None, None, None, "gen-reconcile-me", "unknown")

    monkeypatch.setattr(GeneralLlm, "_mockable_direct_call_to_model", fake_call)
    monkeypatch.setattr(experiment_llm, "resolve_grant_cost", unknown_cost)
    llm = FundedGeneralLlm("openrouter/openai/test", session=session)
    assert await llm.invoke("prompt") == "answer"
    with sqlite3.connect(session.ledger.path) as db:
        assert db.execute("SELECT provider_request_id FROM attempts").fetchone() == ("gen-reconcile-me",)
    assert session.ledger.unreconciled(session.run_id) == [(mock_attempt_id(session), "gen-reconcile-me")]


@pytest.mark.asyncio
async def test_uncertain_grant_failure_is_not_retried(tmp_path, monkeypatch):
    from litellm.exceptions import RateLimitError

    from sibyl.platform.experiment_llm import GrantCallError
    from sibyl.platform.llm_retry import invoke_with_broad_retry

    session = _session(tmp_path)
    calls = 0

    async def rate_limited(self, prompt):
        nonlocal calls
        calls += 1
        raise RateLimitError("sensitive upstream message", llm_provider="openrouter", model=self.model)

    monkeypatch.setattr(GeneralLlm, "_mockable_direct_call_to_model", rate_limited)
    llm = FundedGeneralLlm("openrouter/google/test", session=session)
    with pytest.raises(GrantCallError) as error:
        await invoke_with_broad_retry(
            lambda: llm.invoke("prompt"), wall_timeout=5, label="grant-test", backoffs=(0, 0, 0)
        )
    assert calls == 1
    assert "status=429" in str(error.value)
    assert "sensitive" not in str(error.value)


def mock_attempt_id(session: GrantSession) -> str:
    with sqlite3.connect(session.ledger.path) as db:
        return db.execute("SELECT attempt_id FROM attempts").fetchone()[0]


def _raising(exc: BaseException):
    async def call(self, prompt):
        raise exc

    return call


def _answer(cost: float = 0.2):
    async def call(self, prompt):
        return TextTokenCostResponse(
            data="answer",
            prompt_tokens_used=10,
            completion_tokens_used=5,
            total_tokens_used=15,
            model=self.model,
            cost=cost,
        )

    return call


def _status_error(status: int) -> Exception:
    error = RuntimeError("sensitive upstream message")
    error.status_code = status  # type: ignore[attr-defined]
    return error


def _row(session: GrantSession) -> tuple:
    with sqlite3.connect(session.ledger.path) as db:
        return db.execute("SELECT status, charged_usd, certainty FROM attempts").fetchone()


class TestFailureKinds:
    """Each kind of failed call gets its own consequence; see docs/runbooks/experiment-funding.md."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [401, 402])
    async def test_a_rejected_key_halts_the_run(self, status: int, tmp_path, monkeypatch) -> None:
        from sibyl.platform.experiment_llm import GrantCallError, GrantRunHalted

        session = _session(tmp_path)
        monkeypatch.setattr(GeneralLlm, "_mockable_direct_call_to_model", _raising(_status_error(status)))
        llm = FundedGeneralLlm("openrouter/openai/test", session=session)
        with pytest.raises(GrantCallError):
            await llm.invoke("prompt")
        assert _row(session) == ("rejected", "0", "confirmed"), "a rejected request is not billed"
        monkeypatch.setattr(GeneralLlm, "_mockable_direct_call_to_model", _answer())
        with pytest.raises(GrantRunHalted):
            await llm.invoke("prompt")
        assert session.ledger.summary(session.run_id)["attempts"] == 1, "no further reservation"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [400, 403, 404, 429])
    async def test_a_rejected_request_releases_its_reservation(self, status: int, tmp_path, monkeypatch) -> None:
        from sibyl.platform.experiment_llm import GrantCallError

        session = _session(tmp_path)
        monkeypatch.setattr(GeneralLlm, "_mockable_direct_call_to_model", _raising(_status_error(status)))
        llm = FundedGeneralLlm("openrouter/openai/test", session=session)
        with pytest.raises(GrantCallError):
            await llm.invoke("prompt")
        assert _row(session) == ("rejected", "0", "confirmed")
        assert session.ledger.summary(session.run_id)["remaining_local_usd"] == "95"
        monkeypatch.setattr(GeneralLlm, "_mockable_direct_call_to_model", _answer())
        assert await llm.invoke("prompt") == "answer", "the run carries on"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("error", [_status_error(502), _status_error(408), TimeoutError(), ConnectionError()])
    async def test_a_call_lost_after_sending_holds_its_reservation(self, error, tmp_path, monkeypatch) -> None:
        from sibyl.platform.experiment_llm import GrantCallError

        session = _session(tmp_path)
        monkeypatch.setattr(GeneralLlm, "_mockable_direct_call_to_model", _raising(error))
        llm = FundedGeneralLlm("openrouter/openai/test", session=session)
        with pytest.raises(GrantCallError):
            await llm.invoke("prompt")
        assert _row(session) == ("uncertain", None, "unknown")
        assert session.ledger.summary(session.run_id)["uncertain_usd"] == "1"

    @pytest.mark.asyncio
    async def test_repeated_unknown_costs_halt_the_run(self, tmp_path, monkeypatch) -> None:
        from sibyl.platform.experiment_llm import GrantRunHalted

        session = _session(tmp_path)
        monkeypatch.setattr(GeneralLlm, "_mockable_direct_call_to_model", _answer(cost=0))
        llm = FundedGeneralLlm("openrouter/openai/test", session=session)
        for _ in range(3):
            assert await llm.invoke("prompt") == "answer"
        with pytest.raises(GrantRunHalted, match="cost"):
            await llm.invoke("prompt")

    @pytest.mark.asyncio
    async def test_a_known_cost_resets_the_unknown_streak(self, tmp_path, monkeypatch) -> None:
        session = _session(tmp_path)
        llm = FundedGeneralLlm("openrouter/openai/test", session=session)
        for cost in (0, 0, 0.2, 0, 0):
            monkeypatch.setattr(GeneralLlm, "_mockable_direct_call_to_model", _answer(cost=cost))
            assert await llm.invoke("prompt") == "answer"

    @pytest.mark.asyncio
    async def test_an_overage_keeps_the_answer_and_halts_further_calls(self, tmp_path, monkeypatch) -> None:
        from sibyl.platform import experiment_llm, experiment_usage
        from sibyl.platform.experiment_llm import GrantRunHalted

        session = _session(tmp_path)

        async def over_reserve(*args):
            return experiment_usage.CostResolution(
                Decimal("1.5"), Decimal("1.5"), None, False, "gen-over", "provider_response"
            )

        monkeypatch.setattr(GeneralLlm, "_mockable_direct_call_to_model", _answer())
        monkeypatch.setattr(experiment_llm, "resolve_grant_cost", over_reserve)
        llm = FundedGeneralLlm("openrouter/openai/test", session=session)
        assert await llm.invoke("prompt") == "answer"
        assert _row(session) == ("succeeded", "1.5", "confirmed")
        with pytest.raises(GrantRunHalted, match="reservation"):
            await llm.invoke("prompt")
