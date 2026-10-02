from decimal import Decimal
from pathlib import Path

import pytest
from forecasting_tools import GeneralLlm
from forecasting_tools.ai_models.ai_utils.response_types import TextTokenCostResponse

from sibyl.ops.experiment_grant import _funded_roster
from sibyl.platform.experiment_funding import GrantSession, SpendLedger, grant_context
from sibyl.platform.experiment_llm import FundedGeneralLlm, GrantCostUnknown, build_grant_llm
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
    slots = [*roster.forecasters, roster.summarizer, roster.parser, roster.researcher,
             roster.stacker, roster.stacker_fallback, roster.analyzer]
    assert all(isinstance(llm, FundedGeneralLlm) for llm in slots)
    assert all(llm.litellm_kwargs["api_key"] == "dummy-grant-key" for llm in slots)
    assert roster.forecasters[1].model == "openrouter/anthropic/claude-opus-5"
    assert roster.stacker.model == "openrouter/anthropic/claude-opus-4.8"


@pytest.mark.asyncio
async def test_each_mocked_model_attempt_settles_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = _session(tmp_path)

    async def _fake_call(self: GeneralLlm, prompt: object) -> TextTokenCostResponse:
        return TextTokenCostResponse(
            data="answer", prompt_tokens_used=10, completion_tokens_used=5,
            total_tokens_used=15, model=self.model, cost=0.2,
        )

    monkeypatch.setattr(GeneralLlm, "_mockable_direct_call_to_model", _fake_call)
    with grant_context(session):
        llm = build_grant_llm("openrouter/openai/test", allowed_tries=1)
    response = await llm._mockable_direct_call_to_model("prompt")
    assert response.data == "answer"
    assert session.ledger.summary(session.run_id)["charged_usd"] == "0.2"
    assert session.ledger.summary(session.run_id)["estimated_usd"] == "0.2"
    assert session.ledger.summary(session.run_id)["confirmed_usd"] == "0"


@pytest.mark.asyncio
async def test_missing_cost_retains_full_liability(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = _session(tmp_path)

    async def _fake_call(self: GeneralLlm, prompt: object) -> TextTokenCostResponse:
        return TextTokenCostResponse(
            data="answer", prompt_tokens_used=10, completion_tokens_used=5,
            total_tokens_used=15, model=self.model, cost=0,
        )

    monkeypatch.setattr(GeneralLlm, "_mockable_direct_call_to_model", _fake_call)
    with grant_context(session):
        llm = build_grant_llm("openrouter/openai/test", allowed_tries=1)
    with pytest.raises(GrantCostUnknown):
        await llm._mockable_direct_call_to_model("prompt")
    assert session.ledger.summary(session.run_id)["uncertain_usd"] == "1"
