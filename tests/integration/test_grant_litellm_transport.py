"""Exercise real LiteLLM callbacks with fake HTTP responses and no network."""

import logging
from decimal import Decimal

import httpx
import pytest

from sibyl.platform.experiment_funding import GrantSession, SpendLedger
from sibyl.platform.experiment_llm import FundedGeneralLlm


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["openai/gpt-5.6-sol", "anthropic/claude-opus-5", "google/gemini-3.1-pro-preview"])
async def test_real_litellm_response_confirms_grant_cost(tmp_path, monkeypatch, caplog, model):
    calls = []
    caplog.set_level(logging.INFO, logger="sibyl.platform.experiment_llm")

    async def send(client, request, **kwargs):
        calls.append(str(request.url))
        assert request.url.host == "openrouter.ai"
        assert request.headers["authorization"] == "Bearer dummy-grant"
        if request.url.path == "/api/v1/chat/completions":
            return httpx.Response(
                200,
                request=request,
                json={
                    "id": "gen-offline",
                    "object": "chat.completion",
                    "created": 1,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "offline answer"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                        "cost": 0.12,
                        "is_byok": False,
                    },
                },
            )
        raise AssertionError(f"unexpected request to {request.url.path}")

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    ledger = SpendLedger(tmp_path / "spend.sqlite3")
    ledger.start_run("offline", Decimal("5"))
    llm = FundedGeneralLlm(
        "openrouter/" + model, session=GrantSession("offline", ledger, Decimal("1"), "dummy-grant"), stream=False
    )
    assert await llm.invoke("offline prompt") == "offline answer"
    assert ledger.summary("offline")["confirmed_usd"] == "0.12"
    assert len(calls) == 1, "the billing record is read by reconciliation, not during the call"
    assert "GRANT_CALL_START" in caplog.text
    assert "GRANT_CALL_SETTLED" in caplog.text
    assert "dummy-grant" not in caplog.text
