"""Exercise the funding seam over real LiteLLM callbacks, with fake HTTP and no network."""

import logging
from decimal import Decimal

import httpx
import pytest
from forecasting_tools import GeneralLlm

from sibyl.platform import funding
from sibyl.platform.experiment_funding import SpendLedger


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["openai/gpt-6-sol", "anthropic/claude-opus-5.5", "google/gemini-3.1-pro-preview"])
async def test_a_grant_call_settles_from_the_real_response(tmp_path, monkeypatch, caplog, model):
    calls = []
    caplog.set_level(logging.INFO, logger="sibyl.platform.funding")

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
    monkeypatch.setattr(GeneralLlm, "_mockable_direct_call_to_model", funding._funded_direct_call)
    ledger = SpendLedger(tmp_path / "spend.sqlite3")
    ledger.start_run("offline", Decimal("5"))
    funding.configure_funding(funding.Funding("offline", ledger, "dummy-grant", Decimal("1")))
    llm = GeneralLlm(model="openrouter/" + model, api_key="dummy-own", stream=False)
    assert await llm.invoke("offline prompt") == "offline answer"
    assert ledger.summary("offline")["confirmed_usd"] == "0.12"
    assert len(calls) == 1, "the billing record is read by reconciliation, not during the call"
    assert "SPEND_ATTEMPT" in caplog.text
    assert "dummy-grant" not in caplog.text
