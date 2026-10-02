"""Generation cost reconciliation is mocked; no credential is sent to OpenRouter."""

from decimal import Decimal
from types import SimpleNamespace

import pytest

from sibyl.platform import experiment_usage


@pytest.mark.asyncio
async def test_generation_total_cost_overrides_unpriced_framework_call(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_generation_cost(generation_id: str, api_key: str) -> Decimal:
        assert generation_id == "gen-dummy"
        assert api_key == "dummy-key"
        return Decimal("0.42")

    monkeypatch.setattr(experiment_usage, "generation_cost", fake_generation_cost)
    capture = experiment_usage.CostCapture(generation_id="gen-dummy")
    cost, request_id = await experiment_usage.resolve_grant_cost(capture, 0, "dummy-key")
    assert cost == Decimal("0.42")
    assert request_id == "gen-dummy"


@pytest.mark.asyncio
async def test_callback_captures_only_cost_and_generation_id() -> None:
    response = SimpleNamespace(id="gen-dummy", usage=SimpleNamespace(cost=0.12), choices=["sensitive-output"])
    with experiment_usage.capture_grant_usage() as capture:
        observer = experiment_usage._GrantUsageObserver()
        await observer.async_log_success_event({}, response, None, None)
    assert capture == experiment_usage.CostCapture("gen-dummy", Decimal("0.12"))
    assert "sensitive-output" not in repr(capture)


@pytest.mark.asyncio
async def test_zero_cost_without_generation_record_remains_uncertain() -> None:
    capture = experiment_usage.CostCapture(reported_cost=Decimal("0"))
    cost, request_id = await experiment_usage.resolve_grant_cost(capture, 0, "dummy-key")
    assert cost is None
    assert request_id is None
