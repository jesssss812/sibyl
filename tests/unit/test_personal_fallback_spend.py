"""Existing-key accounting is opt-in and cannot alter fallback routing."""

import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from sibyl.platform import key_fallback, personal_fallback_spend
from sibyl.platform.experiment_usage import _GrantUsageObserver
from sibyl.platform.key_fallback import FallbackOpenRouterLlm
from sibyl.platform.personal_fallback_spend import LEDGER_ENV, PersonalFallbackLedger


def _fallback_llm(monkeypatch: pytest.MonkeyPatch) -> FallbackOpenRouterLlm:
    llm = FallbackOpenRouterLlm(
        model="openrouter/openai/test", primary_api_key="dummy-primary", secondary_api_key="dummy-personal"
    )
    monkeypatch.setattr(llm, "_invoke_once_using_primary", AsyncMock(side_effect=Exception("HTTP 402")))
    monkeypatch.setattr(key_fallback, "record_primary_key_fallback", AsyncMock())
    return llm


@pytest.mark.asyncio
async def test_actual_personal_fallback_is_estimated_separately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "spend.sqlite3"
    monkeypatch.setenv(LEDGER_ENV, str(path))
    llm = _fallback_llm(monkeypatch)

    async def secondary(prompt: str, system_prompt: str | None = None) -> str:
        response = SimpleNamespace(id="gen-dummy", usage=SimpleNamespace(cost=0.12))
        await _GrantUsageObserver().async_log_success_event({}, response, None, None)
        return "personal result"

    monkeypatch.setattr(llm, "_invoke_once_using_secondary", secondary)
    assert await llm.invoke("dummy prompt") == "personal result"
    assert PersonalFallbackLedger(path).summary() == {
        "source": "personal",
        "attempts": 1,
        "estimated_usd": "0.12",
        "unknown_attempts": 0,
    }
    raw = path.read_bytes()
    assert b"dummy-primary" not in raw and b"dummy-personal" not in raw and b"dummy prompt" not in raw


@pytest.mark.asyncio
async def test_failed_secondary_call_keeps_unknown_personal_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "spend.sqlite3"
    monkeypatch.setenv(LEDGER_ENV, str(path))
    llm = _fallback_llm(monkeypatch)
    monkeypatch.setattr(llm, "_invoke_once_using_secondary", AsyncMock(side_effect=RuntimeError("secondary failed")))
    with pytest.raises(RuntimeError, match="secondary failed"):
        await llm.invoke("dummy prompt")
    assert PersonalFallbackLedger(path).summary()["unknown_attempts"] == 1


@pytest.mark.asyncio
async def test_primary_success_never_creates_personal_event(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "spend.sqlite3"
    monkeypatch.setenv(LEDGER_ENV, str(path))
    llm = _fallback_llm(monkeypatch)
    monkeypatch.setattr(llm, "_invoke_once_using_primary", AsyncMock(return_value="primary result"))
    assert await llm.invoke("dummy prompt") == "primary result"
    assert not path.exists()


@pytest.mark.asyncio
async def test_ledger_failure_does_not_block_secondary_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LEDGER_ENV, str(tmp_path / "unusable.sqlite3"))
    llm = _fallback_llm(monkeypatch)
    secondary = AsyncMock(return_value="personal result")
    monkeypatch.setattr(llm, "_invoke_once_using_secondary", secondary)

    def broken_ledger(path: Path) -> PersonalFallbackLedger:
        raise OSError("disk unavailable")

    monkeypatch.setattr(personal_fallback_spend, "PersonalFallbackLedger", broken_ledger)
    assert await llm.invoke("dummy prompt") == "personal result"
    secondary.assert_awaited_once()


def test_summary_filters_personal_fallbacks_by_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from sibyl.ops.experiment_grant import main

    ledger = PersonalFallbackLedger(tmp_path / "spend.sqlite3")
    ledger.start("a", "run-a", "openrouter/openai/test")
    ledger.finish("a", status="succeeded", estimated_usd=Decimal("0.12"), provider_request_id=None)
    ledger.start("b", "run-b", "openrouter/openai/test")
    ledger.finish("b", status="failed", estimated_usd=None, provider_request_id=None)

    assert ledger.summary("run-a") == {
        "source": "personal",
        "attempts": 1,
        "estimated_usd": "0.12",
        "unknown_attempts": 0,
    }
    assert ledger.summary("run-b") == {
        "source": "personal",
        "attempts": 1,
        "estimated_usd": "0",
        "unknown_attempts": 1,
    }
    assert ledger.summary()["attempts"] == 2
    assert main(["report", "--ledger", str(ledger.path), "--run-id", "run-a"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["personal_fallback"]["estimated_usd"] == "0.12"
    assert report["personal_fallback"]["unknown_attempts"] == 0
