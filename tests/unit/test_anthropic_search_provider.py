"""Claude web search on OpenRouter: routing, the grounded floor, and the Sources list."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from sibyl.research import anthropic_search as search


class _FakeLlm:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    async def invoke(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


@pytest.fixture
def built(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Capture what the provider asks OpenRouter for, without building a real client."""
    calls: list[dict[str, Any]] = []

    def fake_build(**kwargs: Any) -> MagicMock:
        calls.append(kwargs)
        return MagicMock()

    monkeypatch.setattr(search, "build_llm_with_openrouter_fallback", fake_build)
    monkeypatch.setattr(search, "label_stage", lambda llm, stage: llm)
    return calls


def test_routes_claude_through_openrouter_native_web_search(built: list[dict[str, Any]]) -> None:
    search.build_search_llm("claude-sonnet-5.5")

    kwargs = built[0]
    assert kwargs["model"] == "openrouter/anthropic/claude-sonnet-5.5"
    assert kwargs["plugins"] == [{"id": "web", "engine": "native", "max_results": search.ANTHROPIC_SEARCH_MAX_USES}]
    assert kwargs["reasoning"] == {"effort": search.ANTHROPIC_SEARCH_EFFORT}


def test_default_model_is_on_the_grant_and_a_vendor_prefix_is_accepted() -> None:
    assert search.ANTHROPIC_SEARCH_DEFAULT_MODEL == "claude-sonnet-5.5"
    assert search._resolve_model("anthropic/claude-opus-5.5") == "claude-opus-5.5"
    assert search._resolve_model(None) == "claude-sonnet-5.5"


@pytest.mark.asyncio
async def test_cited_reply_gets_a_numbered_sources_list(monkeypatch: pytest.MonkeyPatch) -> None:
    reply = (
        "Inflation fell to 2.4% ([BLS](https://www.bls.gov/cpi/)). Markets expect a cut "
        "([Reuters](https://www.reuters.com/markets/x)); see also [BLS](https://www.bls.gov/cpi/)."
    )
    monkeypatch.setattr(search, "build_search_llm", lambda model: _FakeLlm(reply))

    out = await search.invoke_anthropic_grounded("prompt", qid=7)

    assert out.startswith(reply)
    assert out.endswith("### Sources\n[1] bls.gov\n[2] reuters.com")


@pytest.mark.asyncio
async def test_uncited_reply_is_suppressed_and_recorded(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(search, "build_search_llm", lambda model: _FakeLlm("A confident table with no sources."))
    recorded = MagicMock()
    monkeypatch.setattr(search, "record_provider_detail", recorded)

    with caplog.at_level(logging.WARNING, logger=search.__name__):
        out = await search.invoke_anthropic_grounded("prompt", qid=7)

    assert out == ""
    assert "ANTHROPIC_UNGROUNDED_SUPPRESSED: question=7 model=claude-sonnet-5.5" in caplog.text
    recorded.assert_called_once_with(7, "anthropic_search", {"sources": {"grounding": "error(ungrounded_suppressed)"}})


@pytest.mark.asyncio
async def test_empty_reply_returns_empty_without_recording_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(search, "build_search_llm", lambda model: _FakeLlm("   "))
    recorded = MagicMock()
    monkeypatch.setattr(search, "record_provider_detail", recorded)

    assert await search.invoke_anthropic_grounded("prompt", qid=7) == ""
    recorded.assert_not_called()


@pytest.mark.asyncio
async def test_provider_asks_for_markdown_citations_and_threads_qid(monkeypatch: pytest.MonkeyPatch) -> None:
    invoke = AsyncMock(return_value="ok")
    monkeypatch.setattr(search, "invoke_anthropic_grounded", invoke)
    question = MagicMock(question_text="Will X happen?", id_of_question=42)

    result = await search.anthropic_search_provider("claude-opus-5.5")(question)

    assert result == "ok"
    prompt = invoke.await_args.args[0] if invoke.await_args else ""
    assert "[source name](url)" in prompt
    assert invoke.await_args is not None and invoke.await_args.kwargs == {"model_slug": "claude-opus-5.5", "qid": 42}


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.reuters.com/a/b", "reuters.com"),
        ("https://bls.gov", "bls.gov"),
        ("not a url", ""),
    ],
)
def test_domain_extraction(url: str, expected: str) -> None:
    assert search._domain(url) == expected


def test_urls_are_collected_once_in_first_seen_order() -> None:
    text = "[a](https://a.com/1) then https://b.org/x and [a again](https://a.com/1)."
    assert search._collect_urls(text) == ["https://a.com/1", "https://b.org/x"]
