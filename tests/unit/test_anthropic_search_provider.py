"""Tests for the Anthropic grounded search research provider.

These tests fake the `anthropic` SDK at the client seam; no live API calls and no
ANTHROPIC_API_KEY is ever required (every test patches `build_anthropic_client`,
except the one that asserts the missing-key error). Patterns mirror
``tests/test_gemini_search_provider.py``.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from sibyl.research import anthropic_search
from sibyl.research.provider_diagnostics import _is_lost_source, pop_provider_detail


def _make_q(text: str, qid: int | None = 7) -> MagicMock:
    q = MagicMock()
    q.question_text = text
    q.id_of_question = qid
    return q


# ---------------------------------------------------------------------------
# Canned response helpers
# ---------------------------------------------------------------------------


def _text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _thinking_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="thinking", thinking=text, text=None)


def _result(url: str, title: str) -> SimpleNamespace:
    return SimpleNamespace(type="web_search_result", url=url, title=title)


def _search_block(results: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(type="web_search_tool_result", content=results)


def _error_search_block(code: str) -> SimpleNamespace:
    """Server-tool errors arrive as HTTP 200 with content as an OBJECT, not a list."""
    return SimpleNamespace(type="web_search_tool_result", content=SimpleNamespace(error_code=code))


def _message(blocks: list[SimpleNamespace], stop_reason: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(content=blocks, stop_reason=stop_reason, stop_details=None)


class _FakeStream:
    def __init__(self, message: SimpleNamespace) -> None:
        self._message = message

    async def __aenter__(self) -> "_FakeStream":
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def get_final_message(self) -> SimpleNamespace:
        return self._message


def _client_returning(message: SimpleNamespace) -> MagicMock:
    """Fake AsyncAnthropic whose messages.stream(...) yields ``message``."""
    client = MagicMock()
    client.messages.stream = MagicMock(return_value=_FakeStream(message))
    return client


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_formats_text_with_numbered_sources() -> None:
    msg = _message(
        [
            _search_block(
                [_result("https://www.reuters.com/a", "Reuters piece"), _result("https://apnews.com/b", "AP piece")]
            ),
            _text_block("Body of the briefing."),
        ]
    )
    with patch.object(anthropic_search, "build_anthropic_client", return_value=_client_returning(msg)):
        out = await anthropic_search.invoke_anthropic_grounded("prompt", qid=1)

    assert out.startswith("Body of the briefing.")
    assert "### Sources" in out
    # www. is stripped; title and domain are joined.
    assert "[1] Reuters piece — reuters.com" in out
    assert "[2] AP piece — apnews.com" in out


@pytest.mark.asyncio
async def test_thinking_blocks_are_not_forwarded_to_forecasters() -> None:
    """Reasoning is not research — it must never reach the forecaster bundle."""
    msg = _message(
        [
            _thinking_block("secret deliberation"),
            _search_block([_result("https://bbc.co.uk/x", "BBC")]),
            _text_block("Grounded answer."),
        ]
    )
    with patch.object(anthropic_search, "build_anthropic_client", return_value=_client_returning(msg)):
        out = await anthropic_search.invoke_anthropic_grounded("prompt", qid=1)

    assert "secret deliberation" not in out
    assert "Grounded answer." in out


# ---------------------------------------------------------------------------
# The grounded floor (the Q38195 lesson, inherited from the Gemini provider)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ungrounded_response_is_suppressed_and_recorded() -> None:
    """Text with zero search results is a fabrication vector — suppress it."""
    msg = _message([_text_block("Confident but entirely ungrounded prose.")])
    with patch.object(anthropic_search, "build_anthropic_client", return_value=_client_returning(msg)):
        out = await anthropic_search.invoke_anthropic_grounded("prompt", qid=4242)

    assert out == ""
    detail = pop_provider_detail(4242, "anthropic_search")
    assert detail is not None, "suppression must be distinguishable from a healthy empty result"
    assert _is_lost_source(detail["sources"]["grounding"])


@pytest.mark.asyncio
async def test_empty_text_returns_empty_without_recording_loss() -> None:
    msg = _message([_search_block([_result("https://x.com/a", "T")])])
    with patch.object(anthropic_search, "build_anthropic_client", return_value=_client_returning(msg)):
        out = await anthropic_search.invoke_anthropic_grounded("prompt", qid=99)
    assert out == ""


# ---------------------------------------------------------------------------
# Server-tool error shapes (200 OK, error object rather than a list)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_error_block_does_not_crash_and_suppresses_when_alone() -> None:
    msg = _message([_error_search_block("max_uses_exceeded"), _text_block("Some prose.")])
    with patch.object(anthropic_search, "build_anthropic_client", return_value=_client_returning(msg)):
        out = await anthropic_search.invoke_anthropic_grounded("prompt", qid=5)
    # The error block yields no sources, so the grounded floor applies.
    assert out == ""


@pytest.mark.asyncio
async def test_partial_search_error_still_renders_successful_results() -> None:
    msg = _message(
        [
            _error_search_block("unavailable"),
            _search_block([_result("https://ft.com/a", "FT")]),
            _text_block("Partial but grounded."),
        ]
    )
    with patch.object(anthropic_search, "build_anthropic_client", return_value=_client_returning(msg)):
        out = await anthropic_search.invoke_anthropic_grounded("prompt", qid=6)
    assert "Partial but grounded." in out
    assert "[1] FT — ft.com" in out


@pytest.mark.asyncio
async def test_refusal_returns_empty() -> None:
    msg = _message([_text_block("")], stop_reason="refusal")
    msg.stop_details = SimpleNamespace(category="cyber")
    with patch.object(anthropic_search, "build_anthropic_client", return_value=_client_returning(msg)):
        out = await anthropic_search.invoke_anthropic_grounded("prompt", qid=8)
    assert out == ""


# ---------------------------------------------------------------------------
# Wiring / contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_provider_matches_research_callable_contract_and_threads_qid() -> None:
    msg = _message([_search_block([_result("https://npr.org/a", "NPR")]), _text_block("Answer.")])
    client = _client_returning(msg)
    with patch.object(anthropic_search, "build_anthropic_client", return_value=client):
        provider = anthropic_search.anthropic_search_provider()
        out = await provider(_make_q("Will X happen?", qid=555))

    assert "Answer." in out
    kwargs = client.messages.stream.call_args.kwargs
    assert kwargs["model"] == anthropic_search.ANTHROPIC_SEARCH_DEFAULT_MODEL
    tool = kwargs["tools"][0]
    assert tool["type"] == "web_search_20260209", "must use the dynamic-filtering variant"
    assert tool["name"] == "web_search"
    assert tool["max_uses"] == anthropic_search.ANTHROPIC_SEARCH_MAX_USES, "cost bound must be sent"


@pytest.mark.asyncio
async def test_model_slug_override_is_honoured() -> None:
    msg = _message([_search_block([_result("https://x.org/a", "X")]), _text_block("A.")])
    client = _client_returning(msg)
    with patch.object(anthropic_search, "build_anthropic_client", return_value=client):
        provider = anthropic_search.anthropic_search_provider("claude-opus-5")
        await provider(_make_q("Q?"))
    assert client.messages.stream.call_args.kwargs["model"] == "claude-opus-5"


def test_missing_api_key_raises_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        anthropic_search.build_anthropic_client()


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.example.com/path?q=1", "example.com"),
        ("https://sub.example.co.uk/x", "sub.example.co.uk"),
        ("not a url", ""),
        ("", ""),
    ],
)
def test_domain_extraction(url: str, expected: str) -> None:
    assert anthropic_search._domain(url) == expected


# ---------------------------------------------------------------------------
# Regressions from the 2026-08-31 live run (Q43323)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inter_tool_narration_is_stripped_from_the_briefing() -> None:
    """Reproduces the live defect: status mutterings opened the research section.

    The real output began "Let me wait and retry with a single, simple call.I'll
    pause and try again in the next turn to avoid the rate limit.I've hit the search
    rate limit for this turn." before the actual briefing. Only the synthesis after
    the final search may reach forecasters.
    """
    msg = _message(
        [
            _search_block([_result("https://gallup.com/a", "Gallup")]),
            _text_block("Let me wait and retry with a single, simple call."),
            _search_block([_result("https://pew.org/b", "Pew")]),
            _text_block("I've hit the search rate limit for this turn."),
            _error_search_block("max_uses_exceeded"),
            _text_block("# Research Summary\n\nThe actual briefing."),
        ]
    )
    with patch.object(anthropic_search, "build_anthropic_client", return_value=_client_returning(msg)):
        out = await anthropic_search.invoke_anthropic_grounded("prompt", qid=43325)

    assert out.startswith("# Research Summary")
    assert "Let me wait and retry" not in out
    assert "rate limit" not in out
    # Sources gathered across ALL search blocks still survive the text filter.
    assert "Gallup" in out and "Pew" in out


@pytest.mark.asyncio
async def test_text_before_a_trailing_search_is_kept_as_fallback() -> None:
    """A response ending on a tool call must not be silently emptied by the filter."""
    msg = _message(
        [
            _text_block("Everything useful is here."),
            _search_block([_result("https://ap.org/a", "AP")]),
        ]
    )
    with patch.object(anthropic_search, "build_anthropic_client", return_value=_client_returning(msg)):
        out = await anthropic_search.invoke_anthropic_grounded("prompt", qid=2)
    assert "Everything useful is here." in out


def test_sources_are_deduplicated_preserving_first_seen_order() -> None:
    """The live run rendered 76 rows with Gallup/Newsweek/Pew repeated several times."""
    msg = _message(
        [
            _search_block(
                [
                    _result("https://news.gallup.com/x", "Gallup poll"),
                    _result("https://newsweek.com/y", "Newsweek piece"),
                    _result("https://news.gallup.com/x", "Gallup poll"),  # exact repeat
                ]
            ),
            _search_block([_result("https://newsweek.com/y", "Newsweek piece")]),  # repeat across blocks
        ]
    )
    sources, raw_count, error_count = anthropic_search._collect_sources(msg)

    assert [t for t, _ in sources] == ["Gallup poll", "Newsweek piece"], "order must be first-seen"
    assert raw_count == 4, "raw count reports pre-dedup volume"
    assert error_count == 0


def test_dedupe_falls_back_to_title_and_domain_when_url_is_missing() -> None:
    msg = _message(
        [
            _search_block(
                [
                    _result("", "Same Title"),
                    _result("", "Same Title"),
                    _result("", "Other Title"),
                ]
            )
        ]
    )
    sources, _raw, _errors = anthropic_search._collect_sources(msg)
    assert len(sources) == 2


def test_error_blocks_are_counted_not_logged_per_block(caplog: pytest.LogCaptureFixture) -> None:
    """Eight identical warnings buried the signal; one aggregated line replaces them."""
    msg = _message([_error_search_block("max_uses_exceeded") for _ in range(8)])
    with caplog.at_level("WARNING", logger="sibyl.research.anthropic_search"):
        _sources, _raw, error_count = anthropic_search._collect_sources(msg)

    assert error_count == 8
    assert caplog.records == [], "_collect_sources must not log per block; the caller reports once"


@pytest.mark.asyncio
async def test_max_uses_exhaustion_emits_exactly_one_aggregated_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    msg = _message(
        [_error_search_block("max_uses_exceeded") for _ in range(8)]
        + [_search_block([_result("https://x.com/a", "X")]), _text_block("Briefing.")]
    )
    with patch.object(anthropic_search, "build_anthropic_client", return_value=_client_returning(msg)):
        with caplog.at_level("WARNING", logger="sibyl.research.anthropic_search"):
            out = await anthropic_search.invoke_anthropic_grounded("prompt", qid=9)

    assert "Briefing." in out
    warnings = [r for r in caplog.records if "web_search error block" in r.message]
    assert len(warnings) == 1, f"expected one aggregated warning, got {len(warnings)}"
    assert "8 web_search error block(s)" in warnings[0].message


def test_max_uses_is_high_enough_to_not_bind_on_a_real_question() -> None:
    """8 provably throttled real research on 2026-08-31; the cap must stay above it."""
    assert anthropic_search.ANTHROPIC_SEARCH_MAX_USES > 8


@pytest.mark.asyncio
async def test_narration_after_the_final_search_is_also_dropped() -> None:
    """The positional filter alone misses a muttering emitted after the last search.

    Separate text blocks can only arise across a tool call (get_final_message merges
    contiguous streamed text), so the trailing block is the synthesis.
    """
    msg = _message(
        [
            _search_block([_result("https://x.com/a", "X")]),
            _text_block("I've hit the search rate limit. Let me try once more in a fresh call."),
            _text_block("# Research Summary\n\nThe real briefing."),
        ]
    )
    with patch.object(anthropic_search, "build_anthropic_client", return_value=_client_returning(msg)):
        out = await anthropic_search.invoke_anthropic_grounded("prompt", qid=11)

    assert out.startswith("# Research Summary")
    assert "rate limit" not in out


@pytest.mark.asyncio
async def test_a_single_post_search_text_block_is_kept_whole() -> None:
    """The multi-block rule must not truncate a normal one-block answer."""
    body = "# Research Summary\n\nParagraph one.\n\nParagraph two.\n\nParagraph three."
    msg = _message([_search_block([_result("https://x.com/a", "X")]), _text_block(body)])
    with patch.object(anthropic_search, "build_anthropic_client", return_value=_client_returning(msg)):
        out = await anthropic_search.invoke_anthropic_grounded("prompt", qid=12)
    assert "Paragraph one." in out and "Paragraph three." in out
