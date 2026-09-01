"""Anthropic grounded-search research provider.

Uses the `anthropic` SDK directly (NOT via OpenRouter or litellm) so we get the
first-party server-side `web_search` tool rather than OpenRouter's Exa-backed web
plugin. Same reasoning as `gemini_search.py`, which this replaces as the second
search index: the Metaculus Fall 2025 writeup identified research breadth as the
single strongest predictor of winning bots, and breadth only counts when the
indexes genuinely differ. Anthropic's search is a distinct index from the OpenAI
native-search provider, so the ensemble keeps two independent retrieval paths.

**Why this exists at all.** Google AI Studio meters grounded search on a MONTHLY
free allowance, so an exhausted quota stays exhausted until the month turns over.
The operator hit it with no Google billing account attached, and on the 2026-08-31
run every Gemini call returned 429 RESOURCE_EXHAUSTED — the provider contributed
nothing. This path runs on a key the operator already funds.

**Direct SDK, not litellm.** `forecasting_tools` sets `litellm.drop_params = True`
process-globally, which silently discards provider params litellm does not model —
and litellm's Anthropic transform does not model server-side tools at all. Routing
this through `GeneralLlm` would therefore drop the `web_search` tool and return
confident UNGROUNDED text, which is precisely the fabrication mode the grounded
floor below exists to catch. The direct SDK keeps the tool declaration honest.

Mirrors the `gemini_search_provider` / `_native_search_provider` contract
(`MetaculusQuestion -> str`).
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
from typing import Any, cast
from urllib.parse import urlparse

import anthropic
from forecasting_tools.data_models.questions import MetaculusQuestion

from metaculus_bot.constants import (
    ANTHROPIC_API_KEY_ENV,
    ANTHROPIC_SEARCH_DEFAULT_MODEL,
    ANTHROPIC_SEARCH_EFFORT,
    ANTHROPIC_SEARCH_MAX_TOKENS,
    ANTHROPIC_SEARCH_MAX_USES,
    ANTHROPIC_SEARCH_MODEL_ENV,
    ANTHROPIC_SEARCH_TIMEOUT,
)
from metaculus_bot.prompts import web_research_prompt
from metaculus_bot.research.provider_diagnostics import record_provider_detail
from metaculus_bot.research.providers import ResearchCallable
from metaculus_bot.research.raw_log import record_raw_research

logger = logging.getLogger(__name__)

__all__ = [
    "build_anthropic_client",
    "anthropic_search_provider",
    "invoke_anthropic_grounded",
]

# Header-only initializer for the sources list; the sources-present gate compares
# against this length. Same convention as gemini_search._SOURCES_HEADER_LEN.
_SOURCES_HEADER_LEN = 3

# Dynamic-filtering variant. Requires Opus 4.6+ / Sonnet 4.6+ — the default model
# (ANTHROPIC_SEARCH_DEFAULT_MODEL) is chosen to satisfy that. Pointing
# ANTHROPIC_SEARCH_MODEL at an older/smaller model (e.g. Haiku 4.5) will 400 here;
# that is deliberate, because the silent alternative is an ungrounded answer.
_WEB_SEARCH_TOOL_TYPE = "web_search_20260209"


@functools.lru_cache(maxsize=1)
def _cached_client_for_key(api_key: str) -> anthropic.AsyncAnthropic:
    """Process-global cached AsyncAnthropic client keyed on API key.

    SDK clients are designed to be long-lived; one client across a backtest reuses
    TLS connections and HTTP/2 multiplexing over the thousands of calls a round
    makes. Keyed on api_key so a rotated key produces a fresh client.
    """
    return anthropic.AsyncAnthropic(api_key=api_key)


def build_anthropic_client() -> anthropic.AsyncAnthropic:
    """Return the cached AsyncAnthropic client for the operator's Anthropic key.

    Reads ANTHROPIC_API_KEY. There is no Metaculus-donated Anthropic key on this
    path — the donated OpenRouter key only covers OpenRouter-routed Anthropic
    models. Raises ValueError if the key is missing so misconfiguration is loud
    rather than a silent provider drop.

    NOTE for operators: a Claude.ai Pro/Max subscription is NOT an API credential.
    API access is billed separately at console.anthropic.com.
    """
    api_key = os.getenv(ANTHROPIC_API_KEY_ENV)
    if not api_key:
        raise ValueError(f"{ANTHROPIC_API_KEY_ENV} must be set to use the Anthropic search provider")
    return _cached_client_for_key(api_key)


def _resolve_model(model_slug: str | None) -> str:
    return model_slug or os.getenv(ANTHROPIC_SEARCH_MODEL_ENV, ANTHROPIC_SEARCH_DEFAULT_MODEL)


def _domain(url: str) -> str:
    """Registrable-ish host for a result URL, with the www. prefix stripped.

    Rendering the domain rather than the full URL follows the same decision made
    for Gemini (commit d8d4cc4): raw URLs were ~5% of the whole research bundle and
    a text-only forecaster cannot follow them. Returns "" on an unparseable URL so
    the caller can fall back to the title.
    """
    try:
        host = (urlparse(url).netloc or "").strip().lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def _collect_sources(message: Any) -> tuple[list[tuple[str, str]], int, int]:
    """Return (deduped [(title, domain)], raw_result_count, error_block_count).

    Server-tool errors do NOT raise: the API returns HTTP 200 with a
    `web_search_tool_result` block whose `content` is a single error OBJECT rather
    than a LIST of results. Branching on that shape is required — indexing the
    error object as if it were a list is the classic mistake here. Errors are
    COUNTED and reported once by the caller rather than logged here: the
    2026-08-31 live run emitted eight byte-identical `max_uses_exceeded` warnings
    for one question, which is log spam that buries the signal (that the cap was
    too low) under repetition.

    **Deduplication.** The Gemini provider deliberately does NOT dedupe its
    grounding chunks, because its inline `[N]` citation markers are spliced by
    chunk index and dropping a row would misalign every marker after it. This
    provider emits no inline markers, so that constraint does not apply and
    keeping duplicates is pure bundle bloat — the same live run rendered 76 rows
    in which Gallup, Newsweek, Pew, UCSB and Wikipedia each appeared two or more
    times. Dedupe is keyed on the URL where present (the precise identity) and
    falls back to (title, domain); first occurrence wins, so ordering is stable.
    """
    sources: list[tuple[str, str]] = []
    seen: set[str | tuple[str, str]] = set()
    raw_count = 0
    error_count = 0

    for block in getattr(message, "content", None) or []:
        if getattr(block, "type", None) != "web_search_tool_result":
            continue
        content = getattr(block, "content", None)
        if not isinstance(content, list):
            error_count += 1
            continue
        for result in content:
            if getattr(result, "type", None) != "web_search_result":
                continue
            url = (getattr(result, "url", None) or "").strip()
            title = (getattr(result, "title", None) or "").strip()
            domain = _domain(url)
            if not (title or domain):
                continue
            raw_count += 1
            key: str | tuple[str, str] = url or (title.lower(), domain)
            if key in seen:
                continue
            seen.add(key)
            sources.append((title, domain))

    return sources, raw_count, error_count


def _format_source_label(title: str, domain: str) -> str:
    """Render one source as ``title — domain``, degrading to whichever exists."""
    if title and domain and title != domain:
        return f"{title} — {domain}"
    return title or domain


def _response_text(message: Any) -> str:
    """Return the model's FINAL synthesis — the text after the last search call.

    Thinking blocks are skipped: they are reasoning, not research, and forwarding
    them would put un-cited model speculation into the forecaster bundle.

    **Only text emitted after the last `web_search_tool_result` is kept**, because
    a server-side search loop interleaves narration with its tool calls and that
    narration is addressed to nobody. The 2026-08-31 live run opened the
    forecaster-facing section with "Let me wait and retry with a single, simple
    call.I'll pause and try again in the next turn to avoid the rate limit." — three
    concatenated status mutterings ahead of the actual briefing. The Gemini provider
    never hit this because its AFC loop returns a single synthesized answer.

    Falls back to every text block when nothing follows the last search (a response
    that ends on a tool call), so a defensive filter can never silently delete a
    whole briefing.
    """
    blocks = list(getattr(message, "content", None) or [])
    last_search = -1
    for idx, block in enumerate(blocks):
        if getattr(block, "type", None) == "web_search_tool_result":
            last_search = idx

    def _texts(candidates: list[Any]) -> list[str]:
        return [b.text for b in candidates if getattr(b, "type", None) == "text" and getattr(b, "text", None)]

    final = _texts(blocks[last_search + 1 :])
    if not final:
        final = _texts(blocks)
    # More than one surviving text block means TOOL BLOCKS separated them: the SDK's
    # get_final_message() merges contiguous streamed text into a single block, so two
    # blocks can only arise across a tool call. `web_search` is the sole declared tool,
    # therefore the earlier fragments are inter-attempt narration and the last block is
    # the synthesis. This catches the residual case the positional filter alone misses —
    # a muttering emitted AFTER the final search attempt.
    return (final[-1] if len(final) > 1 else "".join(final)).strip()


def _format_grounded_response(message: Any, *, qid: int | None = None, model: str | None = None) -> str:
    """Stitch response text with a Sources section, or suppress if ungrounded.

    Output format:
        <response text>

        ### Sources
        [1] <title> — <domain>
        ...

    **Grounded floor.** A response with zero successful search results is
    suppressed (returns ""), not passed through. This is inherited from the Gemini
    provider's hardest-won lesson (Q38195, 2026-07-19): 30 search queries returned
    0 grounding chunks and the model emitted a confident, fabricated contract table
    with fake ``[primary]`` tags that reached the forecasters. Ungrounded output
    from a provider whose entire premise is retrieval is a fabrication vector, and
    an empty section costs strictly less than a plausible wrong one.

    Unlike Gemini's byte-offset splicing, no inline [N] markers are inserted: the
    model already emits natural-language attribution alongside `citations` on its
    text blocks, so the numbered Sources list is the only addition needed.
    """
    text = _response_text(message)
    if not text:
        return ""

    sources, _raw, _errors = _collect_sources(message)
    if not sources:
        logger.warning(f"ANTHROPIC_UNGROUNDED_SUPPRESSED: question={qid} model={model}")
        # Record the loss so Provider Diagnostics carries a `lost=grounding:...`
        # token. Without it, the "" returned here maps to status `empty` —
        # byte-identical to a healthy call that legitimately found nothing, since
        # nothing raised and no counter moved. Mirrors the Gemini path exactly.
        # Deliberately NOT an alertable counter: changing what CI treats as red is
        # the operator's call, not a side effect of adding visibility.
        record_provider_detail(qid, "anthropic_search", {"sources": {"grounding": "error(ungrounded_suppressed)"}})
        return ""

    lines = ["", "", "### Sources"]
    for idx, (title, domain) in enumerate(sources, start=1):
        label = _format_source_label(title, domain)
        if label:
            lines.append(f"[{idx}] {label}")

    return text + "\n".join(lines) if len(lines) > _SOURCES_HEADER_LEN else text


async def invoke_anthropic_grounded(
    prompt: str,
    *,
    model_slug: str | None = None,
    qid: int | None = None,
) -> str:
    """Invoke Claude with the server-side web_search tool and return formatted text.

    Streaming is used because `max_tokens` here sits well above the SDK's
    non-streaming comfort band and a server-side search loop can hold the
    connection open across several round-trips; `get_final_message()` reassembles
    the whole response, so callers see the same shape either way.

    Raises on SDK errors — callers decide whether to fail hard or soft.
    """
    client = build_anthropic_client()
    model = _resolve_model(model_slug)

    logger.info(f"AnthropicSearch: calling {model} with web_search (max_uses={ANTHROPIC_SEARCH_MAX_USES})")

    async def _call() -> Any:
        async with client.messages.stream(
            model=model,
            max_tokens=ANTHROPIC_SEARCH_MAX_TOKENS,
            thinking={"type": "adaptive"},
            # cast: ANTHROPIC_SEARCH_EFFORT is a plain `str` constant (env-overridable in
            # spirit), while the SDK types `effort` as a Literal. Widening the constant to a
            # Literal would push the same cast up into constants.py for no extra safety —
            # an invalid tier is caught by the API with a 400 either way.
            output_config=cast("Any", {"effort": ANTHROPIC_SEARCH_EFFORT}),
            tools=[
                {
                    "type": _WEB_SEARCH_TOOL_TYPE,
                    "name": "web_search",
                    "max_uses": ANTHROPIC_SEARCH_MAX_USES,
                }
            ],
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            return await stream.get_final_message()

    try:
        message = await asyncio.wait_for(_call(), timeout=ANTHROPIC_SEARCH_TIMEOUT)
    except asyncio.TimeoutError:
        logger.warning(f"AnthropicSearch: {model} timed out after {ANTHROPIC_SEARCH_TIMEOUT}s")
        raise

    # Capture the raw SDK response (text + search results + usage) before
    # formatting drops most of it.
    record_raw_research(qid=qid, provider="anthropic_search", payload=message)

    # A safety refusal is a 200 with stop_reason="refusal" and no useful content —
    # surface it as its own log line rather than letting it read as "found nothing".
    if getattr(message, "stop_reason", None) == "refusal":
        category = getattr(getattr(message, "stop_details", None), "category", None)
        logger.warning(f"AnthropicSearch: refusal (category={category}) question={qid} model={model}")
        return ""

    formatted = _format_grounded_response(message, qid=qid, model=model)
    sources, raw_count, error_count = _collect_sources(message)
    if error_count:
        # One aggregated line, not one per block. A non-zero count here is the
        # signal that ANTHROPIC_SEARCH_MAX_USES is throttling real research —
        # the model keeps asking and every attempt comes back as an error block.
        logger.warning(
            f"AnthropicSearch: {error_count} web_search error block(s) "
            f"(usually max_uses_exceeded at max_uses={ANTHROPIC_SEARCH_MAX_USES}) "
            f"question={qid} model={model}"
        )
    logger.info(
        f"AnthropicSearch: got {len(formatted)} chars, {len(sources)} unique sources "
        f"({raw_count} raw results) from {model}"
    )
    return formatted


def anthropic_search_provider(
    model_slug: str | None = None,
    is_benchmarking: bool = False,
) -> ResearchCallable:
    """Research provider using Claude with the server-side web_search tool.

    Mirrors the `_native_search_provider` contract (`MetaculusQuestion -> str`).
    """

    async def _fetch(question: MetaculusQuestion) -> str:  # noqa: D401
        prompt = web_research_prompt(
            question.question_text,
            is_benchmarking=is_benchmarking,
            citation_style="auto_annotated",
            allow_resolution_source_reading=True,
        )
        return await invoke_anthropic_grounded(
            prompt, model_slug=model_slug, qid=getattr(question, "id_of_question", None)
        )

    return _fetch
