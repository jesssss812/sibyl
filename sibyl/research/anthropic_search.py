"""Claude web-search research provider, on OpenRouter.

Claude answers with OpenRouter's ``web`` plugin on the ``native`` engine, which runs
Anthropic's own server-side web search. That keeps this provider a second search
index, distinct from the OpenAI native-search provider: the Metaculus Fall 2025
writeup identified research breadth as the strongest predictor of winning bots, and
breadth only counts when the indexes genuinely differ. It is paid like every other
model call, the grant first when the run has one, then the ordinary keys.

**History.** 2026-08-31: this replaced Gemini grounded search, whose monthly free
quota was exhausted. 2026-10-18: moved from the Anthropic API (the ``anthropic``
SDK on ``ANTHROPIC_API_KEY``) to OpenRouter, so the grant pays for it. The SDK path
existed because litellm drops server-side tool declarations it does not model; the
OpenRouter plugin is a request parameter OpenRouter acts on itself, the same one the
native-search provider has used throughout.

**Grounded floor.** The prompt asks for inline ``[source](url)`` citations, and a
reply that cites no URL is suppressed (returns ""), not passed through. This is the
Gemini provider's hardest-won lesson (Q38195, 2026-07-19): searches that returned
nothing produced a confident, fabricated table that reached the forecasters. An
empty section costs strictly less than a plausible wrong one.

Mirrors the ``_native_search_provider`` contract (``MetaculusQuestion -> str``).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from urllib.parse import urlparse

from forecasting_tools import GeneralLlm
from forecasting_tools.data_models.questions import MetaculusQuestion

from sibyl.config.settings import (
    ANTHROPIC_SEARCH_DEFAULT_MODEL,
    ANTHROPIC_SEARCH_EFFORT,
    ANTHROPIC_SEARCH_MAX_TOKENS,
    ANTHROPIC_SEARCH_MAX_USES,
    ANTHROPIC_SEARCH_MODEL_ENV,
    ANTHROPIC_SEARCH_TIMEOUT,
)
from sibyl.forecasting.prompts import web_research_prompt
from sibyl.platform.key_fallback import build_llm_with_openrouter_fallback
from sibyl.platform.llm_stage import label_stage
from sibyl.research.provider_diagnostics import record_provider_detail
from sibyl.research.providers import ResearchCallable
from sibyl.research.raw_log import record_raw_research

logger = logging.getLogger(__name__)

# A markdown link target, or a bare URL. Trailing punctuation is not part of a URL.
_URL_RE = re.compile(r"https?://[^\s)\]>\"']+")
_TRAILING = ".,;:!?"


def _resolve_model(model_slug: str | None) -> str:
    """The Anthropic model id, without the vendor prefix OpenRouter puts in front."""
    model = model_slug or os.getenv(ANTHROPIC_SEARCH_MODEL_ENV) or ANTHROPIC_SEARCH_DEFAULT_MODEL
    return model.removeprefix("anthropic/")


def _domain(url: str) -> str:
    """Registrable-ish host for a cited URL, with the www. prefix stripped.

    Rendering the domain rather than the full URL follows the same decision made
    for Gemini (commit d8d4cc4): raw URLs were ~5% of the whole research bundle and
    a text-only forecaster cannot follow them. Returns "" on an unparseable URL.
    """
    try:
        host = (urlparse(url).netloc or "").strip().lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def _collect_urls(text: str) -> list[str]:
    """Every cited URL once, in first-seen order."""
    seen: dict[str, None] = {}
    for match in _URL_RE.findall(text):
        seen.setdefault(match.rstrip(_TRAILING), None)
    return list(seen)


def build_search_llm(model: str) -> GeneralLlm:
    return label_stage(
        build_llm_with_openrouter_fallback(
            model=f"openrouter/anthropic/{model}",
            temperature=None,
            max_tokens=ANTHROPIC_SEARCH_MAX_TOKENS,
            timeout=ANTHROPIC_SEARCH_TIMEOUT,
            # A slow failure is not retried; the wall timeout below bounds the call.
            allowed_tries=1,
            reasoning={"effort": ANTHROPIC_SEARCH_EFFORT},
            plugins=[{"id": "web", "engine": "native", "max_results": ANTHROPIC_SEARCH_MAX_USES}],
        ),
        "anthropic_search",
    )


def _format_grounded_response(text: str, *, qid: int | None = None, model: str | None = None) -> str:
    """The reply with a numbered Sources list of cited domains, or "" when nothing is cited."""
    text = text.strip()
    if not text:
        return ""
    domains = list(dict.fromkeys(d for d in (_domain(url) for url in _collect_urls(text)) if d))
    if not domains:
        logger.warning(f"ANTHROPIC_UNGROUNDED_SUPPRESSED: question={qid} model={model}")
        # Record the loss so Provider Diagnostics carries a `lost=grounding:...` token.
        # Without it the "" maps to status `empty`, identical to a healthy call that
        # found nothing. Deliberately not an alertable counter.
        record_provider_detail(qid, "anthropic_search", {"sources": {"grounding": "error(ungrounded_suppressed)"}})
        return ""
    sources = "\n".join(f"[{i}] {domain}" for i, domain in enumerate(domains, start=1))
    return f"{text}\n\n### Sources\n{sources}"


async def invoke_anthropic_grounded(
    prompt: str,
    *,
    model_slug: str | None = None,
    qid: int | None = None,
) -> str:
    """Ask Claude, with OpenRouter's native web search, and return the cited briefing.

    Raises on call errors and timeouts; callers decide whether to fail hard or soft.
    """
    model = _resolve_model(model_slug)
    llm = build_search_llm(model)
    logger.info(
        f"AnthropicSearch: calling {model} via OpenRouter native web search (max_results={ANTHROPIC_SEARCH_MAX_USES})"
    )
    try:
        text = await asyncio.wait_for(llm.invoke(prompt), timeout=ANTHROPIC_SEARCH_TIMEOUT)
    except asyncio.TimeoutError:
        logger.warning(f"AnthropicSearch: {model} timed out after {ANTHROPIC_SEARCH_TIMEOUT}s")
        raise
    record_raw_research(qid=qid, provider="anthropic_search", payload=text)
    formatted = _format_grounded_response(text, qid=qid, model=model)
    if formatted:
        logger.info(f"AnthropicSearch: got {len(formatted)} chars from {model}")
    return formatted


def anthropic_search_provider(
    model_slug: str | None = None,
    is_benchmarking: bool = False,
) -> ResearchCallable:
    """Research provider using Claude with OpenRouter's native web search."""

    async def _fetch(question: MetaculusQuestion) -> str:  # noqa: D401
        prompt = web_research_prompt(
            question.question_text,
            is_benchmarking=is_benchmarking,
            citation_style="markdown",
            allow_resolution_source_reading=True,
        )
        return await invoke_anthropic_grounded(
            prompt, model_slug=model_slug, qid=getattr(question, "id_of_question", None)
        )

    return _fetch
