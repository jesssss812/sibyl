from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from typing import Any

from litellm import acompletion

from sibyl.config.settings import OAI_ANTH_OPENROUTER_KEY_ENV, OPENROUTER_API_KEY_ENV
from sibyl.platform.key_fallback import (
    record_primary_key_fallback,
    should_retry_with_fallback_key,
    should_route_via_primary_key,
)
from sibyl.research.agentic.types import LoopConfig

LlmCall = Callable[[list[dict[str, Any]], list[dict[str, Any]] | None], Awaitable[Any]]


def build_default_llm_call(config: LoopConfig) -> LlmCall:
    model = config.model if config.model.startswith("openrouter/") else f"openrouter/{config.model}"
    primary_key = os.getenv(OAI_ANTH_OPENROUTER_KEY_ENV)
    fallback_key = os.getenv(OPENROUTER_API_KEY_ENV)
    use_fallback = should_route_via_primary_key(model) and primary_key and fallback_key and primary_key != fallback_key

    async def _call_once(
        messages: list[dict[str, Any]], tools_json: list[dict[str, Any]] | None, api_key: str | None
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": model,
            # Shallow copy: litellm may mutate the caller's list in place in some
            # code paths; copying the container preserves the loop's append-only
            # prefix (dict identity is kept — providers cache on it).
            "messages": list(messages),
            "parallel_tool_calls": True,
            "reasoning_effort": config.reasoning_effort,
            # litellm's OpenrouterConfig doesn't map reasoning_effort; without this
            # it survives only because forecasting_tools sets litellm.drop_params=True
            # globally (silently stripping it). Whitelisting passes the raw param
            # through to OpenRouter (validated live by bottega:scratch/driver_replay_2026-07-17).
            "allowed_openai_params": ["reasoning_effort"],
            "temperature": None,
            # litellm ≥1.92 eagerly imports its proxy MCP-gateway handler (which
            # requires fastapi, a proxy-only extra we don't install) whenever `tools`
            # is passed — even for plain function tools that never touch the gateway.
            # We run our own tool-dispatch loop, so skip the import. Private litellm
            # kwarg, popped before the provider sees it; verified against the locked
            # litellm 1.92 (both the eager-import defect and this skip kwarg are
            # 1.92-era). If a future litellm drops the kwarg, this crashes loudly
            # rather than silently regressing.
            "_skip_mcp_handler": True,
        }
        if tools_json is not None:
            kwargs["tools"] = tools_json
        if api_key:
            kwargs["api_key"] = api_key
        return await acompletion(**kwargs)

    async def _call(messages: list[dict[str, Any]], tools_json: list[dict[str, Any]] | None) -> Any:
        if use_fallback:
            assert primary_key is not None
            assert fallback_key is not None
            try:
                return await _call_once(messages, tools_json, primary_key)
            except Exception as exc:  # classifier re-raises non-key-scoped errors
                if not should_retry_with_fallback_key(exc):
                    raise
                # Same accounting as FallbackOpenRouterLlm.invoke: counted once in the
                # generic total (plus at most one subset) and logged as a PAID
                # FALLBACK-KEY FALLBACK. Without this the highest-volume primary-key
                # path in the bot — v2 runs on every question in all four prod
                # workflows — failed over to the paid key completely silently.
                await record_primary_key_fallback(model, exc)  # noqa: ASYNC120
                return await _call_once(messages, tools_json, fallback_key)

        api_key = primary_key if should_route_via_primary_key(model) and primary_key else fallback_key
        # The counted/logged fallback DECISION is now shared with key_fallback
        # (record_primary_key_fallback). Only the transport differs: this path calls
        # raw litellm.acompletion for tool-loop support, where the wrapper goes
        # through GeneralLlm. Share the transport too if this grows a retry ladder.
        return await _call_once(messages, tools_json, api_key)

    return _call
