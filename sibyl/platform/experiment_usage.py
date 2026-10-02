"""Capture OpenRouter generation IDs and reconcile billed costs after LLM calls.

The pinned LiteLLM map does not price all current OpenRouter model slugs. A
successful model call is therefore charged from OpenRouter's generation record
when available; missing metadata leaves the reservation uncertain.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from threading import Lock
from typing import Any, Iterator

import httpx
import litellm
from litellm.integrations.custom_logger import CustomLogger

_GENERATION_URL = "https://openrouter.ai/api/v1/generation"


@dataclass
class CostCapture:
    generation_id: str | None = None
    reported_cost: Decimal | None = None


_ACTIVE_CAPTURE: ContextVar[CostCapture | None] = ContextVar("sibyl_grant_capture", default=None)
_CALLBACK_LOCK = Lock()


def _reported_cost(raw: Any) -> Decimal | None:
    if raw is None:
        return None
    try:
        cost = Decimal(str(raw))
    except (InvalidOperation, ValueError, TypeError):
        return None
    return cost if cost.is_finite() and cost >= 0 else None


class _GrantUsageObserver(CustomLogger):
    async def async_log_success_event(self, kwargs: dict[str, Any], response_obj: Any, start_time: Any, end_time: Any) -> None:
        capture = _ACTIVE_CAPTURE.get()
        if capture is None:
            return
        response_id = getattr(response_obj, "id", None)
        if isinstance(response_id, str) and response_id:
            capture.generation_id = response_id
        usage = getattr(response_obj, "usage", None)
        cost = usage.get("cost") if isinstance(usage, dict) else getattr(usage, "cost", None)
        capture.reported_cost = _reported_cost(cost)


def _install_observer() -> None:
    with _CALLBACK_LOCK:
        if not any(isinstance(callback, _GrantUsageObserver) for callback in litellm.callbacks):
            litellm.callbacks.append(_GrantUsageObserver())


@contextmanager
def capture_grant_usage() -> Iterator[CostCapture]:
    _install_observer()
    capture = CostCapture()
    token = _ACTIVE_CAPTURE.set(capture)
    try:
        yield capture
    finally:
        _ACTIVE_CAPTURE.reset(token)


async def generation_cost(generation_id: str, api_key: str) -> Decimal | None:
    """Read billed USD from OpenRouter; this GET does not create a generation."""
    if not generation_id.startswith("gen-"):
        return None
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                _GENERATION_URL,
                params={"id": generation_id},
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            data = response.json().get("data", {})
            return _reported_cost(data.get("total_cost"))
    except (httpx.HTTPError, ValueError, TypeError, AttributeError):
        return None


async def resolve_grant_cost(
    capture: CostCapture, framework_cost: float, api_key: str
) -> tuple[Decimal | None, str | None, str]:
    """Return cost, request ID, and evidence class.

    Only the provider's generation billing record establishes a confirmed charge.
    Callback and framework costs are local estimates, even when token usage is known.
    """
    # LiteLLM callbacks can complete one event-loop tick after GeneralLlm's
    # response processing. Wait briefly for the non-text usage metadata only.
    for _ in range(10):
        if capture.generation_id is not None or capture.reported_cost is not None:
            break
        await asyncio.sleep(0.01)
    if capture.generation_id:
        provider_cost = await generation_cost(capture.generation_id, api_key)
        if provider_cost is not None:
            return provider_cost, capture.generation_id, "provider_generation"
    if capture.reported_cost is not None and capture.reported_cost > 0:
        return capture.reported_cost, capture.generation_id, "callback_estimate"
    fallback_cost = _reported_cost(framework_cost)
    if fallback_cost is not None and fallback_cost > 0:
        return fallback_cost, capture.generation_id, "framework_estimate"
    return None, capture.generation_id, "unknown"
