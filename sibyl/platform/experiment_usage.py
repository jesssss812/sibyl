"""Capture OpenRouter generation IDs and reconcile billed costs after LLM calls.

The pinned LiteLLM map does not price all current OpenRouter model slugs. A
successful model call is therefore charged from OpenRouter's generation record
when available; missing metadata leaves the reservation uncertain.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from threading import Lock
from typing import Any, Iterator

import httpx
import litellm
from litellm.integrations.custom_logger import CustomLogger

_GENERATION_URL = "https://openrouter.ai/api/v1/generation"
logger = logging.getLogger(__name__)


@dataclass
class CostCapture:
    generation_id: str | None = None
    reported_cost: Decimal | None = None
    ready: asyncio.Event = field(default_factory=asyncio.Event, repr=False, compare=False)


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
    async def async_log_success_event(
        self, kwargs: dict[str, Any], response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        capture = _ACTIVE_CAPTURE.get()
        if capture is None:
            return
        response_id = getattr(response_obj, "id", None)
        if isinstance(response_id, str) and response_id:
            capture.generation_id = response_id
        usage = getattr(response_obj, "usage", None)
        cost = usage.get("cost") if isinstance(usage, dict) else getattr(usage, "cost", None)
        capture.reported_cost = _reported_cost(cost)
        capture.ready.set()


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
        logger.warning("GRANT_BILLING_LOOKUP skipped: response ID is not an OpenRouter generation ID")
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
            cost = _reported_cost(data.get("total_cost"))
            if cost is None:
                logger.warning("GRANT_BILLING_LOOKUP generation_id=%s missing or invalid total_cost", generation_id)
            return cost
    except httpx.HTTPStatusError as exc:
        logger.warning("GRANT_BILLING_LOOKUP generation_id=%s status=%s", generation_id, exc.response.status_code)
        return None
    except (httpx.HTTPError, ValueError, TypeError, AttributeError) as exc:
        logger.warning("GRANT_BILLING_LOOKUP generation_id=%s error=%s", generation_id, type(exc).__name__)
        return None


async def resolve_grant_cost(
    capture: CostCapture, framework_cost: float, api_key: str
) -> tuple[Decimal | None, str | None, str]:
    """Return cost, request ID, and evidence class.

    Only the provider's generation billing record establishes a confirmed charge.
    Callback and framework costs are local estimates, even when token usage is known.
    """
    # LiteLLM may finish its asynchronous callback after GeneralLlm returns.
    # Wait for that callback before treating a successful call as unpriced.
    if not capture.ready.is_set():
        try:
            await asyncio.wait_for(capture.ready.wait(), timeout=2)
        except asyncio.TimeoutError:
            pass
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
