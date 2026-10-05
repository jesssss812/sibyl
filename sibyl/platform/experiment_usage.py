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


@dataclass(frozen=True)
class GenerationRecord:
    """What OpenRouter's billing record says, with the BYOK split preserved.

    ``openrouter_charge`` is credits drawn from the key; ``upstream_cost`` is what the
    provider billed when the request routed through a BYOK integration. Keeping only the
    former is what made a BYOK run look free.
    """

    openrouter_charge: Decimal | None
    upstream_cost: Decimal | None
    is_byok: bool

    @property
    def billable(self) -> Decimal | None:
        """The money that actually moved, which is what a budget must hold."""
        if self.is_byok:
            return self.upstream_cost
        return self.openrouter_charge


@dataclass(frozen=True)
class CostResolution:
    """A settled view of one call's cost, and how firmly it is known."""

    billable: Decimal | None
    openrouter_charge: Decimal | None
    upstream_cost: Decimal | None
    is_byok: bool | None
    provider_request_id: str | None
    evidence: str

    @property
    def confirmed(self) -> bool:
        """True only when the figure the budget holds came from the provider.

        A zero OpenRouter charge with unknown upstream cost is NOT confirmed: under BYOK
        that is the signature of a real expense we cannot see yet, and confirming it
        would discharge the reservation and report the run as free.
        """
        return self.billable is not None and self.evidence in _CONFIRMING_EVIDENCE


_CONFIRMING_EVIDENCE = frozenset({"provider_response", "provider_generation"})


async def generation_record(generation_id: str, api_key: str) -> GenerationRecord | None:
    """Read the billing record from OpenRouter; this GET does not create a generation."""
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
            charge = _reported_cost(data.get("total_cost"))
            upstream = _reported_cost(data.get("upstream_inference_cost"))
            is_byok = bool(data.get("is_byok"))
            if charge is None and upstream is None:
                logger.warning("GRANT_BILLING_LOOKUP generation_id=%s carried no usable cost", generation_id)
                return None
            return GenerationRecord(openrouter_charge=charge, upstream_cost=upstream, is_byok=is_byok)
    except httpx.HTTPStatusError as exc:
        logger.warning("GRANT_BILLING_LOOKUP generation_id=%s status=%s", generation_id, exc.response.status_code)
        return None
    except (httpx.HTTPError, ValueError, TypeError, AttributeError) as exc:
        logger.warning("GRANT_BILLING_LOOKUP generation_id=%s error=%s", generation_id, type(exc).__name__)
        return None


async def resolve_grant_cost(capture: CostCapture, framework_cost: float, api_key: str) -> CostResolution:
    """Establish what a completed call cost, preferring the provider's own figures.

    Order of evidence, strongest first:

    1. **The generation record** (a GET). Richest -- it alone carries the BYOK split, so
       it is the only source that can give a confirmed billable figure for a BYOK route.
    2. **The completion response's ``usage.cost``.** OpenRouter's account charge, already
       in hand, no extra request. A POSITIVE value confirms. A ZERO does not: zero is
       either a genuinely free route or a BYOK route whose real cost is upstream and not
       yet visible, and those are indistinguishable from the response alone. Zero is
       preserved rather than discarded -- it is a fact, just not a sufficient one.
    3. **The framework estimate** (LiteLLM token maths). A guess; recorded, never confirmed.

    Crucially this no longer decides whether a call may proceed. It reports what is known;
    the caller settles on that and reconciles later. The previous design made the GET a
    precondition of publication, so an eventually-consistent endpoint could discard a
    forecast that had already been paid for.
    """
    # LiteLLM may finish its asynchronous callback after GeneralLlm returns.
    if not capture.ready.is_set():
        try:
            await asyncio.wait_for(capture.ready.wait(), timeout=2)
        except asyncio.TimeoutError:
            pass

    if capture.generation_id:
        record = await generation_record(capture.generation_id, api_key)
        if record is not None and record.billable is not None:
            return CostResolution(
                billable=record.billable,
                openrouter_charge=record.openrouter_charge,
                upstream_cost=record.upstream_cost,
                is_byok=record.is_byok,
                provider_request_id=capture.generation_id,
                evidence="provider_generation",
            )

    if capture.reported_cost is not None:
        # Positive confirms; zero is kept but left unconfirmed for reconciliation.
        return CostResolution(
            billable=capture.reported_cost,
            openrouter_charge=capture.reported_cost,
            upstream_cost=None,
            is_byok=None,
            provider_request_id=capture.generation_id,
            evidence="provider_response" if capture.reported_cost > 0 else "provider_response_zero",
        )

    fallback = _reported_cost(framework_cost)
    if fallback is not None and fallback > 0:
        return CostResolution(
            billable=fallback,
            openrouter_charge=None,
            upstream_cost=None,
            is_byok=None,
            provider_request_id=capture.generation_id,
            evidence="framework_estimate",
        )
    return CostResolution(None, None, None, None, capture.generation_id, "unknown")


async def reconcile_pending(
    ledger: Any,
    run_id: str,
    api_key: str,
    *,
    attempts: int = 4,
    base_delay: float = 2.0,
    sleep: Any = None,
) -> dict[str, int]:
    """Turn settled estimates into confirmed charges from the provider's own records.

    Runs AFTER a run, never on the hot path, and issues only the read-only generation GET
    -- it never repeats a paid request to learn what one cost. Bounded exponential backoff
    exists because that record is eventually consistent: the 2026-10-05 investigation saw
    the same ID return 404 twice and then resolve about eleven minutes later.

    Returns counts of what happened, so an operator can see what is still outstanding
    rather than assuming the ledger is complete.
    """
    pause = sleep or asyncio.sleep
    outstanding = ledger.unreconciled(run_id)
    result = {"pending": len(outstanding), "reconciled": 0, "still_unknown": 0, "no_generation_id": 0}
    for attempt_id, generation_id in outstanding:
        if not generation_id:
            # Nothing to look it up by; it keeps its reservation and stays visible.
            result["no_generation_id"] += 1
            continue
        record = None
        for round_index in range(attempts):
            record = await generation_record(generation_id, api_key)
            if record is not None and record.billable is not None:
                break
            record = None
            if round_index < attempts - 1:
                await pause(base_delay * (2**round_index))
        if record is None or record.billable is None:
            result["still_unknown"] += 1
            logger.warning(
                "GRANT_RECONCILE_UNRESOLVED attempt=%s generation_id=%s; reservation still held",
                attempt_id,
                generation_id,
            )
            continue
        ledger.reconcile(
            attempt_id,
            record.billable,
            openrouter_charge_usd=record.openrouter_charge,
            upstream_cost_usd=record.upstream_cost,
            is_byok=record.is_byok,
            provider_request_id=generation_id,
        )
        result["reconciled"] += 1
        logger.info(
            "GRANT_RECONCILED attempt=%s billable_usd=%s openrouter_usd=%s upstream_usd=%s byok=%s",
            attempt_id,
            record.billable,
            record.openrouter_charge,
            record.upstream_cost,
            record.is_byok,
        )
    return result
