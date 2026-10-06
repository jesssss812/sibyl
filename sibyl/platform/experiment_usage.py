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

from sibyl.platform.experiment_funding import CONFIRMING_EVIDENCE

_GENERATION_URL = "https://openrouter.ai/api/v1/generation"
logger = logging.getLogger(__name__)


@dataclass
class CostCapture:
    generation_id: str | None = None
    reported_cost: Decimal | None = None
    # OpenRouter's usage accounting also reports the BYOK split on the response.
    reported_upstream_cost: Decimal | None = None
    reported_is_byok: bool | None = None
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
        capture.reported_cost = _reported_cost(_field(usage, "cost"))
        is_byok = _field(usage, "is_byok")
        capture.reported_is_byok = is_byok if isinstance(is_byok, bool) else None
        capture.reported_upstream_cost = _reported_cost(
            _field(_field(usage, "cost_details"), "upstream_inference_cost")
        )
        capture.ready.set()


def _field(container: Any, name: str) -> Any:
    if isinstance(container, dict):
        return container.get(name)
    return getattr(container, name, None)


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
        """The money that actually moved, which is what a budget must hold.

        Under BYOK both components count: the grant key's limit meters the upstream
        expense (``include_byok_in_limit``) and OpenRouter's BYOK fee is drawn from
        credits. Without an upstream figure the BYOK cost is not known at all.
        """
        if self.is_byok:
            if self.upstream_cost is None:
                return None
            return self.upstream_cost + (self.openrouter_charge or Decimal(0))
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
        return self.billable is not None and self.evidence in CONFIRMING_EVIDENCE


def _response_record(capture: CostCapture) -> GenerationRecord | None:
    """The response's own billing figures, when they say whether the route was BYOK.

    Without ``is_byok`` a positive ``usage.cost`` could be the whole charge or only
    OpenRouter's BYOK fee, so it cannot settle the budget figure on its own.
    """
    if capture.reported_cost is None or capture.reported_is_byok is None:
        return None
    return GenerationRecord(
        openrouter_charge=capture.reported_cost,
        upstream_cost=capture.reported_upstream_cost,
        is_byok=capture.reported_is_byok,
    )


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

    1. **The completion response's usage accounting**, when it states ``is_byok``.
       OpenRouter's own figures, already in hand: ``usage.cost`` plus, under BYOK,
       ``cost_details.upstream_inference_cost``. This confirms without a request.
    2. **The generation record** (a GET), for responses that do not carry the split.
       The record is eventually consistent, so it is often not there yet.
    3. **``usage.cost`` alone.** Without ``is_byok`` it may be only OpenRouter's BYOK
       fee, so it is recorded as an estimate and never confirms.
    4. **The framework estimate** (LiteLLM token maths). A guess; recorded, never confirmed.

    This reports what is known; the caller settles on it and ``reconcile_pending``
    confirms estimates afterwards from the provider's records.
    """
    # LiteLLM may finish its asynchronous callback after GeneralLlm returns.
    if not capture.ready.is_set():
        try:
            await asyncio.wait_for(capture.ready.wait(), timeout=2)
        except asyncio.TimeoutError:
            pass

    response_record = _response_record(capture)
    if response_record is not None and response_record.billable is not None:
        # The response already carries the full split; no lookup is needed.
        return CostResolution(
            billable=response_record.billable,
            openrouter_charge=response_record.openrouter_charge,
            upstream_cost=response_record.upstream_cost,
            is_byok=response_record.is_byok,
            provider_request_id=capture.generation_id,
            evidence="provider_response",
        )

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
        # The OpenRouter charge alone: under BYOK it is only the fee, so it is kept as an
        # estimate and its reservation is held until reconciliation.
        return CostResolution(
            billable=capture.reported_cost,
            openrouter_charge=capture.reported_cost,
            upstream_cost=capture.reported_upstream_cost,
            is_byok=capture.reported_is_byok,
            provider_request_id=capture.generation_id,
            evidence="provider_response_partial",
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
