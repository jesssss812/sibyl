"""Capture OpenRouter billing evidence during a call, and reconcile it afterwards.

During a call only the response is read: its generation ID, fee, upstream cost and
BYOK flag. OpenRouter's generation record is read after the run by
``reconcile_pending``, which confirms what the response could not.
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

from sibyl.platform.experiment_funding import CONFIRMING_EVIDENCE, provider_billable_cost

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


# How long to wait for LiteLLM's success callback, which can land just after the call returns.
CALLBACK_WAIT_S = 2.0

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


def _field(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)


def _byok_flag(raw: Any) -> bool | None:
    return raw if isinstance(raw, bool) else None


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
        capture.reported_upstream_cost = _reported_cost(
            _field(_field(usage, "cost_details"), "upstream_inference_cost")
        )
        capture.reported_is_byok = _byok_flag(_field(usage, "is_byok"))
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
    is_byok: bool | None

    @property
    def billable(self) -> Decimal | None:
        """The money that actually moved, or None while any component is unknown."""
        return provider_billable_cost(self.openrouter_charge, self.upstream_cost, self.is_byok)


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

        Confirmation needs every applicable billing component from one provider
        record; a reported fee with the BYOK route unknown is not the whole charge.
        """
        return self.billable is not None and self.evidence in CONFIRMING_EVIDENCE


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
            is_byok = _byok_flag(data.get("is_byok"))
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


async def resolve_grant_cost(capture: CostCapture, framework_cost: float) -> CostResolution:
    """Establish what a completed call cost from the response alone.

    The response's usage confirms the charge when it carries every component: the fee
    and, on a BYOK route, the upstream cost. Otherwise the result is a lower bound -- the
    larger of the known components and the framework's token estimate -- recorded as
    an estimate. The generation record is read later by ``reconcile_pending``, never
    here: it is eventually consistent, and a lookup on every call would add a network
    round trip to the forecast for evidence that usually is not there yet.
    """
    # LiteLLM may finish its asynchronous callback after GeneralLlm returns.
    if not capture.ready.is_set():
        try:
            await asyncio.wait_for(capture.ready.wait(), timeout=CALLBACK_WAIT_S)
        except asyncio.TimeoutError:
            pass

    response = GenerationRecord(capture.reported_cost, capture.reported_upstream_cost, capture.reported_is_byok)
    if response.billable is not None:
        return CostResolution(
            response.billable,
            response.openrouter_charge,
            response.upstream_cost,
            response.is_byok,
            capture.generation_id,
            "provider_response",
        )

    # Partial evidence: the known components are a lower bound, never a confirmation.
    known = [response.openrouter_charge] + ([response.upstream_cost] if response.is_byok else [])
    lower_bound = (
        sum((value for value in known if value is not None), Decimal(0))
        if any(value is not None for value in known)
        else None
    )
    estimate = _reported_cost(framework_cost)
    if estimate is not None and estimate > 0 and (lower_bound is None or estimate > lower_bound):
        return CostResolution(
            estimate,
            response.openrouter_charge,
            response.upstream_cost,
            response.is_byok,
            capture.generation_id,
            "framework_estimate",
        )
    if lower_bound is not None:
        return CostResolution(
            lower_bound,
            response.openrouter_charge,
            response.upstream_cost,
            response.is_byok,
            capture.generation_id,
            "provider_response_partial",
        )
    return CostResolution(None, None, None, None, capture.generation_id, "unknown")


async def reconcile_pending(
    ledger: Any,
    run_id: str,
    api_key: str,
    *,
    attempts: int = 4,
    base_delay: float = 2.0,
    concurrency: int = 8,
    sleep: Any = None,
) -> dict[str, int]:
    """Turn settled estimates into confirmed charges from the provider's own records.

    Runs AFTER a run, never on the hot path, and issues only the read-only generation GET
    -- it never repeats a paid request to learn what one cost. Lookups for different
    attempts run concurrently, each with a short bounded backoff. The record is
    eventually consistent (the 2026-10-05 investigation saw one resolve about eleven
    minutes after the call), so an attempt still unresolved is reported, keeps its
    reservation, and is picked up by re-running this later.

    Returns counts of what happened, so an operator can see what is still outstanding
    rather than assuming the ledger is complete.
    """
    pause = sleep or asyncio.sleep
    outstanding = ledger.unreconciled(run_id)
    result = {"pending": len(outstanding), "reconciled": 0, "still_unknown": 0, "no_generation_id": 0}
    gate = asyncio.Semaphore(concurrency)

    async def lookup(generation_id: str) -> GenerationRecord | None:
        async with gate:
            for round_index in range(attempts):
                record = await generation_record(generation_id, api_key)
                if record is not None and record.billable is not None:
                    return record
                if round_index < attempts - 1:
                    await pause(base_delay * (2**round_index))
        return None

    with_ids = [(attempt_id, generation_id) for attempt_id, generation_id in outstanding if generation_id]
    # Nothing to look these up by; they keep their reservation and stay visible.
    result["no_generation_id"] = len(outstanding) - len(with_ids)
    records = await asyncio.gather(*(lookup(generation_id) for _, generation_id in with_ids))
    for (attempt_id, generation_id), record in zip(with_ids, records):
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
