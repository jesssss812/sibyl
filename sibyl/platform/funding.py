"""Which key pays for each paid call, and the record of what it cost.

There is one forecasting pipeline. Funding only decides which credential carries a
request that would be sent anyway, and records the attempt:

1. When the run has a grant budget and the grant can reserve the attempt, the
   OpenRouter grant key pays.
2. Otherwise the request goes out exactly as it would without a grant: on the
   client's own key, which for most models means the Metaculus key first and the
   personal key after it (``FallbackOpenRouterLlm``).

A grant refusal that another key can fix (out of credit, bad credential, rate limit,
key-scoped route) moves the same request to step 2. Any other failure is raised as it
would be without the grant, so the payer never changes the retry policy, the models,
the research or what gets published.

Every attempt is written to the spend ledger under the source that paid, for the
audit. Accounting can stop a grant attempt (no budget left, ledger unwritable), which
then falls to step 2; it never stops a forecast.
"""

from __future__ import annotations

import copy
import logging
import os
import sqlite3
import weakref
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid4

import httpx
from forecasting_tools import GeneralLlm

from sibyl.config.settings import OAI_ANTH_OPENROUTER_KEY_ENV, OPENROUTER_API_KEY_ENV
from sibyl.platform.experiment_funding import (
    GRANT_ALLOCATION_USD,
    GRANT_KEY_ENV,
    GRANT_RESERVE_USD,
    BudgetExceeded,
    FundingSource,
    SpendLedger,
)
from sibyl.platform.experiment_usage import CostResolution, capture_grant_usage, resolve_grant_cost
from sibyl.platform.key_fallback import is_credit_caused_error, is_key_scoped_failure
from sibyl.platform.llm_retry import llm_status_code

logger = logging.getLogger(__name__)
T = TypeVar("T")

SPEND_LEDGER_ENV = "SIBYL_SPEND_LEDGER"
DEFAULT_SPEND_LEDGER = "backtests/experiment_grant_spend.sqlite3"
# Held against the grant for each attempt until the provider's figure replaces it.
# Sized for the largest call the roster makes (a 64k-token Opus completion).
DEFAULT_GRANT_RESERVE_PER_CALL_USD = Decimal("2")


@dataclass
class Funding:
    """The run's spend ledger and, optionally, its grant key."""

    run_id: str
    ledger: SpendLedger
    grant_key: str | None = None
    grant_reserve_per_call_usd: Decimal = DEFAULT_GRANT_RESERVE_PER_CALL_USD
    grant_stopped: str | None = field(default=None, init=False)

    def __repr__(self) -> str:
        return f"Funding(run_id={self.run_id!r}, grant={'yes' if self.grant_key else 'no'})"

    @property
    def grant_usable(self) -> bool:
        return self.grant_key is not None and self.grant_stopped is None

    def stop_grant(self, reason: str) -> None:
        """Send the rest of the run's calls to the ordinary keys."""
        if self.grant_stopped is None:
            self.grant_stopped = reason
            # Later calls in the run use the ordinary keys. A telemetry marker: see
            # sibyl/telemetry/markers.py before changing the format.
            logger.warning("GRANT_STOPPED: run=%s reason=%s", self.run_id, reason)


_FUNDING: Funding | None = None
_QUESTION: ContextVar[str | None] = ContextVar("sibyl_spend_question", default=None)
_STAGES: weakref.WeakKeyDictionary[GeneralLlm, str] = weakref.WeakKeyDictionary()


def configure_funding(funding: Funding | None) -> None:
    """Set the run's funding once, before any client is used. None turns accounting off."""
    global _FUNDING
    _FUNDING = funding


def current_funding() -> Funding | None:
    return _FUNDING


@contextmanager
def spend_question(question_id: int | None) -> Iterator[None]:
    """Attribute the attempts made inside this block to one question."""
    token = _QUESTION.set(str(question_id) if question_id is not None else None)
    try:
        yield
    finally:
        _QUESTION.reset(token)


def label_stage(llm: GeneralLlm, stage: str) -> GeneralLlm:
    """Name the pipeline stage an LLM client serves, for the ledger breakdown."""
    _STAGES[llm] = stage
    return llm


def stage_of(llm: GeneralLlm) -> str:
    return _STAGES.get(llm, "llm")


def _env_key(name: str) -> str:
    return os.getenv(name, "").strip()


# The OpenRouter sources and the env var holding each one's key. A billing record is only
# visible to the key that made the request, so reconciliation reads it through this too.
OPENROUTER_KEY_ENVS: dict[FundingSource, str] = {
    FundingSource.EXPERIMENT_GRANT: GRANT_KEY_ENV,
    FundingSource.METACULUS_KEY: OAI_ANTH_OPENROUTER_KEY_ENV,
    FundingSource.PERSONAL_KEY: OPENROUTER_API_KEY_ENV,
}


def source_for_key(key: str | None) -> FundingSource:
    """The ledger source for an OpenRouter credential. No key means litellm's env default."""
    funding = _FUNDING
    key = (key or "").strip()
    if key and funding is not None and key == funding.grant_key:
        return FundingSource.EXPERIMENT_GRANT
    if key and key == _env_key(OPENROUTER_KEY_ENVS[FundingSource.METACULUS_KEY]):
        return FundingSource.METACULUS_KEY
    return FundingSource.PERSONAL_KEY


def _unreported_cost(estimate: float) -> CostResolution:
    """An estimate for a call whose provider reports no per-call charge."""
    amount = Decimal(str(estimate)) if estimate and estimate > 0 else None
    if amount is not None and amount.is_finite():
        return CostResolution(amount, None, None, None, None, "framework_estimate")
    return CostResolution(Decimal(0), None, None, None, None, "direct_service_unpriced")


# Statuses OpenRouter returns before any model runs: a bad request, a key or credit
# problem, a moderation block, a missing model, an oversized or unprocessable request,
# a rate limit. Nothing is billed for these. A timeout, cancellation or 5xx may have
# reached the model, so those keep their reservation until reconciled or written off.
_REFUSED_STATUSES = frozenset({400, 401, 402, 403, 404, 413, 422, 429})


def _refused_before_serving(exc: Exception) -> bool:
    return llm_status_code(exc) in _REFUSED_STATUSES or is_key_scoped_failure(exc)


class _GrantUnavailable(Exception):
    """The grant cannot account for this attempt, so it must not pay for it."""


def _record(action: Callable[[], None], what: str) -> bool:
    try:
        action()
    except (sqlite3.Error, OSError, ValueError) as exc:
        logger.warning("SPEND_LEDGER_WRITE_FAILED %s (%s); the call is unaffected", what, type(exc).__name__)
        return False
    return True


async def _attempt(
    funding: Funding,
    model: str,
    stage: str,
    source: FundingSource,
    invoke: Callable[[], Awaitable[T]],
    *,
    framework_cost: Callable[[T], float] | None,
    token_counts: Callable[[T], tuple[int | None, int | None]] | None,
    openrouter: bool = True,
) -> T:
    """Record one attempt around ``invoke``.

    ``openrouter=False`` is for calls that never report a charge through LiteLLM's
    callback (direct SDKs, non-OpenRouter models): waiting for it would only add delay,
    so they settle at once on the framework's estimate, or an unpriced zero.
    """
    grant = source is FundingSource.EXPERIMENT_GRANT
    reserve = funding.grant_reserve_per_call_usd if grant else Decimal(0)
    attempt_id = uuid4().hex
    question = _QUESTION.get()
    try:
        funding.ledger.reserve(attempt_id, funding.run_id, model, stage, reserve, question_id=question, source=source)
    except BudgetExceeded:
        raise
    except (sqlite3.Error, OSError, ValueError) as exc:
        if grant:
            raise _GrantUnavailable(type(exc).__name__) from None
        logger.warning("SPEND_LEDGER_WRITE_FAILED reserve (%s); the call is unaffected", type(exc).__name__)
        attempt_id = ""

    generation_id: str | None = None
    try:
        with capture_grant_usage() as capture:
            try:
                result = await invoke()
            finally:
                generation_id = capture.generation_id
    except BaseException as exc:
        status = getattr(exc, "status_code", None)
        diagnostic = f"{type(exc).__name__} status={status if isinstance(status, int) else None}"
        if attempt_id:
            # The refusal test reads OpenRouter's statuses and wording; a direct service's
            # failure is recorded as uncertain (it holds nothing, so nothing is lost).
            refused = openrouter and isinstance(exc, Exception) and _refused_before_serving(exc)
            _record(
                (lambda: funding.ledger.mark_rejected(attempt_id, evidence_source=diagnostic))
                if refused
                # A request that was answered and then failed in the framework (an empty
                # answer, say) was billed; its generation ID lets reconcile price it.
                else (
                    lambda: funding.ledger.mark_uncertain(
                        attempt_id, provider_request_id=generation_id, evidence_source=diagnostic
                    )
                ),
                "failure",
            )
        logger.info(
            "SPEND_ATTEMPT run=%s question=%s source=%s stage=%s model=%s status=failed error=%s",
            funding.run_id,
            question,
            source,
            stage,
            model,
            diagnostic,
        )
        raise

    estimate = framework_cost(result) if framework_cost else 0.0
    resolution = await resolve_grant_cost(capture, estimate) if openrouter else _unreported_cost(estimate)
    prompt_tokens, completion_tokens = token_counts(result) if token_counts else (None, None)
    # An unknown cost settles as an estimated zero: the attempt is still recorded,
    # a grant attempt keeps its full reservation, and reconciliation fills it in.
    billable = resolution.billable if resolution.billable is not None else Decimal(0)
    if attempt_id:
        _record(
            lambda: funding.ledger.settle(
                attempt_id,
                billable,
                prompt_tokens,
                completion_tokens,
                resolution.provider_request_id,
                certainty="confirmed" if resolution.confirmed else "estimated",
                evidence_source=resolution.evidence,
                openrouter_charge_usd=resolution.openrouter_charge,
                upstream_cost_usd=resolution.upstream_cost,
                is_byok=resolution.is_byok,
            ),
            "settlement",
        )
    logger.info(
        "SPEND_ATTEMPT run=%s question=%s source=%s stage=%s model=%s status=succeeded "
        "charged_usd=%s confirmed=%s evidence=%s",
        funding.run_id,
        question,
        source,
        stage,
        model,
        billable,
        resolution.confirmed,
        resolution.evidence,
    )
    if grant and billable > reserve:
        funding.stop_grant(f"a charge of ${billable} exceeded the ${reserve} per-call reservation")
    return result


async def funded_openrouter_call(
    model: str,
    stage: str,
    *,
    with_key: Callable[[str], Awaitable[T]],
    own_key: str | None,
    own_call: Callable[[], Awaitable[T]],
    framework_cost: Callable[[T], float] | None = None,
    token_counts: Callable[[T], tuple[int | None, int | None]] | None = None,
    offer_grant: bool = True,
) -> T:
    """Send one OpenRouter request on the grant if it can pay, otherwise on the client's own key.

    ``with_key(key)`` sends the request on a given key; ``own_call()`` sends it exactly as
    the client would without any grant, on ``own_key``. ``offer_grant=False`` is for a
    caller's own key-fallback retry, whose first try already offered the grant.
    """
    funding = _FUNDING
    if funding is None:
        return await own_call()
    if offer_grant and funding.grant_usable and funding.grant_key is not None:
        grant_key = funding.grant_key
        try:
            return await _attempt(
                funding,
                model,
                stage,
                FundingSource.EXPERIMENT_GRANT,
                lambda: with_key(grant_key),
                framework_cost=framework_cost,
                token_counts=token_counts,
            )
        except BudgetExceeded as exc:
            # Held by requests still in flight: only this call goes to the ordinary key.
            if not exc.in_flight:
                funding.stop_grant(str(exc))
        except _GrantUnavailable as exc:
            logger.warning("GRANT_SKIPPED model=%s: the ledger could not record the attempt (%s)", model, exc)
        except Exception as exc:
            if not is_key_scoped_failure(exc):
                raise
            if is_credit_caused_error(exc) or getattr(exc, "status_code", None) == 401:
                funding.stop_grant(f"the grant key was refused ({type(exc).__name__})")
            logger.warning(
                "GRANT_KEY_FALLBACK model=%s stage=%s: grant key refused (%s); sending on the ordinary key",
                model,
                stage,
                type(exc).__name__,
            )
    return await _attempt(
        funding,
        model,
        stage,
        source_for_key(own_key),
        own_call,
        framework_cost=framework_cost,
        token_counts=token_counts,
    )


async def record_direct_call(service: str, stage: str, invoke: Callable[[], Awaitable[T]]) -> T:
    """Record one call to a service billed outside OpenRouter (Exa, Google, Anthropic, ...).

    These services return no per-call charge, so the attempt is recorded at an
    estimated zero for the operator to price from the service's own invoice.
    """
    funding = _FUNDING
    if funding is None:
        return await invoke()
    return await _attempt(
        funding,
        service,
        stage,
        FundingSource.DIRECT_SERVICE,
        invoke,
        framework_cost=None,
        token_counts=None,
        openrouter=False,
    )


# --- The forecasting-tools seam -------------------------------------------------------

_ORIGINAL_DIRECT_CALL = GeneralLlm._mockable_direct_call_to_model


async def _funded_direct_call(self: GeneralLlm, prompt: Any) -> Any:
    if _FUNDING is None:
        return await _ORIGINAL_DIRECT_CALL(self, prompt)

    def framework_cost(response: Any) -> float:
        return float(getattr(response, "cost", 0) or 0)

    def token_counts(response: Any) -> tuple[int | None, int | None]:
        return getattr(response, "prompt_tokens_used", None), getattr(response, "completion_tokens_used", None)

    if not self.model.startswith("openrouter/"):
        return await _attempt(
            _FUNDING,
            self.model,
            stage_of(self),
            FundingSource.DIRECT_SERVICE,
            lambda: _ORIGINAL_DIRECT_CALL(self, prompt),
            framework_cost=framework_cost,
            token_counts=token_counts,
            openrouter=False,
        )

    async def with_key(key: str) -> Any:
        # A shallow copy with its own kwargs: the roster clients are shared by
        # concurrent questions, so the original's key must never be swapped in place.
        rekeyed = copy.copy(self)
        rekeyed.litellm_kwargs = {**self.litellm_kwargs, "api_key": key}
        return await _ORIGINAL_DIRECT_CALL(rekeyed, prompt)

    return await funded_openrouter_call(
        self.model,
        stage_of(self),
        with_key=with_key,
        own_key=self.litellm_kwargs.get("api_key"),
        own_call=lambda: _ORIGINAL_DIRECT_CALL(self, prompt),
        framework_cost=framework_cost,
        token_counts=token_counts,
        offer_grant=getattr(self, "offer_grant", True),
    )


def apply_funding_seam() -> None:
    """Route every forecasting-tools model call through ``funded_openrouter_call``.

    Patched at ``_mockable_direct_call_to_model`` -- one level below the framework's
    retry loop -- so each retry is its own recorded attempt and the key choice is made
    per attempt. With no funding configured the patch is a pass-through.
    """
    current = GeneralLlm._mockable_direct_call_to_model
    if current is not _funded_direct_call:
        global _ORIGINAL_DIRECT_CALL
        _ORIGINAL_DIRECT_CALL = current
        GeneralLlm._mockable_direct_call_to_model = _funded_direct_call  # type: ignore[method-assign]


# --- Starting a run --------------------------------------------------------------------


def check_grant_key_cap(api_key: str, run_limit: Decimal) -> Decimal:
    """Require the grant key's own OpenRouter cap to bound this run, BYOK included.

    A read-only GET of the key's settings; no model request is made and nothing about
    the key is logged. The local ledger is accounting; this cap is what OpenRouter
    itself enforces, and a request past it is refused and moves to the ordinary keys.
    """
    try:
        response = httpx.get(
            "https://openrouter.ai/api/v1/key",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
            follow_redirects=False,
        )
        response.raise_for_status()
        data = response.json()["data"]
        limit = Decimal(str(data["limit"]))
        remaining = Decimal(str(data["limit_remaining"]))
        includes_byok = data.get("include_byok_in_limit") is True
        # A cap that resets (daily, weekly, monthly) bounds one period, not the grant.
        resets = data.get("limit_reset") is not None
    except (httpx.HTTPError, ValueError, TypeError, KeyError, AttributeError, InvalidOperation) as exc:
        raise ValueError(f"Grant key cap check failed ({type(exc).__name__}); no run started") from None
    if resets or not (
        includes_byok and limit.is_finite() and remaining.is_finite() and 0 < limit <= GRANT_ALLOCATION_USD
    ):
        raise ValueError(
            f"The grant key needs a non-resetting OpenRouter cap at or below ${GRANT_ALLOCATION_USD} that includes BYOK"
        )
    if run_limit + GRANT_RESERVE_USD > remaining:
        raise ValueError(
            f"The grant key has ${remaining} left, which cannot cover ${run_limit} plus the ${GRANT_RESERVE_USD} reserve"
        )
    return remaining


def open_run_funding(
    run_id: str,
    *,
    grant_budget_usd: Decimal | None = None,
    grant_reserve_per_call_usd: Decimal = DEFAULT_GRANT_RESERVE_PER_CALL_USD,
) -> Funding | None:
    """Open the spend ledger for a run and, when a budget is given, enable the grant.

    Without a grant, a ledger that cannot be opened turns accounting off for the run
    and the forecast goes ahead. With a grant, every precondition must hold: the key
    is set and distinct from the ordinary keys, its OpenRouter cap bounds the run, and
    the ledger is writable. Raises ValueError otherwise.
    """
    path = os.getenv(SPEND_LEDGER_ENV, "").strip() or DEFAULT_SPEND_LEDGER
    grant_key: str | None = None
    if grant_budget_usd is not None:
        if not grant_budget_usd.is_finite() or grant_budget_usd <= 0:
            raise ValueError("The grant budget must be a positive amount")
        if grant_budget_usd > GRANT_ALLOCATION_USD - GRANT_RESERVE_USD:
            raise ValueError("The grant budget exceeds the local grant allocation after its reserve")
        if not grant_reserve_per_call_usd.is_finite() or not 0 < grant_reserve_per_call_usd <= grant_budget_usd:
            raise ValueError("The per-call grant reservation must be positive and within the grant budget")
        grant_key = os.getenv(GRANT_KEY_ENV, "").strip() or None
        if grant_key is None:
            raise ValueError(f"{GRANT_KEY_ENV} is required to spend a grant budget")
        if grant_key in {_env_key(OPENROUTER_API_KEY_ENV), _env_key(OAI_ANTH_OPENROUTER_KEY_ENV)}:
            raise ValueError("The grant key must be distinct from the ordinary OpenRouter keys")
        check_grant_key_cap(grant_key, grant_budget_usd)
    try:
        ledger = SpendLedger(Path(path))
        reused = ledger.has_run(run_id)
        if not reused:
            ledger.start_run(run_id, grant_budget_usd or Decimal(0))
    except (sqlite3.Error, OSError) as exc:
        if grant_key is not None:
            raise ValueError(f"The spend ledger at {path} is not writable ({type(exc).__name__})") from None
        logger.warning("SPEND_LEDGER_UNAVAILABLE path=%s (%s); this run is not accounted", path, type(exc).__name__)
        return None
    if reused:
        # Two runs under one ID would share a budget and a SPEND_SUMMARY.
        raise ValueError(f"Run ID {run_id} is already in the spend ledger at {path}; choose another")
    logger.info(
        "SPEND_LEDGER run=%s path=%s grant_budget_usd=%s grant_reserve_per_call_usd=%s",
        run_id,
        path,
        grant_budget_usd or 0,
        grant_reserve_per_call_usd if grant_key else 0,
    )
    return Funding(run_id, ledger, grant_key, grant_reserve_per_call_usd)


def log_spend_summary(funding: Funding) -> None:
    """One line per source that paid for anything in this run, for the archived log."""
    try:
        for source in funding.ledger.sources(funding.run_id):
            totals = funding.ledger.summary(funding.run_id, source=source)
            logger.info(
                # A telemetry marker: see sibyl/telemetry/markers.py before changing it.
                "SPEND_SUMMARY: run=%s source=%s attempts=%s unconfirmed=%s confirmed_usd=%s estimated_usd=%s "
                "uncertain_usd=%s",
                funding.run_id,
                source,
                totals["attempts"],
                totals["unconfirmed_attempts"],
                totals["confirmed_usd"],
                totals["estimated_usd"],
                totals["uncertain_usd"],
            )
    except (sqlite3.Error, OSError, ArithmeticError) as exc:
        # Runs in the CLI's finally: it must never replace the run's own outcome.
        logger.warning("SPEND_SUMMARY unavailable (%s)", type(exc).__name__)
