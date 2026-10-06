"""Explicit grant-key OpenRouter clients with per-attempt local accounting."""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from forecasting_tools import GeneralLlm
from forecasting_tools.ai_models.ai_utils.response_types import TextTokenCostResponse

from sibyl.platform.experiment_funding import (
    BudgetExceeded,
    GrantExecutionError,
    GrantSession,
    active_grant,
    active_grant_question,
)
from sibyl.platform.experiment_usage import capture_grant_usage, resolve_grant_cost

logger = logging.getLogger(__name__)


class GrantCostUnknown(GrantExecutionError):
    """Provider returned a response whose bill could not be established."""


class GrantCallError(GrantExecutionError):
    """A failed request may have incurred spend; its reservation is retained."""


class FundedGeneralLlm(GeneralLlm):
    def __init__(self, model: str, *, session: GrantSession, stage: str = "model", **kwargs: Any) -> None:
        if not model.startswith("openrouter/"):
            raise ValueError("Experiment grant permits OpenRouter routes only")
        if "api_key" in kwargs:
            raise ValueError("Grant client key is selected by the funding session")
        self._grant_session = session
        self._grant_stage = stage
        kwargs.pop("allowed_tries", None)
        super().__init__(model=model, api_key=session.key, allowed_tries=1, **kwargs)

    def __repr__(self) -> str:
        return f"FundedGeneralLlm(model={self.model!r}, funding='experiment-grant')"

    async def _mockable_direct_call_to_model(self, prompt: Any) -> TextTokenCostResponse:
        session = self._grant_session
        attempt_id = uuid4().hex
        session.ledger.reserve(
            attempt_id,
            session.run_id,
            self.model,
            self._grant_stage,
            session.per_call_reserve_usd,
            question_id=active_grant_question(),
        )
        logger.info(
            "GRANT_CALL_START run=%s attempt=%s question=%s stage=%s model=%s reserve_usd=%s",
            session.run_id,
            attempt_id,
            active_grant_question(),
            self._grant_stage,
            self.model,
            session.per_call_reserve_usd,
        )
        try:
            with capture_grant_usage() as capture:
                response = await super()._mockable_direct_call_to_model(prompt)
        except BaseException as exc:
            status = getattr(exc, "status_code", None)
            status = status if isinstance(status, int) else None
            diagnostic = f"{type(exc).__name__} status={status}"
            session.ledger.mark_uncertain(attempt_id, evidence_source=diagnostic)
            logger.warning(
                "GRANT_CALL_FAILED attempt=%s model=%s %s; reservation retained", attempt_id, self.model, diagnostic
            )
            if isinstance(exc, Exception):
                raise GrantCallError(f"OpenRouter grant call failed ({diagnostic}); charge uncertain") from None
            raise
        resolution = await resolve_grant_cost(capture, response.cost, session.key)
        if resolution.billable is None:
            session.ledger.mark_uncertain(
                attempt_id,
                provider_request_id=resolution.provider_request_id,
                evidence_source=resolution.evidence,
            )
            logger.warning(
                "GRANT_COST_UNKNOWN attempt=%s model=%s generation_id=%s callback_received=%s",
                attempt_id,
                self.model,
                resolution.provider_request_id,
                capture.ready.is_set(),
            )
            raise GrantCostUnknown("OpenRouter grant call returned no cost evidence at all; reservation retained")
        session.ledger.settle(
            attempt_id,
            resolution.billable,
            response.prompt_tokens_used,
            response.completion_tokens_used,
            resolution.provider_request_id,
            # Confirmed only when the provider established the figure the budget holds.
            # An unconfirmed settlement keeps its full reservation, so an unseen BYOK
            # expense cannot make the run look free.
            certainty="confirmed" if resolution.confirmed else "estimated",
            evidence_source=resolution.evidence,
            openrouter_charge_usd=resolution.openrouter_charge,
            upstream_cost_usd=resolution.upstream_cost,
            is_byok=resolution.is_byok,
        )
        logger.info(
            "GRANT_CALL_SETTLED attempt=%s model=%s billable_usd=%s openrouter_usd=%s upstream_usd=%s "
            "byok=%s evidence=%s confirmed=%s generation_id=%s",
            attempt_id,
            self.model,
            resolution.billable,
            resolution.openrouter_charge,
            resolution.upstream_cost,
            resolution.is_byok,
            resolution.evidence,
            resolution.confirmed,
            resolution.provider_request_id,
        )
        # The call proceeds on the evidence it has. An unconfirmed cost is a
        # RECONCILIATION task, not a reason to discard a forecast that is already paid
        # for -- which is what the previous `evidence_source != "provider_generation"`
        # raise did, on an endpoint that is only eventually consistent.
        if not resolution.confirmed:
            logger.info(
                "GRANT_RECONCILE_PENDING attempt=%s generation_id=%s reservation held until reconciled",
                attempt_id,
                resolution.provider_request_id,
            )
        if resolution.billable > session.per_call_reserve_usd:
            raise BudgetExceeded("Provider charge exceeded the per-call reservation; execution stopped")
        return response


def build_grant_llm(model: str, *, stage: str = "model", **kwargs: Any) -> FundedGeneralLlm:
    session = active_grant()
    if session is None:
        raise RuntimeError("No active experiment grant session")
    supplied_key = kwargs.pop("api_key", None)
    if supplied_key is not None and supplied_key != session.key:
        raise ValueError("Grant client cannot use a different OpenRouter credential")
    return FundedGeneralLlm(model, session=session, stage=stage, **kwargs)


def grant_route_for(model: str) -> str:
    """The explicit transport mapping shared by preview and client construction."""
    if model == "anthropic/claude-opus-5":
        return "openrouter/anthropic/claude-opus-5"
    if model == "anthropic/claude-opus-4-8":
        return "openrouter/anthropic/claude-opus-4.8"
    if model.startswith("openrouter/"):
        return model
    raise ValueError(f"No explicit grant route configured for {model}")


def clone_for_grant(llm: GeneralLlm, *, stage: str) -> FundedGeneralLlm:
    """Clone a roster slot without mutating the cached production singleton.

    The Anthropic direct transport has an explicit OpenRouter equivalent. Its
    effort parameter needs the OpenRouter `reasoning` form; no other direct route
    is implicitly translated.
    """
    model = grant_route_for(llm.model)
    kwargs = dict(llm.litellm_kwargs)
    kwargs.pop("model", None)
    kwargs.pop("api_key", None)
    effort = kwargs.pop("reasoning_effort", None)
    if effort is not None:
        kwargs["reasoning"] = {"effort": effort}
    return build_grant_llm(
        model,
        stage=stage,
        # One invocation reserves one attempt. Internal retries would hide extra
        # charges; grant failures also stop the pipeline's outer retry loop.
        allowed_tries=1,
        responses_api=llm.responses_api,
        populate_citations=llm.populate_citations,
        **kwargs,
    )
