"""Centralised model configuration for SibylBot.

Keeping these objects in a single module avoids merge-conflicts and makes it
possible to tweak/benchmark models without touching application code.
"""

import logging
import os
from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from typing import Any

from forecasting_tools import GeneralLlm

from sibyl.config.settings import FINANCIAL_CLASSIFIER_MODEL, MODEL_TIER_ENV
from sibyl.platform.key_fallback import build_llm_with_openrouter_fallback

logger = logging.getLogger(__name__)


class ModelTier(StrEnum):
    """Which roster a run uses.

    FREE is every-slot-on-an-OpenRouter-:free-slug and cannot spend. PROD is the paid
    roster below. The distinction is a tier rather than a per-slot toggle because a
    half-free run is the worst of both: it bills, and its output is not comparable to
    either arm.
    """

    FREE = "free"
    PROD = "prod"


# Research providers the FREE tier force-disables regardless of their env flag.
#
# The distinction is not cost, it is substitutability. Every provider here is a PAID
# SEARCH TOOL whose value IS the tool -- AskNews's corpus, OpenAI's and Anthropic's
# server-side web search, Google's grounding, and the two gap-fill loops that drive
# them. Swapping the model underneath does not make them free, it makes them broken:
# there is no `:free` slug that comes with a search index attached.
#
# The providers NOT listed here (financial data, prediction markets, resolution source,
# reference class, time-series anchor) are the opposite shape: free or public data
# sources with an LLM stage bolted on. Pointing that stage at the free utility model
# makes the whole provider free, so they stay available in the free tier.
FREE_TIER_DISABLED_PROVIDERS: frozenset[str] = frozenset(
    {
        "NATIVE_SEARCH_ENABLED",
        "GEMINI_SEARCH_ENABLED",
        "ANTHROPIC_SEARCH_ENABLED",
        "GAP_FILL_ENABLED",
        "GAP_FILL_V2_ENABLED",
    }
)


def active_tier() -> ModelTier:
    """The tier for this run, from ``SIBYL_MODEL_TIER``. Defaults to FREE.

    FREE is the default on purpose, and it is the one deliberate behaviour change this
    phase makes to an existing entry point. A local run that forgets the flag should be
    unable to spend; the prod workflows say ``SIBYL_MODEL_TIER: 'prod'`` explicitly, so
    the expensive path is the one that has to be asked for. Read at call time, so a test
    can monkeypatch it without reloading the module.

    An unrecognised value is a configuration error, not a reason to guess. Falling back
    to FREE would silently downgrade a prod run to free models and publish the result to
    a scored tournament; falling back to PROD would silently spend. Raise instead.
    """
    # os.environ, not os.getenv: several tests patch ``os.getenv`` process-wide to stub a
    # handful of provider flags, and a patched getenv answers "" for every name it was
    # not told about -- which would silently drop those tests onto the free tier and
    # take their providers with it. Reading the mapping directly is unaffected.
    raw = os.environ.get(MODEL_TIER_ENV, "").strip().lower()
    if not raw:
        return ModelTier.FREE
    try:
        return ModelTier(raw)
    except ValueError:
        valid = ", ".join(sorted(t.value for t in ModelTier))
        raise ValueError(f"{MODEL_TIER_ENV}={raw!r} is not a known model tier (expected one of: {valid})") from None


__all__ = [
    "ModelTier",
    "Roster",
    "active_tier",
    "get_roster",
    "FREE_FORECASTER_MODELS",
    "FREE_UTILITY_MODEL",
    "FORECASTER_LLMS",
    "FORECASTER_MODEL_NAMES",
    "SUMMARIZER_LLM",
    "PARSER_LLM",
    "RESEARCHER_LLM",
    "STACKER_LLM",
    "STACKER_FALLBACK_LLM",
    "DISAGREEMENT_ANALYZER_LLM",
    "MARKET_QUERY_AUTHOR_LLM_CONFIG",
    "MARKET_RANKER_LLM_CONFIG",
]
# Reasoning models ignore (or degrade under) explicit sampling params, so we
# defer to provider defaults. temperature=None is explicit but redundant on
# ft 0.2.92, whose GeneralLlm ctor already defaults temperature to None (0.2.54
# injected 0 when the arg was omitted); top_p flows via **kwargs and is never set.
REASONING_MODEL_CONFIG: dict[str, Any] = {
    "temperature": None,
    "max_tokens": 64_000,  # Prevent truncation; all current forecasters/stackers support 64k output
    "stream": False,
    "timeout": 480,
    "allowed_tries": 3,
}
# Low-effort utility slots (parser, summarizer, analyzer). Same sampling-param
# rationale as REASONING_MODEL_CONFIG: temperature=None defers to provider
# defaults (redundant on ft 0.2.92, whose ctor default is already None); top_p
# left unset.
UTILITY_MODEL_CONFIG: dict[str, Any] = {
    "temperature": None,
    "max_tokens": 32_000,
    "stream": False,
    "timeout": 300,
    "allowed_tries": 3,
}
ACCEPTABLE_QUANTS = [
    "fp8",
    "fp16",
    "bf16",
    "fp32",
    "unknown",
]

# Per-instance allowed_tries=1 override (Round-2): forecaster .invoke is wrapped
# in the broad retry gated on TRANSIENT_RETRY_MAX_ELAPSED_S (forecaster_runners.py)
# so we can impose the universal "never retry a slow failure" deadline-safety rule
# that forecasting-tools' un-gated tenacity cannot. Spread per-instance (NOT by mutating
# REASONING_MODEL_CONFIG) so PARSER_LLM / STACKER configs are untouched.
_FORECASTER_CONFIG = {**REASONING_MODEL_CONFIG, "allowed_tries": 1}

FORECASTER_LLMS: list[GeneralLlm] = [
    # 2026-07-20: forecaster roster dropped from 6 to a 3-member latest-per-vendor
    # triple (1 OpenAI / 1 Anthropic / 1 Google). This is the SECOND roster change
    # on 2026-07-20 and supersedes the morning fable-5 → opus-4.7 swap (7a76df6) as
    # the config-era boundary for residual analysis. Removed: gpt-5.5,
    # claude-opus-4.7, grok-4.5. Two adversarially-verified analyses
    # (bottega:scratch/ensemble_3member_audit_2026-07-20/ +
    # bottega:scratch/ensemble_power_model_2026-07-20/) found the triple non-inferior on
    # binary/MC and only a fragile numeric lean toward the full roster (+3.24,
    # 95% CI [-2.5, +9.1], P(loss>1pt/Q)=0.80, driven by 2 questions) — accepted as
    # a ship-and-watch bet; see bottega:FUTURE.md "Frozen-triple numeric watch". Dropping
    # grok (x-ai, 404s on the primary key) also ends routine fallback-key forecaster
    # spend: only the gemini-3.1-pro-preview fallback-key PIN bills
    # OPENROUTER_API_KEY now; the other two slots route via the primary key.
    # (Dates anchor config eras for residual analysis.)
    #
    # OpenAI flagship (5.6 series). 2026-07-20: effort xhigh -> high. The
    # reasoning-effort audit (bottega:scratch/reasoning_effort_audit_2026-07-20/) found
    # default->high clearly worth it but high->xhigh UNMEASURED; sol is ~70% of
    # forecaster reasoning spend, so we stop paying the unmeasured premium on the
    # dominant-cost slot. opus-4.8 keeps xhigh below as the remaining premium bet
    # (bottega:FUTURE.md "Price the high->xhigh reasoning-effort premium"). (2026-07-15
    # had bumped this high -> xhigh.) Live-verified: OpenRouter's effort enum is
    # max|xhigh|high|medium|low|minimal|none and this model accepts high (bogus
    # values 400). NOTE: "max" is Anthropic-only — OpenAI's ceiling is xhigh and
    # OpenAI rejects max upstream even though OpenRouter's enum validation admits it.
    build_llm_with_openrouter_fallback(
        model="openrouter/openai/gpt-5.6-sol",
        reasoning={"effort": "high"},
        **_FORECASTER_CONFIG,
    ),
    # Anthropic slot. 2026-07-15: enabled:True (provider-default adaptive thinking)
    # -> explicit effort=xhigh. Anthropic also exposes "max" one tier above xhigh —
    # held back deliberately for latency: unbounded adaptive thinking caused silent
    # FORECASTER_SOFT_DEADLINE stalls on the retired opus-4.6 slot, e.g. Q14333 on
    # 2026-05-07.
    build_llm_with_openrouter_fallback(
        # 2026-08-31: ROUTING moved from OpenRouter to the Anthropic API direct
        # ("openrouter/anthropic/claude-opus-5" -> "anthropic/claude-opus-5"). The model
        # is unchanged; only the transport is. The primary Metaculus OpenRouter key is
        # exhausted until next season and the fallback OpenRouter balance could not even
        # cover a 64k max_tokens reservation (2026-08-31 run: three 402s, 0/3 forecasters),
        # so the OpenRouter hop bought nothing but a middleman on a key with no credit.
        #
        # `build_llm_with_openrouter_fallback` returns a PLAIN GeneralLlm for any slug
        # that does not start with "openrouter/" (see should_route_via_primary_key), so
        # this correctly bypasses the primary-key wrapper and litellm reads
        # ANTHROPIC_API_KEY from the environment. The call is kept (rather than
        # constructing GeneralLlm directly) so the roster stays uniform and re-routing
        # back through OpenRouter is a one-token change.
        #
        # TWO PARAMETER CHANGES ARE MANDATORY, not cosmetic:
        #   * `reasoning={"effort": ...}` -> `reasoning_effort=...`. The dict form is an
        #     OpenRouter passthrough. litellm's Anthropic transform lists
        #     `reasoning_effort` (not `reasoning`) in get_supported_openai_params, and
        #     forecasting_tools sets `litellm.drop_params = True` process-globally — so
        #     the dict form is SILENTLY DISCARDED, not rejected. The slot would have run
        #     at default effort with no error anywhere in the logs.
        #   * `extra_body={"verbosity": "high"}` is dropped entirely. `verbosity` is an
        #     OpenRouter-only knob with no Anthropic API equivalent; same silent-drop
        #     path.
        # Verified against the litellm model map: claude-opus-5 advertises
        # supports_output_config + supports_xhigh_reasoning_effort, so effort=xhigh
        # survives the translation to output_config.effort.
        #
        # 2026-08-27: claude-opus-4.8 -> claude-opus-5. Anthropic's current flagship,
        # released 2026-07-24, at the SAME OpenRouter list price ($5/M in, $25/M out) and
        # the same 1M context / 128k max-output envelope. Verified against the live
        # /api/v1/models catalogue, including that it advertises the same
        # reasoning/reasoning_effort parameter support, so effort=xhigh carries over.
        # NOT verified by a live call — no forecast has been made with it. Fire one
        # `test_bot_basic` dispatch before trusting a scored run.
        # This starts a new config era for residual analysis (merge date, per AGENTS.md).
        model="anthropic/claude-opus-5",
        reasoning_effort="xhigh",
        **_FORECASTER_CONFIG,
    ),
    # Google slot. No explicit reasoning-effort kwarg — gemini-3.1-pro-preview has
    # no xhigh tier and uses provider defaults. PINNED to the fallback
    # OPENROUTER_API_KEY via the OPENROUTER_PRIMARY_KEY_BLOCKED_MODELS blocklist in
    # fallback_openrouter (the primary key routes it through a free-tier Google
    # AI Studio BYOK integration with quota 0, so it would 429 there); see the
    # OPENROUTER_PRIMARY_KEY_BLOCKED_MODELS default, pending the Metaculus-side BYOK fix.
    build_llm_with_openrouter_fallback(
        model="openrouter/google/gemini-3.1-pro-preview",
        **_FORECASTER_CONFIG,
    ),
]


def _forecaster_display_name(llm: GeneralLlm) -> str:
    """Short label for a forecaster (e.g. 'claude-opus-4.8') — strips the 'openrouter/<provider>/' prefix.

    Used by performance_analysis.parsing to map 'Forecaster N' labels in bot comments
    back to a model name without having to hand-maintain a parallel list.
    """
    return llm.model.rsplit("/", 1)[-1]


FORECASTER_MODEL_NAMES: list[str] = [_forecaster_display_name(llm) for llm in FORECASTER_LLMS]

# Summarizer: compresses raw AskNews article markdown into an analyst briefing
# (AskNews-only; all other providers already emit LLM prose). sol → terra
# 2026-07-18 operator decision: AskNews is an auxiliary/augmenting source
# (content audit: 16% unique content vs native-search 54% / gap-fill 59%), so
# the absolute-frontier tier isn't warranted. The role audit
# (bottega:scratch/research_role_audit_2026-07-17/) had sol 1st but verdict "MARGINAL
# EDGE" with terra 2nd (one attribution blur, no fabrications), and 4/5 briefing
# failures in the AskNews quality audit (bottega:scratch/asknews_quality_audit_2026-07-18/)
# were prompt-era (mini summarizer + missing no-forecast rule), not model-tier.
# Terra: −43% cost, ~50s vs ~118s wall. Effort stays low (latency).
# allowed_tries=1 (Round-2): the summarizer invoke is wrapped in the broad,
# elapsed-gated retry (orchestrator._summarize_asknews) to impose the universal
# "never retry a slow failure" deadline rule. Per-instance override so PARSER_LLM (which
# also uses UTILITY_MODEL_CONFIG) keeps its allowed_tries=3.
SUMMARIZER_LLM: GeneralLlm = build_llm_with_openrouter_fallback(
    "openrouter/openai/gpt-5.6-terra",
    # 2026-08-27: effort low -> high. This is the highest-LEVERAGE call in the pipeline and
    # it was running at the lowest effort tier. It produces the single briefing that ALL
    # THREE forecasters read, so one call conditions three frontier judgments and there is
    # no redundancy to absorb its mistakes.
    #
    # The measured case, not a hunch. The 2026-07-17 content audit found the AskNews section
    # is 44% of bundle tokens, 57% padding, and stale-or-directionally-wrong in 5 of 10
    # judged questions while SMALLER sections had the right answer. The 2026-07-18 follow-up
    # attributed the failures: SUMMARIZE implicated in 5/5 (primary in 3), and "in four of
    # five cases the briefing visibly contains the fresh or correct material it then
    # mishandles". The raw articles were fresh — 48% of cited dates within 30 days. The
    # failure is salience inversion: stale on-topic facts outranking fresh adjacent ones.
    #
    # Salience ranking under a length budget is a reasoning task, which is what the effort
    # tier buys. Cost is bounded: this is ~$0.25 and ~7% of per-question spend, against a
    # forecaster fan-out that is 64%.
    #
    # UNVALIDATED. The 2026-07-18 prompt reform (R1-R5) targeted the same failures and was
    # never measured either, so effort and prompt are now two unmeasured changes stacked on
    # one slot. Measure them together or not at all.
    reasoning={"effort": "high"},
    **{**UTILITY_MODEL_CONFIG, "allowed_tries": 1},
)
# Parser: deterministic extraction of percentiles/JSON from rationales — a
# capability-saturated task, so it rides the cheapest tier that saturates it and
# keeps allowed_tries=3 for robustness. mini → luna 2026-08-03: the per-token
# comparison that used to favor mini inverted. Luna is $0.20/$1.20 vs mini's
# $0.75/$4.50 per 1M, so the newer model is also the ~3.75x cheaper one. (The
# models API showed $0.10/$0.60 behind a "50% off" badge on 2026-08-03; a live
# call on 2026-08-04 billed at double that, so the promo does not apply on this
# route — see the ranker cost comment below. The swap still wins, by less.)
# Effort unchanged at low.
PARSER_LLM: GeneralLlm = build_llm_with_openrouter_fallback(
    "openrouter/openai/gpt-5.6-luna",
    reasoning={"effort": "low"},
    **UTILITY_MODEL_CONFIG,
)
# Researcher slot in the forecasting-tools LLM config dict. Effectively dead
# code in our pipeline — we use research providers (AskNews/Gemini/native_search)
# rather than the framework's researcher path — but the slot must be populated
# to avoid silent framework defaults. Aliasing to SUMMARIZER_LLM rather than
# constructing a duplicate config: same model, same effort, same job tier, no
# reason to maintain two parallel definitions.
RESEARCHER_LLM = SUMMARIZER_LLM

# Stacker meta-model for conditional stacking (invoked only on high-disagreement questions).
#
# allowed_tries=1: a single attempt at REASONING_MODEL_CONFIG's timeout, no
# retries. The outer STACKER_SOFT_DEADLINE catches wholly stuck calls; on failure
# we fall back to STACKER_FALLBACK_LLM rather than burning two more full-timeout
# attempts against the same Anthropic API that just stalled. Retrying against the same
# provider after a stall rarely succeeds (we're almost certainly re-rolling a
# dice with the same distribution), and the budget is better spent on a
# different-provider fallback.
STACKER_LLM: GeneralLlm = build_llm_with_openrouter_fallback(
    # 2026-07-20: fable-5 → opus-4.8 (fable-5 pulled from BOTH roles after
    # content=None failures in the 2026-07-19 test_bot run — see the forecaster-slot
    # comment above + bottega:FUTURE.md). Stacking is prod-disabled, so this is
    # backtest/ablation-only exposure today. opus-4.8 uses effort-based adaptive
    # thinking, not a max_tokens budget. Live-verified OpenRouter effort enum:
    # none/minimal/low/medium/high/xhigh/max. effort=xhigh matches the opus-4.8
    # forecaster slot; "max" (one tier above xhigh) is deliberately held back for
    # latency — the stacker runs under STACKER_SOFT_DEADLINE.
    # 2026-08-31: routed direct alongside the forecaster slot. NOTE the model id
    # changes SHAPE, not identity: Anthropic's own ids are dash-separated
    # ("claude-opus-4-8"), while OpenRouter uses dots ("claude-opus-4.8"). The dotted
    # form 404s against the Anthropic API. Same reasoning_effort / verbosity treatment
    # as the forecaster slot above — see that comment for why the dict form is unsafe.
    "anthropic/claude-opus-4-8",
    reasoning_effort="xhigh",
    **{**REASONING_MODEL_CONFIG, "allowed_tries": 1},
)

# Fallback stacker used when the primary stacker times out or errors.
# Reasoning slot → strongest OpenAI tier (gpt-5.6-sol) at high effort;
# deliberately cross-provider from the Anthropic Fable primary so an Anthropic
# stall doesn't take both attempts down. Tighter timeout and single try since
# we're already running late on the critical path by the time this fires.
# Stays at high (not xhigh) for that same reason — the 2026-07-15 xhigh bump
# covers the primary stacker and forecaster slots, not this tighter-budget path.
STACKER_FALLBACK_LLM: GeneralLlm = build_llm_with_openrouter_fallback(
    "openrouter/openai/gpt-5.6-sol",
    reasoning={"effort": "high"},
    **{**REASONING_MODEL_CONFIG, "allowed_tries": 1, "timeout": 300},
)

# --- The prediction-market provider's two LLM stages ---
#
# Both are RAW DICTS rather than built GeneralLlm singletons, unlike PARSER_LLM and friends:
# the provider is gated OFF by default, so paying construction cost at import would be waste,
# and the tests patch `build_llm_with_openrouter_fallback` at the provider's one invocation
# helper. Both route `openrouter/openai/...` through that wrapper, which tries the primary
# Metaculus key first and falls back to the fallback key on credential / credit / route errors
# — so prod spend lands on OAI_ANTH_OPENROUTER_KEY.
#
# `allowed_tries=1` is required, not decorative: the repo's elapsed-gated `llm_retry` wrapper
# (prediction_market._invoke_market_llm) is the SOLE retry layer, and leaving this unpinned
# inherits forecasting-tools' default of 2 with an UN-GATED `random.uniform(5, 10)` tenacity
# sleep — a large slice of PREDICTION_MARKET_TIMEOUT spent sleeping blind, which is exactly
# what llm_retry exists to eliminate. `temperature=None` defers reasoning models to provider
# defaults (redundant on ft 0.2.92, whose ctor default is already None); top_p left unset. Each
# litellm `timeout` sits ABOVE its elapsed-gated wall cap in constants.py, so the wall is the
# binding bound.
#
# Luna is the cheapest tier that saturates both tasks. The real rate on this route is $0.20/M in
# and $1.20/M out — TWICE the $0.10/$0.60 the bake-off read off the models API on 2026-08-03,
# where a "50% off" badge was displayed that has since lapsed or never applied here. A live
# ranking call reconciled the true rates to 7 significant figures against OpenRouter's own
# `upstream_inference_cost` (26,250 in / 685 out / a 25% cache-WRITE surcharge on the input,
# `bottega:scratch/market_port_2026-08-04/QA_DRY_RUN.md`), so this is measured rather than quoted.
#
# MEASURED cost per question: ranker $0.0074 (26k in at the median post-enrichment,
# full-PredictIt shape + ~685 out, cache write included); author ~1.4k in + ~300 out ≈ $0.0005.
# The two keyword calls they replace measured ~170 tok in / ~50 out ≈ $0.0001, so net new is
# ≈ +$0.008 per question — under a cent per run at the prod shape of 1-2 questions, and ~$0.24
# of ranker spend across a 30-question tournament run. The earlier ~$0.003-0.004 arithmetic in
# the port plan understated by 2.4x purely because of the promo price; the token shapes were
# right. This traffic is ~97% input, so the input rate is the whole cost.

# Prediction-market RANKER: one call per question over the whole ~380-440-candidate pool,
# emitting up to 8 ranked rows with a relation tier and a one-phrase label. Measured completion
# averages 589 tokens including reasoning, max 1,042 (bottega:scratch/bakeoff_run_2026-08-03/results/
# RANKED_ARM_RESULTS.md). max_tokens sits ~3x above that max because a TRUNCATED ranking is a
# fail-open — the whole ranking is lost, not just its tail — and luna's output tokens are cheap.
MARKET_RANKER_LLM_CONFIG: dict = {
    "model": "openrouter/openai/gpt-5.6-luna",
    "temperature": None,
    "max_tokens": 3000,
    "reasoning_effort": "low",
    "timeout": 90,
    "allowed_tries": 1,
}

# Prediction-market QUERY AUTHOR: one call per question emitting the domain vocabulary the
# question's own tokens cannot reach (up to 8 synonyms + 3 framings). Its output is ADDITIVE to
# a deterministic query set, so its failure costs recall nothing. Measured completion max 588
# tokens including reasoning; max_tokens sits ~2.5x above that.
MARKET_QUERY_AUTHOR_LLM_CONFIG: dict = {
    "model": "openrouter/openai/gpt-5.6-luna",
    "temperature": None,
    "max_tokens": 1500,
    "reasoning_effort": "low",
    "timeout": 45,
    "allowed_tries": 1,
}


# Tier-B auxiliary: read-and-synthesize work that needs taste but not deep
# reasoning. Identifies the crux of forecaster disagreement; output text seeds
# the targeted-search query downstream. Runs under CRUX_SOFT_DEADLINE;
# effort deliberately low since 2026-05-20 for latency — the tier was upgraded
# instead (smarter-model-at-lower-effort beats more effort on a smaller model).
# 2026-07-17: sol→terra per the role audit; terra 2nd (sol 3rd) at −49% cost;
# the role fires rarely (stacking disabled in prod).
# allowed_tries=1 (Round-2): the crux-analyzer invoke is wrapped in the broad,
# elapsed-gated retry (targeted.extract_disagreement_crux) to impose the universal
# "never retry a slow failure" deadline rule on the conditional-stacking critical path.
# Per-instance override so PARSER_LLM keeps its allowed_tries=3.
DISAGREEMENT_ANALYZER_LLM: GeneralLlm = build_llm_with_openrouter_fallback(
    "openrouter/openai/gpt-5.6-terra",
    reasoning={"effort": "low"},
    **{**UTILITY_MODEL_CONFIG, "allowed_tries": 1},
)


# ---------------------------------------------------------------------------
# Rosters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Roster:
    """Every model slot a run needs, resolved for one tier.

    A dataclass rather than a dict so a missing slot is a construction error here
    instead of a KeyError deep in a provider halfway through a run, and so adding a
    slot forces both tiers to answer for it.
    """

    forecasters: tuple[GeneralLlm, ...]
    summarizer: GeneralLlm
    parser: GeneralLlm
    researcher: GeneralLlm
    stacker: GeneralLlm
    stacker_fallback: GeneralLlm
    analyzer: GeneralLlm
    market_ranker_config: dict[str, Any]
    market_query_author_config: dict[str, Any]
    # A model NAME rather than a built LLM: financial_data builds its classifier inside
    # a provider factory with its own kwargs, and the provider is off by default, so
    # constructing one at import would be waste. The free tier only needs the slug.
    financial_classifier_model: str


# The free lineup. Every slug ends in `:free`, which is the property the tests assert --
# not the specific models, which will rotate as OpenRouter's free offerings change.
#
# Built as plain GeneralLlm on OPENROUTER_API_KEY, deliberately NOT through
# build_llm_with_openrouter_fallback: most `:free` variants are served only by providers
# outside OPENROUTER_PRIMARY_KEY_PROVIDERS, so the primary key answers 404 "no allowed
# providers", which wastes a fallback attempt and bumps the alerting counter on every
# call. eval/ablation/forecaster_lineup.py's docstring has the long version.
FREE_FORECASTER_MODELS: tuple[str, ...] = (
    "openrouter/google/gemma-4-31b-it:free",
    "openrouter/nvidia/nemotron-3-super-120b-a12b:free",
    "openrouter/qwen/qwen3-next-80b-a3b-instruct:free",
)
# One model for every non-forecaster slot. The utility work (summarize, parse, rank,
# analyze) is capability-saturated rather than reasoning-bound, and a single slug keeps
# the free tier's failure modes to one provider's rate limit instead of five.
FREE_UTILITY_MODEL: str = "openrouter/google/gemma-4-31b-it:free"


def _free_llm(model: str, config: dict[str, Any]) -> GeneralLlm:
    return GeneralLlm(model=model, **config)


def _free_roster() -> Roster:
    utility = _free_llm(FREE_UTILITY_MODEL, UTILITY_MODEL_CONFIG)
    return Roster(
        forecasters=tuple(_free_llm(model, _FORECASTER_CONFIG) for model in FREE_FORECASTER_MODELS),
        summarizer=utility,
        parser=utility,
        researcher=utility,
        stacker=utility,
        stacker_fallback=utility,
        analyzer=utility,
        market_ranker_config={**MARKET_RANKER_LLM_CONFIG, "model": FREE_UTILITY_MODEL},
        market_query_author_config={**MARKET_QUERY_AUTHOR_LLM_CONFIG, "model": FREE_UTILITY_MODEL},
        financial_classifier_model=FREE_UTILITY_MODEL,
    )


def _prod_roster() -> Roster:
    """The paid roster, unchanged. Every value is the module constant it always was."""
    return Roster(
        forecasters=tuple(FORECASTER_LLMS),
        summarizer=SUMMARIZER_LLM,
        parser=PARSER_LLM,
        researcher=RESEARCHER_LLM,
        stacker=STACKER_LLM,
        stacker_fallback=STACKER_FALLBACK_LLM,
        analyzer=DISAGREEMENT_ANALYZER_LLM,
        market_ranker_config=dict(MARKET_RANKER_LLM_CONFIG),
        market_query_author_config=dict(MARKET_QUERY_AUTHOR_LLM_CONFIG),
        financial_classifier_model=FINANCIAL_CLASSIFIER_MODEL,
    )


@cache
def get_roster(tier: ModelTier) -> Roster:
    """The roster for ``tier``. Cached, because a roster is immutable per process."""
    return _prod_roster() if tier is ModelTier.PROD else _free_roster()
