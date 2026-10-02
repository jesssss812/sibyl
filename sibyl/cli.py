import argparse
import asyncio

# ruff: noqa: F401
import logging
import os
import sys
from typing import Any, Literal

from forecasting_tools import MetaculusApi

from sibyl.aggregation.strategies import AggregationStrategy
from sibyl.config.models import (
    ModelTier,
    active_tier,
    get_roster,
)
from sibyl.config.settings import (
    BOT_TESTING_AREA_ID,
    CREDIT_ALERT_RESUME_DATE,
    DEFAULT_MAX_QUESTIONS_PER_RUN,
    FREE_TIER_MAX_QUESTIONS,
    METACULUS_CUP_ID,
    MODEL_TIER_ENV,
    PERSIST_RESEARCH_ENABLED_ENV,
    TEST_QUESTIONS_OVERRIDE_ENV,
    TOURNAMENT_ID,
    check_tournament_dates,
    credit_alerts_active,
    env_flag_enabled,
)
from sibyl.forecasting.bot import SibylBot
from sibyl.platform.fetch_hardening import apply_fetch_hardening
from sibyl.platform.key_fallback import (
    check_deprecation_alerts_and_exit,
    get_credit_key_fallback_count,
    get_fallback_key_count,
    get_primary_route_404_fallback_count,
)
from sibyl.platform.metaculus_preflight import verify_metaculus_api_identity
from sibyl.platform.publish_hardening import apply_publish_hardening
from sibyl.telemetry.credits import CreditTelemetry, get_probed_primary_key_state

logger = logging.getLogger(__name__)


# Canonical mode names, plus every spelling that has ever reached this CLI. The old
# names are kept as aliases rather than removed: they are in the workflow yamls, in
# docs/operations.md and in the operator's shell history, and a rename that breaks a
# dispatch someone types from memory buys nothing.
_MODE_ALIASES: dict[str, str] = {
    "tournament": "tournament",
    "minibench": "minibench",
    "cup": "cup",
    "metaculus_cup": "cup",
    "quarterly_cup": "cup",
    "test": "test",
    "test_questions": "test",
}

# Modes that publish to a SCORED tournament. Free models must never reach these: a free
# forecast is not a cheap version of a prod forecast, it is a different and much worse
# one, and publishing it to a scored tournament damages a public record that cannot be
# retracted. The sandbox mode (`test`) is absent because nothing there is scored.
_SCORED_MODES: frozenset[str] = frozenset({"tournament", "minibench", "cup"})


def _refuse_free_tier_on_scored_tournament(tier: ModelTier, run_mode: str) -> None:
    """Exit non-zero before anything is built if a free run would publish somewhere scored."""
    if tier is ModelTier.FREE and run_mode in _SCORED_MODES:
        logger.error(
            "Refusing to run mode %r on the free model tier: free models must never publish to a "
            "scored tournament. Re-run with --tier prod (or %s=prod) if that is what you meant, "
            "or use `python -m sibyl test` to exercise the pipeline against %s.",
            run_mode,
            MODEL_TIER_ENV,
            BOT_TESTING_AREA_ID,
        )
        sys.exit(2)


def _log_tier_banner(tier: ModelTier) -> None:
    """Say which roster is about to run, loudly, on both tiers.

    A prod run's banner is the last chance to notice that a dispatch is about to spend;
    a free run's is what stops a green log from being mistaken for a prod result later.
    """
    roster = get_roster(tier)
    logger.info(
        "MODEL TIER: %s — forecasters=%s, summarizer=%s, parser=%s",
        tier.value,
        [llm.model for llm in roster.forecasters],
        roster.summarizer.model,
        roster.parser.model,
    )
    if tier is ModelTier.PROD:
        logger.info("This run will SPEND on paid models and publish to Metaculus.")


def _default_max_questions(tier: ModelTier) -> int:
    """Free runs get a small cap; prod keeps the framework default this CLI has always used."""
    return FREE_TIER_MAX_QUESTIONS if tier is ModelTier.FREE else DEFAULT_MAX_QUESTIONS_PER_RUN


def main() -> None:
    """Command-line entry-point for running the SibylBot.

    This code was moved verbatim from the bottom of main.py so external behaviour
    (e.g. GitHub Actions invoking `python -m sibyl`) remains identical.  The only
    difference is that main.py now delegates to this function.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Suppress LiteLLM logging
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    # Forecaster module logs at DEBUG for full per-question tracing; the
    # openai-agents logger is noisy at INFO so pin it to ERROR. Configured here
    # (the runtime entry point) rather than at module import so test imports
    # and library consumers don't inherit these global level mutations.
    logging.getLogger("sibyl.forecasting.bot").setLevel(logging.DEBUG)
    logging.getLogger("openai.agents").setLevel(logging.ERROR)

    # Wrap MetaculusClient publish POSTs with timeout + retry. See
    # sibyl/platform/publish_hardening.py for rationale (a single hung POST
    # blocks the whole batch; we bound it tighter than the upstream default).
    apply_publish_hardening()

    # Wrap MetaculusClient question-list GET with bounded retry. See
    # sibyl/platform/fetch_hardening.py for rationale (a single transient
    # 403/429/5xx would otherwise kill the whole run).
    apply_fetch_hardening()

    # One-shot, unauthenticated identity check before any mode sends the token.
    # See sibyl/platform/metaculus_preflight.py (DNS-parking incident): aborts non-zero
    # if www.metaculus.com isn't answered by the real API, so we never leak
    # METACULUS_TOKEN to a hijacked host.
    verify_metaculus_api_identity()

    parser = argparse.ArgumentParser(
        # Without prog, argparse takes sys.argv[0] and the help header reads
        # "usage: __main__.py" -- an artifact of replacing `python main.py` with
        # `python -m sibyl`, and not a command anyone can type.
        prog="python -m sibyl",
        description="Run the sibyl forecasting bot",
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=sorted(_MODE_ALIASES),
        default=None,
        metavar="{tournament,minibench,cup,test}",
        help="Run mode (default: tournament)",
    )
    parser.add_argument(
        "--mode",
        dest="mode_flag",
        type=str,
        choices=sorted(_MODE_ALIASES),
        default=None,
        # Kept because the workflows, the docs and the operator's shell history all say
        # `--mode X`. The positional form is what the plan specifies; this is the same
        # argument spelled the old way, not a second concept.
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--tier",
        type=str,
        choices=[t.value for t in ModelTier],
        default=None,
        help=f"Model roster to run (default: ${MODEL_TIER_ENV} or 'free')",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Cap questions per run (default: 3 on the free tier, 10 otherwise)",
    )
    args = parser.parse_args()

    if args.mode is not None and args.mode_flag is not None and args.mode != args.mode_flag:
        parser.error(f"mode given twice and they disagree: {args.mode!r} (positional) vs {args.mode_flag!r} (--mode)")
    run_mode: str = _MODE_ALIASES[args.mode or args.mode_flag or "tournament"]

    # Tier resolution and the refusal gate come BEFORE any client is built, any key is
    # read and any question is fetched. A refusal that arrives after the bot is
    # constructed has already done the thing it was meant to prevent in every case where
    # construction is the expensive part.
    tier = ModelTier(args.tier) if args.tier else active_tier()
    _refuse_free_tier_on_scored_tournament(tier, run_mode)
    if run_mode == "tournament":
        from sibyl.config.settings import TOURNAMENT_ENABLED

        if not TOURNAMENT_ENABLED:
            parser.error("Tournament execution is disabled pending operator selection of a new competition")
        check_tournament_dates(logging.getLogger(__name__))
    _log_tier_banner(tier)
    max_questions = args.max_questions if args.max_questions is not None else _default_max_questions(tier)

    # Wire research persistence if enabled (production GHA runs set this env var)
    research_writer = None
    research_sink = None
    if env_flag_enabled(PERSIST_RESEARCH_ENABLED_ENV):
        from sibyl.research.persistence import (  # noqa: PLC0415
            ResearchPersistenceWriter,
        )

        research_writer = ResearchPersistenceWriter(
            run_mode=run_mode,
            tournament_id=str(TOURNAMENT_ID),
            run_id=os.environ.get("GITHUB_RUN_ID", "local"),
        )
        research_sink = research_writer.record

    # "forecasters" holds a list[GeneralLlm]; the helper slots hold single GeneralLlm
    # values. The parent ForecastBot.__init__ annotates llms as dict[str, str | GeneralLlm],
    # which (being invariant) cannot express the list value, so annotate the heterogeneous
    # dict as dict[str, Any]. prepare_llm_config consumes the "forecasters" list at runtime.
    roster = get_roster(tier)
    llms: dict[str, Any] = {
        "forecasters": list(roster.forecasters),
        "stacker": roster.stacker,
        "analyzer": roster.analyzer,
        "summarizer": roster.summarizer,
        "parser": roster.parser,
        "researcher": roster.researcher,
    }
    template_bot = SibylBot(
        research_reports_per_question=1,
        predictions_per_research_report=1,  # Ignored when 'forecasters' present
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        max_questions_per_run=max_questions,
        aggregation_strategy=AggregationStrategy.CONDITIONAL_STACKING,
        research_sink=research_sink,
        llms=llms,
    )

    # Credit-balance telemetry: CREDIT_BALANCE/CREDIT_SPEND marker lines land in
    # the run_logs/ artifact via the workflows' stdout tee, making per-run spend
    # on the shared primary key durably grep-able. The end fetch runs in a
    # finally so a crashed run still logs its spend; the floor check result is
    # consumed AFTER forecasting/publishing below (reminder signal, not abort).
    credit_telemetry = CreditTelemetry()
    credit_telemetry.log_start()
    primary_below_floor = False
    try:
        if run_mode == "tournament":
            check_tournament_dates(logging.getLogger(__name__))  # Warn/error if tournament dates are stale
            # to not risk explosive spend, we won't update preds
            template_bot.skip_previously_forecasted_questions = True
            forecast_reports = asyncio.run(template_bot.forecast_on_tournament(TOURNAMENT_ID, return_exceptions=True))
        elif run_mode == "minibench":
            # to not risk explosive spend, we won't update preds
            template_bot.skip_previously_forecasted_questions = True
            forecast_reports = asyncio.run(
                template_bot.forecast_on_tournament(MetaculusApi.CURRENT_MINIBENCH_ID, return_exceptions=True)
            )
        elif run_mode == "cup":
            # The metaculus cup is a good way to test the bot's performance on regularly open questions
            # to not risk explosive spend, we won't update preds
            template_bot.skip_previously_forecasted_questions = True
            forecast_reports = asyncio.run(
                template_bot.forecast_on_tournament(METACULUS_CUP_ID, return_exceptions=True)
            )
        elif run_mode == "test":
            # Obviously we re-run test predictions in order to test them.
            template_bot.skip_previously_forecasted_questions = False
            # Optional override: a comma/whitespace-separated list of Metaculus URLs to
            # forecast instead of the sandbox tournament. Kept working unchanged -- it is
            # how test_bot_basic pins a specific question, and how anyone reproduces a
            # single bad forecast.
            override_urls = os.environ.get(TEST_QUESTIONS_OVERRIDE_ENV, "").replace(",", " ").split()
            if override_urls:
                logger.info(
                    "TEST_QUESTIONS_OVERRIDE set: forecasting %d override question(s) instead of %s",
                    len(override_urls),
                    BOT_TESTING_AREA_ID,
                )
                questions = [MetaculusApi.get_question_by_url(url) for url in override_urls]
                forecast_reports = asyncio.run(template_bot.forecast_questions(questions, return_exceptions=True))
            else:
                # The open questions of the sandbox tournament, rather than the four
                # hard-coded evergreen URLs this used to carry. Those four were a fixed,
                # slowly-rotting sample that exercised only the by-URL fetch path; the
                # sandbox exercises the same tournament-fetch path prod uses, stays
                # current without anyone editing a list, and is not scored.
                forecast_reports = asyncio.run(
                    template_bot.forecast_on_tournament(BOT_TESTING_AREA_ID, return_exceptions=True)
                )
        else:
            raise ValueError(f"Invalid run mode: {run_mode}")
    finally:
        primary_below_floor = credit_telemetry.log_end_and_check_floor()
        # Flush inside the finally: records accumulate in memory for the whole run,
        # so an exception escaping asyncio.run (an OSError, the invalid-run-mode
        # ValueError above, a KeyboardInterrupt, the 300-minute timeout-minutes
        # SIGTERM) would otherwise discard every question's research — a 40-question
        # run that dies on the last question would archive nothing. The workflows'
        # upload step is `if: always()`, so a crashed run's partial batch still
        # reaches the GHA artifact.
        if research_writer is not None:
            research_writer.flush()

    SibylBot.log_report_summary(forecast_reports)  # type: ignore

    # Alert on degraded runs. Publication has already happened inside
    # forecast_on_tournament / forecast_questions above, so every Q that met
    # MIN_FORECASTERS_TO_PUBLISH is on Metaculus regardless of exit status.
    # Non-zero exit here just triggers the GitHub Actions red-check alert so
    # the operator knows to investigate (forecaster drops, stacker fallback
    # usage, research provider failures, etc. — see
    # `forecaster.py` `alertable_count`).
    bot_alertable = template_bot.alertable_count
    # Primary->fallback key fallback: counted in fallback_openrouter at the
    # wrapper level (process-global, since the wrapper has no link back to the
    # bot). Each fallback was successful — the run completed using the paid
    # fallback key — but a call that should have hit the free primary key
    # billed to the operator instead, so the operator should investigate.
    # ``fallback_count`` counts ALL fallback causes (401/402/429/guardrail/
    # 404); ``primary_route_404`` and ``credit_fallback`` are two disjoint subsets of
    # that total, broken out for diagnostics. Add only ``fallback_count`` to
    # ``alertable`` — adding either subset too would double-count events already
    # inside that total.
    #
    # Credit suppression (until CREDIT_ALERT_RESUME_DATE): the operator is
    # self-funding the rest of the season, so an empty primary key is expected
    # and its fallbacks are SUBTRACTED back out of the total. Every other cause
    # keeps its full weight, because 401/404/429/guardrail each mean real
    # breakage. Each event is still counted exactly once: generic adds it, and at
    # most one subset subtracts it.
    #
    # ``credit_fallback`` counts only the SUPPRESSIBLE credit subset — the
    # primary key genuinely drained. A key that was revoked or re-capped to zero
    # returns the same "Key limit exceeded" text but is classified separately by
    # ``fallback_openrouter.is_suppressible_credit_error`` (which probes
    # /auth/key), so it stays inside the generic total and keeps this run red.
    alerts_active = credit_alerts_active()
    fallback_count = get_fallback_key_count()
    primary_route_404 = get_primary_route_404_fallback_count()
    credit_fallback = get_credit_key_fallback_count()
    suppressed_credit_fallback = 0 if alerts_active else credit_fallback
    alertable = bot_alertable + fallback_count - suppressed_credit_fallback

    suppression_note = (
        ""
        if alerts_active
        else f" with {suppressed_credit_fallback} credit event(s) suppressed until "
        f"{CREDIT_ALERT_RESUME_DATE.isoformat()}"
    )
    # Only rendered when a spend-cap failure actually made the wrapper probe the
    # primary key. Omitted otherwise, because "unknown" would read as a failed
    # probe rather than "no run this shape ever needed one".
    probed_primary_key_state = get_probed_primary_key_state()
    primary_key_note = "" if probed_primary_key_state is None else f", donated_key={probed_primary_key_state.value}"
    # One breakdown, both exit paths. The green path needs it as much as the red
    # one: when every primary-key call fell back and the credit subset cancels the
    # whole generic total, ``alertable`` is 0 — the exact shape of the 2026-07-26
    # drained-key run — and gating this line on the exit status would leave that
    # run's degradation and probe verdict entirely unrecorded.
    # The FIELD NAMES here keep the old donated/personal vocabulary on purpose, even
    # though every identifier around them is now primary/fallback. This line is parsed by
    # the ``run_alertable_summary`` MarkerSpec, whose capture groups become field names in
    # the telemetry archive; renaming them would orphan every archived record and break
    # harvesting on re-run. Same stance markers.py takes about its own key names.
    breakdown = (
        f"Run completed with {alertable} alertable degradation event(s) "
        f"(bot={bot_alertable}, personal_key_fallback={fallback_count} of which "
        f"donated_404={primary_route_404}, credit={credit_fallback}{suppression_note}{primary_key_note});"
    )
    if alertable > 0:
        logger.warning("%s exiting non-zero so CI marks this run red.", breakdown)
        sys.exit(1)
    if fallback_count > 0:
        # Reachable only under suppression with every fallback credit-caused (the
        # subtraction can't otherwise reach zero from a positive total), so state
        # that rather than leaving a reader to derive it from the arithmetic.
        logger.info("%s every fallback was a suppressed credit event, so this run stays green.", breakdown)

    # Primary-key balance below the refill floor (CREDIT_FLOOR_BREACH warning
    # already logged by credit_telemetry). The run completed and published
    # normally; exiting non-zero here is purely the reminder-to-refill signal —
    # and it is suppressed until CREDIT_ALERT_RESUME_DATE, since a drained
    # primary key is the expected state while the operator self-funds. The INFO
    # line keeps the log self-explanatory: a reader who sees the breach WARNING
    # but a green run should not have to guess why.
    if primary_below_floor:
        if alerts_active:
            sys.exit(1)
        logger.info(
            "Primary-key credit floor breached, but credit alerting is suppressed until %s "
            "(operator is self-funding the rest of the season), so this run exits zero.",
            CREDIT_ALERT_RESUME_DATE.isoformat(),
        )

    # Post-submission deprecation tripwire. Runs LAST so submission has fully
    # completed (and so other alertable conditions exit first with their own
    # log lines). When OpenRouter retires a model the bot uses (e.g. the
    # 2026-05-15 x-ai/grok-4.1-fast deprecation that silently 404'd for ~2
    # days), this prints a loud banner + sys.exit(1) so GitHub Actions turns
    # red. Returns silently when no deprecation was observed.
    check_deprecation_alerts_and_exit()


if __name__ == "__main__":
    main()
