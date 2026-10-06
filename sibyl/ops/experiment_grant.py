"""Isolated OpenRouter-grant runners and offline spending inspection.

No production CLI or model singleton is mutated. Run commands publish to their
explicit target and require operator approval for that specific run. Preview
and report make no model or research requests.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Sequence
from uuid import uuid4

from sibyl.platform.experiment_funding import (
    GRANT_KEY_ENV,
    GRANT_RESERVE_USD,
    BudgetExceeded,
    GrantExecutionError,
    GrantSession,
    SpendLedger,
    grant_context,
    grant_question,
)

if TYPE_CHECKING:
    from sibyl.config.models import Roster

_UNSUPPORTED_RESEARCH_FLAGS = (
    "NATIVE_SEARCH_ENABLED",
    "GEMINI_SEARCH_ENABLED",
    "ANTHROPIC_SEARCH_ENABLED",
    "GAP_FILL_ENABLED",
    "GAP_FILL_V2_ENABLED",
    # The initial grant profile is deliberately just one accounted OpenRouter
    # native-search call. These enrichers have independent fetch/LLM paths that
    # have not yet been audited for grant-only billing.
    "FINANCIAL_DATA_ENABLED",
    "TS_ANCHOR_ENABLED",
    "PREDICTION_MARKETS_ENABLED",
    "RESOLUTION_SOURCE_ENABLED",
    "REFERENCE_CLASS_ENABLED",
)


def _enabled(env_name: str) -> bool:
    return os.getenv(env_name, "").strip().lower() in {"1", "true", "yes"}


def preflight_research() -> dict[str, str]:
    enabled = [name for name in _UNSUPPORTED_RESEARCH_FLAGS if _enabled(name)]
    if enabled:
        raise ValueError("Grant run requires these research paths to be disabled: " + ", ".join(enabled))
    native_model = os.getenv("NATIVE_SEARCH_MODEL", "openai/gpt-5.6-terra")
    if native_model.startswith("openrouter/"):
        raise ValueError("NATIVE_SEARCH_MODEL must omit the openrouter/ prefix")
    return {
        "primary_research": "openrouter/" + native_model + " (native web search)",
        "direct_paid_research": "disabled",
    }


def _positive(raw: str, label: str) -> Decimal:
    try:
        value = Decimal(raw)
    except InvalidOperation as exc:
        raise ValueError(f"{label} must be a positive decimal") from exc
    if not value.is_finite() or value <= 0:
        raise ValueError(f"{label} must be a positive decimal")
    return value


def preflight_provider_limit(key: str, run_limit: Decimal) -> Decimal:
    """Refuse a run the grant key's own limit cannot cover.

    OpenRouter's ``limit_remaining`` is authoritative: it meters BYOK spend
    (``include_byok_in_limit``) and spending from other machines, neither of which the
    local ledger sees. The lookup is free and read-only. The local reserve margin is
    kept on top of the run budget.
    """
    import httpx

    from sibyl.telemetry.check_credits import fetch_auth_key

    try:
        data = fetch_auth_key(key)
    except httpx.HTTPStatusError as exc:
        raise ValueError(f"Grant key limit lookup failed (HTTP {exc.response.status_code})") from None
    except httpx.HTTPError as exc:
        raise ValueError(f"Grant key limit lookup failed ({type(exc).__name__})") from None
    raw = data.get("limit_remaining")
    if raw is None:
        raise ValueError("Grant key reports no spending limit; refusing to run without a provider cap")
    try:
        remaining = Decimal(str(raw))
    except InvalidOperation:
        raise ValueError("Grant key reported an unreadable limit_remaining") from None
    if not remaining.is_finite() or remaining < run_limit + GRANT_RESERVE_USD:
        raise BudgetExceeded(
            f"Grant key has ${remaining} remaining; run budget ${run_limit} plus "
            f"${GRANT_RESERVE_USD} reserve does not fit"
        )
    logging.getLogger(__name__).info("GRANT_PREFLIGHT limit_remaining_usd=%s run_budget_usd=%s", remaining, run_limit)
    return remaining


def checked_session(args: argparse.Namespace) -> GrantSession:
    key = os.getenv(GRANT_KEY_ENV, "").strip()
    if not key:
        raise ValueError(f"{GRANT_KEY_ENV} is required for grant execution")
    if key in {os.getenv("OPENROUTER_API_KEY"), os.getenv("OAI_ANTH_OPENROUTER_KEY")}:
        raise ValueError("Experiment grant key must be distinct from existing OpenRouter keys")
    run_limit = _positive(args.max_cost_usd, "--max-cost-usd")
    per_call = _positive(args.reserve_per_call_usd, "--reserve-per-call-usd")
    if per_call > run_limit:
        raise ValueError("The per-call reservation cannot exceed the run budget")
    if run_limit > Decimal("100") - GRANT_RESERVE_USD:
        raise ValueError("Run budget exceeds the local grant allocation after reserve")
    preflight_research()
    preflight_provider_limit(key, run_limit)
    ledger = SpendLedger(args.ledger)
    ledger.start_run(args.run_id, run_limit)
    return GrantSession(args.run_id, ledger, per_call, key)


def _funded_roster(source: Roster) -> Roster:
    from sibyl.platform.experiment_llm import clone_for_grant

    return replace(
        source,
        forecasters=tuple(clone_for_grant(llm, stage="forecaster") for llm in source.forecasters),
        summarizer=clone_for_grant(source.summarizer, stage="summarizer"),
        parser=clone_for_grant(source.parser, stage="parser"),
        researcher=clone_for_grant(source.researcher, stage="researcher"),
        stacker=clone_for_grant(source.stacker, stage="stacker"),
        stacker_fallback=clone_for_grant(source.stacker_fallback, stage="stacker_fallback"),
        analyzer=clone_for_grant(source.analyzer, stage="analyzer"),
    )


def _preview() -> dict[str, object]:
    """Resolve the current roster without a key or a paid provider request."""
    from sibyl.config.models import ModelTier, get_roster
    from sibyl.platform.experiment_llm import grant_route_for

    research = preflight_research()
    roster = get_roster(ModelTier.PROD)
    return {
        "funding": "experiment-grant",
        "tier": "prod",
        "mode": "minibench",
        "forecasters": [grant_route_for(llm.model) for llm in roster.forecasters],
        "stacker": grant_route_for(roster.stacker.model),
        "stacker_fallback": grant_route_for(roster.stacker_fallback.model),
        "summarizer": grant_route_for(roster.summarizer.model),
        "parser": grant_route_for(roster.parser.model),
        "researcher": grant_route_for(roster.researcher.model),
        "analyzer": grant_route_for(roster.analyzer.model),
        "market_ranker": grant_route_for(str(roster.market_ranker_config["model"])),
        "market_query_author": grant_route_for(str(roster.market_query_author_config["model"])),
        "financial_classifier": grant_route_for(roster.financial_classifier_model),
        "research": research,
        "route_validation": "offline configuration only; confirm public catalogue before a live run",
        "allocation_usd": "100",
        "local_reserve_usd": str(GRANT_RESERVE_USD),
    }


async def _run_minibench(args: argparse.Namespace, session: GrantSession) -> None:
    await _run_grant_forecasts(args, session, mode="minibench")


async def _run_test_question(args: argparse.Namespace, session: GrantSession) -> None:
    await _run_grant_forecasts(args, session, mode="test")


async def _run_grant_forecasts(
    args: argparse.Namespace, session: GrantSession, *, mode: Literal["minibench", "test"]
) -> None:
    from forecasting_tools import (
        BinaryQuestion,
        ForecastReport,
        MetaculusApi,
        MetaculusQuestion,
        MultipleChoiceQuestion,
        NumericQuestion,
        QuestionState,
    )

    from sibyl.aggregation.strategies import AggregationStrategy
    from sibyl.config.models import ModelTier, get_roster
    from sibyl.config.settings import BOT_TESTING_AREA_ID
    from sibyl.eval.artifacts import code_identity, question_identity
    from sibyl.forecasting.bot import SibylBot
    from sibyl.platform.metaculus_preflight import verify_metaculus_api_identity

    if args.max_questions < 1:
        raise ValueError("--max-questions must be positive")
    if mode == "test" and (args.max_questions != 1 or args.post_id < 1 or args.question_id < 1):
        raise ValueError("Grant test runs need one question and positive post/question IDs")
    source_roster = get_roster(ModelTier.PROD)
    run_dir = args.reports_dir / session.run_id
    manifest_path = run_dir / "manifest.json"

    class GrantSibylBot(SibylBot):
        async def run_research(self, question: MetaculusQuestion) -> str:
            research = await super().run_research(question)
            if not research.strip() or session.ledger.has_blocking_attempts(
                session.run_id, str(question.id_of_question)
            ):
                raise GrantExecutionError("Research is empty or has a failed or over-reserve call; forecasters refused")
            return research

        async def forecast_questions(self, questions: Sequence[MetaculusQuestion], return_exceptions: bool = False):
            supported = (BinaryQuestion, MultipleChoiceQuestion, NumericQuestion)
            selected = [
                question
                for question in questions
                if isinstance(question, supported)
                and not (self.skip_previously_forecasted_questions and question.already_forecasted)
            ]
            if self.max_questions_per_run is not None:
                selected = selected[: self.max_questions_per_run]
            question_ids = [question.id_of_question for question in selected]
            if any(qid is None for qid in question_ids) or len(set(question_ids)) != len(question_ids):
                raise ValueError("Grant selection needs unique non-null question IDs")
            snapshot_path = run_dir / "scheduled_questions.json"
            with snapshot_path.open("x") as snapshot_file:
                json.dump(
                    {
                        "schema_version": 1,
                        "run_id": session.run_id,
                        "question_ids": question_ids,
                        "questions": [question_identity(question) for question in selected],
                    },
                    snapshot_file,
                    indent=2,
                )
            return await super().forecast_questions(questions, return_exceptions)

        async def _run_individual_question(self, question: MetaculusQuestion) -> ForecastReport:
            logging.getLogger(__name__).info(
                "GRANT_QUESTION_START question=%s url=%s", question.id_of_question, question.page_url
            )
            with grant_question(question.id_of_question):
                report = await super()._run_individual_question(question)
            logging.getLogger(__name__).info("GRANT_QUESTION_COMPLETE question=%s", question.id_of_question)
            return report

        def _create_unified_explanation(self, question, *args, **kwargs):
            # forecasting-tools publishes immediately after creating this text.
            # A failed call may have been dropped from the ensemble, so check the
            # full question ledger instead of relying on prediction errors.
            question_id = question.id_of_question
            if question_id is None or session.ledger.has_blocking_attempts(session.run_id, str(question_id)):
                raise RuntimeError("Grant question has a failed call or an overage; publication refused")
            return super()._create_unified_explanation(question, *args, **kwargs)

    with grant_context(session):
        roster = _funded_roster(source_roster)
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            with manifest_path.open("x") as manifest_file:
                json.dump(
                    {
                        "schema_version": 1,
                        "run_id": session.run_id,
                        "started_at": datetime.now(timezone.utc).isoformat(),
                        "code": code_identity(),
                        "configuration": {**_preview(), "mode": mode},
                        "max_questions": args.max_questions,
                        "max_cost_usd": args.max_cost_usd,
                        "reserve_per_call_usd": str(session.per_call_reserve_usd),
                    },
                    manifest_file,
                    indent=2,
                )
        except FileExistsError as exc:
            raise ValueError(f"Run ID {session.run_id} already has a forecast manifest") from exc
        # Validate every paid model route before the first network request,
        # including the read-only Metaculus identity probe.
        verify_metaculus_api_identity()
        llms = {
            "forecasters": list(roster.forecasters),
            "stacker": roster.stacker,
            "analyzer": roster.analyzer,
            "summarizer": roster.summarizer,
            "parser": roster.parser,
            "researcher": roster.researcher,
        }
        bot = GrantSibylBot(
            research_reports_per_question=1,
            predictions_per_research_report=1,
            publish_reports_to_metaculus=True,
            folder_to_save_reports_to=str(run_dir / "forecasts"),
            skip_previously_forecasted_questions=mode == "minibench",
            max_questions_per_run=args.max_questions,
            aggregation_strategy=AggregationStrategy.CONDITIONAL_STACKING,
            allow_research_fallback=False,
            llms=llms,
        )
        if mode == "test":
            open_questions = bot.metaculus_client.get_all_open_questions_from_tournament(BOT_TESTING_AREA_ID)
            matches = [
                question
                for question in open_questions
                if question.id_of_post == args.post_id
                and question.id_of_question == args.question_id
                and isinstance(question, (BinaryQuestion, MultipleChoiceQuestion, NumericQuestion))
                and question.state == QuestionState.OPEN
                and BOT_TESTING_AREA_ID in question.tournament_slugs
            ]
            if len(matches) != 1:
                raise ValueError("Requested question is not an open question in bot-testing-area")
            reports = await bot.forecast_questions(matches, return_exceptions=True)
        else:
            reports = await bot.forecast_on_tournament(MetaculusApi.CURRENT_MINIBENCH_ID, return_exceptions=True)
        SibylBot.log_report_summary(reports)
        summary = session.ledger.summary(session.run_id)
        if Decimal(str(summary["estimated_usd"])) > 0:
            # Not a failure: the reservations stay held until `reconcile` confirms them.
            logging.getLogger(__name__).warning(
                "GRANT_RECONCILE_NEEDED run=%s estimated_usd=%s; run `reconcile --run-id %s`",
                session.run_id,
                summary["estimated_usd"],
                session.run_id,
            )
        if (
            any(isinstance(report, BaseException) for report in reports)
            or bot.alertable_count > 0
            or Decimal(str(summary["uncertain_usd"])) > 0
            or Decimal(str(summary["reserved_usd"])) > 0
        ):
            raise RuntimeError("Grant run finished with failed forecasts, degradation, or uncertain charges")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m sibyl.ops.experiment_grant")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preview", help="Show grant routes and research preflight without a key or API call")
    report = sub.add_parser("report", help="Show local grant accounting; makes no API call")
    report.add_argument("--ledger", type=Path, default=Path("backtests/experiment_grant_spend.sqlite3"))
    report.add_argument("--run-id")
    rec = sub.add_parser(
        "reconcile",
        help="Confirm settled estimates from OpenRouter's billing records; read-only, makes no paid call",
    )
    rec.add_argument("--ledger", type=Path, default=Path("backtests/experiment_grant_spend.sqlite3"))
    rec.add_argument("--run-id", required=True)
    adjust = sub.add_parser("record-prior-spend", help="Record previous grant spending in the local allocation")
    adjust.add_argument("--ledger", type=Path, default=Path("backtests/experiment_grant_spend.sqlite3"))
    adjust.add_argument("--amount-usd", required=True)
    adjust.add_argument("--note", required=True)
    adjust.add_argument("--id", required=True, help="Stable unique adjustment ID for audit and duplicate protection")
    write_off = sub.add_parser(
        "write-off",
        help="Replace unconfirmed attempts with no generation ID by OpenRouter's own total; makes no API call",
    )
    write_off.add_argument("--ledger", type=Path, default=Path("backtests/experiment_grant_spend.sqlite3"))
    write_off.add_argument("--provider-total-usd", required=True, help="What OpenRouter metered for those attempts")
    write_off.add_argument("--expect-count", type=int, required=True, help="Refuse unless exactly this many match")
    write_off.add_argument("--note", required=True)
    write_off.add_argument("--id", required=True, help="Stable unique adjustment ID for audit and duplicate protection")
    run = sub.add_parser("run-minibench", help="Paid, scored run; requires per-run operator authorization")
    run.add_argument("--ledger", type=Path, default=Path("backtests/experiment_grant_spend.sqlite3"))
    run.add_argument("--run-id", default=None)
    run.add_argument("--max-cost-usd", required=True)
    run.add_argument("--reserve-per-call-usd", required=True)
    run.add_argument("--max-questions", type=int, default=3)
    run.add_argument("--reports-dir", type=Path, default=Path("backtests/minibench_runs"))
    test = sub.add_parser(
        "run-test", help="Paid, unscored bot-testing-area run; requires per-run operator authorization"
    )
    test.add_argument("--ledger", type=Path, default=Path("backtests/experiment_grant_spend.sqlite3"))
    test.add_argument("--run-id", default=None)
    test.add_argument("--max-cost-usd", required=True)
    test.add_argument("--reserve-per-call-usd", required=True)
    test.add_argument("--post-id", type=int, required=True)
    test.add_argument("--question-id", type=int, required=True)
    test.add_argument("--reports-dir", type=Path, default=Path("backtests/grant_test_runs"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from sibyl.config.env import load_environment

    load_environment()
    try:
        if args.command == "preview":
            print(json.dumps(_preview(), indent=2))
            return 0
        if args.command == "report":
            ledger = SpendLedger(args.ledger)
            from sibyl.platform.personal_fallback_spend import PersonalFallbackLedger

            print(
                json.dumps(
                    {
                        "grant": {
                            "totals": ledger.summary(args.run_id),
                            "by_model_stage": ledger.breakdown(args.run_id),
                        },
                        "personal_fallback": PersonalFallbackLedger(args.ledger).summary(args.run_id),
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "reconcile":
            # Read-only: the generation GET reads a billing record, it never creates one.
            # So this needs the grant key but spends nothing, and is safe to re-run.
            key = os.getenv(GRANT_KEY_ENV, "").strip()
            if not key:
                raise ValueError(f"{GRANT_KEY_ENV} is required to read the grant's billing records")
            from sibyl.platform.experiment_usage import reconcile_pending

            # SpendLedger creates a missing file, and an empty ledger would report
            # nothing outstanding; a mistyped path or run ID must not read as success.
            if not args.ledger.is_file():
                raise ValueError(f"No grant ledger at {args.ledger}")
            ledger = SpendLedger(args.ledger)
            if not ledger.has_run(args.run_id):
                raise ValueError(f"Run {args.run_id} is not in the ledger at {args.ledger}")
            outcome = asyncio.run(reconcile_pending(ledger, args.run_id, key))
            print(json.dumps(outcome, indent=2))
            # Non-zero when anything is still unaccounted, so a wrapper notices.
            return 0 if outcome["still_unknown"] == 0 and outcome["no_generation_id"] == 0 else 1
        if args.command == "write-off":
            if not args.ledger.is_file():
                raise ValueError(f"No grant ledger at {args.ledger}")
            ledger = SpendLedger(args.ledger)
            written = ledger.write_off(
                args.id,
                _positive(args.provider_total_usd, "--provider-total-usd"),
                args.note,
                expected_count=args.expect_count,
            )
            print(json.dumps({"written_off": written, "totals": ledger.summary()}, indent=2))
            return 0
        if args.command == "record-prior-spend":
            ledger = SpendLedger(args.ledger)
            ledger.adjust(args.id, _positive(args.amount_usd, "--amount-usd"), args.note)
            print(json.dumps(ledger.summary(), indent=2))
            return 0
        args.run_id = args.run_id or uuid4().hex
        logging.basicConfig(
            level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
        )
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)
        if args.command == "run-test":
            args.max_questions = 1
        session = checked_session(args)
        print(
            f"Grant run ID: {session.run_id}; mode={args.command}; max_questions={args.max_questions}; budget_usd={args.max_cost_usd}",
            flush=True,
        )
        try:
            if args.command == "run-test":
                asyncio.run(_run_test_question(args, session))
            else:
                asyncio.run(_run_minibench(args, session))
        finally:
            print(json.dumps(session.ledger.summary(session.run_id), indent=2))
        return 0
    except (ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
