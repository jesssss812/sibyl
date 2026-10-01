"""Ground truth backtesting: score bot predictions against actual resolved outcomes."""

import argparse
import asyncio
import json
import logging
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import typeguard
from forecasting_tools import Benchmarker, ForecastBot, MonetaryCostManager
from tqdm import tqdm

from sibyl.aggregation.strategies import AggregationStrategy
from sibyl.common.forecast_context import forecast_times
from sibyl.config.env import load_environment
from sibyl.config.models import active_tier, get_roster
from sibyl.config.settings import (
    BACKTEST_DEFAULT_RESOLVED_AFTER,
    BACKTEST_DEFAULT_TOURNAMENT,
    BENCHMARK_BATCH_SIZE,
    HEARTBEAT_INTERVAL,
)
from sibyl.eval.artifacts import code_identity, load_manifest, model_identity, prepare_manifest, write_json
from sibyl.eval.backtest.analysis import (
    BacktestResult,
    aggregate_scores,
    generate_backtest_report,
    save_backtest_data,
)
from sibyl.eval.backtest.leakage import screen_research_for_leakage
from sibyl.eval.backtest.question_prep import fetch_resolved_questions
from sibyl.eval.backtest.scoring import QuestionScore, score_report
from sibyl.eval.benchmark.bot_factory import (
    BENCHMARK_BOT_CONFIG,
    create_individual_bots,
)
from sibyl.eval.benchmark.fixture_llm import FixtureLlm
from sibyl.eval.benchmark.heartbeat import install_benchmarker_heartbeat
from sibyl.eval.benchmark.logging_setup import configure_benchmark_logging
from sibyl.eval.performance.id_mapping import QuestionIds
from sibyl.eval.power import validate_preflight
from sibyl.forecasting.bot import SibylBot  # noqa: F401  # used in annotations below
from sibyl.platform.aiohttp_cleanup import enable_aiohttp_session_autoclose
from sibyl.platform.metaculus_preflight import verify_metaculus_api_identity
from sibyl.platform.scoring_patches import apply_scoring_patches

logger: logging.Logger = logging.getLogger(__name__)

load_environment()


enable_aiohttp_session_autoclose()


_progress_state: dict[str, Any] = {
    "total_predictions": 0,
    "start_time": 0,
    "completed_batches": 0,
    "total_batches": 0,
    "pbar": None,
}


install_benchmarker_heartbeat(HEARTBEAT_INTERVAL, _progress_state)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ground truth backtesting against resolved questions")
    parser.add_argument(
        "--num-questions",
        type=int,
        default=20,
        help="Number of resolved questions to backtest (default: 20)",
    )
    parser.add_argument(
        "--resolved-after",
        type=str,
        default=BACKTEST_DEFAULT_RESOLVED_AFTER,
        help=f"Only use questions resolved after this date (default: {BACKTEST_DEFAULT_RESOLVED_AFTER})",
    )
    parser.add_argument(
        "--tournament",
        type=str,
        default=BACKTEST_DEFAULT_TOURNAMENT,
        help=f"Tournament slug to fetch resolved questions from (default: {BACKTEST_DEFAULT_TOURNAMENT})",
    )
    parser.add_argument(
        "--include-models",
        nargs="*",
        default=None,
        help="Only include models matching these substrings (case-insensitive)",
    )
    parser.add_argument(
        "--exclude-models",
        nargs="*",
        default=None,
        help="Exclude models by substring match (case-insensitive)",
    )
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="Run with deterministic fixture forecasters instead of real models. Makes NO "
        "LLM or research call, so the run is free by construction rather than by discipline. "
        "Questions and resolutions are still fetched live (read-only Metaculus API, free), so "
        "the whole pipeline below the model — extraction, PCHIP, aggregation, the A3 guard, "
        "scoring — runs for real against real outcomes. Use it to exercise and validate the "
        "eval path, never to make a forecasting claim.",
    )
    parser.add_argument(
        "--fixture-seed",
        type=int,
        default=0,
        help="Seed for --fixture. Two runs with the same seed are identical (that is the A/A "
        "null test); different seeds give genuinely different arms.",
    )
    parser.add_argument(
        "--fixture-forecasters",
        type=int,
        default=3,
        help="How many fixture forecasters to ensemble (default 3, matching the prod roster).",
    )
    parser.add_argument(
        "--research-dir",
        type=str,
        default=None,
        help="Path to pre-cached research archive (e.g. backtests/research_archive/latest). "
        "When set, skips live research + leakage screening and replays stored research.",
    )
    parser.add_argument("--tier", choices=["free", "prod"])
    parser.add_argument("--bot-mode", choices=["ensemble", "individual"], default="ensemble")
    parser.add_argument("--manifest", type=Path, help="Frozen question/research snapshot; no fetching")
    parser.add_argument("--prepare-only", type=Path, help="Write a prepared snapshot, then exit")
    parser.add_argument("--power-plan", type=Path, help="Approved prospective preflight output")
    parser.add_argument("--arm-id", help="Named treatment arm for a registered real run")
    parser.add_argument("--output-dir", type=Path, default=Path("backtests"))
    parser.add_argument("--seed", type=int, default=0, help="Question selection seed")
    return parser


def _create_real_bots(research_cache: dict[int, str], *, individual: bool = False) -> list[SibylBot]:
    """Use the authoritative roster for the selected tier; never a stale paid catalog."""
    roster = get_roster(active_tier())
    helpers = {"parser": roster.parser, "researcher": roster.researcher, "summarizer": roster.summarizer,
               "stacker": roster.stacker, "analyzer": roster.analyzer}
    if individual:
        return create_individual_bots(
            [{"name": llm.model, "forecaster": llm} for llm in roster.forecasters], helpers,
            BENCHMARK_BOT_CONFIG, batch_size=BENCHMARK_BATCH_SIZE, research_cache=research_cache)
    bot = SibylBot(**BENCHMARK_BOT_CONFIG, aggregation_strategy=AggregationStrategy.CONDITIONAL_STACKING,
                  llms=cast("Any", {"forecasters": list(roster.forecasters), **helpers}),
                  max_concurrent_research=BENCHMARK_BATCH_SIZE, research_cache=research_cache)
    bot.name = f"{active_tier().value}-production-ensemble"
    return [bot]


def _create_fixture_bots(*, count: int, seed: int, research_cache: dict[int, str], individual: bool = True) -> list[SibylBot]:
    """Deterministic forecasters plus canned research. Cannot make a paid call.

    Every LLM slot is a FixtureLlm — forecasters AND the helper slots (summarizer, parser,
    researcher). Leaving even one helper real would reintroduce spend through a back door,
    which is exactly the kind of partial guard that reads as safe and isn't.

    ``research_provider`` is overridden too, so the research fan-out never runs. That also
    keeps the run offline: with `is_benchmarking=True` the leakage-sensitive providers are
    already hard-disabled, but AskNews and the search providers would otherwise still fire.
    """
    from sibyl.eval.benchmark.fixture_llm import (
        build_fixture_forecasters,
        disable_all_research_providers,
        fixture_research_provider,
    )

    # Overriding `research_provider` only replaces the PRIMARY provider — every add-on
    # (Gemini grounded search, prediction markets, gap-fill v2, ...) still fans out and
    # still bills. A smoke run caught this: the forecast succeeded while litellm raised an
    # OpenrouterException underneath, i.e. a paid call was attempted. Flags off first.
    disable_all_research_providers()

    forecasters = build_fixture_forecasters(count, seed=seed)
    helper = FixtureLlm(model="fixture/helper", seed=seed)
    config = {**BENCHMARK_BOT_CONFIG, "research_provider": fixture_research_provider}

    if not individual:
        bot = SibylBot(**config, aggregation_strategy=AggregationStrategy.MEDIAN,
                      llms=cast("Any", {"forecasters": forecasters, "default": helper,
                                      "summarizer": helper, "parser": helper, "researcher": helper}),
                      research_cache=research_cache)
        bot.name = "fixture-median-ensemble"
        return [bot]

    return create_individual_bots(
        cast("Any", [{"name": f.model, "forecaster": f} for f in forecasters]),
        # "default" is required by the framework's LLM-purpose validation; omitting it
        # raises rather than silently falling back, which is the right behaviour and is
        # why it must be listed explicitly here too.
        cast(
            "Any",
            {"default": helper, "summarizer": helper, "parser": helper, "researcher": helper},
        ),
        config,
        batch_size=BENCHMARK_BATCH_SIZE,
        research_cache=research_cache,
    )


def _filter_bots(
    bots: list[SibylBot],
    include_models: list[str] | None,
    exclude_models: list[str] | None,
) -> list[SibylBot]:
    """Filter bots by include/exclude substring matching on bot name."""
    filtered = list(bots)

    # `name` is set on each bot by create_individual_bots (bot.name = spec["name"]) and
    # declared on SibylBot, so reads are statically typed.
    if include_models:
        filtered = [b for b in filtered if any(token.lower() in b.name.lower() for token in include_models)]

    if exclude_models:
        filtered = [b for b in filtered if not any(token.lower() in b.name.lower() for token in exclude_models)]

    if not filtered:
        available_names = [b.name for b in bots]
        raise ValueError(
            f"No bots remaining after model filtering. "
            f"Available: {available_names}, include={include_models}, exclude={exclude_models}"
        )

    logger.info(f"Model filtering: {len(bots)} -> {len(filtered)} bots: {[b.name for b in filtered]}")
    return filtered


def _load_research_from_archive(
    research_dir: str, questions: list, evidence_records: dict[int, dict] | None = None
) -> dict[int, str]:
    """Load pre-cached research from a local archive directory into a research_cache dict.

    The returned cache is keyed on ``id_of_question`` — the id the orchestrator reads
    it back by (``_lookup_research_cache``). But the archive FILE for a question may be
    named after EITHER id: live persistence + comment-backfill write ``<question_id>.json``
    while ``backfill_research_from_logs`` writes ``<post_id>.json`` (its qid is parsed from
    the page URL). On a divergent post (post_id != question_id) a question-id-only lookup
    silently misses a log-backfilled record, so we try both ids (question-id first) via
    ``QuestionIds.lookup_order`` and cache under ``id_of_question`` regardless.

    Because the filename keyspace mixes both id spaces, a filename hit can be a FOREIGN
    question's record (question A's post id colliding with question B's question id).
    Each hit is therefore validated against the record's self-declared identity
    (``QuestionIds.matches_archive_record``); a mismatch skips to the next rung, leaving
    the question uncached (live research) rather than silently replaying B's research as A's.
    """

    cache: dict[int, str] = {}
    archive_path = Path(research_dir)
    if not archive_path.exists():
        logger.warning(f"Research archive dir does not exist: {research_dir}")
        return cache

    # Provenance of what we replay, logged below. A comment-backfilled record is
    # middle-trimmed text with sections re-headed and Provider Diagnostics leaked in,
    # so a replay's own log should state how much of it is that rather than the
    # artifact capture. "unknown" covers records written before build_archive stamped
    # the field. See sibyl/ops/download_research.py `record_source`.
    source_counts: Counter[str] = Counter()
    loaded = 0
    for q in questions:
        ids = QuestionIds.from_question(q)
        cache_key = ids.question_id
        if cache_key is None:
            continue
        for file_id in ids.lookup_order():
            record_file = archive_path / f"{file_id}.json"
            if not record_file.exists():
                continue
            record = json.loads(record_file.read_text())
            if not ids.matches_archive_record(record):
                logger.warning(
                    f"Archive file {record_file.name} exists but identifies a different post "
                    f"(question ids {ids}) — skipping"
                )
                continue
            cache[cache_key] = record.get("research_text", "")
            if evidence_records is not None:
                as_of = q.open_time + (q.scheduled_resolution_time - q.open_time) / 2 if q.open_time and q.scheduled_resolution_time else None
                raw_time = record.get("timestamp")
                try:
                    retrieval = datetime.fromisoformat(raw_time.replace("Z", "+00:00")) if isinstance(raw_time, str) else None
                    post_cutoff = retrieval is None or as_of is None or retrieval > as_of
                except (TypeError, ValueError):
                    retrieval, post_cutoff = None, True
                evidence_records[cache_key] = {
                    "category": str(record.get("record_source", record.get("source", "unknown"))),
                    "retrieved_at": raw_time,
                    "post_cutoff_or_unknown": post_cutoff,
                    "question_id": cache_key,
                }
            source_counts[str(record.get("source", "unknown"))] += 1
            loaded += 1
            break

    uncached = len(questions) - loaded
    logger.info(
        f"Loaded {loaded} cached research records from {research_dir} ({uncached} questions uncached); "
        f"sources: {dict(source_counts)}"
    )
    if uncached > 0:
        logger.warning(f"{uncached} question(s) have no cached research; replay must refuse them")
    return cache


async def run_backtest(args: argparse.Namespace) -> None:
    """Run the full backtest pipeline."""
    if args.prepare_only and args.manifest:
        raise ValueError("Choose either --prepare-only or --manifest")
    if not args.fixture and not args.prepare_only and (not args.manifest or not args.power_plan or not args.arm_id):
        raise ValueError("Real evaluation requires --manifest, --power-plan and --arm-id before model calls")
    # Fetch resolved questions and extract ground truths
    logger.info(
        f"Fetching {args.num_questions} resolved questions "
        f"(tournament={args.tournament}, resolved_after={args.resolved_after})"
    )
    sys.stdout.flush()

    if args.tier:
        os.environ["SIBYL_MODEL_TIER"] = args.tier
    manifest = None
    if args.manifest:
        manifest, question_set = load_manifest(args.manifest)
    else:
        question_set = await fetch_resolved_questions(
            total_questions=args.num_questions, resolved_after=args.resolved_after,
            tournament=args.tournament, seed=args.seed)


    logger.info(
        f"Prepared {len(question_set.questions)} questions with {len(question_set.ground_truths)} ground truths"
    )

    # Research: either load from pre-cached archive or run live with leakage screening
    if manifest is not None:
        clean_questions, clean_ground_truths = question_set.questions, question_set.ground_truths
        research_cache = question_set.research_cache
    elif args.fixture:
        from sibyl.eval.benchmark.fixture_llm import fixture_research_provider

        clean_questions = question_set.questions
        clean_ground_truths = question_set.ground_truths
        research_cache = {
            q.id_of_question: await fixture_research_provider(q)
            for q in clean_questions if q.id_of_question is not None
        }
    elif args.research_dir:
        archive_evidence: dict[int, dict] = {}
        research_cache = _load_research_from_archive(args.research_dir, question_set.questions, archive_evidence)
        question_set.fetch_metadata["research_provenance"] = archive_evidence
        clean_questions = question_set.questions
        clean_ground_truths = question_set.ground_truths
        missing = [q.id_of_question for q in clean_questions if not research_cache.get(q.id_of_question or -1)]
        if missing:
            raise ValueError(f"Missing frozen research for {missing}; refusing live fallback")
    else:
        screening_records = {}
        clean_questions, clean_ground_truths, research_cache = await screen_research_for_leakage(
            question_set.questions,
            question_set.ground_truths,
            screening_records=screening_records,
        )
        question_set.fetch_metadata["screening"] = screening_records
        question_set.fetch_metadata["research_provenance"] = {
            qid: {"category": "live_after_resolution", "post_cutoff_or_unknown": True}
            for qid in research_cache}
        question_set.questions = clean_questions
        question_set.ground_truths = clean_ground_truths

    if args.fixture and manifest is None:
        question_set.fetch_metadata["research_provenance"] = {
            qid: {"category": "fixture", "post_cutoff_or_unknown": False} for qid in research_cache}
    question_set.research_cache = research_cache
    logger.info(f"After research setup: {len(clean_questions)} questions, {len(research_cache)} cached")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.prepare_only:
        excluded = []
        eligible = []
        for q in question_set.questions:
            qid = q.id_of_question
            if qid is None or q.open_time is None or q.scheduled_resolution_time is None:
                excluded.append({"question_id": qid, "reason": "missing_forecast_time"})
                continue
            resolved_at = question_set.ground_truths[qid].actual_resolution_time
            as_of = q.open_time + (q.scheduled_resolution_time - q.open_time) / 2
            if resolved_at is not None and as_of >= resolved_at:
                excluded.append({"question_id": qid, "reason": "resolved_before_simulated_forecast"})
                continue
            eligible.append(q)
        question_set.fetch_metadata["preparation_exclusions"] = excluded
        question_set.questions = eligible
        question_set.ground_truths = {q.id_of_question: question_set.ground_truths[q.id_of_question]
                                      for q in eligible if q.id_of_question is not None}
        question_set.research_cache = {q.id_of_question: research_cache[q.id_of_question]
                                       for q in eligible if q.id_of_question in research_cache}
        manifest = manifest or prepare_manifest(question_set, source="fixture" if args.fixture else "retrospective-research")
        write_json(args.prepare_only, manifest)
        return
    if not args.fixture:
        if manifest is None or args.power_plan is None:
            raise ValueError("Real evaluation requires --manifest and --power-plan; prepare and preflight first")
        endpoint = json.loads(args.power_plan.read_text())["plan"]["endpoint"]
        if endpoint == "crps":
            raise ValueError("Backtest does not yet score CRPS; choose numeric_log_score")
        endpoint_type = "BinaryQuestion" if endpoint in ("brier", "binary_log_score") else "MultipleChoiceQuestion" if endpoint == "mc_log_score" else "NumericQuestion"
        eligible_count = sum(entry["question_class"] == endpoint_type or
                             (endpoint_type == "NumericQuestion" and entry["question_class"] == "DiscreteQuestion")
                             for entry in manifest["entries"])
        validate_preflight(args.power_plan, eligible_count, manifest["manifest_id"])

    started_at = datetime.now().astimezone().isoformat()
    # Create bots
    if args.fixture:
        bots = _create_fixture_bots(
            count=args.fixture_forecasters,
            seed=args.fixture_seed,
            research_cache=research_cache,
            individual=args.bot_mode == "individual",
        )
        logger.warning(
            "FIXTURE MODE: %d deterministic forecasters, seed=%d. No LLM or research call will "
            "be made. Scores from this run measure the PIPELINE, not forecasting skill — never "
            "report them as accuracy.",
            len(bots),
            args.fixture_seed,
        )
    else:
        bots = _create_real_bots(research_cache, individual=args.bot_mode == "individual")

    bots = typeguard.check_type(bots, list[SibylBot])

    bots = _filter_bots(bots, args.include_models, args.exclude_models)

    # Apply scoring patches for mixed question types
    apply_scoring_patches()
    historical_times = {int(e["question_id"]): datetime.fromisoformat(e["as_of"]) for e in manifest["entries"]} if manifest else {}
    with forecast_times(historical_times), MonetaryCostManager() as cost_manager:
        # Run Benchmarker
        total_predictions = len(bots) * len(clean_questions)
        logger.info(
            f"Starting backtest: {len(bots)} bots x {len(clean_questions)} questions "
            f"= {total_predictions} total predictions"
        )
        sys.stdout.flush()

        _progress_state.update(
            {
                "total_predictions": total_predictions,
                "start_time": time.time(),
                "completed_batches": 0,
                "total_batches": len(bots),
                "pbar": tqdm(total=total_predictions, desc="Backtesting", unit="predictions"),
            }
        )

        benchmarks = await Benchmarker(
            questions_to_use=clean_questions,
            forecast_bots=cast("list[ForecastBot]", bots),
            file_path_to_save_reports=str(args.output_dir),
            concurrent_question_batch_size=BENCHMARK_BATCH_SIZE,
        ).run_benchmark()

        if _progress_state["pbar"] is not None:
            _progress_state["pbar"].close()
            _progress_state["pbar"] = None

        logger.info("Benchmarker completed, scoring against ground truth...")
        sys.stdout.flush()

        # Score each bot's reports against ground truth
        results: list[BacktestResult] = []
        for benchmark in benchmarks:
            bot_scores: list[QuestionScore] = []
            num_scored = 0
            num_failed = 0
            statuses: dict[int, dict[str, Any]] = {qid: {"status": "missing_report"} for qid in clean_ground_truths}

            for report in benchmark.forecast_reports:
                qid = report.question.id_of_question
                if qid not in clean_ground_truths:
                    num_failed += 1
                    logger.warning(f"No ground truth for question {qid}, skipping")
                    continue

                from sibyl.eval.performance.parsing import parse_forecasters_used_marker
                used = parse_forecasters_used_marker(report.explanation)
                degraded = bool(report.errors) or (used is not None and used[0] < used[1])
                statuses[qid] = {"status": "degraded" if degraded else "ok",
                                 "errors": list(report.errors), "forecasters_used": used}
                report_scores = score_report(report, clean_ground_truths[qid])
                if report_scores and any(not math.isfinite(float(row.bot_score)) for row in report_scores):
                    statuses[qid] = {"status": "invalid_score"}
                    num_failed += 1
                    continue
                if report_scores:
                    bot_scores.extend(report_scores)
                    num_scored += 1
                else:
                    statuses[qid] = {"status": "unscored"}
                    num_failed += 1

            result = BacktestResult(
                bot_name=benchmark.name,
                scores=bot_scores,
                num_questions=len(clean_questions),
                num_scored=num_scored,
                num_failed=len(clean_questions) - num_scored,
                question_status=statuses,
            )
            results.append(result)

            aggregated = aggregate_scores(bot_scores)
            logger.info(f"Bot '{benchmark.name}': scored={num_scored}, failed={num_failed}")
            for metric_name, agg in aggregated.items():
                community_str = ""
                if agg.get("community_mean") is not None:
                    community_str = f" | Community: {agg['community_mean']:.4f}"
                logger.info(f"  {metric_name}: Bot mean = {agg['bot_mean']:.4f} (n={agg['n']}){community_str}")

        # Generate report and save data
        question_set.fetch_metadata["experiment"] = {
            **code_identity(), "manifest_id": manifest["manifest_id"] if manifest else None,
            "scoring_version": "per-type-v1", "fixture": args.fixture,
            "bot_mode": args.bot_mode, "arm_id": args.arm_id or "fixture", "started_at": started_at, "tier": active_tier().value,
            "power_plan": json.loads(args.power_plan.read_text()) if args.power_plan else None,
            "research_policy": "frozen retrospective research; differs from live fan-out",
            "roster": [] if args.fixture else [model_identity(m) for m in get_roster(active_tier()).forecasters],
        }
        report_text = generate_backtest_report(results, question_set, output_path=str(args.output_dir / "backtest_report.md"))
        save_backtest_data(question_set, results, output_dir=str(args.output_dir))

        logger.info(f"\nTotal Cost: {cost_manager.current_usage}")
        logger.info("\n" + "=" * 60)
        logger.info("BACKTEST REPORT")
        logger.info("=" * 60)
        logger.info(report_text)


def main() -> None:
    """Console entry point.

    Extracted from the `if __name__ == "__main__"` block during the move into this
    package: under `python -m sibyl.eval.backtest`, `__name__` belongs to `__main__.py`,
    so a bare guard here would silently never run. Naming it makes the entry point
    callable and testable rather than reachable only by running the file.
    """
    os.environ["PYTHONUNBUFFERED"] = "1"

    configure_benchmark_logging(log_dir="backtests")

    parser = _build_parser()
    args = parser.parse_args()

    # Confirm www.metaculus.com is the real API before pulling resolved
    # questions (which send METACULUS_TOKEN). See sibyl/platform/metaculus_preflight.py
    # for the DNS-parking incident that motivated this guard.
    if not args.manifest:
        verify_metaculus_api_identity()

    asyncio.run(run_backtest(args))


if __name__ == "__main__":
    main()
