"""Golden parity harness: a recording of what this code does, so a fork can prove it still does it.

The sibyl clean fork (``docs/plans/2026-09-clean-fork.md``) moves nearly every module
in ``sibyl/`` to a new package layout. The risk of a move that large is not that
it fails loudly — ``make test`` catches that — but that it changes a number by 1e-6, or
reorders an aggregation, or drops a counter, in a way no existing unit test happens to
assert. Those tests were written to pin *intent*; this file exists to pin *output*.

So every scenario below runs real production code end to end and writes its result to
``tests/e2e/golden/<name>.json``. The fork re-runs the same scenarios and compares. Floats
match to an absolute 1e-9; strings match exactly.

**Zero API calls, by construction, not by convention.** Every LLM slot is a
``FixtureLlm`` (``sibyl/eval/benchmark/fixture_llm.py``) — seeded, deterministic, and
overriding the one method that would reach the network. Research is the canned fixture
provider and every add-on provider flag is off. The suite's autouse egress guard
(``tests/conftest.py``) is the backstop, not the plan.

Recording is deliberate and separate: ``uv run pytest tests/e2e/test_golden.py --record-golden``.
A mismatch during a normal run is a finding, and auto-healing it would record the
regression rather than catch it.
"""

from __future__ import annotations

import asyncio
import json
import math
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from forecasting_tools import BinaryQuestion, MultipleChoiceQuestion, NumericQuestion
from forecasting_tools.data_models.questions import DiscreteQuestion

from sibyl.aggregation.strategies import AggregationStrategy
from sibyl.eval.benchmark.fixture_llm import (
    FixtureLlm,
    build_fixture_forecasters,
    fixture_research_provider,
)
from sibyl.eval.performance.parsing import parse_stacker_outcome_marker
from sibyl.forecasting.bot import SibylBot

GOLDEN_DIR = Path(__file__).parent / "golden"

# Absolute, not relative. The values recorded here span probabilities (~1e-1) and CDF
# heights (~1e-3), and a relative tolerance would be far laxer on the former than the
# latter for no principled reason. 1e-9 is well inside float64 noise for arithmetic of
# this depth but far tighter than any real behaviour change would be.
FLOAT_TOL = 1e-9

# The instant every pipeline scenario runs at.
#
# The forecaster prompt embeds "Today: <date>" and the question's elapsed/remaining day
# counts, and FixtureLlm seeds itself on the prompt -- so an unfrozen clock makes every
# recorded prediction change at UTC midnight. That is not hypothetical: these records
# were written on 2026-09-01 and began failing the moment the process crossed into
# 2026-09-02 UTC, which read as a parity break in an unrelated change.
GOLDEN_NOW = datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc)

_OPEN_TIME = datetime(2026, 1, 1)
_RESOLVE_TIME = datetime(2026, 12, 31)


# ---------------------------------------------------------------------------
# Comparison and recording
# ---------------------------------------------------------------------------


def _compare(expected: Any, actual: Any, path: str) -> None:
    """Structural comparison that reports WHERE it diverged, not just that it did.

    ``assert expected == actual`` on a nested dict of 600 floats produces an unreadable
    diff, which in practice means the failure gets re-recorded rather than investigated.
    The path prefix ("cdf[137].value") is what makes a mismatch actionable.
    """
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: expected object, got {type(actual).__name__}"
        assert set(expected) == set(actual), (
            f"{path}: key set changed. missing={sorted(set(expected) - set(actual))} "
            f"added={sorted(set(actual) - set(expected))}"
        )
        for key in expected:
            _compare(expected[key], actual[key], f"{path}.{key}" if path else str(key))
        return

    if isinstance(expected, list):
        assert isinstance(actual, list), f"{path}: expected list, got {type(actual).__name__}"
        assert len(expected) == len(actual), f"{path}: length {len(expected)} -> {len(actual)}"
        for i, (e, a) in enumerate(zip(expected, actual)):
            _compare(e, a, f"{path}[{i}]")
        return

    # bool before float: bool is a subclass of int, and True == 1.0 would let a flag
    # flip to a number without tripping anything.
    if isinstance(expected, bool) or isinstance(actual, bool):
        assert expected == actual, f"{path}: {expected!r} -> {actual!r}"
        return

    if isinstance(expected, float) or isinstance(actual, float):
        assert isinstance(actual, (int, float)), f"{path}: expected number, got {type(actual).__name__}"
        if math.isnan(expected) or math.isnan(actual):
            assert math.isnan(expected) and math.isnan(actual), f"{path}: {expected!r} -> {actual!r}"
            return
        assert abs(expected - actual) <= FLOAT_TOL, f"{path}: {expected!r} -> {actual!r} (tol {FLOAT_TOL})"
        return

    assert expected == actual, f"{path}: {expected!r} -> {actual!r}"


def assert_golden(request: pytest.FixtureRequest, name: str, actual: Any) -> None:
    """Compare ``actual`` against ``tests/e2e/golden/<name>.json``, or record it."""
    path = GOLDEN_DIR / f"{name}.json"
    # Round-trip through JSON so the comparison sees exactly what a fork will load,
    # rather than Python objects that serialise lossily (tuples, NamedTuples, Decimal).
    actual = json.loads(json.dumps(actual, sort_keys=True, allow_nan=False))

    if request.config.getoption("--record-golden"):
        GOLDEN_DIR.mkdir(exist_ok=True)
        path.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n")
        pytest.skip(f"recorded {path.relative_to(Path(__file__).parents[2])}")

    assert path.exists(), f"missing golden file {path}; run pytest --record-golden to create it"
    _compare(json.loads(path.read_text()), actual, name)


# ---------------------------------------------------------------------------
# The five synthetic questions
# ---------------------------------------------------------------------------
#
# Fixed ids and fixed text, because the fixture LLM seeds itself on the prompt: change a
# question's wording and every recorded number moves. They are deliberately NOT reused
# from ``pipeline_test_helpers`` — that module serves tests that may legitimately want
# their questions edited, and an edit there must not silently invalidate a parity record.


def _binary_question() -> BinaryQuestion:
    return BinaryQuestion(
        question_text="Will the golden harness question resolve YES by December 2026?",
        id_of_question=9001,
        id_of_post=19001,
        page_url="https://www.metaculus.com/questions/9001/",
        background_info="A synthetic question that exists only to pin pipeline output.",
        resolution_criteria="Resolves YES if the recorded value matches the golden file.",
        fine_print="Synthetic; never published.",
        open_time=_OPEN_TIME,
        scheduled_resolution_time=_RESOLVE_TIME,
        api_json={"my_forecasts": {"latest": {"forecast_values": None}}},
    )


def _mc_question() -> MultipleChoiceQuestion:
    return MultipleChoiceQuestion(
        question_text="Which golden option describes the outcome?",
        id_of_question=9002,
        id_of_post=19002,
        page_url="https://www.metaculus.com/questions/9002/",
        background_info="A synthetic question that exists only to pin pipeline output.",
        resolution_criteria="Resolves to the option that matches the golden file.",
        fine_print="Synthetic; never published.",
        open_time=_OPEN_TIME,
        scheduled_resolution_time=_RESOLVE_TIME,
        options=["Option A", "Option B", "Option C"],
        api_json={"my_forecasts": {"latest": {"forecast_values": None}}},
    )


def _numeric_question(
    qid: int,
    *,
    open_upper_bound: bool,
    cls: type[NumericQuestion] = NumericQuestion,
    **extra: Any,
) -> NumericQuestion:
    return cls(
        question_text=f"What value will the golden numeric series {qid} take in December 2026?",
        id_of_question=qid,
        id_of_post=qid + 10000,
        page_url=f"https://www.metaculus.com/questions/{qid}/",
        background_info="A synthetic question that exists only to pin pipeline output.",
        resolution_criteria="Resolves to the value recorded in the golden file.",
        fine_print="Synthetic; never published.",
        open_time=_OPEN_TIME,
        scheduled_resolution_time=_RESOLVE_TIME,
        lower_bound=0.0,
        upper_bound=20.0,
        open_lower_bound=False,
        open_upper_bound=open_upper_bound,
        zero_point=None,
        unit_of_measure="%",
        api_json={"my_forecasts": {"latest": {"forecast_values": None}}},
        **extra,
    )


def golden_questions() -> list[Any]:
    """Binary, multiple choice, numeric closed, numeric open-upper, discrete numeric.

    ``DiscreteQuestion`` subclasses ``NumericQuestion`` and is distinguished by a
    ``cdf_size`` other than the continuous 201, which is what routes it down the
    discrete branch of the numeric pipeline. It is in the set because that branch is
    reachable in production and would otherwise move unobserved.
    """
    return [
        _binary_question(),
        _mc_question(),
        _numeric_question(9003, open_upper_bound=False),
        _numeric_question(9004, open_upper_bound=True),
        _numeric_question(9005, open_upper_bound=False, cls=DiscreteQuestion, cdf_size=21),
    ]


# ---------------------------------------------------------------------------
# Bot construction
# ---------------------------------------------------------------------------


def build_golden_bot(strategy: AggregationStrategy, *, n_forecasters: int = 3, seed: int = 0, **overrides: Any):
    """A fully-fixture bot. Every LLM slot, including the helpers, and the research provider.

    Leaving one helper slot real would reintroduce spend through a back door — the same
    partial guard ``backtest/cli.py:_create_fixture_bots`` documents. ``default`` is
    listed explicitly because the framework's LLM-purpose validation raises without it.
    """
    helper = FixtureLlm(model="fixture/helper", seed=seed)
    config: dict[str, Any] = {
        "publish_reports_to_metaculus": False,
        "is_benchmarking": True,
        "research_reports_per_question": 1,
        "predictions_per_research_report": 1,
        "aggregation_strategy": strategy,
        "research_provider": fixture_research_provider,
        "max_questions_per_run": None,
        "min_forecasters_to_publish": 1,
        # Order feeds the stacker prompt, and a shuffled prompt reseeds the fixture.
        "stacking_randomize_order": False,
        "stacking_fallback_on_failure": True,
        "llms": {
            "forecasters": build_fixture_forecasters(n_forecasters, seed=seed),
            "stacker": FixtureLlm(model="fixture/stacker", seed=seed),
            "analyzer": helper,
            "default": helper,
            "summarizer": helper,
            "parser": helper,
            "researcher": helper,
        },
    }
    config.update(overrides)
    return SibylBot(**config)


@pytest.fixture
def frozen_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the prompt's notion of "now" so the recorded predictions are reproducible.

    Patches the ``datetime`` bound in ``sibyl.forecasting.prompts`` rather than trying to
    freeze the clock globally: that module is the only place the pipeline reads the wall
    clock on this path, and a narrow patch cannot silently change what anything else
    does.
    """
    from sibyl.forecasting import prompts

    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz: Any = None) -> datetime:  # type: ignore[override]
            return GOLDEN_NOW if tz is None else GOLDEN_NOW.astimezone(tz)

    monkeypatch.setattr(prompts, "datetime", _FrozenDatetime)


@pytest.fixture
def no_retry_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the real 1s/10s/30s backoff between retries of a failing LLM call.

    A forecaster that always raises waits out every backoff before it is dropped, which
    cost the guard scenario 41s. Like ``frozen_clock``, the patch is narrow: only
    ``llm_retry``'s ``asyncio`` is replaced, its ``sleep`` yields without waiting, and
    the attempts, their order and the recorded drop are unchanged.
    """
    from sibyl.platform import llm_retry

    class _NoBackoff:
        def __getattr__(self, name: str) -> Any:
            return getattr(asyncio, name)

        @staticmethod
        async def sleep(_seconds: float, *_args: Any) -> None:
            await asyncio.sleep(0)

    monkeypatch.setattr(llm_retry, "asyncio", _NoBackoff())


@pytest.fixture
def all_providers_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every research/provider flag off, including the three stacking gates.

    ``tests/conftest.py`` already clears the provider flags and force-enables the
    stacking gates suite-wide. A parity record must not depend on that: it states its
    own environment so it means the same thing if the shared fixture changes.
    """
    from sibyl.config import settings as _constants

    for attr in dir(_constants):
        if attr.endswith("_ENABLED_ENV"):
            value = getattr(_constants, attr)
            if isinstance(value, str):
                monkeypatch.setenv(value, "false")


@pytest.fixture
def per_type_stacking_on(monkeypatch: pytest.MonkeyPatch) -> None:
    from sibyl.config import settings as _constants

    for env_name in (
        _constants.BINARY_STACKING_ENABLED_ENV,
        _constants.MC_STACKING_ENABLED_ENV,
        _constants.NUMERIC_STACKING_ENABLED_ENV,
    ):
        monkeypatch.setenv(env_name, "true")


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def _serialize_prediction(report: Any) -> Any:
    prediction = report.prediction

    if isinstance(prediction, float):
        return {"kind": "binary", "probability": prediction}

    options = getattr(prediction, "predicted_options", None)
    if options is not None:
        # A list, not a dict: option ORDER is part of the published payload, and a dict
        # would let a reordering pass silently.
        return {
            "kind": "multiple_choice",
            "options": [{"name": o.option_name, "probability": o.probability} for o in options],
        }

    # Numeric. ``cdf`` is what actually gets published, so it is the field that must not
    # move; ``declared_percentiles`` is recorded alongside it because it localises a
    # regression to before or after the PCHIP fit.
    return {
        "kind": "numeric",
        "declared_percentiles": [
            {"percentile": p.percentile, "value": p.value} for p in prediction.declared_percentiles
        ],
        "cdf": [{"percentile": p.percentile, "value": p.value} for p in prediction.cdf],
    }


def _serialize_report(report: Any) -> dict[str, Any]:
    return {
        "question_id": report.question.id_of_question,
        "report_type": type(report).__name__,
        "prediction": _serialize_prediction(report),
        "explanation": report.explanation,
    }


def _serialize_counters(bot: SibylBot) -> dict[str, Any]:
    """The eleven alertable counters, the five stacking counters, and the alertable sum.

    Named individually rather than harvested by ``dir()``: the point is to fail when a
    counter DISAPPEARS, and a derived list would quietly shrink with the thing it measures.
    """
    return {
        "forecasters_dropped": bot._forecasters_dropped_count,
        "questions_failed_to_publish": bot._questions_failed_to_publish,
        "stacker_primary_failed": bot._stacker_primary_failed_count,
        "stacker_fallback_used": bot._stacker_fallback_used_count,
        "stacker_fallback_failed": bot._stacker_fallback_failed_count,
        "research_provider_failures": bot._research_provider_failure_count,
        "summarizer_failures": bot._summarizer_failure_count,
        "gap_fill_v2_errors": bot._gap_fill_v2_error_count,
        "prediction_market_degraded": bot._prediction_market_degraded_count,
        "prediction_market_source_losses": bot._prediction_market_source_loss_count,
        "provider_degradation": bot._provider_degradation_count,
        "conditional_stacking_triggered": bot._conditional_stacking_triggered_count,
        "conditional_stacking_skipped": bot._conditional_stacking_skipped_count,
        "conditional_stacking_skipped_single_forecaster": bot._conditional_stacking_skipped_single_forecaster_count,
        "conditional_stacking_crux_failures": bot._conditional_stacking_crux_failures,
        "conditional_stacking_search_failures": bot._conditional_stacking_search_failures,
        "alertable_count": bot.alertable_count,
    }


# ---------------------------------------------------------------------------
# Scenario 1 and 2: the pipeline
# ---------------------------------------------------------------------------


async def _run_pipeline(strategy: AggregationStrategy) -> dict[str, Any]:
    bot = build_golden_bot(strategy)
    reports = await bot.forecast_questions(golden_questions(), return_exceptions=True)

    failures = [r for r in reports if isinstance(r, BaseException)]
    assert not failures, f"golden pipeline raised: {failures!r}"
    # Re-bound rather than reusing ``reports``: the return type is a union with
    # BaseException, and the assertion above does not narrow it for the type checker.
    succeeded = [r for r in reports if not isinstance(r, BaseException)]

    payload: dict[str, Any] = {
        "strategy": strategy.value,
        "reports": [_serialize_report(r) for r in succeeded],
        "counters": _serialize_counters(bot),
    }
    if strategy is AggregationStrategy.CONDITIONAL_STACKING:
        # Read back out of the published comment rather than off the bot. ``_stacker_outcome``
        # is POPPED per question during comment annotation (forecaster.py:836), so by the end
        # of the batch it is empty by design. Parsing the comment with the production parser
        # records the outcome AND pins the marker contract the two sides share -- which is the
        # more valuable of the two, since that marker is what the telemetry archive harvests.
        # Keyed by question id as a string: JSON object keys are always strings, and an int
        # key would compare a recorded "9001" against a fresh 9001.
        payload["stacker_outcome"] = {
            str(r.question.id_of_question): parse_stacker_outcome_marker(r.explanation) for r in succeeded
        }
    return payload


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_golden_pipeline_median(
    request: pytest.FixtureRequest, all_providers_off: None, frozen_clock: None
) -> None:
    """Scenario 1: five question types through ``forecast_questions`` under MEDIAN."""
    assert_golden(request, "pipeline_median", await _run_pipeline(AggregationStrategy.MEDIAN))


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_golden_pipeline_conditional_stacking(
    request: pytest.FixtureRequest,
    all_providers_off: None,
    per_type_stacking_on: None,
    frozen_clock: None,
) -> None:
    """Scenario 2: the same questions with all three per-type stacking gates open.

    ``all_providers_off`` runs first and clears the gates; ``per_type_stacking_on``
    re-sets them. Fixture argument order is the dependency order, so this is stable
    rather than a coin toss.
    """
    assert_golden(
        request,
        "pipeline_conditional_stacking",
        await _run_pipeline(AggregationStrategy.CONDITIONAL_STACKING),
    )


# ---------------------------------------------------------------------------
# Scenario 3: the min-forecasters guard
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_golden_min_forecasters_guard(
    request: pytest.FixtureRequest, all_providers_off: None, frozen_clock: None, no_retry_backoff: None
) -> None:
    """Scenario 3: a raising forecaster drops the question, and the drop is attributed.

    Three forecasters, one of which always raises, against a guard of 3. The surviving
    two are below the threshold, so nothing publishes — and the recorded artifact is the
    drop telemetry, which is the part a restructure could silently lose.
    """
    from sibyl.telemetry.drops import ForecasterDrop

    class RaisingFixtureLlm(FixtureLlm):
        """Raises where a real provider would, so the drop is classified, not swallowed."""

        async def invoke(self, prompt: Any, system_prompt: str | None = None) -> str:
            raise RuntimeError("golden: forecaster unavailable")

    forecasters = build_fixture_forecasters(2, seed=0) + [RaisingFixtureLlm(model="fixture/raises", seed=0)]
    bot = build_golden_bot(
        AggregationStrategy.MEDIAN,
        min_forecasters_to_publish=3,
        llms={
            "forecasters": forecasters,
            "stacker": FixtureLlm(model="fixture/stacker", seed=0),
            "analyzer": FixtureLlm(model="fixture/helper", seed=0),
            "default": FixtureLlm(model="fixture/helper", seed=0),
            "summarizer": FixtureLlm(model="fixture/helper", seed=0),
            "parser": FixtureLlm(model="fixture/helper", seed=0),
            "researcher": FixtureLlm(model="fixture/helper", seed=0),
        },
    )

    reports = await bot.forecast_questions([_binary_question()], return_exceptions=True)

    assert_golden(
        request,
        "min_forecasters_guard",
        {
            "report_types": sorted(type(r).__name__ for r in reports),
            "published_report_count": sum(1 for r in reports if not isinstance(r, BaseException)),
            "drops": sorted(
                ({"model": d.model, "qid": d.qid, "cause": d.cause} for d in bot._forecaster_drops),
                key=lambda d: (d["model"], str(d["qid"]), d["cause"]),
            ),
            "counters": _serialize_counters(bot),
        },
    )
    assert all(isinstance(d, ForecasterDrop) for d in bot._forecaster_drops)


# ---------------------------------------------------------------------------
# Scenario 4: CLI exit codes
# ---------------------------------------------------------------------------
#
# The exit code is the whole contract between this bot and CI, and it is computed from
# module-global counters that a package move relocates. Recorded as a matrix rather than
# four separate files so a rule that disappears shows up as a missing key.

# Chosen relative to CREDIT_ALERT_RESUME_DATE (2026-09-10). Fixed literals, not offsets
# from today, so the record does not change meaning as the wall clock crosses the date.
_DURING_SUPPRESSION = date(2026, 7, 25)
_AFTER_RESUME_DATE = date(2026, 10, 1)


def _cli_exit_code(
    *,
    alertable_count: int,
    fallback_count: int = 0,
    credit_fallback: int = 0,
    today: date,
) -> int | None:
    """Run ``cli.main()`` under full stubs and report the exit code (``None`` = clean).

    Mirrors ``tests/test_cli.py::_cli_main_test_mode``. ``check_deprecation_alerts_and_exit``
    is stubbed here although that helper does not stub it: it reads a module-global list
    that another test file can leave populated, which would turn the clean case red for a
    reason having nothing to do with parity.
    """
    import sys
    from unittest.mock import AsyncMock, MagicMock, patch

    import sibyl.platform.key_fallback as fb_module
    from sibyl.cli import main as cli_main
    from sibyl.config.settings import credit_alerts_active

    stub_bot = MagicMock()
    stub_bot.alertable_count = alertable_count
    stub_bot.forecast_questions = AsyncMock(return_value=[])
    stub_bot.forecast_on_tournament = AsyncMock(return_value=[])

    stub_telemetry = MagicMock()
    stub_telemetry.log_end_and_check_floor.return_value = False

    fb_module._fallback_key_count = fallback_count
    fb_module._credit_key_fallback_count = credit_fallback
    fb_module._primary_route_404_fallback_count = 0

    argv_backup = sys.argv
    sys.argv = ["cli", "--mode", "test_questions"]
    try:
        with (
            patch("sibyl.cli.credit_alerts_active", lambda: credit_alerts_active(today)),
            patch("sibyl.cli.SibylBot", return_value=stub_bot),
            patch("sibyl.cli.MetaculusApi", MagicMock()),
            patch("sibyl.cli.apply_publish_hardening"),
            patch("sibyl.cli.apply_fetch_hardening"),
            patch("sibyl.cli.check_tournament_dates"),
            patch("sibyl.cli.verify_metaculus_api_identity"),
            patch("sibyl.cli.check_deprecation_alerts_and_exit"),
            patch.object(type(stub_bot), "log_report_summary", create=True, return_value=None),
            patch("sibyl.cli.CreditTelemetry", return_value=stub_telemetry),
        ):
            try:
                cli_main()
            except SystemExit as exc:
                return exc.code if isinstance(exc.code, int) else 1
        return None
    finally:
        sys.argv = argv_backup
        fb_module._fallback_key_count = 0
        fb_module._credit_key_fallback_count = 0
        fb_module._primary_route_404_fallback_count = 0


def test_golden_cli_exit_codes(request: pytest.FixtureRequest) -> None:
    """Scenario 4: clean, alertable, and credit fallback inside/outside the window."""
    matrix = {
        "clean_run": _cli_exit_code(alertable_count=0, today=_DURING_SUPPRESSION),
        "alertable_nonzero": _cli_exit_code(alertable_count=1, today=_DURING_SUPPRESSION),
        "credit_fallback_inside_suppression": _cli_exit_code(
            alertable_count=0, fallback_count=1, credit_fallback=1, today=_DURING_SUPPRESSION
        ),
        "credit_fallback_outside_suppression": _cli_exit_code(
            alertable_count=0, fallback_count=1, credit_fallback=1, today=_AFTER_RESUME_DATE
        ),
    }
    assert_golden(request, "cli_exit_codes", matrix)


# ---------------------------------------------------------------------------
# Scenario 5: eval
# ---------------------------------------------------------------------------


def test_golden_eval_aggregate_scores(request: pytest.FixtureRequest) -> None:
    from sibyl.eval.backtest.analysis import aggregate_scores
    from sibyl.eval.backtest.scoring import QuestionScore

    scores = [
        QuestionScore(1, "binary", 0.25, 0.16, "brier"),
        QuestionScore(2, "binary", 0.09, 0.12, "brier"),
        QuestionScore(3, "binary", 0.49, None, "brier"),
        QuestionScore(4, "numeric", -1.25, -1.10, "numeric_log_score"),
        QuestionScore(5, "numeric", -0.80, -0.95, "numeric_log_score"),
        QuestionScore(6, "multiple_choice", -1.05, None, "mc_log_score"),
    ]
    assert_golden(request, "eval_aggregate_scores", aggregate_scores(scores))


def test_golden_eval_aggregate_paired(request: pytest.FixtureRequest) -> None:
    """``seed=0`` with a fixed comparison, so the bootstrap draw is pinned too.

    ``aggregate_paired`` derives a per-group seed by SHA-256 of the group key rather than
    builtin ``hash``, which is what makes this reproducible across processes and so
    legitimate to record at all.
    """
    from sibyl.eval.ablation.scoring import PairedScore, aggregate_paired

    def row(qid: int, delta: float) -> PairedScore:
        return PairedScore(
            qid=qid,
            question_type="binary",
            metric="brier",
            comparison="median-stack",
            score_stack=0.20,
            score_stack_aug=0.20,
            score_median=0.20 + delta,
            delta=delta,
            higher_is_better=False,
        )

    deltas = [0.05, -0.02, 0.01, 0.00, -0.04, 0.03, 0.02, -0.01, 0.06, -0.03]
    stats = aggregate_paired([row(i, d) for i, d in enumerate(deltas)], n_bootstrap=200, seed=0)

    assert_golden(
        request,
        "eval_aggregate_paired",
        [
            {
                "metric": s.metric,
                "question_type": s.question_type,
                "comparison": s.comparison,
                "n": s.n,
                "mean_delta": s.mean_delta,
                "bootstrap_ci_low": s.bootstrap_ci_low,
                "bootstrap_ci_high": s.bootstrap_ci_high,
                "sign_test_p": s.sign_test_p,
                "wilcoxon_p": s.wilcoxon_p,
                "higher_is_better": s.higher_is_better,
                "median_delta": s.median_delta,
                "median_ci_low": s.median_ci_low,
                "median_ci_high": s.median_ci_high,
                "n_clean": s.n_clean,
                "mean_delta_clean": s.mean_delta_clean,
            }
            for s in stats
        ],
    )


def test_golden_eval_performance_parsing(request: pytest.FixtureRequest) -> None:
    """Every parser against every record of the mini comment fixture.

    The parsers are the read side of the published comment format: if a restructure
    changes what a comment says, this is where it shows, because the fixture's twelve
    records were selected to cover one distinct parser-relevant shape each.
    """
    from sibyl.eval.performance.parsing import (
        parse_forecaster_model_map,
        parse_inferred_stacker_outcome,
        parse_per_model_forecasts,
        parse_per_model_mc_option_probs,
        parse_per_model_numeric_percentiles,
        parse_stacked_marker,
    )

    fixture = Path(__file__).parents[1] / "data" / "performance_comments_mini.jsonl"
    records = [json.loads(line) for line in fixture.read_text().splitlines() if line.strip()]
    assert records, f"{fixture} is empty"

    out = {}
    for record in sorted(records, key=lambda r: r["post_id"]):
        comment = record["comment_text"]
        outcome, source = parse_inferred_stacker_outcome(comment)
        out[str(record["post_id"])] = {
            "type": record["type"],
            "per_model_forecasts": parse_per_model_forecasts(comment),
            # int keys become strings through JSON; do it here so the recorded shape and
            # the fresh one are the same object, not equal-looking ones.
            "forecaster_model_map": {str(k): v for k, v in parse_forecaster_model_map(comment).items()},
            "numeric_percentiles": {
                model: [list(pair) for pair in pairs]
                for model, pairs in parse_per_model_numeric_percentiles(comment).items()
            },
            "mc_option_probs": parse_per_model_mc_option_probs(comment),
            "inferred_stacker_outcome": [outcome, source],
            "stacked_marker": parse_stacked_marker(comment),
        }
    assert_golden(request, "eval_performance_parsing", out)


def test_golden_comment_trimming(request: pytest.FixtureRequest) -> None:
    """A comment inflated past ``COMMENT_CHAR_LIMIT``, then trimmed.

    Inflation is required, not decorative: ``trim_comment`` returns its input unchanged
    below the limit, so trimming a fixture record directly would pin the identity
    function and pass forever.
    """
    from sibyl.config.settings import COMMENT_CHAR_LIMIT
    from sibyl.publishing.comment.trimming import trim_comment

    fixture = Path(__file__).parents[1] / "data" / "performance_comments_mini.jsonl"
    records = [json.loads(line) for line in fixture.read_text().splitlines() if line.strip()]
    # post_id 41537 is the numeric, named, intact, boundary-marker-present record — the
    # shape that exercises every branch of the section-aware strategy chain.
    comment = next(r["comment_text"] for r in records if r["post_id"] == 41537)

    marker = "### Research Summary"
    filler = "filler_token " * 15_000
    inflated = comment.replace(marker, marker + "\n" + filler, 1) if marker in comment else comment + filler
    assert len(inflated) > COMMENT_CHAR_LIMIT, "fixture no longer inflates past the limit"

    trimmed = trim_comment(inflated)
    assert_golden(
        request,
        "comment_trimming",
        {
            "input_length": len(inflated),
            "output_length": len(trimmed),
            "comment_char_limit": COMMENT_CHAR_LIMIT,
            "head": trimmed[:2000],
            "tail": trimmed[-2000:],
        },
    )


def test_golden_numeric_pipeline(request: pytest.FixtureRequest) -> None:
    """Three fixed percentile sets through sanitize + PCHIP, across three bound shapes."""
    from forecasting_tools.data_models.numeric_report import Percentile

    from sibyl.forecasting.numeric.pipeline import build_numeric_distribution, sanitize_percentiles

    ladders = {
        "narrow": [3.0, 3.2, 3.4, 3.6, 3.8, 4.1, 4.3, 4.5, 5.0, 5.6, 6.2, 7.0, 7.8],
        "wide": [0.5, 1.0, 2.0, 3.5, 5.0, 7.5, 9.0, 11.0, 13.5, 16.0, 17.5, 18.5, 19.5],
        "skewed": [1.0, 1.1, 1.2, 1.4, 1.8, 2.6, 3.4, 4.8, 8.0, 12.0, 15.0, 17.0, 19.0],
    }
    levels = [0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 0.95, 0.975, 0.99]
    questions = {
        "closed": _numeric_question(9003, open_upper_bound=False),
        "open_upper": _numeric_question(9004, open_upper_bound=True),
        "discrete": _numeric_question(9005, open_upper_bound=False, cls=DiscreteQuestion, cdf_size=21),
    }

    out: dict[str, Any] = {}
    for ladder_name, values in ladders.items():
        for question_name, question in questions.items():
            percentiles = [Percentile(percentile=p, value=v) for p, v in zip(levels, values)]
            sanitized, zero_point = sanitize_percentiles(percentiles, question)
            distribution = build_numeric_distribution(sanitized, question, zero_point)
            out[f"{ladder_name}/{question_name}"] = {
                "zero_point": zero_point,
                "sanitized": [{"percentile": p.percentile, "value": p.value} for p in sanitized],
                "cdf": [{"percentile": p.percentile, "value": p.value} for p in distribution.cdf],
            }
    assert_golden(request, "numeric_pipeline", out)


# ---------------------------------------------------------------------------
# Scenario 6: surface inventory
# ---------------------------------------------------------------------------


def _repo_files() -> list[Path]:
    import subprocess

    root = Path(__file__).parents[2]
    listing = subprocess.run(
        ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    return [root / name for name in listing if name]


def test_golden_surface_inventory(request: pytest.FixtureRequest) -> None:
    """Scenario 6: the names the outside world uses.

    Env vars, Makefile targets, ``python -m`` entry points and telemetry markers are the
    surface a workflow, a runbook or an operator's shell history depends on. A rename
    inside the package is free; a rename here breaks something that is not in this repo.
    Later phases assert this list is unchanged except for an explicit allowlist.
    """
    import re

    from sibyl.config import settings as _constants
    from sibyl.telemetry.archive import MARKER_NAMES

    root = Path(__file__).parents[2]

    env_constants = {
        name: getattr(_constants, name)
        for name in dir(_constants)
        if name.endswith("_ENV") and isinstance(getattr(_constants, name), str)
    }

    makefile_targets = sorted(
        set(re.findall(r"^([A-Za-z_][A-Za-z0-9_-]*):", (root / "Makefile").read_text(), re.MULTILINE))
    )

    python_m: set[str] = set()
    for path in _repo_files():
        if path.suffix in {".png", ".jsonl", ".lock"} or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        python_m.update(re.findall(r"python3? -m ([A-Za-z_][\w.]*)", text))

    assert_golden(
        request,
        "surface_inventory",
        {
            "env_constants": dict(sorted(env_constants.items())),
            "makefile_targets": makefile_targets,
            "python_m_modules": sorted(python_m),
            "telemetry_markers": sorted(MARKER_NAMES),
        },
    )
