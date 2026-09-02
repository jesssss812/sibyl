"""Live coverage of the FREE model tier, against real OpenRouter :free models.

Exercises the actual inference path end-to-end with no mocks. Every test requires
OPENROUTER_API_KEY and is gated behind ``@pytest.mark.live``, so it never runs in CI or
during ``make test`` (addopts excludes ``live`` by default).

These cost nothing: every model here is an OpenRouter ``:free`` slug. That is what makes
a live test defensible in a repo whose first rule is that no test may spend. Nothing
here publishes -- the one end-to-end case runs ``forecast_questions`` on a synthetic
question with ``publish_reports_to_metaculus=False``. Publishing is ``make e2e_free``,
which is a deliberate operator action against the unscored sandbox tournament.

Opt-in: ``pytest -m live``
"""

import asyncio
import os
import re
from datetime import datetime, timedelta
from typing import Any, cast

import pytest
from forecasting_tools import (
    BinaryQuestion,
    GeneralLlm,
    MultipleChoiceQuestion,
    NumericQuestion,
    structure_output,
)
from forecasting_tools.data_models.binary_report import BinaryPrediction

from sibyl.forecasting.prompts import binary_prompt, multiple_choice_prompt, numeric_prompt

FREE_MODEL = "openrouter/google/gemma-4-31b-it:free"
SKIP_REASON = "OPENROUTER_API_KEY not set"

skip_no_key = pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason=SKIP_REASON)

_NOW = datetime.now()
_OPEN = _NOW - timedelta(days=30)
_RESOLVE = _NOW + timedelta(days=180)


def _make_binary_question() -> BinaryQuestion:
    return BinaryQuestion(
        question_text="Will the global average temperature in 2026 exceed the 2025 average?",
        id_of_post=99999,
        id_of_question=99999,
        background_info="Climate scientists track annual global mean surface temperature anomalies.",
        resolution_criteria="Resolves YES if the NASA GISS annual mean for 2026 exceeds 2025.",
        fine_print="Uses the January release of the prior year's annual mean.",
        open_time=_OPEN,
        scheduled_resolution_time=_RESOLVE,
    )


def _make_numeric_question() -> NumericQuestion:
    return NumericQuestion(
        question_text="What will the US unemployment rate be in December 2026?",
        id_of_post=99998,
        id_of_question=99998,
        background_info="The BLS publishes monthly unemployment figures.",
        resolution_criteria="Resolves to the seasonally-adjusted U-3 rate published by BLS.",
        fine_print="",
        unit_of_measure="percent",
        lower_bound=0.0,
        upper_bound=20.0,
        open_lower_bound=False,
        open_upper_bound=True,
        open_time=_OPEN,
        scheduled_resolution_time=_RESOLVE,
    )


def _make_mc_question() -> MultipleChoiceQuestion:
    return MultipleChoiceQuestion(
        question_text="Which party will win the next UK general election?",
        id_of_post=99997,
        id_of_question=99997,
        options=["Labour", "Conservative", "Liberal Democrats", "Other"],
        background_info="UK general elections use first-past-the-post constituency voting.",
        resolution_criteria="Resolves to the party winning the most seats.",
        fine_print="",
        open_time=_OPEN,
        scheduled_resolution_time=_RESOLVE,
    )


def _make_llm() -> GeneralLlm:
    return GeneralLlm(model=FREE_MODEL, temperature=0.0, timeout=120)


@pytest.mark.live
@skip_no_key
async def test_free_model_binary_forecast():
    llm = _make_llm()
    question = _make_binary_question()
    prompt = binary_prompt(question, research="No relevant research found.")

    response = await llm.invoke(prompt)

    assert len(response) > 100, f"Response too short ({len(response)} chars)"
    prob_pattern = re.compile(r"(\d{1,3})%|0\.\d+|1\.0")
    assert prob_pattern.search(response), "No probability-like value found in response"


@pytest.mark.live
@skip_no_key
async def test_free_model_numeric_percentiles():
    llm = _make_llm()
    question = _make_numeric_question()
    prompt = numeric_prompt(
        question,
        research="No relevant research found.",
        lower_bound_message="The lower bound is 0.0% (hard floor).",
        upper_bound_message="The upper bound is 20.0% (open — values above are possible).",
    )

    response = await llm.invoke(prompt)

    percentile_pattern = re.compile(r"Percentile\s+[\d.]+:\s*([\d.]+)", re.IGNORECASE)
    matches = percentile_pattern.findall(response)
    assert len(matches) >= 5, f"Expected >=5 percentile lines, found {len(matches)}"

    values = [float(m) for m in matches]
    assert all(isinstance(v, float) for v in values)
    inversions = sum(1 for i in range(len(values) - 1) if values[i] > values[i + 1])
    assert inversions <= 2, f"Too many inversions ({inversions}) — values not roughly monotonic"


@pytest.mark.live
@skip_no_key
async def test_free_model_mc_forecast():
    llm = _make_llm()
    question = _make_mc_question()
    prompt = multiple_choice_prompt(question, research="No relevant research found.")

    response = await llm.invoke(prompt)

    options_mentioned = sum(1 for opt in question.options if opt.lower() in response.lower())
    assert options_mentioned >= 2, f"Only {options_mentioned} options mentioned in response"

    pct_pattern = re.compile(r"\d{1,3}%")
    assert pct_pattern.search(response), "No percentage-like number found"


@pytest.mark.live
@skip_no_key
async def test_free_model_structured_output_parseable():
    llm = _make_llm()
    question = _make_binary_question()
    prompt = binary_prompt(question, research="No relevant research found.")

    raw_response = await llm.invoke(prompt)

    parser_llm = _make_llm()
    prediction = await structure_output(
        text_to_structure=raw_response,
        output_type=BinaryPrediction,
        model=parser_llm,
    )

    assert isinstance(prediction, BinaryPrediction)
    assert 0.0 < prediction.prediction_in_decimal < 1.0


@pytest.mark.live
@skip_no_key
async def test_full_pipeline_single_question_free_model():
    from forecasting_tools.data_models.forecast_report import ForecastReport

    from sibyl.aggregation.strategies import AggregationStrategy
    from sibyl.forecasting.bot import SibylBot

    llm = _make_llm()

    async def fake_research(question):
        await asyncio.sleep(0)
        return "No relevant research found. The question is speculative."

    # The "forecasters" slot accepts a list[GeneralLlm] at runtime (consumed by
    # llm_setup), which the parent ForecastBot's dict[str, str | GeneralLlm]
    # signature can't express; cast to satisfy the type checker.
    llms = cast(
        "dict[str, str | GeneralLlm]",
        {
            "forecasters": [llm],
            "default": llm,
            "parser": llm,
            "researcher": llm,
            "summarizer": llm,
        },
    )
    forecaster = SibylBot(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        publish_reports_to_metaculus=False,
        aggregation_strategy=AggregationStrategy.MEAN,
        research_provider=fake_research,
        llms=llms,
        min_forecasters_to_publish=1,
    )

    question = _make_binary_question()
    reports = await forecaster.forecast_questions([question])

    assert len(reports) == 1
    report = reports[0]
    assert isinstance(report, ForecastReport)
    assert isinstance(report.prediction, float)
    assert 0.0 < report.prediction < 1.0
    assert report.explanation and len(report.explanation) > 0


@pytest.mark.live
@pytest.mark.asyncio
async def test_free_roster_runs_the_real_pipeline_without_publishing():
    """The roster the free lane actually ships, driving the real pipeline once.

    The other tests here pin individual model behaviours; this one pins that
    ``get_roster(FREE)`` produces a set of slots that the pipeline can actually run
    together. That is a different failure mode -- every slug can be individually
    healthy while the assembled roster still fails the publish guard, or hands the
    parser output no forecaster agrees on.

    Publishing is off. This asserts a forecast came back, not that it was any good:
    free models are not expected to forecast well, and an assertion about the VALUE
    would be a flaky test dressed up as a quality bar.
    """
    from sibyl.aggregation.strategies import AggregationStrategy
    from sibyl.config.models import ModelTier, get_roster
    from sibyl.forecasting.bot import SibylBot

    roster = get_roster(ModelTier.FREE)

    async def _canned_research(_question) -> str:
        return "No external research was performed for this smoke test."

    bot = SibylBot(
        publish_reports_to_metaculus=False,
        research_reports_per_question=1,
        predictions_per_research_report=1,
        aggregation_strategy=AggregationStrategy.MEAN,
        research_provider=_canned_research,
        max_questions_per_run=1,
        min_forecasters_to_publish=1,
        # dict[str, Any] for the same reason cli.py annotates it that way: "forecasters"
        # holds a list while every other slot holds a single LLM, and dict values are
        # invariant, so the framework's dict[str, str | GeneralLlm] cannot express it.
        llms=cast(
            "dict[str, Any]",
            {
                "forecasters": list(roster.forecasters),
                "stacker": roster.stacker,
                "analyzer": roster.analyzer,
                "default": roster.parser,
                "summarizer": roster.summarizer,
                "parser": roster.parser,
                "researcher": roster.researcher,
            },
        ),
    )

    question = BinaryQuestion(
        question_text="Will the sun rise tomorrow?",
        id_of_question=999_001,
        id_of_post=999_001,
        page_url="https://www.metaculus.com/questions/999001/",
        background_info="A deliberately trivial question; this test is about wiring, not skill.",
        resolution_criteria="Resolves YES if the sun rises.",
        fine_print="",
        open_time=datetime.now() - timedelta(days=1),
        scheduled_resolution_time=datetime.now() + timedelta(days=30),
        api_json={"my_forecasts": {"latest": None}},
    )

    reports = await bot.forecast_questions([question], return_exceptions=True)

    failures = [r for r in reports if isinstance(r, BaseException)]
    assert not failures, f"free roster failed to complete a forecast: {failures!r}"
    assert reports, "no report returned"
    prediction = cast("float", reports[0].prediction)  # type: ignore[union-attr]
    assert 0.0 <= prediction <= 1.0, prediction
