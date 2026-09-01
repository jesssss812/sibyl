"""Tests for FORECASTER_SPREAD telemetry (sibyl/aggregation/agreement.py).

Why this exists: `compute_spread` already runs on every multi-forecaster question, but its
value was only ever logged when conditional stacking TRIGGERED. Stacking is disabled on all
three question types in prod, so the spread has been computed and discarded on every
question the bot has ever forecast. The distribution of how often the ensemble agrees with
itself — the single most useful number for deciding whether 3 forecasters is the right
count, or whether a disagreement-triggered escalation would ever fire — has never been
recorded.

This is deliberately measurement only. It changes no forecast and takes no routing
decision; it makes the existing signal durable so a policy can be designed against real
numbers rather than an assumption.
"""

from __future__ import annotations

import pytest

from sibyl.aggregation.agreement import format_spread_marker


class TestMarkerShape:
    """The marker is a parsed contract, same stance as CLOSE_MARGIN and RESOLUTION_GUARD."""

    def test_full_marker(self) -> None:
        assert format_spread_marker(
            question_id=44620,
            question_type="binary",
            n_forecasters=3,
            spread=0.0821,
            threshold=0.15,
            exceeds=False,
        ) == ("FORECASTER_SPREAD: question=44620 qtype=binary n=3 spread=0.0821 threshold=0.1500 exceeds=False")

    def test_exceeds_is_rendered_truthfully(self) -> None:
        marker = format_spread_marker(
            question_id=1,
            question_type="numeric",
            n_forecasters=3,
            spread=0.42,
            threshold=0.15,
            exceeds=True,
        )
        assert "exceeds=True" in marker

    def test_threshold_may_be_unknown(self) -> None:
        """A strategy with no threshold still has a spread worth recording."""
        marker = format_spread_marker(
            question_id=7,
            question_type="mc",
            n_forecasters=2,
            spread=0.31,
            threshold=None,
            exceeds=None,
        )
        assert "threshold=n/a" in marker and "exceeds=n/a" in marker

    def test_four_decimal_places_are_pinned(self) -> None:
        """Spread values are small; two decimals would collapse distinct questions."""
        marker = format_spread_marker(
            question_id=1,
            question_type="binary",
            n_forecasters=3,
            spread=0.00049,
            threshold=0.15,
            exceeds=False,
        )
        assert "spread=0.0005" in marker


class TestQuestionTypeLabel:
    """The label must match the cohort names the eval harness already slices on."""

    @pytest.mark.parametrize(
        "cls_name,expected",
        [
            ("BinaryQuestion", "binary"),
            ("MultipleChoiceQuestion", "mc"),
            ("NumericQuestion", "numeric"),
            ("DiscreteQuestion", "numeric"),
        ],
    )
    def test_label_for(self, cls_name: str, expected: str) -> None:
        from sibyl.aggregation.agreement import question_type_label

        assert question_type_label(cls_name) == expected

    def test_unknown_type_is_labelled_not_guessed(self) -> None:
        from sibyl.aggregation.agreement import question_type_label

        assert question_type_label("SomeFutureQuestion") == "unknown"


class TestWiring:
    """It must emit on EVERY multi-forecaster question, not only when stacking fires.

    Emitting only on the triggered path is the bug this replaces: with stacking off, that
    path never runs and the log is empty.
    """

    @pytest.mark.asyncio
    async def test_marker_logged_when_stacking_is_disabled(self, caplog, monkeypatch) -> None:
        from sibyl.config import settings as constants

        for flag in (
            constants.BINARY_STACKING_ENABLED_ENV,
            constants.MC_STACKING_ENABLED_ENV,
            constants.NUMERIC_STACKING_ENABLED_ENV,
        ):
            monkeypatch.setenv(flag, "false")

        with caplog.at_level("INFO", logger="sibyl.aggregation.routing"):
            await _route_three_binary_forecasts()

        markers = [r.message for r in caplog.records if "FORECASTER_SPREAD:" in r.message]
        assert len(markers) == 1, markers
        assert "n=3" in markers[0]
        assert "exceeds=False" in markers[0]

    @pytest.mark.asyncio
    async def test_no_marker_for_a_single_survivor(self, caplog) -> None:
        """Spread is undefined at n=1 — compute_spread raises. Must not emit a fake zero."""
        with caplog.at_level("INFO", logger="sibyl.aggregation.routing"):
            await _route_three_binary_forecasts(n=1)

        assert not any("FORECASTER_SPREAD:" in r.message for r in caplog.records)


async def _route_three_binary_forecasts(n: int = 3):
    """Drive route_after_forecasts with n canned binary forecasts and no LLM."""
    from collections import defaultdict
    from typing import cast
    from unittest.mock import AsyncMock, Mock

    from forecasting_tools import BinaryQuestion, GeneralLlm, ReasonedPrediction
    from forecasting_tools.data_models.data_organizer import PredictionTypes

    from sibyl.aggregation.pipeline import AggregationPipeline
    from sibyl.aggregation.routing import route_after_forecasts
    from sibyl.aggregation.strategies import AggregationStrategy

    llm = GeneralLlm(model="test-model", temperature=0.0)
    question = BinaryQuestion(
        question_text="Will X?",
        id_of_question=44620,
        id_of_post=44620,
        api_json={"type": "binary"},
    )
    values = [0.30, 0.34, 0.32][:n]
    # cast: ReasonedPrediction is invariant in its type param, so a list of the binary
    # specialisation is not a list[ReasonedPrediction[PredictionTypes]] to the checker
    # even though it is exactly what the runtime expects here.
    preds = cast(
        "list[ReasonedPrediction[PredictionTypes]]",
        [ReasonedPrediction(prediction_value=v, reasoning=f"Model: m{i}") for i, v in enumerate(values)],
    )

    bot = Mock()
    bot.aggregation_strategy = AggregationStrategy.CONDITIONAL_STACKING
    bot._get_threshold_for_question = Mock(return_value=0.15)
    # A real number: _skip_stacking_for_budget compares the remaining wall-clock
    # budget against a minimum, and a Mock would raise on the comparison.
    bot._remaining_budget_seconds = Mock(return_value=9999.0)
    bot._conditional_stacking_skipped_count = 0
    bot._conditional_stacking_skipped_single_forecaster_count = 0
    bot._conditional_stacking_triggered_count = 0
    bot._stacker_outcome = {}
    bot._register_expected_base_combine = Mock()
    bot._pipeline = AggregationPipeline(
        strategy=AggregationStrategy.CONDITIONAL_STACKING,
        stacker_llm=llm,
        parser_llm=llm,
        stacking_fallback_on_failure=True,
        stacking_randomize_order=False,
        stacking_spread_thresholds={"binary": 0.15, "mc": 0.20, "numeric": 0.15},
        discrete_integer_votes=defaultdict(list),
    )
    bot._aggregate_predictions = AsyncMock(return_value=0.32)

    return await route_after_forecasts(
        bot,
        question=question,
        qid=44620,
        valid_predictions=preds,
        errors=[],
        research="r",
        summary_report="s",
        diagnostics_block=None,
        per_q_start=0.0,
    )
