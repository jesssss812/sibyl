"""The A3 guard is wired into the ONE post-aggregation funnel, so every path gets it.

``AggregationPipeline`` reaches its post-processing step from eight different call
sites (base-combine, both stacking branches, the simple MEAN/MEDIAN path, ...), all of
which funnel through a single private method. These tests pin that funnelling: they
assert the guard applies on the paths a forecast can actually take to publication,
rather than testing the pure function again (that is test_resolution_state_guard.py).

The earlier version of this file only exercised ``AggregationStrategy.MEDIAN`` while
claiming to cover "eight call sites", and so passed while two base-combine returns
bypassed the guard entirely — including the single-survivor path, which is the highest
risk case A3 exists for. TestBaseCombinePathsAreGuarded below covers those explicitly.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone

import pytest
from forecasting_tools import BinaryQuestion, GeneralLlm, PredictedOption, PredictedOptionList

from metaculus_bot.aggregation_pipeline import AggregationPipeline
from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.constants import RESOLUTION_GUARD_CAP_HIGH, RESOLUTION_GUARD_ENABLED_ENV

# Relative to the REAL clock, not a fixed date. The guard reads
# `datetime.now(timezone.utc)` internally, so fixtures anchored to a hard-coded day
# drift out of the firing window and the suite starts failing on a calendar date rather
# than on a code change.
NOW = datetime.now(timezone.utc)


def _wide_open_binary(qid: int = 100) -> BinaryQuestion:
    """A binary question with half its open->close window still to run."""
    return BinaryQuestion(
        question_text="Will X happen?",
        id_of_question=qid,
        page_url="https://metaculus.com/questions/100",
        api_json={"type": "binary"},
        open_time=NOW - timedelta(days=10),
        close_time=NOW + timedelta(days=10),
    )


def _closing_soon_binary(qid: int = 101) -> BinaryQuestion:
    """Same question 95% of the way through its window."""
    return BinaryQuestion(
        question_text="Will X happen?",
        id_of_question=qid,
        page_url="https://metaculus.com/questions/101",
        api_json={"type": "binary"},
        open_time=NOW - timedelta(days=19),
        close_time=NOW + timedelta(days=1),
    )


def _pipeline() -> AggregationPipeline:
    test_llm = GeneralLlm(model="test-model", temperature=0.0)
    return AggregationPipeline(
        strategy=AggregationStrategy.MEDIAN,
        stacker_llm=test_llm,
        parser_llm=test_llm,
        stacking_fallback_on_failure=True,
        stacking_randomize_order=False,
        stacking_spread_thresholds={"binary": 0.15, "mc": 0.20, "numeric": 0.15},
        discrete_integer_votes=defaultdict(list),
    )


@pytest.mark.asyncio
class TestGuardAppliesThroughAggregate:
    async def test_early_extreme_median_is_railed(self) -> None:
        """Three forecasters agreeing at 0.99 on a wide-open question get railed to 0.95."""
        result = await _pipeline().aggregate([0.99, 0.99, 0.98], _wide_open_binary())
        assert result == RESOLUTION_GUARD_CAP_HIGH

    async def test_late_extreme_median_is_untouched(self) -> None:
        result = await _pipeline().aggregate([0.99, 0.99, 0.98], _closing_soon_binary())
        assert result == pytest.approx(0.99)

    async def test_non_extreme_median_is_untouched(self) -> None:
        result = await _pipeline().aggregate([0.60, 0.65, 0.70], _wide_open_binary())
        assert result == pytest.approx(0.65)

    async def test_disabled_via_env_passes_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(RESOLUTION_GUARD_ENABLED_ENV, "false")
        result = await _pipeline().aggregate([0.99, 0.99, 0.98], _wide_open_binary())
        assert result == pytest.approx(0.99)

    async def test_multiple_choice_is_out_of_scope_and_unchanged(self) -> None:
        """MC needs its own design (clamp interacts with ballot renormalization)."""
        from forecasting_tools import MultipleChoiceQuestion

        question = MultipleChoiceQuestion(
            question_text="Which?",
            id_of_question=200,
            page_url="https://metaculus.com/questions/200",
            options=["A", "B"],
            api_json={"type": "multiple_choice"},
            open_time=NOW - timedelta(days=10),
            close_time=NOW + timedelta(days=10),
        )
        ballots = [
            PredictedOptionList(
                predicted_options=[
                    PredictedOption(option_name="A", probability=0.99),
                    PredictedOption(option_name="B", probability=0.01),
                ]
            )
        ] * 3
        result = await _pipeline().aggregate(list(ballots), question)
        assert isinstance(result, PredictedOptionList)
        top = max(result.predicted_options, key=lambda o: o.probability)
        assert top.probability > RESOLUTION_GUARD_CAP_HIGH


@pytest.mark.asyncio
class TestGuardEmitsTelemetry:
    async def test_marker_logged_when_fired(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("INFO"):
            await _pipeline().aggregate([0.99, 0.99, 0.98], _wide_open_binary())
        assert any("RESOLUTION_GUARD: question=100" in r.message for r in caplog.records)

    async def test_no_marker_when_not_fired(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("INFO"):
            await _pipeline().aggregate([0.60, 0.65, 0.70], _wide_open_binary())
        assert not any("RESOLUTION_GUARD:" in r.message for r in caplog.records)


@pytest.mark.asyncio
class TestBaseCombinePathsAreGuarded:
    """`_base_combine` has two returns that skip Platt by design. They must NOT skip A3.

    Both are reachable in production: the single-prediction branch is the
    ``MIN_FORECASTERS_TO_PUBLISH = 1`` short-circuit in stacking_route.py, where one
    surviving forecaster's raw probability publishes with no median to temper it.
    """

    def _pipeline(self, strategy: AggregationStrategy) -> AggregationPipeline:
        test_llm = GeneralLlm(model="test-model", temperature=0.0)
        return AggregationPipeline(
            strategy=strategy,
            stacker_llm=test_llm,
            parser_llm=test_llm,
            stacking_fallback_on_failure=True,
            stacking_randomize_order=False,
            stacking_spread_thresholds={"binary": 0.15, "mc": 0.20, "numeric": 0.15},
            discrete_integer_votes=defaultdict(list),
        )

    async def test_single_survivor_is_guarded(self) -> None:
        result = await self._pipeline(AggregationStrategy.STACKING).aggregate([0.99], _wide_open_binary())
        assert result == RESOLUTION_GUARD_CAP_HIGH

    async def test_conditional_stacking_single_survivor_is_guarded(self) -> None:
        result = await self._pipeline(AggregationStrategy.CONDITIONAL_STACKING).aggregate([0.99], _wide_open_binary())
        assert result == RESOLUTION_GUARD_CAP_HIGH

    async def test_plain_stacking_base_combine_is_guarded(self) -> None:
        result = await self._pipeline(AggregationStrategy.STACKING).aggregate([0.99, 0.99, 0.98], _wide_open_binary())
        assert result == RESOLUTION_GUARD_CAP_HIGH

    async def test_late_extreme_still_untouched_on_base_combine(self) -> None:
        """The guard is wired in, but its firing condition is unchanged."""
        result = await self._pipeline(AggregationStrategy.STACKING).aggregate([0.99], _closing_soon_binary())
        assert result == pytest.approx(0.99)
