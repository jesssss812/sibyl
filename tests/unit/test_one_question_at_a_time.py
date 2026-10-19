"""One question at a time, soonest close first: the MiniBench queue order.

MiniBench questions are open for about three hours. Forecasting them one by one
keeps a single question's calls in flight, and taking the soonest close first
spends the time on the question that would be lost first.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import cast
from unittest.mock import MagicMock

import pytest
from forecasting_tools import BinaryQuestion, MetaculusQuestion
from forecasting_tools.forecast_bots.forecast_bot import ForecastBot

from sibyl.forecasting.bot import SibylBot

_NOW = datetime.now(timezone.utc)


def _question(name: str, closes_in_minutes: float | None) -> MetaculusQuestion:
    q = MagicMock(spec=BinaryQuestion)
    q.already_forecasted = False
    q.page_url = name
    q.close_time = None if closes_in_minutes is None else _NOW + timedelta(minutes=closes_in_minutes)
    return cast(MetaculusQuestion, q)


def _bot(**kwargs) -> SibylBot:
    return SibylBot(
        llms={"default": "mock", "summarizer": "mock_sum", "parser": "mock_parser", "researcher": "mock_researcher"},
        one_question_at_a_time=True,
        **kwargs,
    )


@pytest.fixture
def forecast_calls(monkeypatch):
    calls: list[list[str]] = []
    in_flight = 0
    peak = 0

    async def stub(self, questions_arg, return_exceptions=False):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        calls.append([q.page_url for q in questions_arg])
        await asyncio.sleep(0)
        in_flight -= 1
        return [MagicMock() for _ in questions_arg]

    monkeypatch.setattr(ForecastBot, "forecast_questions", stub, raising=True)
    yield calls
    assert peak <= 1, "two questions were in flight at once"


@pytest.mark.asyncio
async def test_forecasts_one_at_a_time_soonest_close_first(forecast_calls):
    questions = [_question("late", 170), _question("no-close", None), _question("soon", 20), _question("mid", 90)]

    results = await _bot().forecast_questions(questions)

    assert forecast_calls == [["soon"], ["mid"], ["late"], ["no-close"]]
    assert len(results) == 4


@pytest.mark.asyncio
async def test_cap_keeps_the_soonest_closing_questions(forecast_calls):
    questions = [_question(f"q{m}", m) for m in (150, 30, 120, 60)]

    await _bot(max_questions_per_run=2).forecast_questions(questions)

    assert forecast_calls == [["q30"], ["q60"]]


@pytest.mark.asyncio
async def test_question_closed_before_its_turn_is_skipped(forecast_calls):
    questions = [_question("closed", -1), _question("open", 60)]

    results = await _bot().forecast_questions(questions, return_exceptions=True)

    assert forecast_calls == [["open"]]
    assert len(results) == 1


def test_off_by_default():
    bot = SibylBot(llms={"default": "mock", "summarizer": "s", "parser": "p", "researcher": "r"})
    assert bot.one_question_at_a_time is False


@pytest.mark.asyncio
async def test_question_closing_too_soon_to_finish_is_skipped(forecast_calls, caplog):
    # A forecast takes about three minutes; one that cannot finish is wasted spend.
    questions = [_question("closing", 2), _question("open", 60)]

    with caplog.at_level("WARNING"):
        results = await _bot().forecast_questions(questions, return_exceptions=True)

    assert forecast_calls == [["open"]]
    assert len(results) == 1
    assert "too soon to finish" in caplog.text
