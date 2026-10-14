"""The CLI's run archive is what the MiniBench scorer reads."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from sibyl.eval.minibench_live import _read_run
from sibyl.eval.run_archive import RunArchive
from sibyl.forecasting.bot import SibylBot
from tests.conftest import make_mock_binary_question


def test_the_scorer_accepts_the_archive_layout(tmp_path: Path) -> None:
    archive = RunArchive(tmp_path, "run-1", {"mode": "minibench"})
    questions = [make_mock_binary_question(11), make_mock_binary_question(12)]
    for index, question in enumerate(questions):
        question.id_of_post = 100 + index
    with patch(
        "sibyl.eval.run_archive.question_identity",
        lambda q: {"id_of_question": q.id_of_question, "id_of_post": q.id_of_post},
    ):
        archive.record_selection(questions)
    manifest, schedule = _read_run(archive.run_dir)
    assert manifest["run_id"] == "run-1" and "code" in manifest
    assert schedule["question_ids"] == [11, 12]
    assert archive.forecasts_dir == tmp_path / "run-1" / "forecasts"


def test_a_run_id_cannot_be_archived_twice(tmp_path: Path) -> None:
    RunArchive(tmp_path, "run-1", {})
    with pytest.raises(ValueError, match="already has an archived manifest"):
        RunArchive(tmp_path, "run-1", {})


@pytest.mark.asyncio
async def test_the_bot_reports_exactly_the_questions_it_will_forecast(test_llms: dict[str, Any]) -> None:
    selected: list[list[int | None]] = []
    bot = SibylBot(
        llms=test_llms,
        max_questions_per_run=2,
        skip_previously_forecasted_questions=True,
        selection_sink=lambda questions: selected.append([q.id_of_question for q in questions]),
    )
    questions = [make_mock_binary_question(qid) for qid in (1, 2, 3, 4)]
    for question in questions:
        question.already_forecasted = question.id_of_question == 1
    with patch("forecasting_tools.ForecastBot.forecast_questions", return_value=[]):
        await bot.forecast_questions(questions, return_exceptions=True)
    assert selected == [[2, 3]], "already-forecast questions are skipped and the cap applied first"


@pytest.mark.asyncio
async def test_a_failing_selection_sink_does_not_stop_the_forecasts(test_llms: dict[str, Any]) -> None:
    def broken(questions: Any) -> None:
        raise ValueError("An archived run needs unique, non-null question IDs")

    bot = SibylBot(llms=test_llms, selection_sink=broken)
    with patch("forecasting_tools.ForecastBot.forecast_questions", return_value=["report"]) as forecast:
        assert await bot.forecast_questions([make_mock_binary_question(1)], return_exceptions=True) == ["report"]
    forecast.assert_called_once()
