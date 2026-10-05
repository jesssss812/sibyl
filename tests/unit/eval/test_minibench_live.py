"""Offline MiniBench scoring from the grant archive and frozen resolutions."""

import json
from datetime import datetime
from pathlib import Path

import pytest
from forecasting_tools import BinaryQuestion, MultipleChoiceQuestion, NumericQuestion
from forecasting_tools.data_models.binary_report import BinaryReport
from forecasting_tools.data_models.multiple_choice_report import (
    MultipleChoiceReport,
    PredictedOption,
    PredictedOptionList,
)
from forecasting_tools.data_models.numeric_report import NumericDistribution, NumericReport, Percentile
from forecasting_tools.data_models.questions import QuestionState
from forecasting_tools.helpers.metaculus_client import MetaculusClient

from sibyl.eval.artifacts import fingerprint, write_json
from sibyl.eval.minibench_live import collect_resolutions, score_archive
from sibyl.eval.scoring import binary_log_score, brier_score


def _question(qid: int, *, resolved: bool = False) -> BinaryQuestion:
    return BinaryQuestion(
        question_text=f"Will event {qid} occur?",
        id_of_question=qid,
        id_of_post=qid + 100,
        page_url=f"https://www.metaculus.com/questions/{qid + 100}/",
        background_info="",
        resolution_criteria="Resolves yes if the event occurs.",
        fine_print="",
        open_time=datetime(2026, 10, 5),
        scheduled_resolution_time=datetime(2026, 10, 19),
        actual_resolution_time=datetime(2026, 10, 19) if resolved else None,
        resolution_string="yes" if resolved else None,
    )


def _archive(tmp_path: Path, *, include_second: bool = False) -> tuple[Path, Path]:
    run_dir = tmp_path / "run-a"
    forecast = _question(101)
    selected = [forecast, _question(102)] if include_second else [forecast]
    write_json(run_dir / "manifest.json", {"schema_version": 1, "run_id": "run-a", "code": {"git_sha": "test"}})
    schedule = {
        "schema_version": 1,
        "run_id": "run-a",
        "question_ids": [q.id_of_question for q in selected],
        "questions": [q.model_dump(mode="json") for q in selected],
    }
    write_json(run_dir / "scheduled_questions.json", schedule)
    report = BinaryReport(question=forecast, prediction=0.6, explanation="# Synthetic forecast")
    write_json(run_dir / "forecasts" / "Forecasts-for-test.json", [report.to_json()])
    payload = {
        "schema_version": 1,
        "run_id": "run-a",
        "schedule_id": fingerprint(schedule),
        "collected_at": "2026-10-20T00:00:00Z",
        "questions": [_question(101, resolved=True).model_dump(mode="json")],
    }
    snapshot = tmp_path / "resolutions.json"
    write_json(snapshot, {**payload, "snapshot_id": fingerprint(payload)})
    return run_dir, snapshot


def test_scores_binary_and_counts_missing_forecasts(tmp_path: Path) -> None:
    run_dir, snapshot = _archive(tmp_path, include_second=True)
    result = score_archive(run_dir, snapshot)
    assert result["scheduled_count"] == 2
    assert result["status_counts"] == {"missing_forecast": 1, "scored": 1}
    assert result["inference_status"] == "descriptive_only_single_arm"
    assert result["rows"][0]["scores"] == {
        "brier": brier_score(0.6, True),
        "log_score": binary_log_score(0.6, True),
    }
    assert result["endpoint_means_descriptive_only"]["brier"]["n"] == 1


def test_degraded_published_forecast_remains_in_descriptive_mean(tmp_path: Path) -> None:
    run_dir, snapshot = _archive(tmp_path)
    report = BinaryReport(
        question=_question(101),
        prediction=0.6,
        explanation="# Synthetic forecast\n<!-- FORECASTERS_USED=2/3 -->",
    )
    write_json(run_dir / "forecasts" / "Forecasts-for-test.json", [report.to_json()])
    result = score_archive(run_dir, snapshot)
    assert result["status_counts"] == {"degraded": 1}
    assert result["endpoint_means_descriptive_only"]["brier"]["n"] == 1
    assert result["rows"][0]["degradation_status"] == "degraded"


def test_modified_resolution_snapshot_is_refused(tmp_path: Path) -> None:
    run_dir, snapshot = _archive(tmp_path)
    contents = snapshot.read_text().replace('"resolution_string": "yes"', '"resolution_string": "no"')
    snapshot.write_text(contents)
    with pytest.raises(ValueError, match="identity mismatch"):
        score_archive(run_dir, snapshot)


def test_duplicate_forecast_is_refused(tmp_path: Path) -> None:
    run_dir, snapshot = _archive(tmp_path)
    original = (run_dir / "forecasts" / "Forecasts-for-test.json").read_text()
    (run_dir / "forecasts" / "Forecasts-for-second.json").write_text(original)
    with pytest.raises(ValueError, match="duplicate forecast"):
        score_archive(run_dir, snapshot)


def test_unresolved_question_is_not_scored(tmp_path: Path) -> None:
    run_dir, snapshot_path = _archive(tmp_path)
    schedule = json.loads((run_dir / "scheduled_questions.json").read_text())
    payload = {
        "schema_version": 1,
        "run_id": "run-a",
        "schedule_id": fingerprint(schedule),
        "collected_at": "2026-10-10T00:00:00Z",
        "questions": [_question(101).model_dump(mode="json")],
    }
    write_json(snapshot_path, {**payload, "snapshot_id": fingerprint(payload)})
    result = score_archive(run_dir, snapshot_path)
    assert result["status_counts"] == {"unresolved_or_canceled": 1}
    assert result["endpoint_means_descriptive_only"] == {}


def test_resolved_question_without_visible_outcome_is_explicit(tmp_path: Path) -> None:
    run_dir, snapshot_path = _archive(tmp_path)
    schedule = json.loads((run_dir / "scheduled_questions.json").read_text())
    question = _question(101).model_copy(update={"state": QuestionState.RESOLVED})
    payload = {
        "schema_version": 1,
        "run_id": "run-a",
        "schedule_id": fingerprint(schedule),
        "collected_at": "2026-10-20T00:00:00Z",
        "questions": [question.model_dump(mode="json")],
    }
    write_json(snapshot_path, {**payload, "snapshot_id": fingerprint(payload)})
    result = score_archive(run_dir, snapshot_path)
    assert result["status_counts"] == {"resolved_without_outcome": 1}


def test_multiple_choice_keeps_a_separate_endpoint(tmp_path: Path) -> None:
    run_dir, snapshot_path = _archive(tmp_path)

    def mc_question(resolved: bool) -> MultipleChoiceQuestion:
        return MultipleChoiceQuestion(
            question_text="Which option wins?",
            id_of_question=101,
            id_of_post=201,
            page_url="https://www.metaculus.com/questions/201/",
            options=["A", "B"],
            background_info="",
            resolution_criteria="B wins",
            fine_print="",
            open_time=datetime(2026, 10, 5),
            scheduled_resolution_time=datetime(2026, 10, 19),
            actual_resolution_time=datetime(2026, 10, 19) if resolved else None,
            resolution_string="B" if resolved else None,
        )

    forecast = mc_question(False)
    schedule = {
        "schema_version": 1,
        "run_id": "run-a",
        "question_ids": [101],
        "questions": [forecast.model_dump(mode="json")],
    }
    write_json(run_dir / "scheduled_questions.json", schedule)
    prediction = PredictedOptionList(
        predicted_options=[
            PredictedOption(option_name="A", probability=0.25),
            PredictedOption(option_name="B", probability=0.75),
        ]
    )
    report = MultipleChoiceReport(question=forecast, prediction=prediction, explanation="# Synthetic forecast")
    write_json(run_dir / "forecasts" / "Forecasts-for-test.json", [report.to_json()])
    payload = {
        "schema_version": 1,
        "run_id": "run-a",
        "schedule_id": fingerprint(schedule),
        "collected_at": "2026-10-20T00:00:00Z",
        "questions": [mc_question(True).model_dump(mode="json")],
    }
    write_json(snapshot_path, {**payload, "snapshot_id": fingerprint(payload)})
    result = score_archive(run_dir, snapshot_path)
    assert result["status_counts"] == {"scored": 1}
    assert set(result["endpoint_means_descriptive_only"]) == {"mc_log_score"}


def test_numeric_keeps_a_separate_endpoint(tmp_path: Path) -> None:
    run_dir, snapshot_path = _archive(tmp_path)

    def numeric_question(resolved: bool) -> NumericQuestion:
        return NumericQuestion(
            question_text="What value?",
            id_of_question=101,
            id_of_post=201,
            page_url="https://www.metaculus.com/questions/201/",
            background_info="",
            resolution_criteria="Resolves to the measured value.",
            fine_print="",
            unit_of_measure="units",
            lower_bound=0,
            upper_bound=10,
            open_lower_bound=False,
            open_upper_bound=False,
            open_time=datetime(2026, 10, 5),
            scheduled_resolution_time=datetime(2026, 10, 19),
            actual_resolution_time=datetime(2026, 10, 19) if resolved else None,
            resolution_string="5" if resolved else None,
        )

    forecast = numeric_question(False)
    schedule = {
        "schema_version": 1,
        "run_id": "run-a",
        "question_ids": [101],
        "questions": [forecast.model_dump(mode="json")],
    }
    write_json(run_dir / "scheduled_questions.json", schedule)
    distribution = NumericDistribution(
        declared_percentiles=[Percentile(percentile=p, value=p * 10) for p in (0.1, 0.25, 0.5, 0.75, 0.9)],
        open_lower_bound=False,
        open_upper_bound=False,
        lower_bound=0,
        upper_bound=10,
        zero_point=None,
    )
    report = NumericReport(question=forecast, prediction=distribution, explanation="# Synthetic forecast")
    write_json(run_dir / "forecasts" / "Forecasts-for-test.json", [report.to_json()])
    payload = {
        "schema_version": 1,
        "run_id": "run-a",
        "schedule_id": fingerprint(schedule),
        "collected_at": "2026-10-20T00:00:00Z",
        "questions": [numeric_question(True).model_dump(mode="json")],
    }
    write_json(snapshot_path, {**payload, "snapshot_id": fingerprint(payload)})
    result = score_archive(run_dir, snapshot_path)
    assert result["status_counts"] == {"scored": 1}
    assert set(result["endpoint_means_descriptive_only"]) == {"numeric_log_score"}


def test_collect_reads_selected_posts_without_comments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir, _snapshot = _archive(tmp_path)
    from sibyl.platform import metaculus_preflight

    monkeypatch.setattr(metaculus_preflight, "verify_metaculus_api_identity", lambda: None)
    calls: list[tuple[int, str]] = []

    def fake_get(_self: MetaculusClient, post_id: int, group_question_mode: str):
        calls.append((post_id, group_question_mode))
        return _question(101, resolved=True)

    monkeypatch.setenv("METACULUS_TOKEN", "dummy-token")
    monkeypatch.setattr(MetaculusClient, "get_question_by_post_id", fake_get)
    output = tmp_path / "fresh-resolutions.json"
    result = collect_resolutions(run_dir, output)
    assert calls == [(201, "unpack_subquestions")]
    assert result["snapshot_id"]
    assert score_archive(run_dir, output)["status_counts"] == {"scored": 1}
