"""Archive and descriptively score a grant MiniBench run without the comments API.

The forecast archive and selected-question snapshot come from the live grant run.
Resolution collection is read-only. Scoring is offline and never labels a single
arm as a confirmatory experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from forecasting_tools.data_models.binary_report import BinaryReport
from forecasting_tools.data_models.multiple_choice_report import MultipleChoiceReport
from forecasting_tools.data_models.numeric_report import DiscreteReport, NumericReport
from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    DiscreteQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
    QuestionState,
)

from sibyl.eval.artifacts import fingerprint
from sibyl.eval.backtest.question_prep import _extract_ground_truth
from sibyl.eval.backtest.scoring import score_report
from sibyl.eval.performance.parsing import parse_forecasters_used_marker

_REPORT_TYPES = {
    "binary": BinaryReport,
    "numeric": NumericReport,
    "discrete": DiscreteReport,
    "multiple_choice": MultipleChoiceReport,
}
_QUESTION_TYPES = {
    "binary": BinaryQuestion,
    "numeric": NumericQuestion,
    "discrete": DiscreteQuestion,
    "multiple_choice": MultipleChoiceQuestion,
}


def _read_run(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    schedule = json.loads((run_dir / "scheduled_questions.json").read_text())
    if manifest.get("schema_version") != 1 or schedule.get("schema_version") != 1:
        raise ValueError("Unknown MiniBench archive version")
    if manifest.get("run_id") != schedule.get("run_id") or manifest.get("run_id") != run_dir.name:
        raise ValueError("Run manifest and selected-question snapshot disagree")
    ids = schedule.get("question_ids")
    questions = schedule.get("questions")
    if not isinstance(ids, list) or not isinstance(questions, list) or len(ids) != len(questions):
        raise ValueError("Selected-question snapshot is incomplete")
    if any(not isinstance(qid, int) for qid in ids) or len(set(ids)) != len(ids):
        raise ValueError("Selected-question IDs must be unique integers")
    for qid, question in zip(ids, questions, strict=True):
        if question.get("id_of_question") != qid or not isinstance(question.get("id_of_post"), int):
            raise ValueError(f"Q{qid}: selected-question identity mismatch")
    return manifest, schedule


def _write_new_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as output_file:
        json.dump(value, output_file, indent=2, allow_nan=False)


def collect_resolutions(run_dir: Path, output: Path) -> dict[str, Any]:
    """Fetch exact selected questions by post ID and freeze a new outcome snapshot."""
    from sibyl.config.env import load_environment

    manifest, schedule = _read_run(run_dir)
    if output.exists():
        raise FileExistsError(f"Resolution snapshot already exists: {output}")
    load_environment()
    from forecasting_tools.helpers.metaculus_client import MetaculusClient  # noqa: PLC0415

    from sibyl.platform.metaculus_preflight import verify_metaculus_api_identity  # noqa: PLC0415

    client = MetaculusClient()
    if client.token is None:
        raise ValueError("METACULUS_TOKEN is required to collect MiniBench resolutions")
    verify_metaculus_api_identity()
    resolved: list[dict[str, Any]] = []
    for index, (qid, selected) in enumerate(zip(schedule["question_ids"], schedule["questions"], strict=True)):
        fetched = client.get_question_by_post_id(selected["id_of_post"], group_question_mode="unpack_subquestions")
        candidates = fetched if isinstance(fetched, list) else [fetched]
        matches = [question for question in candidates if question.id_of_question == qid]
        if len(matches) != 1:
            raise ValueError(f"Q{qid}: Metaculus returned no unique matching question")
        resolved.append(matches[0].model_dump(mode="json"))
        if index + 1 < len(schedule["question_ids"]):
            time.sleep(0.5)
    payload = {
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "schedule_id": fingerprint(schedule),
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "questions": resolved,
    }
    snapshot = {**payload, "snapshot_id": fingerprint(payload)}
    _write_new_json(output, snapshot)
    return snapshot


def _load_reports(run_dir: Path, scheduled_ids: set[int]) -> dict[int, Any]:
    reports: dict[int, Any] = {}
    for path in sorted((run_dir / "forecasts").glob("*.json")):
        raw_reports = json.loads(path.read_text())
        if not isinstance(raw_reports, list):
            raise ValueError(f"{path}: forecast archive must contain a list")
        for raw in raw_reports:
            question = raw.get("question", {})
            qid, kind = question.get("id_of_question"), question.get("question_type")
            if not isinstance(qid, int) or qid not in scheduled_ids or qid in reports:
                raise ValueError(f"{path}: unexpected or duplicate forecast Q{qid}")
            report_type = _REPORT_TYPES.get(kind)
            if report_type is None:
                raise ValueError(f"{path}: unsupported forecast type {kind!r}")
            report = report_type.from_json(raw)
            if report.question.id_of_question != qid:
                raise ValueError(f"{path}: forecast question identity mismatch")
            reports[qid] = report
    return reports


def score_archive(run_dir: Path, snapshot_path: Path) -> dict[str, Any]:
    """Score all selected questions, preserving missing and unresolved rows."""
    manifest, schedule = _read_run(run_dir)
    snapshot = json.loads(snapshot_path.read_text())
    payload = {key: value for key, value in snapshot.items() if key != "snapshot_id"}
    if (
        snapshot.get("schema_version") != 1
        or snapshot.get("run_id") != manifest["run_id"]
        or snapshot.get("schedule_id") != fingerprint(schedule)
        or snapshot.get("snapshot_id") != fingerprint(payload)
    ):
        raise ValueError("Resolution snapshot identity mismatch")
    scheduled_ids = set(schedule["question_ids"])
    outcomes: dict[int, Any] = {}
    for raw in snapshot["questions"]:
        qid, kind = raw.get("id_of_question"), raw.get("question_type")
        question_type = _QUESTION_TYPES.get(kind)
        if not isinstance(qid, int) or qid not in scheduled_ids or qid in outcomes or question_type is None:
            raise ValueError(f"Unexpected, duplicate, or unsupported outcome Q{qid}")
        outcomes[qid] = question_type.model_validate(raw)
    reports = _load_reports(run_dir, scheduled_ids)
    rows: list[dict[str, Any]] = []
    endpoint_values: dict[str, list[float]] = {}
    for qid in schedule["question_ids"]:
        report = reports.get(qid)
        question = outcomes.get(qid)
        if report is None:
            rows.append({"question_id": qid, "status": "missing_forecast", "scores": {}})
            continue
        if question is None:
            rows.append({"question_id": qid, "status": "missing_outcome", "scores": {}})
            continue
        if type(report.question) is not type(question) or report.question.id_of_post != question.id_of_post:
            raise ValueError(f"Q{qid}: forecast and outcome question identity mismatch")
        ground_truth = _extract_ground_truth(question)
        if ground_truth is None:
            status = (
                "resolved_without_outcome"
                if question.state is QuestionState.RESOLVED and question.resolution_string is None
                else "unresolved_or_canceled"
            )
            rows.append({"question_id": qid, "status": status, "scores": {}})
            continue
        metrics = {score.metric_name: score.bot_score for score in score_report(report, ground_truth)}
        if not metrics or any(not math.isfinite(value) for value in metrics.values()):
            rows.append({"question_id": qid, "status": "invalid_score", "scores": {}})
            continue
        forecasters_used = parse_forecasters_used_marker(report.explanation)
        degraded = bool(report.errors) or (forecasters_used is not None and forecasters_used[0] < forecasters_used[1])
        status = "degraded" if degraded else "scored"
        degradation_status = "degraded" if degraded else "unknown" if forecasters_used is None else "none_reported"
        rows.append({"question_id": qid, "status": status, "question_type": ground_truth.question_type,
                     "degradation_status": degradation_status, "forecasters_used": forecasters_used,
                     "errors": report.errors, "scores": metrics})
        # A degraded forecast was still published and belongs in the live baseline.
        # Keep its status visible instead of selecting it out of the mean.
        for metric, value in metrics.items():
            endpoint_values.setdefault(metric, []).append(value)
    return {
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "code": manifest["code"],
        "schedule_id": fingerprint(schedule),
        "snapshot_id": snapshot["snapshot_id"],
        "scheduled_count": len(schedule["question_ids"]),
        "status_counts": {status: sum(row["status"] == status for row in rows) for status in sorted({row["status"] for row in rows})},
        "endpoint_means_descriptive_only": {
            metric: {"n": len(values), "mean": sum(values) / len(values)}
            for metric, values in sorted(endpoint_values.items())
        },
        "rows": rows,
        "inference_status": "descriptive_only_single_arm",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect", help="Read-only Metaculus resolution snapshot; no comments request")
    collect.add_argument("--run-dir", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)
    score = sub.add_parser("score", help="Offline descriptive scoring from a frozen resolution snapshot")
    score.add_argument("--run-dir", type=Path, required=True)
    score.add_argument("--snapshot", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "collect":
        result = collect_resolutions(args.run_dir, args.output)
        print(f"Collected {len(result['questions'])} question snapshots: {args.output}")
        return 0
    result = score_archive(args.run_dir, args.snapshot)
    _write_new_json(args.output, result)
    scored = result["status_counts"].get("scored", 0) + result["status_counts"].get("degraded", 0)
    print(f"Scored {scored} of {result['scheduled_count']} selected questions: {args.output}")
    return 0
