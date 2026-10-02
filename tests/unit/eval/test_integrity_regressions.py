"""Offline reproductions from the September 2026 evaluation review."""

import json
import math
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from sibyl.eval.compare import CompareError, _verdict, render_report
from sibyl.eval.stats import bootstrap_mean_ci, bootstrap_median_ci, paired_summary


@pytest.mark.parametrize("n", [1, 2, 4])
def test_small_samples_do_not_certify_a_winner(n):
    for bootstrap in (bootstrap_mean_ci, bootstrap_median_ci):
        point, lo, hi = bootstrap([-0.1] * n)
        assert point == pytest.approx(-0.1)
        assert math.isnan(lo) and math.isnan(hi)
    summary = paired_summary([-0.1] * n)
    assert "insufficient" in _verdict("brier", summary)
    if n == 1:
        assert math.isnan(summary["mde"])


def test_shared_failures_are_counted_against_scheduled_questions(tmp_path):
    paths = []
    for arm, score in [("a", 0.2), ("b", 0.1)]:
        path = tmp_path / f"{arm}.json"
        path.write_text(json.dumps({
            "ground_truths": {str(q): {} for q in range(100)},
            "results": [{"bot_name": arm, "num_questions": 100, "num_failed": 99,
                         "scores": [{"question_id": 0, "metric_name": "brier", "bot_score": score}]}],
        }))
        paths.append(path)
    with pytest.raises(CompareError, match="99.*dropped"):
        render_report(*paths)


async def test_fixture_runner_never_prepares_real_research(monkeypatch):
    from sibyl.eval.backtest import cli
    from sibyl.eval.backtest.question_prep import BacktestQuestionSet

    monkeypatch.setattr(cli, "fetch_resolved_questions", AsyncMock(return_value=BacktestQuestionSet([], {})))
    screen = AsyncMock(side_effect=AssertionError("paid research reached"))
    monkeypatch.setattr(cli, "screen_research_for_leakage", screen)

    def stop_after_preparation(**kwargs):
        raise RuntimeError("fixture factory reached safely")

    monkeypatch.setattr(cli, "_create_fixture_bots", stop_after_preparation)
    with pytest.raises(RuntimeError, match="fixture factory reached safely"):
        await cli.run_backtest(cli._build_parser().parse_args(["--fixture"]))
    screen.assert_not_called()


def test_historical_prompt_uses_one_date():
    from sibyl.eval.ablation.window_patch import patched_window_for_question
    from sibyl.forecasting.prompts import binary_prompt

    q = SimpleNamespace(id_of_question=1, question_text="Will X occur?", background_info="",
                        resolution_criteria="X occurs by close", fine_print="",
                        open_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
                        scheduled_resolution_time=datetime(2025, 3, 1, tzinfo=timezone.utc))
    with patched_window_for_question(q):
        prompt = binary_prompt(q, "research")  # type: ignore[arg-type]
    assert "Today: 2025-01-30" in prompt
    assert "open and unresolved as of 2025-01-30" in prompt


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_invalid_deltas_cannot_disappear_silently(value):
    with pytest.raises(ValueError, match="finite"):
        paired_summary([0.1, value])


def test_cache_requires_matching_upstream_inputs(tmp_path):
    from sibyl.eval.ablation.cache import AblationCache

    cache = AblationCache(tmp_path)
    cache.bind_inputs("stacker", 1, "median", {"predictions": [0.1, 0.2, 0.3]})
    cache.write_stacker_output(1, "median", {"success": True, "prediction": 0.2})
    assert cache.read_stacker_output(1, "median") is not None
    cache.bind_inputs("stacker", 1, "median", {"predictions": [0.7, 0.8, 0.9]})
    assert cache.read_stacker_output(1, "median") is None


async def test_historical_clocks_are_task_local():
    import asyncio

    from sibyl.common.forecast_context import forecast_time, forecast_times
    q = SimpleNamespace(id_of_question=1)

    async def clock(year):
        moment = datetime(year, 1, 1, tzinfo=timezone.utc)
        with forecast_times({1: moment}):
            await asyncio.sleep(0)
            assert forecast_time(q) == moment
        assert forecast_time(q) is None

    await asyncio.gather(clock(2024), clock(2025))


def test_registered_primary_missing_fails_instead_of_reporting_secondary(tmp_path):
    from sibyl.eval.artifacts import fingerprint
    plan = {"endpoint": "brier", "hypothesis": "synthetic", "variance_source": "independent",
            "manifest_id": "m1", "n": 5, "scheduled_n": 5, "attrition_allowance": 0.0, "sd": 0.1, "effect": 0.2}
    saved = {"plan": plan, "plan_id": fingerprint(plan), "decision": "proceed",
             "assessed_at": "2025-01-01T00:00:00+00:00"}
    paths = []
    for arm in ("a", "b"):
        path = tmp_path / f"{arm}.json"
        path.write_text(json.dumps({"ground_truths": {str(q): {} for q in range(5)},
             "experiment": {"manifest_id": "m1", "scoring_version": "per-type-v1", "power_plan": saved, "arm_id": arm, "started_at": "2026-01-01T00:00:00+00:00"},
             "results": [{"bot_name": arm, "question_status": {str(q): {"status": "ok"} for q in range(5)},
                          "scores": [{"question_id": q, "question_type": "binary", "metric_name": "binary_log_score", "bot_score": -0.2} for q in range(5)]}]}))
        paths.append(path)
    with pytest.raises(CompareError, match="primary brier"):
        render_report(*paths)


def test_matching_plan_id_cannot_hide_different_registration_payloads(tmp_path):
    paths = []
    for arm, effect in (("a", 0.02), ("b", 0.03)):
        path = tmp_path / f"{arm}.json"
        path.write_text(json.dumps({
            "ground_truths": {"1": {}},
            "experiment": {"manifest_id": "frozen", "scoring_version": "per-type-v1",
                           "power_plan": {"plan_id": "same", "plan": {"effect": effect}}},
            "results": [{"bot_name": arm, "scores": [
                {"question_id": 1, "metric_name": "brier", "bot_score": 0.2}]}],
        }))
        paths.append(path)
    with pytest.raises(CompareError, match="payload differs"):
        render_report(*paths)


def _registered_binary_log_pair(tmp_path, *, fixture_a=False, fixture_b=False):
    from sibyl.eval.artifacts import fingerprint

    plan = {"endpoint": "binary_log_score", "hypothesis": "synthetic", "variance_source": "independent",
            "manifest_id": "m1", "n": 5, "scheduled_n": 5, "attrition_allowance": 0.0,
            "sd": 0.1, "effect": 0.2}
    saved = {"plan": plan, "plan_id": fingerprint(plan), "decision": "proceed",
             "assessed_at": "2025-01-01T00:00:00+00:00"}
    paths = []
    for arm, fixture, score in (("a", fixture_a, -0.3), ("b", fixture_b, -0.2)):
        path = tmp_path / f"{arm}.json"
        path.write_text(json.dumps({
            "ground_truths": {str(q): {} for q in range(5)},
            "experiment": {"manifest_id": "m1", "scoring_version": "per-type-v1", "power_plan": saved,
                           "arm_id": arm, "fixture": fixture, "started_at": "2026-01-01T00:00:00+00:00"},
            "results": [{"bot_name": arm, "question_status": {str(q): {"status": "ok"} for q in range(5)},
                         "scores": [{"question_id": q, "question_type": "binary", "metric_name": "log_score",
                                     "bot_score": score} for q in range(5)]}],
        }))
        paths.append(path)
    return paths


def test_registered_binary_log_endpoint_matches_persisted_metric(tmp_path):
    report = render_report(*_registered_binary_log_pair(tmp_path))
    assert "log_score:" in report
    assert "B is better than A" in report


def test_fixture_and_real_arms_cannot_be_compared(tmp_path):
    with pytest.raises(CompareError, match="fixture and real"):
        render_report(*_registered_binary_log_pair(tmp_path, fixture_a=True))


def test_fixture_power_plan_does_not_certify_inference(tmp_path):
    report = render_report(*_registered_binary_log_pair(tmp_path, fixture_a=True, fixture_b=True))
    assert "FIXTURE PIPELINE CHECK" in report
    assert "exploratory only; no confirmatory verdict" in report


def test_archive_research_after_forecast_time_is_labeled(tmp_path):
    from sibyl.eval.backtest.cli import _load_research_from_archive
    q = SimpleNamespace(id_of_question=55, id_of_post=55, page_url="https://metaculus.com/questions/55",
                        open_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
                        scheduled_resolution_time=datetime(2024, 3, 1, tzinfo=timezone.utc))
    (tmp_path / "55.json").write_text(json.dumps({"qid": 55, "research_text": "known outcome",
                             "timestamp": "2025-01-01T00:00:00Z", "source": "live_capture"}))
    evidence = {}
    assert _load_research_from_archive(str(tmp_path), [q], evidence)[55] == "known outcome"
    assert evidence[55]["post_cutoff_or_unknown"] is True


async def test_prepare_manifest_records_early_resolution_exclusion(tmp_path, monkeypatch):
    from forecasting_tools import BinaryQuestion

    from sibyl.eval.backtest import cli
    from sibyl.eval.backtest.question_prep import BacktestQuestionSet
    from sibyl.eval.backtest.scoring import GroundTruth
    q = BinaryQuestion(question_text="Synthetic early close", id_of_question=8,
                       resolution_criteria="Event occurs", open_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
                       scheduled_resolution_time=datetime(2024, 3, 1, tzinfo=timezone.utc))
    gt = GroundTruth(8, "binary", True, "Yes", None,
                     datetime(2024, 1, 10, tzinfo=timezone.utc), q.question_text)
    monkeypatch.setattr(cli, "fetch_resolved_questions", AsyncMock(return_value=BacktestQuestionSet([q], {8: gt})))
    output = tmp_path / "manifest.json"
    await cli.run_backtest(cli._build_parser().parse_args(["--fixture", "--prepare-only", str(output)]))
    artifact = json.loads(output.read_text())
    assert artifact["entries"] == []
    assert artifact["preparation"]["preparation_exclusions"] == [
        {"question_id": 8, "reason": "resolved_before_simulated_forecast"}]
