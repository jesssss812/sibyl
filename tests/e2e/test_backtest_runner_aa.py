"""Real offline runner A/A: immutable inputs through ensemble and persisted scoring."""
import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock

from forecasting_tools import BinaryQuestion

from sibyl.eval.artifacts import load_manifest, prepare_manifest, write_json
from sibyl.eval.backtest.cli import _build_parser, run_backtest
from sibyl.eval.backtest.question_prep import BacktestQuestionSet
from sibyl.eval.backtest.scoring import GroundTruth
from sibyl.eval.compare import render_report


async def test_real_fixture_runner_aa(tmp_path, monkeypatch):
    import sibyl.eval.backtest.cli as cli
    question_set = BacktestQuestionSet([], {})
    for qid in range(1, 7):
        q = BinaryQuestion(question_text=f"Will event {qid} happen?", id_of_question=qid,
                           background_info="Synthetic fixture", resolution_criteria="Event happens",
                           fine_print="", open_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
                           scheduled_resolution_time=datetime(2024, 3, 1, tzinfo=timezone.utc),
                           community_prediction_at_access_time=0.5)
        question_set.questions.append(q)
        question_set.ground_truths[qid] = GroundTruth(qid, "binary", bool(qid % 2),
                                                      str(bool(qid % 2)), 0.5, None, q.question_text)
        question_set.research_cache[qid] = "Canned research for offline pipeline validation."
    manifest = tmp_path / "manifest.json"
    write_json(manifest, prepare_manifest(question_set, source="fixture"))
    assert len(load_manifest(manifest)[1].questions) == 6
    tripwire = AsyncMock(side_effect=AssertionError("live preparation forbidden"))
    monkeypatch.setattr(cli, "fetch_resolved_questions", tripwire)
    monkeypatch.setattr(cli, "screen_research_for_leakage", tripwire)
    paths = []
    for arm in ("a", "b"):
        out = tmp_path / arm
        args = _build_parser().parse_args(["--fixture", "--manifest", str(manifest), "--output-dir", str(out)])
        await asyncio.wait_for(run_backtest(args), timeout=15)
        paths.append(next(out.glob("backtest_data_*.json")))
    a, b = [json.loads(p.read_text()) for p in paths]
    assert a["results"][0]["scores"] == b["results"][0]["scores"]
    assert a["results"][0]["num_scored"] == 6
    assert all(s["forecasters_used"] == [3, 3] for s in a["results"][0]["question_status"].values())
    report = render_report(*paths)
    assert "0 dropped" in report and "+0.00000" in report
    tripwire.assert_not_called()
