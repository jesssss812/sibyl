"""`eval compare` over two synthetic backtest files.

The cases that matter are the refusals. A comparison tool that prints a number for
mismatched inputs is worse than one that crashes, because the number gets quoted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sibyl.eval.compare import MAX_DROP_RATE, CompareError, pair_scores, render_report


def _write(tmp_path: Path, name: str, scores: list[dict], bot: str = "bot") -> Path:
    path = tmp_path / name
    path.write_text(
        json.dumps(
            {
                "timestamp": "2026-09-01T00:00:00",
                "fetch_metadata": {},
                "ground_truths": {},
                "results": [
                    {
                        "bot_name": bot,
                        "num_questions": len(scores),
                        "num_scored": len(scores),
                        "num_failed": 0,
                        "scores": scores,
                        "aggregated": {},
                    }
                ],
            }
        )
    )
    return path


def _rows(values: dict[int, float], metric: str = "brier", qtype: str = "binary") -> list[dict]:
    return [
        {
            "question_id": qid,
            "question_type": qtype,
            "bot_score": value,
            "community_score": None,
            "metric_name": metric,
        }
        for qid, value in values.items()
    ]


class TestPairing:
    def test_matches_on_question_id_and_metric(self) -> None:
        a = {(1, "brier"): 0.2, (2, "brier"): 0.3}
        b = {(1, "brier"): 0.1, (2, "brier"): 0.4}
        deltas, dropped = pair_scores(a, b)
        assert dropped == set()
        assert deltas["brier"] == [(1, pytest.approx(-0.1)), (2, pytest.approx(0.1))]

    def test_a_question_missing_from_either_arm_is_dropped_from_both(self) -> None:
        a = {(1, "brier"): 0.2, (2, "brier"): 0.3}
        b = {(1, "brier"): 0.1}
        deltas, dropped = pair_scores(a, b)
        assert dropped == {2}
        assert [qid for qid, _ in deltas["brier"]] == [1]

    def test_metrics_are_kept_apart(self) -> None:
        a = {(1, "brier"): 0.2, (1, "log_score"): -0.5}
        b = {(1, "brier"): 0.1, (1, "log_score"): -0.4}
        deltas, _dropped = pair_scores(a, b)
        assert set(deltas) == {"brier", "log_score"}


class TestIdenticalArms:
    def test_two_identical_files_report_exactly_no_difference(self, tmp_path: Path) -> None:
        """The A/A null. If this ever reports a difference, the harness is broken."""
        values = {qid: 0.1 + qid / 100 for qid in range(1, 21)}
        a = _write(tmp_path, "a.json", _rows(values))
        b = _write(tmp_path, "b.json", _rows(values))
        report = render_report(a, b)
        assert "n=20" in report
        assert "mean Δ   +0.00000" in report
        assert "0 dropped" in report
        assert "no detectable difference" in report


class TestRefusals:
    def test_refuses_above_the_drop_rate_limit(self, tmp_path: Path) -> None:
        # 20 questions, 4 unscored in B -> 20% dropped, well over the 5% limit.
        a = _write(tmp_path, "a.json", _rows({qid: 0.2 for qid in range(1, 21)}))
        b = _write(tmp_path, "b.json", _rows({qid: 0.2 for qid in range(1, 17)}))
        with pytest.raises(CompareError) as exc:
            render_report(a, b)
        message = str(exc.value)
        assert "REFUSING" in message
        assert f"{MAX_DROP_RATE:.0%}" in message
        # The refusal must still say WHICH questions, or it cannot be acted on.
        assert "17" in message

    def test_tolerates_a_drop_rate_at_or_below_the_limit(self, tmp_path: Path) -> None:
        # 40 questions, 2 dropped = 5.0%, exactly at the limit.
        a = _write(tmp_path, "a.json", _rows({qid: 0.2 for qid in range(1, 41)}))
        b = _write(tmp_path, "b.json", _rows({qid: 0.2 for qid in range(1, 39)}))
        report = render_report(a, b)
        assert "n=38" in report

    def test_refuses_a_file_holding_more_than_one_bot(self, tmp_path: Path) -> None:
        path = tmp_path / "two.json"
        path.write_text(
            json.dumps(
                {
                    "results": [
                        {"bot_name": "alpha", "scores": _rows({1: 0.2})},
                        {"bot_name": "beta", "scores": _rows({1: 0.3})},
                    ]
                }
            )
        )
        other = _write(tmp_path, "one.json", _rows({1: 0.2}))
        with pytest.raises(CompareError, match="exactly one per file"):
            render_report(path, other)

    def test_refuses_a_missing_file_by_name(self, tmp_path: Path) -> None:
        real = _write(tmp_path, "a.json", _rows({1: 0.2}))
        with pytest.raises(CompareError, match="no such backtest data file"):
            render_report(tmp_path / "absent.json", real)

    def test_refuses_a_file_that_is_not_backtest_data(self, tmp_path: Path) -> None:
        junk = tmp_path / "junk.json"
        junk.write_text(json.dumps({"something": "else"}))
        real = _write(tmp_path, "a.json", _rows({1: 0.2}))
        with pytest.raises(CompareError, match="no 'results' key"):
            render_report(junk, real)

    def test_disjoint_metrics_refuse_as_a_total_drop(self, tmp_path: Path) -> None:
        """Same question, different metric, pairs on neither -- so every question drops.

        The drop-rate refusal fires rather than a separate "no shared metric" one, and
        that is the more useful message: it names the questions and the rate.
        """
        a = _write(tmp_path, "a.json", _rows({1: 0.2}, metric="brier"))
        b = _write(tmp_path, "b.json", _rows({1: -0.5}, metric="log_score"))
        with pytest.raises(CompareError, match="100.0% of questions were dropped"):
            render_report(a, b)

    def test_two_empty_files_refuse_with_nothing_to_compare(self, tmp_path: Path) -> None:
        a = _write(tmp_path, "a.json", [])
        b = _write(tmp_path, "b.json", [])
        with pytest.raises(CompareError, match="nothing to compare"):
            render_report(a, b)


class TestReportContent:
    def test_reports_a_real_improvement_with_the_right_sign(self, tmp_path: Path) -> None:
        # Brier: lower is better, so B scoring lower is B being better.
        a = _write(tmp_path, "a.json", _rows({qid: 0.30 for qid in range(1, 21)}))
        b = _write(tmp_path, "b.json", _rows({qid: 0.20 for qid in range(1, 21)}))
        report = render_report(a, b)
        assert "mean Δ   -0.10000" in report
        assert "B is better than A" in report

    def test_log_score_direction_is_inverted_relative_to_brier(self, tmp_path: Path) -> None:
        # log_score: higher is better, so the SAME negative delta means B is worse.
        a = _write(tmp_path, "a.json", _rows({qid: -0.30 for qid in range(1, 21)}, metric="log_score"))
        b = _write(tmp_path, "b.json", _rows({qid: -0.40 for qid in range(1, 21)}, metric="log_score"))
        report = render_report(a, b)
        assert "higher is better" in report
        assert "B is worse than A" in report

    def test_never_prints_an_absolute_score(self, tmp_path: Path) -> None:
        """Rule 3: absolute backtest numbers are not reportable, so they are not printed."""
        a = _write(tmp_path, "a.json", _rows({qid: 0.37123 for qid in range(1, 21)}))
        b = _write(tmp_path, "b.json", _rows({qid: 0.29456 for qid in range(1, 21)}))
        report = render_report(a, b)
        assert "0.37123" not in report and "0.29456" not in report

    def test_carries_the_caveats_block(self, tmp_path: Path) -> None:
        a = _write(tmp_path, "a.json", _rows({qid: 0.2 for qid in range(1, 21)}))
        b = _write(tmp_path, "b.json", _rows({qid: 0.2 for qid in range(1, 21)}))
        report = render_report(a, b)
        assert "CAVEATS" in report and "leak" in report.lower() and "MDE" in report

    def test_reports_the_mde_alongside_every_null(self, tmp_path: Path) -> None:
        a = _write(tmp_path, "a.json", _rows({qid: 0.2 for qid in range(1, 21)}))
        b = _write(tmp_path, "b.json", _rows({qid: 0.2 for qid in range(1, 21)}))
        assert "MDE at this n:" in render_report(a, b)

    def test_by_question_type_slices_the_comparison(self, tmp_path: Path) -> None:
        rows_a = _rows({1: 0.2, 2: 0.2}, qtype="binary") + _rows({3: 0.4, 4: 0.4}, qtype="numeric")
        rows_b = _rows({1: 0.1, 2: 0.1}, qtype="binary") + _rows({3: 0.4, 4: 0.4}, qtype="numeric")
        a, b = _write(tmp_path, "a.json", rows_a), _write(tmp_path, "b.json", rows_b)
        report = render_report(a, b, by_question_type=True)
        assert "question_type=binary" in report and "question_type=numeric" in report

    def test_without_the_flag_only_the_overall_section_appears(self, tmp_path: Path) -> None:
        a = _write(tmp_path, "a.json", _rows({1: 0.2, 2: 0.3}))
        b = _write(tmp_path, "b.json", _rows({1: 0.2, 2: 0.3}))
        assert "question_type=" not in render_report(a, b)

    def test_is_reproducible_for_a_fixed_seed(self, tmp_path: Path) -> None:
        values_a = {qid: 0.2 + (qid % 5) / 50 for qid in range(1, 31)}
        values_b = {qid: 0.2 + (qid % 7) / 50 for qid in range(1, 31)}
        a, b = _write(tmp_path, "a.json", _rows(values_a)), _write(tmp_path, "b.json", _rows(values_b))
        assert render_report(a, b, seed=0) == render_report(a, b, seed=0)
