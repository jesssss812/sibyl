"""The A/A null test: two identical fixture arms, compared, must show exactly nothing.

This is the test that validates the eval harness rather than the bot. Two arms built from
the same seeded fixture produce the same forecasts by construction, so the paired
comparison MUST report a zero delta, a CI containing zero, and no drops. If it reports a
difference, the harness is broken -- and that is only detectable when both arms are
controlled, which is exactly what the fixture arm exists to provide.

An A/B test cannot catch this. A real difference and a harness bug look identical there.

Zero API calls: the scores are produced by the same seeded generator the fixture
forecaster uses, and nothing here opens a socket.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sibyl.eval.compare import CompareError, render_report

# A deterministic score set standing in for one backtest arm. Values vary across
# questions and types so a real difference WOULD be visible -- an A/A test over constant
# scores would pass even if the comparison silently returned zeros.
_QUESTION_TYPES = ["binary", "numeric", "multiple_choice"]


def _arm_scores(n: int = 24) -> list[dict]:
    rows = []
    for i in range(1, n + 1):
        qtype = _QUESTION_TYPES[i % len(_QUESTION_TYPES)]
        metric = "brier" if qtype == "binary" else "numeric_log_score"
        # Deterministic and spread out; the exact values do not matter, only that they
        # differ per question so a spurious delta could not hide in a constant.
        value = 0.10 + ((i * 7) % 23) / 100 if metric == "brier" else -0.5 - ((i * 5) % 17) / 10
        rows.append(
            {
                "question_id": 40000 + i,
                "question_type": qtype,
                "bot_score": value,
                "community_score": value - 0.02,
                "metric_name": metric,
            }
        )
    return rows


def _write_arm(path: Path, scores: list[dict], bot_name: str) -> Path:
    path.write_text(
        json.dumps(
            {
                "timestamp": "2026-09-01T00:00:00",
                "fetch_metadata": {"source": "fixture"},
                "ground_truths": {},
                "results": [
                    {
                        "bot_name": bot_name,
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


@pytest.fixture
def identical_arms(tmp_path: Path) -> tuple[Path, Path]:
    scores = _arm_scores()
    return (
        _write_arm(tmp_path / "arm_a.json", scores, "fixture/alpha"),
        _write_arm(tmp_path / "arm_b.json", list(scores), "fixture/alpha"),
    )


class TestFixtureAaNull:
    def test_reports_a_zero_delta(self, identical_arms: tuple[Path, Path]) -> None:
        report = render_report(*identical_arms)
        # Every printed mean and median delta must be a signed zero at 5dp.
        deltas = [line for line in report.splitlines() if "mean Δ" in line or "median Δ" in line]
        assert deltas, "no delta lines in the report"
        for line in deltas:
            assert "+0.00000" in line, line

    def test_the_ci_contains_zero(self, identical_arms: tuple[Path, Path]) -> None:
        report = render_report(*identical_arms)
        assert "[+0.00000, +0.00000]" in report

    def test_drops_nothing(self, identical_arms: tuple[Path, Path]) -> None:
        assert "0 dropped (0.0%)" in render_report(*identical_arms)

    def test_calls_it_no_detectable_difference(self, identical_arms: tuple[Path, Path]) -> None:
        report = render_report(*identical_arms)
        assert "exploratory only; no confirmatory verdict" in report
        assert "B is better" not in report and "B is worse" not in report

    def test_holds_per_question_type_too(self, identical_arms: tuple[Path, Path]) -> None:
        # A bug that cancels in the pooled mean can still show in a slice.
        report = render_report(*identical_arms, by_question_type=True)
        assert "question_type=binary" in report
        for line in report.splitlines():
            if "mean Δ" in line:
                assert "+0.00000" in line, line

    def test_every_metric_is_compared_not_just_the_first(self, identical_arms: tuple[Path, Path]) -> None:
        report = render_report(*identical_arms)
        assert "brier:" in report and "numeric_log_score:" in report

    def test_the_arms_really_would_show_a_difference_if_one_existed(self, tmp_path: Path) -> None:
        """Guards the guard: an A/A test over inputs the tool ignores proves nothing."""
        scores = _arm_scores()
        shifted = [{**row, "bot_score": row["bot_score"] + 0.05} for row in scores]
        a = _write_arm(tmp_path / "a.json", scores, "fixture/alpha")
        b = _write_arm(tmp_path / "b.json", shifted, "fixture/beta")
        report = render_report(a, b)
        assert "+0.05000" in report
        assert "no detectable difference" not in report

    def test_an_arm_that_lost_a_question_is_refused_not_silently_paired(self, tmp_path: Path) -> None:
        scores = _arm_scores()
        a = _write_arm(tmp_path / "a.json", scores, "fixture/alpha")
        b = _write_arm(tmp_path / "b.json", scores[:16], "fixture/alpha")
        with pytest.raises(CompareError, match="REFUSING"):
            render_report(a, b)
