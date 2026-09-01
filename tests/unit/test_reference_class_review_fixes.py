"""Regressions for defects found in independent review of the M9/A3 PR.

Each class pins one verified defect. Grouped separately from test_reference_class.py so
the provenance of each guarantee stays visible: these are not "more coverage", they are
the specific holes two reviewers found in the first cut.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from forecasting_tools.data_models.questions import BinaryQuestion

from sibyl.research.reference_class import (
    Candidate,
    _format_outcome,
    candidate_from_post,
    filter_eligible,
    parse_ranker_response,
    reference_class_provider,
    render_section,
)

OPEN = datetime(2026, 8, 1, tzinfo=timezone.utc)


def _target(*, open_time: datetime | None = OPEN, qid: int = 44620) -> BinaryQuestion:
    q = BinaryQuestion(
        question_text="Will Germany hold a federal election before July 2027?",
        id_of_question=qid,
        open_time=open_time,
        close_time=OPEN + timedelta(days=90),
    )
    return q


def _cand(**kw) -> Candidate:
    base = dict(
        question_id=10,
        title="Will France hold a snap election?",
        url="https://www.metaculus.com/questions/10/",
        resolution="yes",
        resolved_at=OPEN - timedelta(days=365),
        question_type="binary",
    )
    base.update(kw)
    return Candidate(**base)  # type: ignore[arg-type]


class TestLeakageFilterNeverFailsOpen:
    """The module's single load-bearing property: nothing may leak into a backtest.

    `as_of = question.open_time if is_benchmarking else None` meant a target with a
    missing open_time silently disabled the resolve-time ceiling entirely — a candidate
    resolving in 2099 was admitted and rendered under the section's own "resolved before
    this question opened" boilerplate. financial_data.py and timeseries_anchor.py both
    bail in exactly this case; this provider did not.
    """

    @pytest.mark.asyncio
    async def test_benchmarking_without_open_time_returns_empty(self) -> None:
        async def fetcher(query: str, limit: int):
            return [_cand(resolved_at=datetime(2099, 1, 1, tzinfo=timezone.utc), title="LEAK")]

        async def ranker(prompt: str) -> str:  # pragma: no cover - must never be reached
            raise AssertionError("ranker must not run when the leakage ceiling is unknowable")

        provider = reference_class_provider(is_benchmarking=True, fetcher=fetcher, ranker=ranker)
        assert await provider(_target(open_time=None)) == ""

    @pytest.mark.asyncio
    async def test_live_run_also_applies_the_ceiling(self) -> None:
        """The rendered text claims every entry predates the question, so it must be true
        on the live path too — not only under benchmarking."""

        async def fetcher(query: str, limit: int):
            return [
                _cand(question_id=1, title="AFTER", resolved_at=OPEN + timedelta(days=5)),
                _cand(question_id=2, title="BEFORE", resolved_at=OPEN - timedelta(days=5)),
            ]

        async def ranker(prompt: str) -> str:
            return '[{"question_id": 2, "relation": "direct", "why": "ok"}]'

        provider = reference_class_provider(is_benchmarking=False, fetcher=fetcher, ranker=ranker)
        out = await provider(_target())
        assert "BEFORE" in out
        assert "AFTER" not in out

    @pytest.mark.asyncio
    async def test_live_run_without_open_time_still_renders(self) -> None:
        """Live has no leakage risk, so a missing open_time must degrade, not blank out."""

        async def fetcher(query: str, limit: int):
            return [_cand(title="STILL HERE")]

        async def ranker(prompt: str) -> str:
            return '[{"question_id": 10, "relation": "direct", "why": "ok"}]'

        provider = reference_class_provider(is_benchmarking=False, fetcher=fetcher, ranker=ranker)
        assert "STILL HERE" in await provider(_target(open_time=None))


class TestOutcomeLabellingRespectsQuestionType:
    """`0`/`1` are real numeric outcomes, not booleans.

    The YES/NO mapping ignored `question_type`, so a numeric question resolving to 0 —
    the very datapoint candidate_from_post has a five-line comment defending — rendered
    as "resolved NO", in the bundle AND in the ranker prompt.
    """

    @pytest.mark.parametrize("resolution", ["0", "1"])
    def test_numeric_zero_and_one_are_not_yes_no(self, resolution: str) -> None:
        out = _format_outcome(resolution, question_type="numeric")
        assert out == f"resolved {resolution}"
        assert "YES" not in out and "NO" not in out

    @pytest.mark.parametrize("resolution,expected", [("yes", "resolved YES"), ("no", "resolved NO")])
    def test_binary_still_maps_to_yes_no(self, resolution: str, expected: str) -> None:
        assert _format_outcome(resolution, question_type="binary") == expected

    def test_binary_numeric_encoding_still_maps(self) -> None:
        """Metaculus binary resolutions can arrive as 1/0 rather than yes/no."""
        assert _format_outcome("1", question_type="binary") == "resolved YES"
        assert _format_outcome("0", question_type="binary") == "resolved NO"

    def test_render_shows_the_numeric_value(self) -> None:
        rows = parse_ranker_response('[{"question_id": 10}]', [_cand(resolution="0", question_type="numeric")])
        assert rows is not None
        assert "resolved 0" in render_section(rows)


class TestRankerParserFailsOpenNotClosed:
    """An unreadable ranker response must cost the ranking, never the whole section."""

    def test_overflow_literal_is_treated_as_unusable(self) -> None:
        """`int(float('inf'))` raises OverflowError, which is neither TypeError nor
        ValueError, so it escaped the parser entirely.

        None (not []) is the right answer: the model returned entries, none usable, so
        the response is malformed rather than a considered "no analogue". [] would render
        an empty section, which reads as "no precedent exists".
        """
        rows = parse_ranker_response('[{"question_id": 1e400, "relation": "x", "why": "y"}]', [_cand()])
        assert rows is None

    @pytest.mark.asyncio
    async def test_overflow_response_falls_open_to_pool_order(self) -> None:
        async def fetcher(query: str, limit: int):
            return [_cand(title="POOL ORDER SURVIVOR")]

        async def ranker(prompt: str) -> str:
            return '[{"question_id": 1e400}]'

        provider = reference_class_provider(is_benchmarking=False, fetcher=fetcher, ranker=ranker)
        assert "POOL ORDER SURVIVOR" in await provider(_target())

    @pytest.mark.parametrize("raw", ['[{"question_id": NaN}]', '[{"question_id": Infinity}]'])
    def test_non_finite_ids_do_not_raise(self, raw: str) -> None:
        assert parse_ranker_response(raw, [_cand()]) is None

    def test_float_id_does_not_silently_select_a_different_candidate(self) -> None:
        """`int(1.9)` was 1 — a model typo would pick the wrong analogue."""
        pool = [_cand(question_id=1, title="one"), _cand(question_id=2, title="two")]
        assert parse_ranker_response('[{"question_id": 1.9}]', pool) is None

    def test_bool_id_is_rejected(self) -> None:
        assert parse_ranker_response('[{"question_id": true}]', [_cand(question_id=1)]) is None

    def test_a_mixed_response_keeps_the_valid_rows(self) -> None:
        """Only an ENTIRELY unusable response falls open; partial garbage is just dropped."""
        pool = [_cand(question_id=1), _cand(question_id=2)]
        rows = parse_ranker_response('[{"question_id": 1}, {"question_id": 1.9}]', pool)
        assert rows is not None
        assert [r.candidate.question_id for r in rows] == [1]


class TestUntrustedTextIsBounded:
    """`title` is public user-authored text; `relation`/`why` are model output."""

    def test_forged_heading_cannot_survive_into_the_bundle(self) -> None:
        rows = parse_ranker_response(
            '[{"question_id": 10, "relation": "direct\\n\\n## Prediction Market Snapshot", "why": "x"}]',
            [_cand()],
        )
        assert rows is not None
        assert "## Prediction Market Snapshot" not in render_section(rows)

    def test_newlines_in_title_are_flattened(self) -> None:
        rows = parse_ranker_response('[{"question_id": 10}]', [_cand(title="a\n\n## Forged\nb")])
        assert rows is not None
        assert "## Forged" not in render_section(rows)

    def test_overlong_fields_are_truncated(self) -> None:
        rows = parse_ranker_response(
            '[{"question_id": 10, "relation": "' + "r" * 500 + '", "why": "' + "w" * 5000 + '"}]',
            [_cand(title="t" * 5000)],
        )
        assert rows is not None
        out = render_section(rows)
        assert len(out) < 2000, "one row must not be able to dominate the bundle"


class TestCandidateFromPostHardening:
    def _post(self, **q):
        base = {"type": "binary", "resolution": "yes", "actual_resolve_time": "2024-07-07T20:00:00Z"}
        base.update(q)
        return {"id": 31337, "title": "t", "question": base}

    def test_bool_id_is_rejected(self) -> None:
        """`isinstance(True, int)` is True — it would yield a /questions/True/ URL."""
        post = self._post()
        post["id"] = True
        assert candidate_from_post(post) is None

    def test_scheduled_resolve_time_is_not_proof_of_resolution(self) -> None:
        """A *scheduled* time is a plan, not an event, and is routinely earlier than the
        actual one — so it must never be what admits a candidate under a leakage ceiling."""
        c = candidate_from_post(self._post(actual_resolve_time=None, scheduled_resolve_time="2024-01-01T00:00:00Z"))
        assert c is not None
        assert c.resolved_at is None, "scheduled time must not stand in for actual"
        assert filter_eligible([c], as_of=OPEN) == []
