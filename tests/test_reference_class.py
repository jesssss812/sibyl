"""Tests for reference-class retrieval (metaculus_bot/research/reference_class.py).

Design principle M9: "Look up similar previously-resolved questions." 34% of Fall-2025
winners did this; 0% of non-winners did (Fisher p=.04) — the sharpest categorical split
in the survey, and the one Moderate item this fork had not implemented.

The bot already *prompts* for reference-class reasoning in all three question-type
prompts, but the model has to supply the reference class from memory. This provider
retrieves actual resolved Metaculus questions and hands over their real outcomes.

Shape follows research/market_retrieval/: recall-maximal candidate generation, with all
the judgment concentrated in ONE LLM ranking call, because that subsystem's bake-off
measured selection (not generation) as the binding constraint.

**The eligibility filter is the load-bearing part.** Candidates must have resolved
STRICTLY BEFORE the target question opened. That is what makes this the rare research
provider that is backtest-safe: prediction markets and the resolution-source fetcher are
both hard-disabled under `is_benchmarking` and so cannot be measured offline at all.

No test here touches the network or an LLM: fetch and rank are injected.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from forecasting_tools.data_models.questions import BinaryQuestion

from metaculus_bot.research.reference_class import (
    Candidate,
    build_search_queries,
    filter_eligible,
    parse_ranker_response,
    reference_class_provider,
    render_section,
)

OPEN = datetime(2026, 8, 1, 0, 0, 0, tzinfo=timezone.utc)


def _target(text: str = "Will Germany hold a federal election before July 2027?") -> BinaryQuestion:
    return BinaryQuestion(
        question_text=text,
        id_of_question=44620,
        open_time=OPEN,
        close_time=OPEN + timedelta(days=90),
    )


# Sentinel so a caller can pass resolved_at=None explicitly and mean it. Using None as
# the default would make "unset" and "explicitly unknown" indistinguishable — which
# silently defeats the unknown-resolve-time test.
_UNSET = object()


def _candidate(
    *,
    qid: int,
    title: str = "Will France hold a snap election in 2024?",
    resolved_at: datetime | None | object = _UNSET,
    resolution: str = "yes",
) -> Candidate:
    return Candidate(
        question_id=qid,
        title=title,
        url=f"https://www.metaculus.com/questions/{qid}/",
        resolution=resolution,
        resolved_at=(OPEN - timedelta(days=365)) if resolved_at is _UNSET else resolved_at,  # type: ignore[arg-type]
        question_type="binary",
    )


class TestBuildSearchQueries:
    """Deterministic keyword extraction — no LLM, so it costs nothing and cannot fail."""

    def test_produces_at_least_one_query(self) -> None:
        assert build_search_queries(_target())

    def test_strips_stopwords_and_interrogatives(self) -> None:
        queries = build_search_queries(_target("Will the United States enter a recession in 2027?"))
        joined = " ".join(queries).lower()
        assert "will" not in joined.split()
        assert "the" not in joined.split()

    def test_keeps_proper_nouns(self) -> None:
        queries = build_search_queries(_target("Will Germany hold a federal election before July 2027?"))
        assert any("germany" in q.lower() for q in queries)

    def test_queries_are_deduped(self) -> None:
        queries = build_search_queries(_target("Election election ELECTION in Germany?"))
        assert len(queries) == len(set(q.lower() for q in queries))

    def test_empty_question_text_yields_no_queries(self) -> None:
        assert build_search_queries(_target("")) == []

    def test_ladder_is_full_phrase_then_prefixes(self) -> None:
        """The previous assertion (<= 8) was tautological: the builder emits at most 3."""
        from metaculus_bot.constants import REFERENCE_CLASS_QUERY_PREFIX_WIDTHS

        long_q = _target(" ".join(f"Topic{i}" for i in range(200)) + "?")
        queries = build_search_queries(long_q)
        assert len(queries) == 1 + len(REFERENCE_CLASS_QUERY_PREFIX_WIDTHS)
        assert len(queries[0].split()) == 200, "first query is the full content phrase"
        for query, width in zip(queries[1:], REFERENCE_CLASS_QUERY_PREFIX_WIDTHS):
            assert len(query.split()) == width

    def test_short_question_yields_fewer_queries_not_repeats(self) -> None:
        queries = build_search_queries(_target("Germany election?"))
        assert len(queries) == len(set(queries))


class TestFilterEligible:
    """The leakage guard. This is the property that makes the provider backtest-safe."""

    def test_keeps_candidate_resolved_before_open(self) -> None:
        c = _candidate(qid=1, resolved_at=OPEN - timedelta(days=1))
        assert filter_eligible([c], as_of=OPEN) == [c]

    def test_drops_candidate_resolved_after_open(self) -> None:
        c = _candidate(qid=2, resolved_at=OPEN + timedelta(days=1))
        assert filter_eligible([c], as_of=OPEN) == []

    def test_drops_candidate_resolved_exactly_at_open(self) -> None:
        """Strictly-before: a same-instant resolution is not safely in the past."""
        c = _candidate(qid=3, resolved_at=OPEN)
        assert filter_eligible([c], as_of=OPEN) == []

    def test_drops_candidate_with_unknown_resolve_time(self) -> None:
        """Unknown resolution time cannot be proven safe, so it is excluded."""
        c = _candidate(qid=4, resolved_at=None)
        assert filter_eligible([c], as_of=OPEN) == []

    def test_drops_the_target_question_itself(self) -> None:
        c = _candidate(qid=44620, resolved_at=OPEN - timedelta(days=10))
        assert filter_eligible([c], as_of=OPEN, exclude_question_id=44620) == []

    def test_drops_unresolved_candidates(self) -> None:
        c = _candidate(qid=5, resolved_at=OPEN - timedelta(days=10), resolution="")
        assert filter_eligible([c], as_of=OPEN) == []

    def test_no_as_of_keeps_everything_resolved(self) -> None:
        """Live runs have no as_of ceiling; only the resolved-ness check applies."""
        future = _candidate(qid=6, resolved_at=OPEN + timedelta(days=5))
        assert filter_eligible([future], as_of=None) == [future]

    def test_dedupes_by_question_id_keeping_first(self) -> None:
        a = _candidate(qid=7, title="first")
        b = _candidate(qid=7, title="second")
        assert filter_eligible([a, b], as_of=OPEN) == [a]

    def test_naive_datetimes_are_treated_as_utc(self) -> None:
        c = _candidate(qid=8, resolved_at=(OPEN - timedelta(days=1)).replace(tzinfo=None))
        assert filter_eligible([c], as_of=OPEN) == [c]


class TestParseRankerResponse:
    POOL = [_candidate(qid=10), _candidate(qid=11), _candidate(qid=12)]

    def test_parses_fenced_json(self) -> None:
        raw = '```json\n[{"question_id": 11, "relation": "direct", "why": "same mechanism"}]\n```'
        rows = parse_ranker_response(raw, self.POOL)
        assert rows is not None
        assert len(rows) == 1
        assert rows[0].candidate.question_id == 11
        assert rows[0].relation == "direct"
        assert rows[0].why == "same mechanism"

    def test_parses_bare_json(self) -> None:
        raw = '[{"question_id": 10, "relation": "analogous", "why": "adjacent domain"}]'
        rows = parse_ranker_response(raw, self.POOL)
        assert rows is not None
        assert len(rows) == 1

    def test_preserves_model_ordering(self) -> None:
        raw = '[{"question_id": 12, "relation": "a", "why": "w"}, {"question_id": 10, "relation": "b", "why": "w"}]'
        rows = parse_ranker_response(raw, self.POOL)
        assert rows is not None
        assert [r.candidate.question_id for r in rows] == [12, 10]

    def test_empty_array_is_a_valid_answer_not_a_failure(self) -> None:
        """Mirrors the market ranker: 'no relevant analogue' is a real answer."""
        assert parse_ranker_response("[]", self.POOL) == []

    def test_ignores_ids_not_in_pool(self) -> None:
        """A hallucinated id must never reach the render.

        With every entry unusable the response is malformed, so this reports None and the
        caller falls open to pool order — better than an empty section, which a forecaster
        would read as "no precedent exists".
        """
        raw = '[{"question_id": 999, "relation": "direct", "why": "invented"}]'
        assert parse_ranker_response(raw, self.POOL) is None

    def test_unparseable_output_returns_none_to_signal_fail_open(self) -> None:
        assert parse_ranker_response("I could not determine anything.", self.POOL) is None

    def test_missing_fields_are_tolerated(self) -> None:
        rows = parse_ranker_response('[{"question_id": 10}]', self.POOL)
        assert rows is not None
        assert len(rows) == 1
        assert rows[0].relation == "unspecified"

    def test_duplicate_ids_collapse(self) -> None:
        raw = '[{"question_id": 10, "why": "a"}, {"question_id": 10, "why": "b"}]'
        rows = parse_ranker_response(raw, self.POOL)
        assert rows is not None
        assert len(rows) == 1


class TestRenderSection:
    def test_empty_rows_render_empty_string(self) -> None:
        """An empty section must not add a bare header to the bundle."""
        assert render_section([]) == ""

    def test_renders_outcome_and_relation(self) -> None:
        rows = parse_ranker_response(
            '[{"question_id": 10, "relation": "direct", "why": "same mechanism"}]', [_candidate(qid=10)]
        )
        assert rows is not None
        out = render_section(rows)
        assert "resolved YES" in out
        assert "direct" in out
        assert "same mechanism" in out
        assert "metaculus.com/questions/10" in out

    def test_renders_unusual_but_real_resolutions_verbatim(self) -> None:
        """A numeric/MC outcome that is not yes/no still renders as its own value."""
        pool = [_candidate(qid=10, resolution="42.5")]
        rows = parse_ranker_response('[{"question_id": 10}]', pool)
        assert rows is not None
        assert "resolved 42.5" in render_section(rows)

    def test_includes_a_leakage_provenance_line(self) -> None:
        """Readers (and reviewers) must be able to see the eligibility rule that applied."""
        rows = parse_ranker_response('[{"question_id": 10}]', [_candidate(qid=10)])
        assert rows is not None
        assert "resolved before" in render_section(rows).lower()


@pytest.mark.asyncio
class TestProviderOrchestration:
    """End-to-end with fetch and rank injected — no network, no LLM."""

    async def _run(self, *, fetcher, ranker, is_benchmarking=False):
        provider = reference_class_provider(
            is_benchmarking=is_benchmarking,
            fetcher=fetcher,
            ranker=ranker,
        )
        return await provider(_target())

    async def test_happy_path_renders_ranked_rows(self) -> None:
        async def fetcher(query: str, limit: int):
            return [_candidate(qid=10, title="Will France hold a snap election in 2024?")]

        async def ranker(prompt: str) -> str:
            return '[{"question_id": 10, "relation": "direct", "why": "same electoral mechanism"}]'

        out = await self._run(fetcher=fetcher, ranker=ranker)
        assert "same electoral mechanism" in out
        assert "France" in out

    async def test_no_candidates_returns_empty_string(self) -> None:
        async def fetcher(query: str, limit: int):
            return []

        async def ranker(prompt: str) -> str:  # pragma: no cover - must not be called
            raise AssertionError("ranker must not run on an empty pool")

        assert await self._run(fetcher=fetcher, ranker=ranker) == ""

    async def test_fetch_failure_soft_fails_to_empty(self) -> None:
        async def fetcher(query: str, limit: int):
            raise RuntimeError("metaculus down")

        async def ranker(prompt: str) -> str:  # pragma: no cover
            raise AssertionError("unreachable")

        assert await self._run(fetcher=fetcher, ranker=ranker) == ""

    async def test_one_query_failing_does_not_lose_the_others(self) -> None:
        """Per-query failure isolation, same posture as the venue-native market search."""
        calls: list[str] = []

        async def fetcher(query: str, limit: int):
            calls.append(query)
            if len(calls) == 1:
                raise RuntimeError("transient")
            return [_candidate(qid=10)]

        async def ranker(prompt: str) -> str:
            return '[{"question_id": 10, "relation": "direct", "why": "ok"}]'

        out = await self._run(fetcher=fetcher, ranker=ranker)
        assert "ok" in out

    async def test_ranker_failure_fails_open_to_pool_order(self) -> None:
        """A transient ranker error must cost the ranking, never the whole snapshot."""

        async def fetcher(query: str, limit: int):
            return [_candidate(qid=10, title="Pool order survivor")]

        async def ranker(prompt: str) -> str:
            raise RuntimeError("llm blew up")

        out = await self._run(fetcher=fetcher, ranker=ranker)
        assert "Pool order survivor" in out

    async def test_unparseable_ranker_output_fails_open(self) -> None:
        async def fetcher(query: str, limit: int):
            return [_candidate(qid=10, title="Pool order survivor")]

        async def ranker(prompt: str) -> str:
            return "sorry, no idea"

        assert "Pool order survivor" in await self._run(fetcher=fetcher, ranker=ranker)

    async def test_benchmarking_applies_the_open_time_ceiling(self) -> None:
        """Under benchmarking, a candidate resolved after the target opened is dropped."""

        async def fetcher(query: str, limit: int):
            return [
                _candidate(qid=10, title="LEAKY", resolved_at=OPEN + timedelta(days=5)),
                _candidate(qid=11, title="SAFE", resolved_at=OPEN - timedelta(days=5)),
            ]

        async def ranker(prompt: str) -> str:
            return '[{"question_id": 11, "relation": "direct", "why": "eligible"}]'

        out = await self._run(fetcher=fetcher, ranker=ranker, is_benchmarking=True)
        assert "SAFE" in out
        assert "LEAKY" not in out

    async def test_leaky_candidate_never_reaches_the_ranker_prompt(self) -> None:
        """Filter runs INSIDE pool assembly, ahead of ranking — not as a post-hoc cut."""
        seen: dict[str, str] = {}

        async def fetcher(query: str, limit: int):
            return [_candidate(qid=10, title="LEAKYTITLE", resolved_at=OPEN + timedelta(days=5))]

        async def ranker(prompt: str) -> str:
            seen["prompt"] = prompt
            return "[]"

        await self._run(fetcher=fetcher, ranker=ranker, is_benchmarking=True)
        assert "LEAKYTITLE" not in seen.get("prompt", "")


class TestCandidateFromPost:
    """Parsing the Metaculus /api/posts/ payload. Untrusted JSON: never raise, just skip."""

    def _post(self, **overrides) -> dict:
        post = {
            "id": 31337,
            "title": "Will France hold a snap election in 2024?",
            "question": {
                "type": "binary",
                "resolution": "yes",
                "actual_resolve_time": "2024-07-07T20:00:00Z",
            },
        }
        post.update(overrides)
        return post

    def test_parses_a_well_formed_post(self) -> None:
        from metaculus_bot.research.reference_class import candidate_from_post

        c = candidate_from_post(self._post())
        assert c is not None
        assert c.question_id == 31337
        assert c.resolution == "yes"
        assert c.question_type == "binary"
        assert c.resolved_at == datetime(2024, 7, 7, 20, 0, tzinfo=timezone.utc)
        assert c.url == "https://www.metaculus.com/questions/31337/"

    def test_missing_id_is_skipped(self) -> None:
        from metaculus_bot.research.reference_class import candidate_from_post

        post = self._post()
        del post["id"]
        assert candidate_from_post(post) is None

    def test_missing_question_block_is_skipped(self) -> None:
        from metaculus_bot.research.reference_class import candidate_from_post

        assert candidate_from_post(self._post(question=None)) is None

    def test_unparseable_resolve_time_becomes_none(self) -> None:
        """A bad timestamp must not raise — it makes the candidate ineligible instead."""
        from metaculus_bot.research.reference_class import candidate_from_post

        c = candidate_from_post(
            self._post(question={"type": "binary", "resolution": "yes", "actual_resolve_time": "not-a-date"})
        )
        assert c is not None
        assert c.resolved_at is None
        assert filter_eligible([c], as_of=OPEN) == []

    def test_does_not_fall_back_to_scheduled_resolve_time(self) -> None:
        """A scheduled resolve time is a plan, not an event.

        It is routinely earlier than the actual one, so treating it as proof of when a
        question resolved would admit candidates that really resolved AFTER the target
        opened — defeating the leakage ceiling with a value that merely looks
        authoritative. An undated candidate is dropped instead, which is provable.
        """
        from metaculus_bot.research.reference_class import candidate_from_post

        c = candidate_from_post(
            self._post(
                question={"type": "binary", "resolution": "no", "scheduled_resolve_time": "2024-01-01T00:00:00Z"}
            )
        )
        assert c is not None
        assert c.resolved_at is None
        assert filter_eligible([c], as_of=OPEN) == []

    def test_garbage_input_is_skipped_not_raised(self) -> None:
        from metaculus_bot.research.reference_class import candidate_from_post

        for junk in [None, [], "string", 42, {}]:
            assert candidate_from_post(junk) is None  # type: ignore[arg-type]


class TestNonOutcomesAreExcluded:
    """A question that ended without a real answer teaches nothing about a base rate.

    Each member of _NON_OUTCOMES is asserted individually: the original bug was a single
    trailing space on "annulled " that made that one entry dead while "ambiguous" kept
    working, so a test over only one member would not have caught it.
    """

    @pytest.mark.parametrize("resolution", ["annulled", "ambiguous", "ANNULLED", "  Annulled  "])
    def test_non_outcome_is_filtered_out(self, resolution: str) -> None:
        c = _candidate(qid=1, resolution=resolution)
        assert filter_eligible([c], as_of=OPEN) == []

    def test_every_non_outcome_member_actually_matches(self) -> None:
        """Guards the guard: a stray space or case in the frozenset makes an entry dead."""
        from metaculus_bot.research.reference_class import _NON_OUTCOMES, _is_real_outcome

        for member in _NON_OUTCOMES:
            assert member == member.strip().lower(), f"{member!r} can never match a stripped value"
            assert _is_real_outcome(member) is False

    @pytest.mark.parametrize("resolution", ["yes", "no", "42.5", "0"])
    def test_real_outcomes_survive(self, resolution: str) -> None:
        c = _candidate(qid=1, resolution=resolution)
        assert filter_eligible([c], as_of=OPEN) == [c]


class TestZeroResolutionIsARealOutcome:
    """`resolution or ""` coerced a legitimate 0.0 to empty, dropping the candidate."""

    @pytest.mark.parametrize("raw,expected", [(0.0, "0.0"), (0, "0"), (42.0, "42.0"), (False, "False")])
    def test_falsy_numeric_resolution_is_preserved(self, raw, expected: str) -> None:
        from metaculus_bot.research.reference_class import candidate_from_post

        c = candidate_from_post(
            {
                "id": 1,
                "title": "t",
                "question": {"type": "numeric", "resolution": raw, "actual_resolve_time": "2024-01-01T00:00:00Z"},
            }
        )
        assert c is not None
        assert c.resolution == expected

    def test_zero_resolved_question_survives_the_filter(self) -> None:
        from metaculus_bot.research.reference_class import candidate_from_post

        c = candidate_from_post(
            {
                "id": 1,
                "title": "How many X occurred?",
                "question": {"type": "numeric", "resolution": 0.0, "actual_resolve_time": "2024-01-01T00:00:00Z"},
            }
        )
        assert c is not None
        assert filter_eligible([c], as_of=OPEN) == [c]

    def test_missing_resolution_is_still_dropped(self) -> None:
        from metaculus_bot.research.reference_class import candidate_from_post

        c = candidate_from_post(
            {"id": 1, "title": "t", "question": {"type": "binary", "actual_resolve_time": "2024-01-01T00:00:00Z"}}
        )
        assert c is not None
        assert c.resolution == ""
        assert filter_eligible([c], as_of=OPEN) == []
