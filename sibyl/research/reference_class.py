"""Reference-class retrieval: resolved Metaculus questions as an outside-view anchor.

Design principle M9 — "look up similar previously-resolved questions." 34% of Fall-2025
tournament winners did this and 0% of non-winners did (Fisher p=.04), the sharpest
categorical split in the bot-maker survey.

The forecaster prompts already ask every model to "list plausible reference classes and
evaluate suitability", but the model has to supply that reference class *from memory*.
This provider retrieves real resolved questions and hands over their actual outcomes, so
the outside view is evidence rather than recall.

Shape deliberately mirrors ``research/market_retrieval/``: generation is recall-maximal
and cheap, and ALL the judgment sits in one LLM ranking call — that subsystem's bake-off
measured selection, not generation, as the binding constraint. Query building here is
fully deterministic (no LLM), so the cheap half of the pipeline cannot fail or bill.

Backtest safety — the strategically important property
------------------------------------------------------
Candidates must have resolved **strictly before** the target question opened. That single
filter makes this the rare research provider that stays on under ``is_benchmarking``:
the prediction-market snapshot and the resolution-source fetcher are both *hard-disabled*
during benchmarking (post-resolution prices and present-day page content leak), so their
forecasting value cannot be measured by the standard backtest gate at all. This one can
be, which means a paired comparison on it is actually runnable.

The filter runs INSIDE pool assembly, ahead of ranking — not as a post-hoc cut over an
already-truncated list. Applying it afterwards would let an ineligible row consume a slot
the width had spent, exactly the bug the market-retrieval `as_of` filter had to fix.

Failure posture: every path soft-fails to ``""``. A dead Metaculus API or a broken ranker
costs this section, never the forecast.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime

from forecasting_tools.data_models.questions import MetaculusQuestion

from sibyl.common.time_utils import _as_utc
from sibyl.config.settings import (
    REFERENCE_CLASS_HTTP_TIMEOUT,
    REFERENCE_CLASS_MAX_CANDIDATES_PER_QUERY,
    REFERENCE_CLASS_MAX_RETRIES,
    REFERENCE_CLASS_MAX_ROWS,
    REFERENCE_CLASS_QUERY_PREFIX_WIDTHS,
    REFERENCE_CLASS_RANKER_WALL_TIMEOUT,
    REFERENCE_CLASS_RETRY_BACKOFF_SECS,
)

logger = logging.getLogger(__name__)

SECTION_HEADER = "## Reference Class (Resolved Metaculus Questions)"

# Interrogatives, auxiliaries and general-purpose glue. Dropping these is what turns a
# question sentence into a search query; keeping them would match every question on the
# platform. Deliberately conservative — anything domain-bearing stays.
_STOPWORDS = frozenset(
    """
    a an and any are as at be been before being below by can could did do does for from
    had has have how if in into is it its many more most much of on or over per since so
    than that the their then there these they this those to under until up was were what
    when where whether which while who whom why will with within would you your
    """.split()
)

# A token must contain a letter or digit to be searchable; "?" and "—" are not queries.
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'&.\-/]*")
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

# Resolution strings Metaculus uses for a binary YES / NO outcome.
# A question that ended without a real answer teaches nothing about a base rate, so
# these are dropped alongside the unresolved ones. Compared against a
# ``.strip().lower()`` value, so every member must itself be stripped and lowercase —
# tests pin each one against _is_real_outcome, because a stray space silently makes an
# entry unmatchable while its neighbours keep working.
_NON_OUTCOMES = frozenset({"annulled", "ambiguous"})

# Metaculus binary questions can report their resolution as 1/0 rather than yes/no.
_BINARY_YES = frozenset({"yes", "true", "1"})
_BINARY_NO = frozenset({"no", "false", "0"})

# Untrusted-text caps. `title` is public user-authored text from the Metaculus API;
# `relation` and `why` are raw model output. Both reach the forecaster bundle verbatim,
# so both are flattened and capped the way market_retrieval/rendering.py `_cell()` does.
# Newline flattening is the load-bearing half: `_demote_inner_headings` only PREFIXES a
# heading with extra hashes, so an embedded "## Prediction Market Snapshot" survives as a
# substring — and prompts.py gates a clause on exactly that kind of substring test.
_TITLE_MAX_CHARS = 160
_RELATION_MAX_CHARS = 32
_WHY_MAX_CHARS = 160


def _flatten(text: str, limit: int) -> str:
    """Collapse whitespace, strip heading markers, and hard-cap length.

    Stripping ``#`` is not belt-and-braces. Flattening newlines alone leaves the literal
    ``## Prediction Market Snapshot`` sitting inline, and prompts.py gates a prompt clause
    on a bare SUBSTRING test (``TS_ANCHOR_SECTION_HEADER in research``) rather than on a
    line-start match — so an inline forged header still trips downstream consumers. A
    ``#`` carries no meaning in a question title, so removing it costs nothing real.
    """
    collapsed = " ".join(str(text or "").replace("#", "").split())
    return collapsed[:limit]


@dataclass(frozen=True)
class Candidate:
    """One resolved Metaculus question, as retrieved and before any ranking."""

    question_id: int
    title: str
    url: str
    resolution: str
    resolved_at: datetime | None
    question_type: str


@dataclass(frozen=True)
class RankedRow:
    """A candidate the ranker kept, with its stated relation to the target question."""

    candidate: Candidate
    relation: str
    why: str


# A fetcher takes (query, limit) and returns candidates. Injected so tests never touch
# the network and so the Metaculus transport can be swapped without touching this logic.
Fetcher = Callable[[str, int], Awaitable[list[Candidate]]]
# A ranker takes a prompt and returns raw model text.
Ranker = Callable[[str], Awaitable[str]]


def build_search_queries(question: MetaculusQuestion) -> list[str]:
    """Deterministic search queries derived from the question's own words.

    No LLM: this half of the pipeline is free, cannot fail, and cannot bill. The
    market-retrieval subsystem pays for an LLM query author because venue catalogues use
    domain vocabulary a question's own tokens cannot reach; Metaculus questions are
    written in the same register as each other, so the target's own nouns are already the
    right index terms.

    Returns the full content phrase first (highest precision), then the prefixes named by
    ``REFERENCE_CLASS_QUERY_PREFIX_WIDTHS`` (higher recall). Duplicates collapse, so a
    short question yields fewer queries rather than repeats.
    """
    text = (question.question_text or "").strip()
    if not text:
        return []

    tokens = [t for t in _TOKEN_RE.findall(text) if t.lower() not in _STOPWORDS]
    if not tokens:
        return []

    queries: list[str] = []
    seen: set[str] = set()
    for width in (len(tokens), *REFERENCE_CLASS_QUERY_PREFIX_WIDTHS):
        candidate = " ".join(tokens[:width]).strip()
        if candidate and candidate.lower() not in seen:
            seen.add(candidate.lower())
            queries.append(candidate)

    return queries


def _is_real_outcome(resolution: str) -> bool:
    value = (resolution or "").strip().lower()
    if not value:
        return False
    return value not in _NON_OUTCOMES


def filter_eligible(
    candidates: list[Candidate],
    *,
    as_of: datetime | None,
    exclude_question_id: int | None = None,
) -> list[Candidate]:
    """Keep only candidates that are resolved, distinct, and safely in the past.

    ``as_of`` is the leakage ceiling — the target question's ``open_time`` under
    benchmarking, and ``None`` on a live run (where "already resolved" is by definition
    already public, so no ceiling is needed).

    The comparison is **strictly before**: a candidate resolving at the same instant the
    target opened is not demonstrably prior information. A candidate with no known
    resolve time is dropped for the same reason — it cannot be *proven* safe, and an
    unprovable candidate is exactly the kind that quietly contaminates a backtest.
    """
    eligible: list[Candidate] = []
    seen_ids: set[int] = set()

    for candidate in candidates:
        if candidate.question_id in seen_ids:
            continue
        if exclude_question_id is not None and candidate.question_id == exclude_question_id:
            continue
        if not _is_real_outcome(candidate.resolution):
            continue

        if as_of is not None:
            if candidate.resolved_at is None:
                continue
            if _as_utc(candidate.resolved_at) >= _as_utc(as_of):
                continue

        seen_ids.add(candidate.question_id)
        eligible.append(candidate)

    return eligible


def _extract_json_array(raw: str) -> list | None:
    """Pull a JSON array out of model text, fenced or bare. None when there isn't one."""
    fenced = _FENCED_JSON_RE.search(raw or "")
    body = fenced.group(1) if fenced else (raw or "")
    start, end = body.find("["), body.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(body[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, list) else None


def parse_ranker_response(raw: str, pool: list[Candidate]) -> list[RankedRow] | None:
    """Turn the ranker's output into rows, or ``None`` when the output is unusable.

    The two failure modes are deliberately distinguished, matching the market ranker:

    * ``[]`` — a VALID answer meaning "no analogue in this pool is worth showing".
    * ``None`` — unreadable output; the caller fails open to pool order.

    Ordering is the model's, verbatim: nothing downstream re-scores or re-sorts. Ids not
    present in the pool are dropped, so a hallucinated question can never be rendered.
    """
    parsed = _extract_json_array(raw)
    if parsed is None:
        return None

    by_id = {c.question_id: c for c in pool}
    rows: list[RankedRow] = []
    used: set[int] = set()

    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        raw_id = entry.get("question_id")
        # Only an exact integer identifies a candidate. Rejected deliberately:
        #   bool  - `isinstance(True, int)` is True, and True would select candidate 1
        #   float - `int(1.9)` silently selected a DIFFERENT analogue than the model named
        #   inf/nan - `int(float("inf"))` raises OverflowError, which is neither TypeError
        #             nor ValueError, so it escaped this parser entirely and cost the whole
        #             section instead of failing open to pool order
        if isinstance(raw_id, bool) or not isinstance(raw_id, int):
            continue
        qid = raw_id
        if qid in used or qid not in by_id:
            continue
        used.add(qid)
        rows.append(
            RankedRow(
                candidate=by_id[qid],
                relation=str(entry.get("relation") or "unspecified"),
                why=str(entry.get("why") or ""),
            )
        )

    # A non-empty array that yielded no usable row is a MALFORMED response, not the
    # model's considered "no analogue here". Returning [] there would render an empty
    # section — which reads to a forecaster as "we looked and found no precedent" — when
    # the truth is we have eligible candidates and only the ranking failed. Signal
    # unusable so the caller falls open to pool order.
    if parsed and not rows:
        return None

    return rows[:REFERENCE_CLASS_MAX_ROWS]


def _format_outcome(resolution: str, question_type: str) -> str:
    """Render an outcome, mapping to YES/NO only for BINARY questions.

    The type check is load-bearing. `0` and `1` are ordinary numeric outcomes, and a
    numeric question resolving to 0 is often the most informative datapoint in a
    reference class — labelling it "resolved NO" corrupts it both in the bundle and in
    the ranker prompt.
    """
    value = (resolution or "").strip()
    if question_type == "binary":
        lowered = value.lower()
        if lowered in _BINARY_YES:
            return "resolved YES"
        if lowered in _BINARY_NO:
            return "resolved NO"
    return f"resolved {value}"


def render_section(rows: list[RankedRow]) -> str:
    """Render ranked rows as the provider's section body, or "" when there are none.

    Returns the body only — the orchestrator prepends ``SECTION_HEADER``. An empty result
    must render as "" rather than a bare header, or the bundle gains an empty section
    that reads to a forecaster as "we looked and there is no precedent".
    """
    if not rows:
        return ""

    lines = [
        "Previously-resolved Metaculus questions selected as reference-class analogues "
        "for this question, with their actual outcomes. Every entry below **resolved "
        "before this question opened**, so each is information that was already "
        "available at forecast time.",
        "",
        "Weight these by the stated relation: a `direct` analogue is a near-repeat of "
        "the same mechanism, while a looser one only bounds the plausible range. An "
        "outcome here is evidence about the base rate, not about this question's answer.",
        "",
    ]

    for row in rows:
        candidate = row.candidate
        resolved_on = (
            _as_utc(candidate.resolved_at).date().isoformat() if candidate.resolved_at is not None else "unknown date"
        )
        lines.append(
            f"- **{_flatten(candidate.title, _TITLE_MAX_CHARS)}** — "
            f"{_format_outcome(candidate.resolution, candidate.question_type)} "
            f"({resolved_on}, {_flatten(candidate.question_type, _RELATION_MAX_CHARS)})"
        )
        detail = f"  - relation: `{_flatten(row.relation, _RELATION_MAX_CHARS)}`"
        why = _flatten(row.why, _WHY_MAX_CHARS)
        if why:
            detail += f" — {why}"
        lines.append(detail)
        lines.append(f"  - {candidate.url}")

    return "\n".join(lines)


def _build_ranker_prompt(question: MetaculusQuestion, pool: list[Candidate]) -> str:
    """One call, whole pool, model chooses the width. Empty array is a valid answer."""
    listing = "\n".join(
        f"- id={c.question_id} | {_flatten(c.title, _TITLE_MAX_CHARS)} | "
        f"{_format_outcome(c.resolution, c.question_type)} | type={c.question_type}"
        for c in pool
    )
    return f"""You are selecting reference-class analogues for a forecasting question.

TARGET QUESTION:
{question.question_text}

CANDIDATE RESOLVED QUESTIONS (all resolved before the target opened):
{listing}

Select up to {REFERENCE_CLASS_MAX_ROWS} candidates that genuinely inform the OUTSIDE VIEW
for the target — the historical frequency of outcomes for questions of this KIND. Rank
them best-first.

A good analogue shares the target's causal mechanism, actor type, or decision process. A
candidate that merely shares a topic word is NOT an analogue and must be omitted.

Returning an EMPTY array is a correct and expected answer when nothing in the pool is a
genuine analogue. Do not pad the list.

Respond with ONLY a JSON array:
[{{"question_id": <int>, "relation": "direct|analogous|weak", "why": "<one short phrase>"}}]
"""


def reference_class_provider(
    *,
    is_benchmarking: bool = False,
    fetcher: Fetcher,
    ranker: Ranker,
):
    """Build the research callable. ``fetcher`` and ``ranker`` are injected.

    Injection is not just for tests: it keeps this module free of transport and LLM
    wiring, so the Metaculus client and the ranker model can each change without
    touching the eligibility rule — which is the part that must stay correct.
    """

    async def _provider(question: MetaculusQuestion) -> str:
        try:
            return await _run(question)
        except Exception:  # noqa: BLE001 - soft-fail is the contract
            logger.warning("REFERENCE_CLASS: provider failed; returning empty section", exc_info=True)
            return ""

    async def _run(question: MetaculusQuestion) -> str:
        # Under benchmarking an unknowable ceiling must fail CLOSED. `as_of=None` means
        # "no ceiling" to filter_eligible, so a target with no open_time would silently
        # admit anything — including a candidate resolving after the backtest window —
        # while the rendered text still claimed everything predated the question. Same
        # posture as financial_data.py and timeseries_anchor.py, which both bail here.
        if is_benchmarking and not isinstance(question.open_time, datetime):
            logger.warning(
                "REFERENCE_CLASS: is_benchmarking but qid=%s has no open_time; skipping (leakage-safe)",
                question.id_of_question,
            )
            return ""

        queries = build_search_queries(question)
        if not queries:
            return ""

        # Per-query failure isolation: one transient search error must not zero the
        # others. Same posture as the venue-native market search.
        results = await asyncio.gather(
            *(fetcher(q, REFERENCE_CLASS_MAX_CANDIDATES_PER_QUERY) for q in queries),
            return_exceptions=True,
        )

        pool: list[Candidate] = []
        failures = 0
        for query, result in zip(queries, results):
            if isinstance(result, BaseException):
                failures += 1
                logger.warning("REFERENCE_CLASS: query %r failed: %s", query, result)
                continue
            pool.extend(result)

        if failures == len(queries):
            logger.warning("REFERENCE_CLASS: every query failed; returning empty section")
            return ""

        # Eligibility BEFORE ranking, so an ineligible row never consumes a slot and
        # never reaches the model.
        #
        # The ceiling applies on BOTH paths, not just under benchmarking. render_section
        # tells every forecaster "Every entry below resolved before this question opened";
        # skipping the filter live would make that claim false in the only mode that
        # normally ships. It also costs nothing live — a question resolved after the
        # target opened is simply not outside-view evidence.
        as_of = question.open_time
        eligible = filter_eligible(pool, as_of=as_of, exclude_question_id=question.id_of_post)
        if not eligible:
            logger.info("REFERENCE_CLASS: no eligible analogues after filtering %d candidate(s)", len(pool))
            return ""

        rows = await _rank(question, eligible)
        return render_section(rows)

    async def _rank(question: MetaculusQuestion, eligible: list[Candidate]) -> list[RankedRow]:
        """Rank, failing open to pool order on any error or unreadable output."""
        fallback = [RankedRow(candidate=c, relation="unranked", why="") for c in eligible[:REFERENCE_CLASS_MAX_ROWS]]
        try:
            raw = await ranker(_build_ranker_prompt(question, eligible))
        except Exception as exc:  # noqa: BLE001 - fail open, never fail the section
            logger.warning("REFERENCE_CLASS: ranker call failed (%s); falling back to pool order", exc)
            return fallback

        rows = parse_ranker_response(raw, eligible)
        if rows is None:
            logger.warning("REFERENCE_CLASS: ranker output unreadable; falling back to pool order")
            return fallback
        return rows

    return _provider


# ---------------------------------------------------------------------------
# Transport: Metaculus search + the ranker LLM
#
# Kept below the pure logic and behind the injected Fetcher/Ranker seams, so the
# eligibility rule above stays independently testable with no network in the path.
# ---------------------------------------------------------------------------

METACULUS_POSTS_URL = "https://www.metaculus.com/api/posts/"


def _parse_iso8601(value: object) -> datetime | None:
    """Lenient ISO-8601 parse over untrusted JSON. Never raises.

    Returning None on a bad timestamp is not a silent swallow: an undated candidate
    fails ``filter_eligible``, so a parse failure degrades to exclusion rather than to
    an unverifiable row sneaking into a backtest.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return _as_utc(datetime.fromisoformat(value.strip().replace("Z", "+00:00")))
    except (ValueError, TypeError):
        return None


def candidate_from_post(post: object) -> Candidate | None:
    """Project one ``/api/posts/`` entry onto a Candidate, or None if unusable.

    Defensive by construction: this reads an external API's JSON, so every field is
    treated as absent-or-wrong-typed until proven otherwise, and a malformed entry is
    skipped rather than allowed to raise through the provider's gather.
    """
    if not isinstance(post, dict):
        return None

    post_id = post.get("id")
    # `isinstance(True, int)` is True in Python, and a bool id would build the nonsense
    # URL `/questions/True/` and collide with candidate 1 in the pool.
    if isinstance(post_id, bool) or not isinstance(post_id, int):
        return None

    question = post.get("question")
    if not isinstance(question, dict):
        return None

    # `is None` rather than truthiness: a numeric question resolving to exactly 0 (or a
    # boolean False) is a real outcome, and `resolution or ""` would coerce it to the
    # empty string, after which _is_real_outcome drops it as unresolved. A zero is often
    # the most informative base-rate datapoint in a reference class.
    raw_resolution = question.get("resolution")
    resolution = "" if raw_resolution is None else str(raw_resolution)
    # ONLY the actual resolve time. A `scheduled_resolve_time` is a plan, not an event,
    # and is routinely earlier than the real one — so using it as a fallback would admit
    # candidates that actually resolved AFTER the target opened, defeating the leakage
    # ceiling with a value that merely looks authoritative. An undated candidate is
    # dropped by filter_eligible, which is the correct, provable outcome.
    resolved_at = _parse_iso8601(question.get("actual_resolve_time"))

    return Candidate(
        question_id=post_id,
        title=str(post.get("title") or "").strip() or f"Metaculus question {post_id}",
        url=f"https://www.metaculus.com/questions/{post_id}/",
        resolution=resolution.strip(),
        resolved_at=resolved_at,
        question_type=str(question.get("type") or "unknown"),
    )


async def _fetch_from_metaculus(query: str, limit: int) -> list[Candidate]:
    """One bounded search against the Metaculus posts API.

    Read-only and free — the same endpoint ``sibyl/eval/performance/collector.py`` already
    uses. No LLM in this path.
    """
    import os  # noqa: PLC0415

    import aiohttp  # noqa: PLC0415

    token = os.getenv("METACULUS_TOKEN", "")
    headers = {"Authorization": f"Token {token}"} if token else {}
    params = {"search": query, "statuses": "resolved", "limit": str(limit)}

    timeout = aiohttp.ClientTimeout(total=REFERENCE_CLASS_HTTP_TIMEOUT)
    payload: object = None
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Metaculus rate-limits, and this provider issues several queries per question
        # across a whole tournament run. sibyl/eval/performance/collector.py retries 429 on
        # this same API for exactly that reason. Without it a 429 empties the section,
        # which reads downstream as "no analogue exists" — a false negative that looks
        # like a real finding, which is worse than a visible failure.
        for attempt in range(REFERENCE_CLASS_MAX_RETRIES):
            async with session.get(METACULUS_POSTS_URL, params=params, headers=headers) as resp:
                if resp.status == 429 and attempt < REFERENCE_CLASS_MAX_RETRIES - 1:
                    wait = REFERENCE_CLASS_RETRY_BACKOFF_SECS * (attempt + 1)
                    logger.warning("REFERENCE_CLASS: 429 on %r, retrying in %.1fs", query, wait)
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                payload = await resp.json()
                break

    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        return []

    candidates = [candidate_from_post(post) for post in results]
    return [c for c in candidates if c is not None]


async def _rank_with_llm(prompt: str) -> str:
    """Single ranking call, under the same elapsed-gated retry wrapper the market ranker uses."""
    from sibyl.config.models import (  # noqa: PLC0415
        MARKET_RANKER_LLM_CONFIG,
    )
    from sibyl.platform.key_fallback import (  # noqa: PLC0415
        build_llm_with_openrouter_fallback,
    )
    from sibyl.platform.llm_retry import (  # noqa: PLC0415
        invoke_with_transient_retry,
    )

    # Reuses the market ranker's slot rather than adding a roster entry: identical job
    # (rank a candidate pool, emit a small JSON array) and identical output budget, so a
    # separate config would be a duplicate that could silently drift.
    llm = build_llm_with_openrouter_fallback(**MARKET_RANKER_LLM_CONFIG)
    return await invoke_with_transient_retry(
        lambda: llm.invoke(prompt),
        wall_timeout=REFERENCE_CLASS_RANKER_WALL_TIMEOUT,
        label="reference_class_ranker",
        backoffs=(1.0, 2.0),
    )


def default_reference_class_provider(is_benchmarking: bool = False):
    """Production wiring: the live Metaculus search plus the ranker LLM."""
    return reference_class_provider(
        is_benchmarking=is_benchmarking,
        fetcher=_fetch_from_metaculus,
        ranker=_rank_with_llm,
    )
