"""Shared reader for Gemini's ``url_context`` retrieval telemetry.

The grounded-search provider (``research/gemini_search.py``, off since 2026-08-31) hands
URLs to Gemini's ``url_context`` tool and uses the answer as evidence, so it needs to know
whether any URL was actually retrieved: Gemini answers fluently from parametric memory
when every retrieval failed. That is the Q38195 failure (2026-07-19: 30 search queries,
0 grounding chunks, a confident fabricated contract table with fake ``[primary]`` tags
reached forecasters). Gap-fill v2's ``read_document`` used this too until 2026-10-18; it
now fetches documents itself and reads them on OpenRouter.

The extraction reads only declared fields on the typed SDK pydantic models
(``url_context_metadata``, ``url_metadata``, and each ``UrlMetadata``'s
``url_retrieval_status`` / ``retrieved_url``), so a future SDK rename fails loudly here
rather than silently reporting zero retrievals forever.
"""

from __future__ import annotations

from google.genai import types as genai_types

URL_RETRIEVAL_SUCCESS = "URL_RETRIEVAL_STATUS_SUCCESS"


def coerce_status_name(status: object) -> str:
    """Coerce a url_retrieval_status (enum-with-.name, plain string, or None) to a string name."""
    name = getattr(status, "name", None)
    if isinstance(name, str):
        return name
    if isinstance(status, str):
        return status
    return str(status)


def extract_url_context_telemetry(
    response: genai_types.GenerateContentResponse,
) -> tuple[bool, int, int, list[tuple[str, str]]]:
    """Pull url_context retrieval telemetry off a Gemini response.

    Returns ``(reported, n_total, n_success, [(status_name, retrieved_url), ...])``. ``reported``
    is whether the SDK attached a url_context_metadata object at all — the tool ran and reported
    back, even with an empty fetch list — which lets callers keep 'fired but fetched nothing'
    greppably distinct from 'no url_context signal'. ``n_total`` is the url_metadata entry count
    and ``n_success`` those with status ``URL_RETRIEVAL_SUCCESS``.

    A None-valued (but present) entry still coerces gracefully — ``coerce_status_name(None)``
    maps to ``"None"`` and ``None or ""`` to ``""`` — so telemetry never takes down the
    research path it only observes.
    """
    candidates = response.candidates
    if not candidates:
        return (False, 0, 0, [])

    url_context_metadata = candidates[0].url_context_metadata
    if url_context_metadata is None:
        return (False, 0, 0, [])

    url_metadata = url_context_metadata.url_metadata
    if not url_metadata:
        return (True, 0, 0, [])

    entries: list[tuple[str, str]] = []
    n_success = 0
    for meta in url_metadata:
        status_name = coerce_status_name(meta.url_retrieval_status)
        retrieved_url = meta.retrieved_url or ""
        if status_name == URL_RETRIEVAL_SUCCESS:
            n_success += 1
        entries.append((status_name, retrieved_url))

    return (True, len(url_metadata), n_success, entries)
