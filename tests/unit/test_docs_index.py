"""Every document under docs/ is reachable from the docs index.

A doc nobody links to is a doc nobody reads, and an agent that starts from
docs/README.md cannot find it at all. test_docs_paths.py proves the links that exist
resolve; this proves the ones that should exist do.
"""

from __future__ import annotations

import re
from pathlib import Path

DOCS = Path(__file__).parents[2] / "docs"
INDEX = DOCS / "README.md"
_LINK = re.compile(r"\[[^\]]*\]\(([^)\s#]+)(?:#[^)\s]*)?\)")


def _linked_from_index() -> set[Path]:
    return {(DOCS / href).resolve() for href in _LINK.findall(INDEX.read_text()) if not href.startswith("http")}


def test_every_doc_is_listed_in_the_index() -> None:
    linked = _linked_from_index()
    unlisted = sorted(
        str(doc.relative_to(DOCS)) for doc in DOCS.rglob("*.md") if doc != INDEX and doc.resolve() not in linked
    )
    assert not unlisted, f"docs/README.md does not link to: {unlisted}"


def test_the_index_links_to_something() -> None:
    """A guard on the guard: a link regex that matched nothing would pass every doc."""
    assert len(_linked_from_index()) >= 15
