"""Every path the docs point at must exist.

A restructure this size breaks links silently: nothing fails, the file just stops being
there, and the reader finds out. These two checks are cheap and catch the whole class.

Two kinds of reference are checked, and one kind is deliberately not:

* backticked repo paths (`sibyl/config/models.py`) -- the form the prose uses when it
  names a file rather than linking to it;
* relative markdown links between docs.

``bottega:`` citations are SKIPPED by construction: they name files that live only in
the bottega repo, which is the whole point of the prefix. Anything under a directory
that is gitignored here (backtests/, run_logs/, research_outputs/) is skipped too --
those are runtime outputs, and a doc may legitimately name one that does not exist yet.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).parents[2]
# docs/plans/ is excluded: those are records of what was PLANNED, and they name
# bottega's layout and files a later phase creates. Checking them would assert that the
# plan was never written down.
DOC_FILES = sorted(
    [
        REPO / "README.md",
        REPO / "AGENTS.md",
        *(p for p in (REPO / "docs").rglob("*.md") if "plans" not in p.parts),
    ]
)

# Directories that hold generated or per-operator output, not source.
RUNTIME_DIRS = ("backtests/", "run_logs/", "research_outputs/", "benchmarks/", "scratch/")

# A backticked token that looks like a repo path: starts with a tracked top-level dir.
_BACKTICKED = re.compile(r"`([^`\s]+)`")
_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
_PATH_ROOTS = ("sibyl/", "tests/", "scripts/", "docs/", ".github/")


def _is_checkable_path(token: str) -> bool:
    if token.startswith("bottega:") or token.startswith(RUNTIME_DIRS):
        return False
    if not token.startswith(_PATH_ROOTS):
        return False
    # Prose often backticks a path with a symbol or a line number attached.
    return not any(ch in token for ch in "*<>{}|")


def _strip_locator(token: str) -> str:
    """Drop a trailing :line, ::member or #anchor so the file itself is what is checked."""
    return re.split(r"[:#]", token, maxsplit=1)[0].rstrip(".,;")


def _doc_id(path: Path) -> str:
    return str(path.relative_to(REPO))


@pytest.mark.parametrize("doc", DOC_FILES, ids=_doc_id)
def test_backticked_repo_paths_exist(doc: Path) -> None:
    missing = []
    for token in _BACKTICKED.findall(doc.read_text()):
        if not _is_checkable_path(token):
            continue
        target = _strip_locator(token)
        if not (REPO / target).exists():
            missing.append(token)
    assert not missing, f"{_doc_id(doc)} names paths that do not exist: {sorted(set(missing))}"


@pytest.mark.parametrize("doc", DOC_FILES, ids=_doc_id)
def test_relative_links_resolve(doc: Path) -> None:
    missing = []
    for href in _LINK.findall(doc.read_text()):
        if href.startswith(("http://", "https://", "#", "mailto:", "bottega:")):
            continue
        target = (doc.parent / _strip_locator(href)).resolve()
        if not target.exists():
            missing.append(href)
    assert not missing, f"{_doc_id(doc)} links to files that do not exist: {sorted(set(missing))}"


def test_the_check_actually_has_documents_to_check() -> None:
    """A guard on the guard: an empty DOC_FILES would make every test above vacuous."""
    assert len(DOC_FILES) >= 15
