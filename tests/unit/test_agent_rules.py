"""Every ``paths:`` glob in .claude/rules/ matches at least one file.

A rule loads only when an agent reads a file its globs match. Rename or delete that
file and the rule stops loading without anything failing, which is how a rule goes
quiet exactly when the code it guards is being changed. test_docs_paths.py checks the
backticked paths in a rule's body; this checks the front matter that decides when the
rule is read at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).parents[2]
RULES = sorted((REPO / ".claude" / "rules").glob("*.md"))


def _paths(rule: Path) -> list[str]:
    text = rule.read_text()
    if not text.startswith("---\n"):
        return []
    front_matter = text.split("---\n", 2)[1]
    return list((yaml.safe_load(front_matter) or {}).get("paths", []))


@pytest.mark.parametrize("rule", RULES, ids=lambda p: p.name)
def test_every_rule_glob_matches_a_file(rule: Path) -> None:
    unmatched = [pattern for pattern in _paths(rule) if not any(REPO.glob(pattern))]
    assert not unmatched, f"{rule.name} has paths: globs that match no file: {unmatched}"


def test_the_rules_have_globs() -> None:
    """A guard on the guard: front matter that failed to parse would pass every rule."""
    assert RULES
    assert all(_paths(rule) for rule in RULES)
