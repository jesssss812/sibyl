---
paths:
  - "docs/**"
  - "README.md"
  - "AGENTS.md"
---

# Docs

- **Reference vs. intent.** `docs/architecture/` and `docs/runbooks/` describe the code
  as it is; when one disagrees with the code, the doc is wrong. `docs/strategy/`,
  `docs/decisions/` and `docs/plans/` are intent and records: a mismatch with the code is
  not a bug, so they are not rewritten to match it.
- **Intent changes deliberately.** `docs/strategy/eval-strategy.md` is the spec every
  comparison follows: a change to how something is measured updates it in the same
  change. When a file moves, every doc's links and paths are updated.
- **One home per fact.** Before adding a fact, find where it already lives and link to
  it. A second copy drifts.
- **Every doc is in the index.** A new doc gets a row in `docs/README.md`;
  `tests/unit/test_docs_index.py` enforces it, and `tests/unit/test_docs_paths.py`
  checks every link and backticked repo path.
- **AGENTS.md is for always-true rules.** Reference detail goes in `docs/`, rules for one
  area go in `.claude/rules/`.
- Things not done go in `docs/roadmap.md`, one line each, with a pointer.
- `bottega:` marks a path in the bottega repo. It is not checked and must not be
  rewritten to a local path.
