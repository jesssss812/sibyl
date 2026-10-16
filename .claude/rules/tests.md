---
paths:
  - "tests/**"
  - "pyproject.toml"
---

# Tests

- **Lanes.** `tests/unit/`, `tests/integration/`, `tests/e2e/`, `tests/live/`. Shared fakes
  go in `tests/helpers/`, JSON fixtures in `tests/data/`. Both `tests/data/*.json` and
  `tests/e2e/golden/*.json` are un-ignored in `.gitignore`; a fixture anywhere else that
  matches `*.json` is silently left out of the commit.
- **The network-egress guard in `tests/conftest.py` stays autouse** for every test
  outside `tests/live/`. It is the money-safety backstop. A test that needs a real socket
  marks itself `allow_network`; never weaken the guard itself.
- **Live tests** carry the `live` marker, are deselected by default, and need the
  operator's go for the specific run, except `make e2e_free_no_publish` (AGENTS.md).
- **Golden parity.** `tests/e2e/test_golden.py` compares against
  `tests/e2e/golden/*.json`. A mismatch is a finding: investigate before re-recording.
  `--record-golden` is a deliberate act, never a fix for a red test.
  See `docs/decisions/0001-parity-contract.md`.
- **Docs are tested too.** `tests/unit/test_docs_paths.py` fails on a broken link or a
  backticked path that does not exist; `tests/unit/test_docs_index.py` fails on a doc
  missing from `docs/README.md`.
- Never skip, disable or quarantine a test to get green.
