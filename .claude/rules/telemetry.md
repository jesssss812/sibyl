---
paths:
  - "sibyl/telemetry/**"
  - "sibyl/cli.py"
  - "sibyl/forecasting/value_extraction.py"
  - "sibyl/forecasting/numeric/diagnostics.py"
  - "sibyl/research/agentic/loop.py"
  - "sibyl/research/provider_health.py"
  - "sibyl/research/prediction_market.py"
  - "sibyl/aggregation/agreement.py"
  - "sibyl/aggregation/resolution_guard.py"
  - "sibyl/publishing/close_margin.py"
  - "sibyl/platform/key_fallback.py"
---

# Telemetry markers

Run-log markers are an archive format, not log text.

- `sibyl/telemetry/markers.py` parses every marker with a regex written against the
  emitted format string. The marker name is the archive file's stem, and each named
  group is an archive field. Changing an emitted format string, a marker name or a group
  name orphans every record already in `backtests/telemetry_archive/`.
- The module docstring of `markers.py` lists which file emits each marker. A new marker
  goes in that list, gets a `MarkerSpec`, and gets a case in
  `tests/unit/test_telemetry_markers.py` built from the real format string.
- Three identifiers keep the pre-rename vocabulary on purpose: the
  `PAID PERSONAL-KEY FALLBACK` line and `personal_key_fallback=` / `donated_404=` /
  `donated_key=` in the end-of-run summary. Do not rename them (AGENTS.md).
- What each marker means to an operator is in `docs/runbooks/operations.md`, "Reading run
  logs". Update it in the same change.
