# 0004 — Primary and fallback keys

*2026-09-01. Status: in force.*

## Context

Two OpenRouter keys: one donated by Metaculus with server-side allowed-provider
preferences, one the operator's own. The code named them for their *provenance* —
"donated" and "personal"/"general" — which is a fact about who paid rather than about
what the routing does, and which stops being true when the subsidy ends.

The routing policy itself was two frozensets in `fallback_openrouter.py`: which providers
prefer the donated key, and which models are pinned off it. Both changed twice in three
months, and each change was a code edit and a redeploy.

## Decision

Rename by role: **primary** and **fallback**. Env var names do not change —
`OAI_ANTH_OPENROUTER_KEY` is still the primary key and `OPENROUTER_API_KEY` the fallback,
and `GEMINI_USE_DONATED_OPENROUTER_KEY` keeps its name.

The two lists become configuration: `OPENROUTER_PRIMARY_KEY_PROVIDERS` and
`OPENROUTER_PRIMARY_KEY_BLOCKED_MODELS`, read at call time, defaulting to the old
literals so an unset var reproduces today's routing exactly.

An env var set to the empty string yields the **default**, not the empty set. Emptying a
routing list is a real thing an operator might want, but `""` is far more often an unset
secret or a stray `export FOO=`, and silently sending every call to the fallback key is
not a failure mode worth inferring from a typo. To genuinely empty one, name a provider
that does not exist.

## Consequences

- **Three identifiers keep the old vocabulary on purpose**, each with a comment saying so:
  the `PAID PERSONAL-KEY FALLBACK: donated OpenRouter key failed for model=` log line, and
  the `personal_key_fallback=` / `donated_404=` / `donated_key=` field names in the
  end-of-run summary. Both lines are parsed by `MarkerSpec` regexes in
  `sibyl/telemetry/markers.py`, and those capture groups become field names in the
  telemetry archive. Renaming them would orphan every archived record and break harvesting
  on re-run — the exact failure that module's own docstring warns about.
- That conflict was not theoretical. The rename initially caught both regexes and their
  sample log lines, and the marker tests caught all four.
- The `gemini-3.1-pro` blocklist entry was tagged `TODO(gemini-3.1-pro-donated)` when
  clearing it meant editing a frozenset. It is a config value now, so the tag was wrong
  about what the fix costs; the prose says which condition to check and which variable to
  change instead.
- `OPENROUTER_CREDIT_ALERT_RESUME_DATE` was already env-driven but undocumented. Its
  default is unchanged and `.env.template` now explains what the suppression window is:
  the operator is self-funding the season, so an empty subsidised wallet is expected
  rather than alertable, while a fallback from any other cause stays alertable throughout.
