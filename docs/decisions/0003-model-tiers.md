# 0003 — Model tiers and the spend guard

*2026-09-01. Status: in force.*

## Context

The repo's first rule is that nothing spends money without the operator asking. That rule
lived in prose, and prose depends on remembering. Meanwhile there was no way to exercise
the pipeline end to end without paying for it: the only rosters were the paid ones, so
"does the whole thing still work" and "spend real money" were the same action.

## Decision

`SIBYL_MODEL_TIER` selects a `Roster` — a frozen dataclass with one field per model slot.
Two tiers: `free`, every slot an OpenRouter `:free` slug, and `prod`, the paid roster
unchanged.

**Unset means free.** A run that forgets the flag cannot spend; the expensive path is the
one that has to be asked for. The prod workflows say `SIBYL_MODEL_TIER: 'prod'`
explicitly, and a test asserts all four still do.

An unrecognised value raises rather than defaulting, because neither fallback is safe:
defaulting to free silently downgrades a prod run and publishes free-model forecasts to a
scored tournament, and defaulting to prod silently spends.

The free tier **refuses** every scored mode (`tournament`, `minibench`, `cup`) before the
roster is built, any key is read or any question is fetched. A free forecast is not a
cheap version of a prod forecast; it is a much worse one, and publishing it to a scored
tournament damages a public record that cannot be retracted. `test` is the only
non-scored mode, and a test asserts that partition directly so a new mode cannot land
outside the guard.

Providers are vetoed by **substitutability, not cost**. AskNews, the OpenAI and Anthropic
server-side web searches, Google grounding and the two gap-fill loops are paid *search
tools* — no `:free` slug comes with a search index attached, so pointing them at a
cheaper model makes them broken rather than free. Financial data, prediction markets,
resolution source, reference class and time-series anchor are the opposite shape: free or
public data with an LLM stage bolted on, so they stay and that stage moves to the free
utility model.

## Amendment (2026-09-02): free-tier forecaster drops are not alertable

`alertable_total` excludes `forecasters_dropped` in the FREE tier, and only there.

Every OpenRouter `:free` slug shares one upstream pool with the whole of that tier, so a
429 tracks the hour rather than anything in this repo. Two consecutive `make e2e_free`
runs each lost a forecaster to one, which made the lane permanently red for a reason that
said nothing about the code — and a signal that is always red stops being read, which
costs more than the signal was worth.

**The caveat, stated plainly, because this is a real trade and it cuts against
[eval-strategy.md](../eval-strategy.md) Rule 6 ("degraded runs fail loud"):** a free-tier
run now exits 0 with forecasters missing. If a free model breaks systematically — a
withdrawal like the qwen 404, a refusal class, a routing change — the exit status will not
tell you. It will look like a clean run.

What still tells you, and what to actually read:

- `SYSTEMATIC_FORECASTER_FAILURE` at **WARNING**, naming the model, the question ids and
  the cause, whenever one model fails across multiple questions in a run. This is the line
  that caught both faults so far.
- `Degradation counters: forecasters_dropped=N` in the run summary, unchanged and still a
  parsed telemetry contract.
- `FORECASTERS_SURVIVED: survived=N/M` per question.
- `tests/live/test_free_tier.py`, which checks every roster slug against OpenRouter's live
  catalogue — the only thing that catches a withdrawal, since a withdrawn slug still ends
  in `:free`.

The exemption is deliberately narrow, and tests hold it there: it covers exactly
`forecasters_dropped`, in exactly the free tier. Every other counter — a failure to
publish, a research provider erroring, a stacker fallback — still exits non-zero in the
free tier, because none of those is a fact of life about shared quotas. Prod is untouched.

## Consequences

- **The plan's provider list was incomplete, and this is the finding worth keeping.** The
  *primary* research ladder — AskNews → Exa → Perplexity — has no feature flag at all. It
  runs whether or not any add-on is enabled, and every rung is paid, so a free run would
  have billed on `openrouter/perplexity/sonar-reasoning-pro`: the one provider nobody
  opted into. It is disabled in the free tier and has its own test, because it is exactly
  the paid path a flag-shaped guard misses.
- The financial-data classifier is the one paid LLM call reachable in the free tier (its
  provider stays enabled because FRED and yfinance cost nothing), so it reads its model
  from the roster.
- `make e2e_free` runs the whole pipeline at zero API spend against bot-testing-area. It
  still **publishes**: free of money is not free of consequence. Nothing there is scored,
  which is why it is the only mode the free tier may run.
- The test suite is pinned to `prod` in `conftest.py`. Every test predates the split and
  was written against the prod provider set; inheriting the new free default would have
  turned dozens of provider tests into assertions about a provider that never ran.
- `active_tier()` reads `os.environ` rather than `os.getenv`, because several tests patch
  `os.getenv` process-wide to stub a few flags and a patched getenv answers `""` for every
  name it was not told about.
