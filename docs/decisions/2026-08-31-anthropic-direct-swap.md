# Moving Anthropic off OpenRouter, and Gemini search off Google

**Date:** 2026-08-31 · **Status:** shipped · **Trigger:** a failed live smoke test

## What broke

The first paid single-question run (Q43323, numeric, bot-testing-area) produced
**zero forecasts**. Three separate credential failures, not one:

| Provider | Failure | Root cause |
|---|---|---|
| All three forecasters | `402 Payment Required` ×3 | OpenRouter balance |
| AskNews | `401 Unauthorized` | bad credentials (since fixed) |
| Gemini grounded search | `429 RESOURCE_EXHAUSTED` | monthly free bucket exhausted |

## The OpenRouter failure is subtler than "no credit"

The error was not "balance is zero". It was:

> `You requested up to 64000 tokens, but can only afford 1594.`

**OpenRouter reserves the full `max_tokens` up front** and refuses the request if
the balance cannot cover the reservation — regardless of how many tokens the call
would actually use. The proof is in the same log: the prediction-market ranker
**succeeded** on that run. It declares `max_tokens: 3000`; the forecasters declare
`64_000`. Small calls cleared the reservation, large ones did not.

Lowering `max_tokens` would "fix" this and was rejected: 64k exists to stop
reasoning models truncating mid-answer, and a truncated forecast is worse than a
missing one.

## Decision 1 — Anthropic slots route direct

`openrouter/anthropic/claude-opus-5` → `anthropic/claude-opus-5` (forecaster) and
`openrouter/anthropic/claude-opus-4.8` → `anthropic/claude-opus-4-8` (stacker).
**Models are unchanged; only the transport is.** With the primary Metaculus key
exhausted until next season and the fallback OpenRouter balance unable to cover a
reservation, the OpenRouter hop was a middleman on a key with no credit.

`build_llm_with_openrouter_fallback` is deliberately still the constructor:
`should_route_via_primary_key` gates on an `openrouter/` prefix, so a bare
`anthropic/` slug already falls through to a plain `GeneralLlm` and litellm reads
`ANTHROPIC_API_KEY`. Routing back is a one-token change.

### The trap this created

Two kwargs do **not** survive the swap, and neither one errors:

* `reasoning={"effort": "xhigh"}` — the OpenRouter passthrough dict. litellm's
  Anthropic transform advertises `reasoning_effort`, not `reasoning`.
* `extra_body={"verbosity": "high"}` — OpenRouter-only, no Anthropic equivalent.

`forecasting_tools` sets **`litellm.drop_params = True` process-globally**, so both
are **silently discarded**. Left unchanged, the flagship forecaster would have run
at default effort with nothing in any log to say so. They are converted to
`reasoning_effort="xhigh"` and dropped, respectively, and
`tests/unit/test_anthropic_direct_routing.py` plus the updated
`test_ft_llm_seams.py` pin both shapes so the regression cannot return quietly.

Also note the id shape: Anthropic uses dashes (`claude-opus-4-8`), OpenRouter uses
dots (`claude-opus-4.8`). The dotted form 404s upstream.

**Out of scope:** the ablation/backtest harness (`sibyl/eval/ablation/`) still
routes Anthropic through OpenRouter. Its cache-key scheme strips the `openrouter/`
prefix and its lineup constants are offline-replay fixtures — rerouting them would
break the cache for no benefit on runs that are not being made.

## Decision 2 — Claude web search replaces Gemini grounded search

Google meters grounded search on a **monthly** free bucket (~5k prompts on the 3.x
family), not a daily one, so an exhausted quota stays exhausted until the month
rolls over. There is no separate free grounding tier to discover — the free tier
*is* that bucket. With no billing account attached, this provider was contributing
nothing.

`research/anthropic_search.py` fills the same role: a search index the OpenAI
native-search provider does not share, keeping two genuinely independent retrieval
paths in the ensemble.

**Model: `claude-sonnet-5` at `medium` effort, `max_uses: 15`.** Deliberately the
middle tier — this slot retrieves and synthesises, it does not forecast, so Opus
would be 2.5× the price for search-and-summarise. It cannot go lower: the
dynamic-filtering `web_search_20260209` variant requires Sonnet 4.6+/Opus 4.6+, and
Haiku 4.5 only gets the older basic tool. `max_uses` is the hard cost bound
(searches bill at $10/1k on top of tokens); Gemini's AFC loop had no equivalent knob.

**Direct SDK, not litellm** — same `drop_params` hazard as above. litellm's
Anthropic transform does not model server-side tools at all, so routing this through
`GeneralLlm` would drop the `web_search` tool and return confident *ungrounded*
text, which is exactly the failure the grounded floor exists to catch.

**Gemini is not deleted.** Module, tests and constants are intact and default-off;
one env flag restores it if quota returns or billing is added. Both flags are
independent, so running both for a side-by-side comparison is supported.

### Inherited safeguard

The grounded floor carries over from the Gemini provider's hardest lesson (Q38195,
2026-07-19: 30 queries, 0 grounding chunks, a confident fabricated contract table
with fake `[primary]` tags reached the forecasters). A response with zero search
results is suppressed, and — because an empty section is otherwise byte-identical to
a healthy call that found nothing — the loss is recorded to Provider Diagnostics and
emitted as `ANTHROPIC_UNGROUNDED_SUPPRESSED`, registered in
`sibyl/telemetry/markers.py` from day one so the suppression *rate* is directly
comparable between the two search indexes. That comparison is the evidence for
whether this swap was like-for-like.

## What this does not fix

`ANTHROPIC_API_KEY` still has to be funded. **A Claude.ai Pro/Max subscription is
not an API credential** — the Anthropic API is billed separately at
console.anthropic.com. Until a key is set, the Anthropic forecaster slot, the
stacker and this search provider all fail the same way the OpenRouter slots did.

## Post-ship correction (same day, first live run)

The swap worked — `anthropic/claude-opus-5` forecast successfully and the Anthropic
search provider returned 76 results — but the first live run (Q43323) exposed three
defects in the new provider, all now fixed:

1. **Inter-attempt narration reached forecasters.** The research section opened with
   *"Let me wait and retry with a single, simple call.I'll pause and try again in the
   next turn…"* — three concatenated status mutterings. A server-side search loop
   interleaves narration with its tool calls; Gemini never did this because its AFC
   loop returns one synthesized answer. Only text after the last
   `web_search_tool_result` is kept now, and where several text blocks survive, only
   the last — separate blocks can only arise across a tool call, since
   `get_final_message()` merges contiguous streamed text.

2. **`max_uses: 8` bound the research, not just the cost.** The model exhausted the
   budget, emitted eight consecutive `max_uses_exceeded` blocks still trying, and
   closed with *"I could not access more forward-looking data … due to search tool
   constraints."* A cost bound that changes the answer is not a cost bound. Raised to
   15 ($0.15 worst case, against a measured ~$1.61+ per-question run).

3. **76 undeduplicated sources.** Gallup, Newsweek, Pew, UCSB and Wikipedia each
   appeared two or more times. This provider had inherited Gemini's deliberate
   *no-dedupe* rule without inheriting the reason for it — Gemini splices inline `[N]`
   markers by chunk index, so dropping a row misaligns every later marker. This
   provider emits no inline markers, so the constraint never applied and the
   duplicates were pure bundle bloat on an already-trimmed comment.

Also: the eight identical `max_uses_exceeded` warnings are now one aggregated line.
Repetition buried the actual signal, which was that the cap was too low.

### Known gap, not fixed here

`CREDIT_SPEND` polls **OpenRouter only**. Now that the flagship forecaster and the
search provider bill Anthropic, per-run cost telemetry undercounts by the whole
Anthropic share. The 2026-08-31 run reported `run_delta_usd=1.61` for OpenRouter with
the Anthropic spend invisible.
