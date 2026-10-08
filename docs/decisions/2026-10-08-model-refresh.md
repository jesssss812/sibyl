# Production model refresh, 2026-10-08

**Status:** shipped with the one-forecast-route change; the configuration era starts at
its merge.

This changes the production forecasting method. It is a new configuration era for
forecast comparisons. The model identities are pinned in
`sibyl/config/models.py` and `sibyl/config/settings.py`. The payer does not
change which model a slot calls. No paid inference or publishing validation was
performed while making these swaps.

| Role | Previous | New | Reason |
| --- | --- | --- | --- |
| OpenAI forecaster | GPT-5.6 Sol | GPT-6 Sol | Newer Sol-tier model at the same listed $2/$10 per million input/output tokens. |
| Anthropic forecaster | Claude Opus 5 | Claude Opus 5.5 | Latest verified Opus OpenRouter route. |
| Conditional stacker | Claude Opus 4.8 | Claude Opus 5.5 | Same Anthropic frontier route as the forecaster. |
| Stacker fallback | GPT-5.6 Sol | GPT-6 Sol | Keeps the cross-vendor Sol-tier fallback. |
| Parser, market ranker, market query author, financial classifier, leakage detector | GPT-5.6 Luna | GPT-6 Luna | Same fast/low-cost role tier; listed price falls from $0.20/$1.20 to $0.10/$0.50 per million tokens. |

GPT-5.6 Terra remains in the summarizer, disagreement analyzer, native web
search, and both gap-fill drivers/resolvers. No GPT-6 Terra route was verified.
The OpenAI forecaster stays in the Sol cost tier: [GPT-6 Astra](https://openrouter.ai/openai/gpt-6-astra)
is the new flagship but its listed $10/$50 per million-token rate is five times
Sol's $2/$10. Moving to Astra would change the budget shape as well as the
model generation and needs a separately sized reservation and quality test.
Claude web search stays on Claude Sonnet 5. It calls the Anthropic API
directly, and the catalogue check above was of OpenRouter routes only; the
direct model identifier and its server-side search tool need their own check.
Gemini 3.1 Pro Preview remains the Google forecaster; it is still the current
Pro-class text route listed by OpenRouter. The Google direct grounded-search and
document-reader models remain pinned pending a separate live capability check.

[Opus 5.5](https://openrouter.ai/anthropic/claude-opus-5.5) was OpenRouter's newest
Opus route on 2026-10-08. Other catalogue checks:
[GPT-6 Sol](https://openrouter.ai/openai/gpt-6-sol),
[GPT-6 Luna](https://openrouter.ai/openai/gpt-6-luna),
and [Gemini 3.1 Pro Preview](https://openrouter.ai/google/gemini-3.1-pro-preview).
Catalogue availability and listed prices do not prove that every key can serve
every route or that tool use works end to end. Validate on a separately
authorized paid, unscored run before relying on this roster for MiniBench.
