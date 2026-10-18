# The first MiniBench round: grant-only models, one question at a time

*2026-10-18.* Sibyl enters its first scored MiniBench round, which opens on Monday
19 October 2026. This records what was changed for it and why.

## Context

- MiniBench questions launch at random hours and are open for about three hours.
  Usually five or fewer are open at once, occasionally about ten, and most launch in a
  round's first few days ([Metaculus FutureEval resources](https://www.metaculus.com/notebooks/38928/docs.adj.news)).
- The round is paid from the $100 OpenRouter grant. Its account routes only OpenAI,
  Anthropic and Google models, and its Google route has no quota: Gemini 3.1 Pro got an
  immediate 429 on every grant call and was paid by the personal key instead. The
  account's settings cannot be changed by us.
- The personal key is the last resort. The MiniBench GitHub workflow has no grant key,
  so a round run there would bill the personal key for everything.

## Decisions

1. **The third forecaster is Claude Sonnet 5.5, everywhere.** It replaces Gemini 3.1
   Pro in every mode, not only on the grant, so a grant run and an ordinary run use one
   method. Sonnet costs about what Gemini did. The swap is unmeasured, and the only
   evidence points against a same-vendor replacement: the July composition screen
   (`bottega:scratch/ensemble_composition_2026-07-15/`) found that dropping Gemini cost
   about 2 log points even with a second OpenAI and a second Anthropic member present.
   An offline re-screen with the six-member history was not possible: Metaculus returns
   403 on the bot's comment reads, and the cached July dataset is gone.
2. **Every model call goes through OpenRouter.** Claude web search moved from the
   Anthropic API to OpenRouter's native web search (Sonnet 5.5). The gap-fill document
   reader moved from Gemini 3.7 Flash on Google's API to GPT-6 Luna on OpenRouter, reading
   documents the bot fetches itself. The automatic research ladder no longer calls
   Perplexity's own API. AskNews, Exa, FRED and the market and resolution-source fetches
   are data services and keep their own keys.
3. **One question at a time, soonest close first** (`--one-at-a-time`). A run sorts its
   questions by scheduled close before the per-run cap and forecasts them one after
   another, so only one question's calls are in flight and the question closest to
   closing goes first.
4. **A 15-minute polling loop on the operator's machine** (`python -m sibyl.ops.minibench_loop`).
   It never overlaps runs, uses the MiniBench workflow's own settings, and stops on a
   round grant budget, a grant floor, a personal-key spend cap, a refused run or an end
   time. Run, settings and stops: [minibench-readiness.md](../runbooks/minibench-readiness.md).

## Consequences

- The round starts a new configuration era for every comparison: new third forecaster,
  new Claude-search and reader routes.
- Two vendors carry the ensemble. Opus and Sonnet now start together on each question,
  so a rate limit on the grant's Anthropic route would push both to the personal key.
- A page that blocks the bot's own fetches now blocks the document reader too; Gemini
  used to fetch from Google's side.
- A free-tier run used to pay for GPT-6 Luna in the prediction-market and
  reference-class rankers; those stages now follow the run's tier.
