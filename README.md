# sibyl

A forecasting bot for [Metaculus](https://www.metaculus.com/). It fetches open questions,
researches them, asks an ensemble of models for a forecast, aggregates the answers, and
publishes a prediction with its reasoning.

## Attribution

sibyl is a fork of [nostreambot](https://github.com/No-Stream/metaculus-bot) by
No-Stream, used under the MIT license, which is itself built on the
[Metaculus bot template](https://github.com/Metaculus/metac-bot-template). nostreambot
placed 9th in Fall 2025 (top open-source) and 10th in Spring 2026; the forecasting engine,
the numeric PCHIP pipeline and the ensemble design are its work. The `LICENSE` file keeps
the original copyright alongside this fork's.

What is new here, relative to that upstream:

- **Reference-class retrieval** — surfacing resolved Metaculus questions similar to the
  one being forecast, as a base-rate anchor.
- **Open-vs-resolved guard** — refusing to treat an already-resolved question as open.
- **Zero-cost fixture backtest** — the whole eval path on a seeded fake model, so the
  harness can be validated without spending anything.
- **Ensemble agreement telemetry** — measuring how much the forecasters actually disagree,
  which is what conditional stacking routes on.
- **Anthropic direct routing and Claude web search** — off OpenRouter for the Anthropic
  slots, after an exhausted key cost a run three forecasters.
- **Model tiers and the spend guard** — see below.
- **The free test lane** — the entire pipeline, end to end, at zero API spend.
- **Eval consolidation** — one entry point, and a paired `compare` that refuses to report
  a summary it cannot support.
- **This restructure** — ten purpose-named packages, proven equivalent to the original by
  a golden parity harness rather than by assertion.

## Quickstart

```bash
uv sync --dev
cp .env.template .env    # then fill in METACULUS_TOKEN and OPENROUTER_API_KEY
make test
```

Run the pipeline end to end at **zero API spend**, against the unscored
`bot-testing-area` tournament:

```bash
make e2e_free
```

A single mode, explicitly:

```bash
uv run python -m sibyl test --tier free --max-questions 3
```

## The cost rules

1. **Free is the default.** With `SIBYL_MODEL_TIER` unset every model slot is an
   OpenRouter `:free` slug. Spending requires asking, via `--tier prod` or
   `SIBYL_MODEL_TIER=prod`, which is what the production workflows set.
2. **Free models never publish to a scored tournament.** `tournament`, `minibench` and
   `cup` are refused on the free tier before anything is built. Only `test`, which
   publishes to the unscored sandbox, is allowed.
3. **Tests cannot spend.** An autouse guard in `tests/conftest.py` blocks every non-local
   socket for the whole suite. The live tests, which opt out, use `:free` models only.

`make e2e_free` costs nothing but it does **publish** — real comments on real questions,
in a tournament where nothing is scored.

## Where to look next

- [docs/README.md](docs/README.md) — the documentation index.
- [AGENTS.md](AGENTS.md) — the always-true rules, for humans and coding agents alike.
- [docs/architecture.md](docs/architecture.md) — how a forecast is actually produced.
- [docs/operations.md](docs/operations.md) — running it, and what the telemetry means.
- [docs/decisions/](docs/decisions/) — why things are the way they are.
- [docs/roadmap.md](docs/roadmap.md) — what is not done.

Licensed under the MIT License. See [LICENSE](LICENSE).
