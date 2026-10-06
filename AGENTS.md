# AGENTS.md

Always-true rules for anyone working in this repo, human or agent. Everything here is
load-bearing; the detail lives in `docs/`.

## The cost rules

These are the only rules that can cost real money if broken.

1. **No paid calls without the operator asking.** Never invoke a paid model or a paid
   research API — OpenRouter non-free slugs, Anthropic direct, OpenAI, AskNews, Exa,
   Perplexity, Google grounded search. The network-egress guard in `tests/conftest.py`
   stays autouse for every test outside `tests/live/`.
2. **Free is the default.** With `SIBYL_MODEL_TIER` unset, every model slot is an
   OpenRouter `:free` slug. `--tier prod` or `SIBYL_MODEL_TIER=prod` is how a run asks to
   spend, and the production workflows set it explicitly.
3. **Free models never publish to a scored tournament.** `tournament`, `minibench` and
   `cup` are refused on the free tier before any client is built. Only `test`, which
   publishes to the unscored `bot-testing-area`, is allowed.
4. **"Operator go" means for one specific run.** `make e2e_free` and live tests other
   than the nonpublishing `make e2e_free_no_publish` need the operator to say go for
   *that* run. Prior approval of a similar run is not approval of this one.
   `make e2e_free` costs nothing but publishes real comments on real questions.
   The operator has authorized `make e2e_free_no_publish` as the required pre-push
   check after code changes: it uses only OpenRouter `:free` models on a synthetic
   question and blocks all Metaculus publish calls.
5. **Never** re-enable a `schedule:` block, dispatch a workflow, or commit a secret.

## Working rules

6. **Functional parity is proven, not asserted.** `tests/e2e/test_golden.py` records what
   the pipeline actually produces. Scenarios 1–5 are byte-identical to the records made
   in bottega before the restructure. If a change moves one of them, that is the finding
   — investigate before re-recording. `--record-golden` is a deliberate act.
7. **Keep commits small.** One reviewable change each — a rename, a test, an extraction —
   never a whole workstream in one blob. Separate mechanical renames from behaviour
   changes. Small commits are what make `git bisect` and `git log --follow` worth having.
8. **Tests before restructuring.** Move or write the test first, then change the code.
9. **Leave what looks wrong.** If a module seems wrong but fixing it is not the task, add
   a line to `docs/roadmap.md` instead. A restructure whose diff also contains behaviour
   changes cannot be checked against a parity record.
10. **Never commit on `main`.** Work on a feature branch; `clean-fork` is the integration
    branch.
11. **Show publishing runs live.** For an authorized test or forecast that publishes,
    open a visible terminal or live log view before starting, use unbuffered output,
    and stream progress while retaining a local log. Do not hide the run until it ends.

## Before every commit

```bash
make test        # suite; live tests deselected by default
make lint        # ruff
make typecheck   # basedpyright, must report 0 errors
```

## Before pushing code changes

Run `make e2e_free_no_publish` after the final code edit and before every push.
It requires `OPENROUTER_API_KEY`, makes live free-model calls, and must never
publish. If the key, network, or a free provider is unavailable, do not substitute
`make e2e_free` or claim the check passed; report the blocker. A feature branch may
still be pushed so the work is not lost, with the skipped check stated in the
report, but it must not be merged into `main` or `clean-fork` until the check has
passed on its final commit. This standing authorization applies only to this
nonpublishing check.

## Layout

| Package | Holds |
| --- | --- |
| `sibyl/common/` | Types and helpers with no dependency on the rest of the bot. |
| `sibyl/config/` | Environment loading, tunable settings, and the model roster. |
| `sibyl/platform/` | Adapters to third parties: API keys, retries, and the patches and hardening applied to forecasting-tools and Metaculus. |
| `sibyl/research/` | Research providers and the orchestrator that selects them. |
| `sibyl/forecasting/` | Prompts, per-type runners, extraction, and `SibylBot` itself. |
| `sibyl/aggregation/` | Combining several forecasters into one published forecast. |
| `sibyl/publishing/` | Rendering and posting the comment that accompanies a forecast. |
| `sibyl/telemetry/` | Counters, markers, and the archive that harvests them from run logs. |
| `sibyl/eval/` | Offline measurement: backtest, ablation, ensemble, performance. |
| `sibyl/ops/` | Operator scripts. Not imported by the bot. |

Tests are laned: `tests/unit/`, `tests/integration/`, `tests/e2e/`, `tests/live/`, with
shared fakes in `tests/helpers/` and fixtures in `tests/data/`.

## Entry points

```bash
python -m sibyl <tournament|minibench|cup|test> [--tier free|prod] [--max-questions N]
python -m sibyl.eval <backtest|ablation|ensemble|performance|fit-platt|compare> [args]
```

Every pre-fork invocation still works: `--mode metaculus_cup`, `--mode test_questions`,
`python -m sibyl.eval.backtest` and the rest are kept as aliases.

## Things that look like mistakes and are not

- **Three identifiers keep a "donated"/"personal" vocabulary** that was renamed everywhere
  else: the `PAID PERSONAL-KEY FALLBACK` log line, and `personal_key_fallback=` /
  `donated_404=` / `donated_key=` in the end-of-run summary. Those two lines are parsed by
  regexes in `sibyl/telemetry/markers.py` and their capture groups are archive field
  names. Renaming them orphans every archived telemetry record.
- **An empty env var means the default, not an empty list**, for the key-routing
  settings. `""` is far more often an unset secret than a deliberate choice.
- **`sibyl/forecasting/bot.py` imports from `sibyl/eval/`** for comment annotation. Known;
  in the roadmap.
- **`forecasting-tools` is pinned to `==0.2.92`** and `tests/integration/test_framework_*`
  pin that exact version's behaviour. A bump means re-reading them, not re-recording them.
- **A green free-tier run can still have lost forecasters.** `forecasters_dropped` is not
  alertable in the free tier (shared upstream `:free` quotas 429 constantly), so
  `make e2e_free` exits 0 with models missing. Read `SYSTEMATIC_FORECASTER_FAILURE` and
  `forecasters_dropped=` in the log, not the exit code. Prod alerts on drops as before.
- **`bottega:` in a comment or doc** points at the [bottega](https://github.com/jesssss812/bottega)
  repo, which holds the design log and study notes this fork deliberately does not carry.

## Where detail lives

- [README.md](README.md) — what sibyl is, and the quickstart.
- [docs/README.md](docs/README.md) — the documentation index.
- [docs/architecture.md](docs/architecture.md) — how a forecast is produced.
- [docs/operations.md](docs/operations.md) — running it; what the telemetry means.
- [docs/eval-strategy.md](docs/eval-strategy.md) — the rules every comparison follows.
- [docs/decisions/](docs/decisions/) — why things are the way they are.
- [docs/roadmap.md](docs/roadmap.md) — what is not done.
