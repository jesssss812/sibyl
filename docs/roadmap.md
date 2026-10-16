# Roadmap

One line per open item, with a pointer to where the detail lives. No prose is copied
here: this is an index of things not done, not a design document.

`bottega:` paths are in the [bottega](https://github.com/jesssss812/bottega) repo, whose
`FUTURE.md` is the long-form design log this fork deliberately does not carry.

## Inherited from bottega's FUTURE.md

High-priority items, unchanged in status by the fork. See `bottega:FUTURE.md` for the
argument behind each.

- Frozen-triple numeric watch — the 3-member roster's fragile numeric lean vs. the full
  roster was accepted as a ship-and-watch bet. `bottega:FUTURE.md`
- Price the high→xhigh reasoning-effort premium — currently unmeasured on the one slot
  still paying it. `bottega:FUTURE.md`
- Gemini on the primary OpenRouter key — the `gemini-3.1-pro` blocklist entry is a
  workaround for Metaculus-side BYOK routing, now a documented setting rather than a code
  TODO. `bottega:FUTURE.md`, and the default of `OPENROUTER_PRIMARY_KEY_BLOCKED_MODELS`.
- Gap-fill v1 vs v2 comparison on ~20 prod questions — the gate v2 was meant to clear.
  `bottega:FUTURE.md`
- Summarizer effort and prompt were changed together and neither was measured. Measure
  them together or not at all. `bottega:FUTURE.md`
- Promote the core pipeline to strict type checking. `bottega:FUTURE.md`

## Deferred by the clean fork

Found during the restructure, deliberately not fixed under the plan's "no rewrites the
plan does not name" rule.

- **Ops scripts point at the wrong GitHub repo.** `sibyl/ops/{download_research,
  download_run_logs,download_raw_research,backfill_research_from_logs,sync_all}.py` all
  default `--repo` to `No-Stream/metaculus-bot`, the upstream nostreambot repo, rather
  than the operator's own. Every artifact download therefore targets a repo whose Actions
  artifacts the operator does not own. Inherited from bottega; likely a latent bug there
  too.
- **Cross-file test isolation leak, pre-existing.** Running
  `tests/unit/test_provider_flag_and_logging.py` before
  `tests/unit/test_research_orchestrator.py` leaves a process-global counter set, so a
  freshly constructed bot reports `alertable_count == 4`. Reproduces on bottega. Does not
  fire in full-suite collection order, so the suite is green. Probably the provider-health
  or prediction-market source-loss counter; `test_cli.py`'s reset fixture covers the CLI
  tests only.
- **Two lines of `value_extraction.py`'s JSON-repair error branch lost coverage** in the
  full-suite run, though the same test files give identical coverage in isolation. An
  ordering artifact rather than a lost test; see
  [decisions/0001-parity-contract.md](decisions/0001-parity-contract.md).
- **`forecasting/bot.py` imports `eval/performance/parsing.py`** for comment annotation —
  a dependency from the forecasting path onto the eval layer. Noted in the plan as
  known-and-left-alone.
- **The eval `compare` command reads only backtest data.** Ablation and performance runs
  write different shapes and cannot be compared with it yet.
- **docs/architecture-notes.md was never written.** [vision.md](vision.md) makes it the
  comprehension checkpoint gating repo surgery; the surgery happened first.
- **A golden scenario relies on the network guard.** `pipeline_conditional_stacking`
  does not stub `run_targeted_search`; its record (`search_failures=5`) holds only
  because the guard makes the real call fail. Stub it so parity never depends on egress.

## Free roster (repaired 2026-09-02, verification pending)

The first end-to-end run found two faults, both now fixed in `sibyl/config/models.py`:

- `qwen/qwen3-next-80b-a3b-instruct:free` had been **withdrawn** by OpenRouter and
  answered 404 on every question. Replaced.
- `gemma-4-31b-it:free` was 429 rate-limited every time because it served as a forecaster
  *and* all five utility slots. The forecasters are now three distinct vendors
  (nemotron / minimax-m3 / glm-5.2), none of them the utility model, so the load spreads
  across four free pools instead of one.

Two tests enforce the shape that broke: no forecaster may share the utility model, and
forecaster vendors must be distinct. A third, `tests/live/test_free_tier.py`, checks the
roster against OpenRouter's live catalogue — the only check that catches a withdrawal,
since a withdrawn slug still ends in `:free`.

Verified by a second run: survivors went 1/3 -> 2/3 and gemma was never rate-limited.

**Still open, and not a roster bug:** `z-ai/glm-5.2:free` is 429 rate-limited on its
UPSTREAM shared pool, which every `:free` slug shares with the whole of OpenRouter's free
tier. That is a function of the hour, not of our configuration, and swapping models to
chase it is a treadmill. Two live questions follow from it:

- ~~Should the free tier treat a dropped forecaster as alertable?~~ **Decided 2026-09-02:
  no.** `alertable_total` excludes `forecasters_dropped` in the free tier only. The caveat
  — a green free run can have lost forecasters, so read the log rather than the exit code
  — is in [decisions/0003-model-tiers.md](decisions/0003-model-tiers.md), AGENTS.md and
  the README.
- `/api/comments/` returns 403 for the current `METACULUS_TOKEN` while `/api/posts/<id>/`
  succeeds, so the token is valid and that endpoint is specifically restricted. Nothing in
  the bot's publishing path needs it, but `sibyl/eval/performance/collector.py` fetches
  comments to score past forecasts, so performance analysis will hit the same wall until
  the token's scope is widened.

## Dependency advisories (inherited)

`osv-scanner.toml` suppresses five cryptography advisories, all on 46.0.6, all with fixes
that are unreachable while `asknews 0.13.54` caps `cryptography<46.0.7`. They go the
moment that cap lifts — re-run `make audit` at the next asknews or forecasting-tools bump
and prune whatever no longer fires. The backlog is inherited rather than introduced:
bottega's CI fails its audit job on the same lockfile.

The three aiohttp advisories found at the same time were NOT suppressed, because their
fix was reachable: bumped 3.14.1 -> 3.14.3, a lock diff touching that package alone.

## Constraints worth knowing before changing things

- `forecasting-tools` is pinned to `==0.2.92`, and the six
  `tests/integration/test_framework_*.py` files pin live behaviour of that exact version.
  A version bump means re-reading them, not re-recording them.
- Two log lines are parsed telemetry contracts and keep the pre-fork
  donated/personal vocabulary on purpose: the `PAID PERSONAL-KEY FALLBACK` line and the
  end-of-run summary's `personal_key_fallback=` / `donated_404=` / `donated_key=` fields.
  Renaming them orphans every archived telemetry record. See
  `sibyl/telemetry/markers.py`.
- `eval/ablation/forecaster_lineup.py` re-exports the config roster's free lineup. It
  shrank from four models to three at the fork, so ablation results either side of that
  are not directly comparable.

## Evaluation study registration pending

The historical backtest harness now records frozen manifests, per-question results,
research hashes, endpoint plans, and explicit drop accounting. Legacy score and
comment-derived artifacts remain descriptive where they lack those identities.
The live performance collector still receives only each question’s latest forecast
and latest comment without an event-level join; its new
`forecast_comment_link_status=unverified_legacy` field makes that limit explicit.
Persist event IDs, timestamped forecast histories, variant IDs, and research
identities before using live data for a forward rank-prediction study.

The cheap-agreement study still needs a prespecified spread threshold, loss margin,
primary contrast, tail measure, multiplicity rule, and power calculation for the
bucket endpoint. Cross-tier disagreement alone cannot prove correlated error.
No router is enabled. Pooled scoring remains pending a versioned definition.

## Repo standardization

- Add `basedpyright` as a pre-push hook. The codebase is type-clean and CI enforces it, so
  only the hook's run time stands in the way. `.pre-commit-config.yaml`
- Fold the five remaining "Agent-facing detail (relocated from `AGENTS.md`)" sections
  into the body of their docs, as was done for `docs/operations.md`'s key routing. Each
  overlaps the prose above it. `docs/architecture.md`, `docs/numeric_pipeline.md`,
  `docs/performance_analysis.md`, `docs/probabilistic_tools.md`, `docs/research.md`
