# Sibyl — Improvement Plan (conceptual)

> Written 2026-08-24, after reading the nostreambot author docs (`architecture`,
> `research`, `agentic_gap_fill`, `numeric_pipeline`, `operations`), the five Sibyl
> planning docs, `docs/study-notes.md`, and three of the owner's prior-project
> learning docs.
>
> **No code changes proposed here.** This is the conceptual layer: what the fork
> actually is, where the current plan's premises are wrong, and three paths forward.

---

## Part 1 — What the author intended

Read across the five author docs, nostreambot has one governing idea and three
consequences.

**The idea: reliability is the product, cleverness is optional.** Every subsystem is
built so that a failure is *bounded, attributed, and loud*. Research providers fan
out in parallel and each fails independently. Both gap-fill passes soft-fail without
zeroing each other. Every forecaster runs under its own soft deadline inside a
shared per-question wall-clock. A min-forecasters guard skips a question rather than
publishing a thin ensemble. Degradation counters sum into a **non-zero process exit**.
This is not incidental polish — it is the architecture.

**Consequence 1 — the sophisticated parts are switched off.** `CONDITIONAL_STACKING`
is the configured strategy but all four workflows set the three stacking flags to
`false`, so prod runs plain MEDIAN. `PLATT_CALIBRATION_ENABLED` is unset everywhere,
so the calibration hook is a passthrough. The author built both, wired both, measured
enough to distrust both in prod, and left them exercised only in backtest and
ablation. The pattern repeats in `timeseries_anchor`: a CV-gated model picker was
built, measured at 43% win rate out-of-sample, and the naive empirical band shipped
instead. **The author's revealed method is: build it, measure it, ship the simpler
one.** That is the same method your `eval-strategy.md` argues for.

**Consequence 2 — research breadth, not research brand.** Eight providers, each
independently env-gated: AskNews (primary, LLM-summarized into an analyst briefing),
OpenAI native search, Gemini grounded search (a genuinely distinct index, first-party
not via OpenRouter), financial data, prediction-market snapshot, resolution-source
fetcher, time-series anchor, plus two concurrent gap-fill passes. This matches
`design-principles.md` Moderate #6 and Anti-pattern #1 exactly. Note the two design
details that carry the most craft: the **diagnostics seam** (provider health is
computed but deliberately withheld from forecaster prompts, then re-attached to the
published comment), and the **deterministic-extraction backstop** in financial data
(the resolving series is pulled from URLs in the resolution criteria even if the LLM
classifier misreads the question).

**Consequence 3 — leakage is handled per-provider, not globally.** Three postures
coexist: hard-off under benchmarking (prediction markets, resolution-source, agentic
gap-fill), point-in-time pinning to `open_time` (time-series anchor via ALFRED
vintages, financial data), and unguarded (the news providers). The author is explicit
that the hard-off providers' forecasting value **cannot be measured by the backtest
gate at all** and were validated by live runs instead. That is a real constraint on
any paired experiment you run offline.

**On cost, the author is in your position and solves it with prose.** Credit alerts
are suppressed until 2026-09-10 while the operator self-funds; `operations.md` has a
long, careful cost-discipline section. See Finding 3 for why that is the weak point.

---

## Part 2 — Three findings that change the plan

### Finding 1 — The differentiation claim is factually wrong. This is the big one.

`vision.md` says nostreambot's analysis "stops at means and standard deviations, with
**no confidence intervals, no paired comparison, and no significance testing
anywhere**," and scopes your contribution as a ~200-line inference layer.
`nostreambot-reading-guide.md` repeats it as a table of ❌s.

That is not what is in the repo. `sibyl/eval/ablation/` — **which the reading guide
put on the skip list** — already contains:

| Claimed missing | Actually present |
|---|---|
| Paired bootstrap | `bootstrap_mean_ci`, `bootstrap_median_ci`, `aggregate_paired` (`ablation/scoring.py`) |
| Confidence intervals | percentile bootstrap CIs, rendered per comparison |
| Significance testing | `sign_test` (binomtest), `wilcoxon_signed_rank` |
| Dual Brier + log scoring | `scoring_common.py`: `brier_score`, `binary_log_score`, `mc_log_score`, `numeric_log_score` — plus `numeric_crps` |
| Cohort slicing | `aggregate_paired` groups by (metric, question_type, comparison) |
| — | per-question paired diagnostic table, per-arm fallback-rate confounder tracking, score-saturation flags, and an explicit written caveat that percentile CIs under-cover at n<30 |

The report generator even writes the methodology caveats your Rule 7 asks for. The
`bootstrap_mean_ci` / `sign_test` / `wilcoxon_signed_rank` functions are **generic over
`list[float]`** — nothing about them is ablation-specific.

What *is* genuinely absent, verified by repo-wide search:

- **MDE / power analysis — zero occurrences anywhere.** Rule 2 has no implementation.
- **Horizon cohort slicing** — `horizon` appears only in prompts and `ts_estimators`,
  never in any scoring path. Rule 4's first and most load-bearing bullet is unbuilt.
- **Generalization** — `aggregate_paired` consumes a `PairedScore` dataclass welded to
  hardcoded arm labels (`COMPARISONS_3ARM/5ARM/6ARM`: stack / stack_aug / median /
  mean) for one specific experiment. The statistics are general; the harness around
  them is not.
- **The backtest path never got the treatment.** `backtest/analysis.py::aggregate_scores`
  really is means-only (`bot_mean`, `bot_std`, `bot_minus_community`) — the reading
  guide was right about that one file. The good statistics live in a sibling package
  that was never wired to it.

**What this means.** The contribution is not "build the inference layer." It is
**"generalize an inference layer that exists, connect it to the backtest path it was
never wired to, and add the two pieces genuinely missing (MDE, horizon slicing)."**
That is smaller engineering and, framed honestly, a *better* interview story — reading
a large unfamiliar codebase well enough to find that the capability already existed in
a package the reading guide told you to skip is a stronger demonstration than writing
200 lines of numpy. Framed dishonestly it is exactly `vision.md`'s non-goal #1.

Both docs need correcting before any agent reads them as ground truth.

### Finding 2 — The two headline experiments are underpowered by your own rule

`eval-strategy.md` Rule 2 says only test changes expected to move pooled score by
>0.015, with ±0.016 half-width at n=60 (one MiniBench round).

- **Platt on/off.** Measured effect in the literature you cite: −0.016 binary Brier.
  That sits *exactly at* the n=60 detection threshold — a coin flip on whether you see
  it. Worse, `design-principles.md` states Platt needs ~100+ resolved questions to
  *fit* two stable parameters. You will not have 100 resolved questions in the Aug–Sep
  window. **The flagship Sprint-2 experiment is not runnable as specified**, and Rule 2
  says declining it is itself a reportable finding.
- **Cheap vs frontier.** Large enough to clear MDE (model choice was "the single
  largest differentiator"). But Rule 3's third bullet — leakage may *interact* with the
  treatment — is not a footnote here, it is the central threat: a frontier model has
  memorized more of the outcome space than a cheap one, so on a leaky backtest the
  frontier arm is advantaged by leakage *and* by capability, and the paired difference
  does not separate them. This experiment is only clean **live, forward-looking**.

That reordering has a schedule consequence, below.

### Finding 3 — The cost guard is convention; the repo contains its own counterexample

Study note #1 is correct and the evidence is stronger than the note claims.
`tests/conftest.py` installs an autouse fixture that monkeypatches `socket.connect`
to raise on any non-loopback connect, so `make test` **cannot** spend money even if a
new test tries. The author calls this "safe by construction, not by convention."

There is no equivalent on the live path. `make run`, `make backtest_*`, `make ablation_*`
are protected by prose in `AGENTS.md` and `operations.md` and nothing else. At a $50
cap with agents doing the typing, that asymmetry is the single largest budget risk in
the project — and the fix has prior art in two places: this repo's own conftest, and
Lumi's F7 fixture mode.

Partial prior art also exists for the free lane: `ablation/forecaster_lineup.py` already
ships `FREE_FORECASTER_MODELS` (`:free` OpenRouter slugs on the fallback key) precisely
so a 100-question backtest doesn't burn budget on the forecaster stage.

Study note #2 is also confirmed: `AGENTS.md` is 74KB (~20k tokens per agent invocation)
and `bottega:FUTURE.md` is 127KB.

---

## Part 3 — What the prior-project learnings say about this plan

Mapping the three attached docs onto the current state, only the non-obvious ones:

**"Because human effort cost is low, one tends to think less and ask it to implement —
the result is mumble jumble you need to clean up."** (Atlas) This is the strongest
argument in any of your docs for the comprehension checkpoint, and it is the specific
risk of forking 182 files you have not read. `vision.md` already makes
`architecture-notes.md` non-negotiable. Finding 1 is what happens when that step is
skipped: a plan built on a skim.

**"Always have a test/validation in mind before accepting an output."** (Atlas)
Nothing in `eval-strategy.md` applies this *to the eval harness itself*. The concrete
version: run the paired harness on **two identical arms** and confirm it reports no
significant difference. An A/A null test is ~10 lines, catches alignment bugs, seed
bugs, and accidental unpairing, and is a genuinely good thing to be able to say you
did. Add it as Rule 8.

**"Set up eval of the model from the get-go, and examine manually to confirm."** (Lumi)
`eval-strategy.md` is entirely automated statistics. There is no manual-inspection
step anywhere — no "read 10 forecasts and their research bundles end to end before
trusting any number." Given that the repo publishes full reasoning in every Metaculus
comment, this is nearly free and it is how you would actually catch a units bug or a
resolved-question-read-as-open error, neither of which any Brier delta will surface.

**"A degraded mode you can't see is a bug factory."** (Lumi) This is the one place the
fork is *ahead* of your docs. Rule 6 is already implemented natively — degradation
counters, drop telemetry, min-forecasters guard, non-zero exit. Adopt, don't rebuild,
and say so in the docs so no agent re-implements it.

**"Ensure the data layer works first."** (Apify) Reading-guide question 8 — *confirm the
shape of the backtest JSON, that's your input contract* — is still unanswered in the
study notes. Every statistic you plan depends on that file containing per-question
ids, resolved outcomes, and a horizon field. **Horizon is the one you should expect to
be missing**, since `horizon` appears nowhere in any scoring path. Check before
planning around Rule 4.

**"Don't be afraid to burn tokens — use the best model."** (Apify) vs. the $50 cap.
These resolve cleanly and the docs never state the resolution: **the $50 cap is on
forecast-time API spend, not on coding-agent spend.** Burn freely on agents reading and
writing code; be frugal on live forecast calls. Worth writing down, because an agent
reading `AGENTS.md`'s cost rules may otherwise economize on the wrong axis.

**The retrospective's central finding applies directly.** Items with a time, a named
output, and a calendar slot ran 15/15. Items without — reading, the career synthesis,
the comfort-zone act — slipped three weeks running, and the diagnosis was explicit:
*"the items that slipped are not low-value items, they're loose-shape items."*

`architecture-notes.md` is currently a loose-shape item. So is "build the eval harness."
Both are exactly the shape that slipped. The retrospective's own prescription —
*"treat it as a scheduled output, not a recurring habit: a 60-min block, one specific
date, deliverable = a one-page doc"* — is the fix, and it is already written down in
your own hand. The study notes standing at two entries with an empty open-questions
section, nine days into a window, is the same pattern repeating.

---

## Part 4 — The schedule reality

Stated plan: Sprint 1 = Aug 15–22, "get live fast," Milestone 1 = submitting by
Aug 16–17. MiniBench window = Aug 15–28.

Today is **Aug 24**. Sprint 1's window has closed and there is no live record.
**Four days of MiniBench remain.**

One asymmetry should drive everything from here:

> **The live track record is the only input that cannot be reconstructed later.**

Statistics, backtests, Platt fitting, harness generalization, the whole inference
layer — all of it runs in September against archived data, and none of it decays. A
question that closes on Aug 28 unforecast is gone permanently. Finding 2 sharpens this
further: cheap-vs-frontier is only clean *live*, so the live arm is not merely the
holdout, it is the flagship experiment's only uncontaminated substrate.

**Sequence by irreversibility, not by interest.**

---

## Part 5 — Three paths

### Path A — Ship now, measure in September *(recommended)*

Spend the remaining four days on exactly two things: (1) get the fork live and
submitting to MiniBench, and (2) the structural cost guard from Finding 3, because it
is what makes the next four days safe to hand to an agent. Nothing else. Accept a
partial MiniBench round.

Then Sept/Oct, when credits refresh and questions have resolved: correct the two docs
per Finding 1, generalize the ablation harness, wire it to the backtest path, add MDE
and horizon slicing, add the A/A null test and the manual-inspection pass.

- **Serves:** goal 3 immediately, goals 1–2 fully but later. Preserves the only
  perishable asset.
- **Costs:** `architecture-notes.md` gets written *after* first contact rather than
  before, which softens the comprehension checkpoint. Mitigate by writing it in
  September from the live record, which is arguably a better document anyway.
- **Risk:** four days is tight for credentials + config + a first submission. If it
  slips past Aug 28 you have spent the window and gotten nothing.

### Path B — Comprehension first, eval as the product, skip MiniBench

Declare the Aug window lost. Spend the next two weeks doing what `vision.md` says is
non-negotiable: write `architecture-notes.md` properly, correct Findings 1–3 in the
docs, and build the generalized eval layer against nostreambot's *existing archived
backtest data* — which requires no live run and no spend at all. Go live for the
autumn tournament with credits, not cash.

- **Serves:** goals 1–2 maximally and honestly. Zero budget risk. Uses the $50 in
  September on the frontier arm rather than on a rushed August baseline.
- **Costs:** goal 3 deferred entirely; no live record until autumn, which pushes the
  backtest-predicts-live question out past the timebox's hard stop.
- **Risk:** the classic failure mode of this project type — an immaculate measurement
  apparatus with nothing measured. `vision.md` non-goal #1 in a different costume.

### Path C — Re-scope the artifact around what's actually differentiated

Take Finding 1 seriously and change what the deliverable *is*. The inference layer is
no longer scarce. Three things still are, in descending order of value:

1. **The backtest-predicts-live question.** Still genuinely unanswered for a hobbyist
   budget — and now *more* tractable than `vision.md` assumed, because both halves
   already exist in the repo: the offline paired harness (`ablation/`) and the live
   calibration analysis (`performance_analysis/`, with Jeffreys-prior CIs and
   era-bucketing already implemented). The missing piece is the **linkage** — run ids,
   variants, and question sets kept comparable across both regimes — not either side.
   This is the strongest interview story available and nobody has done it.
2. **The power/MDE discipline layer plus the A/A null test.** Genuinely absent, small,
   and it is precisely what separates a real evaluation from a plausible one.
   *"Here are the four experiments I declined to run as underpowered, and the
   arithmetic"* is a sharper answer than any Brier number.
3. **The structural cost guard.** Least glamorous, most transferable, and it is the
   one thing here that generalizes to production LLM work of any kind.

Concretely this means: go live in the four remaining days *for the sake of the linkage*
(item 1 needs a live arm and cannot be backfilled), then spend September on items 1–3.

- **Serves:** all four goals, and fixes the credibility problem in `vision.md` rather
  than shipping on top of it.
- **Costs:** requires rewriting two docs before an agent uses them as ground truth.
- **Risk:** re-scoping mid-project is itself a scope risk. Guard: items 1–3 are
  *narrower* than the original scope, not wider.

### Recommendation

**Path C, executed with Path A's ordering.** They are compatible and differ only in
what you tell yourself September is for.

The next four days are Path A verbatim: get live, add the spend guard, touch nothing
else. The re-scoping in Path C is doc work, not code work, and it can be done in the
Sept window at no cost to the live record. Path B is the one to avoid — not because
comprehension doesn't matter, but because it trades the only perishable asset in the
project for work that keeps.

---

## Concrete next actions (in order)

1. **Correct `vision.md` and `nostreambot-reading-guide.md`** per Finding 1, before any
   coding agent treats them as ground truth. ~30 min. Do this first — it is cheap and
   everything downstream depends on it.
2. **Answer reading-guide question 8 for real**: confirm the backtest JSON's shape and
   whether it carries horizon. This is the "data layer first" lesson and it gates Rule 4.
3. **Decide the four-day scope** and put the live-submission attempt on the calendar
   with a date, per the retrospective's own prescription.
4. **The spend guard** — the smallest structural version that makes an accidental live
   run impossible rather than merely discouraged.
5. **Defer everything else to September.** Write down what was deferred and why; per
   Rule 2, that list is itself part of the artifact.

---

# Appendix — `design-principles.md` vs. the repo today

Every item audited against the code on 2026-08-24. Verdicts are from grep/read, not
from the author's docs.

## Full audit

| # | Principle | Status |
|---|---|---|
| **S1** | Current frontier reasoning model | ✅ `gpt-5.6-sol`, `claude-opus-4.8`, `gemini-3.1-pro-preview` |
| **S2** | Invest in harness/scaffolding | ✅ extensively — this is the repo's whole character |
| **S3** | Agentic / iterative search | ✅ gap-fill v2 bounded tool loop |
| **S4** | Ensemble 3–7 across model families, MEDIAN default | ⚠️ 3 members (floor of the band) across 3 families; **Grok — the low-correlation member `design-principles.md` names — is no longer in the roster** |
| **S5** | Platt scaling post-hoc | ✅ built (`calibration/platt.py`) · ❌ **disabled in prod** |
| **M6** | 2+ research providers | ✅ eight |
| **M7** | Cap predictions at min/max | ✅ `BINARY_PROB_MIN/MAX = 0.02/0.98`, MC `0.01/0.99`; correctly re-clamped *after* Platt |
| **M8** | Explicitly compute base rates | ✅ two-phase outside/inside-view prompt structure **+** `probabilistic_tools/base_rate.py` (Beta-Binomial posteriors) |
| **M9** | **Look up similar previously-resolved questions** | ❌ **NOT BUILT** — see Gap 1 |
| **M10** | ~28 LLM calls/question | ✅ roughly (research fan-out + 2 gap-fill passes + 3 forecasters + support models) |
| **M11** | Keep the model current | ✅ pinned in `llm_configs.py`, rotates |
| **A1** | Don't chase a single best search provider | ✅ compliant — breadth by construction |
| **A2** | Don't scaffold before reliability | ✅ compliant — reliability *is* the architecture |
| **A3** | **Guard against open-vs-resolved confusion** | ❌ **NOT BUILT** — and a prompt line leans the other way. See Gap 2 |
| **A4** | No multi-persona aggregation | ✅ compliant — `binary_prompt(question, research)` takes no model or persona argument; every ensemble member gets an identical prompt. Diversity is purely the model |
| **A5** | No "Bayesian" framing in prompts | ✅ compliant — **zero** occurrences in `prompts.py` |
| **A6** | Don't ship the default numeric pipeline | ✅ PCHIP done · ❌ **(mean, std) elicitation not done** — 13 fixed percentiles. See Gap 3 |

**Score: 14 of 17 fully satisfied.** The author independently converged on nearly the
entire evidence-graded playbook. That is itself worth noting in `architecture-notes.md`.

---

## Gap 1 — Reference-class retrieval *(the one worth building)*

**M9: "Look up similar previously-resolved questions." 34% of winners vs 0% of
non-winners, Fisher p=.04.**

Reference-class reasoning is heavily *prompted* — "list plausible reference classes and
evaluate suitability" appears in all three question-type prompts. But the model is
being asked to produce reference classes **from its own memory**. Nothing in the repo
ever *retrieves* an analogous resolved question and hands the forecaster its actual
outcome. There is no such provider in `research/`.

Note the evidence shape: that statistic isn't a correlation, it's a categorical split.
**Zero non-winners did this.** It's the sharpest single discriminator in the entire
survey, and it's the one Moderate item the fork skipped.

Why this is the right build here, specifically:

- **The repo already contains the template.** `research/market_retrieval/` is exactly
  this pipeline pointed at a different index: fetch a catalogue → assemble a candidate
  pool from multiple channels → **one LLM ranking call over the whole pool** → render
  ranked rows with a relation tier and a one-phrase reason. The author's own bake-off
  found that *selection, not generation, is the binding constraint* — a lesson that
  transfers directly and saves you from rediscovering it.
- **Metaculus resolved-question fetching is already wired.**
  `performance_analysis/collector.py::fetch_resolved_questions` does the paginated pull
  with retry handling.
- **Architecturally additive, zero blast radius.** A new env-gated provider that
  soft-fails and appends its own `##` section. It cannot destabilize anything (A2).
- **It is one of the few providers that is backtest-safe** — filter candidates to
  questions that resolved *before* the target question's `open_time` and the leakage
  problem disappears. Prediction markets and the resolution-source fetcher are both
  hard-off under `is_benchmarking=True` and their value is therefore unmeasurable
  offline. This one is measurable. For a project whose deliverable is the evaluation,
  **a new component you can actually run a clean paired comparison on is worth more
  than one you can't** — and it clears Rule 2's >0.015 bar as an architectural change.

That last point is the argument. It's the rare item that serves goal 3 *and* goal 1
without trading against either.

## Gap 2 — The open-vs-resolved guard *(cheapest, and a reliability item)*

**A3: "Add an explicit guard that checks question state before allowing extreme
confidence."**

No such guard exists in the live path. The only "already resolved" logic in the repo is
in `backtest/leakage.py`, which is the offline leakage detector, not a runtime check.

More than absent — `prompts.py:668` actively instructs the shortcut:

> *"Does the research already contain evidence that the resolution condition has been
> met (or is now impossible to meet)? If so, assign a near-extreme probability (≥95% or
> ≤5%) … and skip to the final answer. Do not perform full reference-class analysis for
> questions whose answers are already deterministic from current evidence."*

That is a deliberate and defensible efficiency choice by the author. But it is precisely
the pathway A3 warns about, and it has **no counterweight**: nothing verifies the
question is still open, nothing checks that the resolution window has actually closed,
and the model is explicitly told to skip the analysis that would catch the error.

Existing mitigations are indirect and partial:
- capping at `[0.02, 0.98]` bounds the damage but does not prevent it;
- the AskNews summarizer's `[PRE-WINDOW]` tagging — events predating question open are
  flagged as unable to satisfy the criteria — is aimed at exactly this failure and is
  the strongest thing in place, but it only covers facts arriving via AskNews;
- `prompts.py:501` — "if the resolution date or criteria differ, discount it."

**Why it's worth the hour.** Under log score a capped 0.98 on a question that resolves
NO costs ≈ 3.9 — roughly forty times a typical well-forecast question. Your own
`design-principles.md` states the case: *"Bugs cost more than cleverness gains"* —
nostream lost 80 points to a single units bug. This is the same class of defect, it is
one prompt-level check plus a question-status assertion, and it is measurable as a rare
tail event rather than a mean shift (which is exactly the loss-distribution analysis
Rule 4 asks for).

## Gap 3 — (mean, std) elicitation *(build as an experiment arm, not a fix)*

**A6's second half.** Forecasters declare 13 fixed percentiles
(`numeric/config.py::STANDARD_PERCENTILES`); the PCHIP half of A6 is done, the
elicitation half is not.

**Recommend against implementing this as a replacement.** The percentile path is ~13
files of hard-won engineering — sanitizing, tail widening, min/max step enforcement,
bound pinning, discrete-integer snapping, a unit-mismatch guard. Swapping the
elicitation format destabilizes all of it, for a change `design-principles.md` grades
only as an anti-pattern footnote rather than a Strong finding.

If it interests you, it's a clean **paired experiment arm** — same questions, two
elicitation formats, both through the same PCHIP machinery — which is exactly the
posture `design-principles.md` prescribes for Moderate/Suggestive items: *"treat every
Moderate/Suggestive item as a hypothesis to test with a paired comparison rather than a
feature to assume."*

## Config-only items (no build required)

- **Enable Platt (S5).** Already known. But see Finding 2 — it can't be *fitted* on the
  data you'll have, so this stays a September item at the earliest.
- **Add a 4th ensemble family (S4).** The roster is at the floor of the 3–7 band and has
  lost the low-correlation member `design-principles.md` names. This is one line in
  `llm_configs.py`, and `ensemble_analysis/correlation_analysis.py` already exists to
  measure whether the addition actually decorrelates. Cheap, measurable, and it raises
  per-question cost — so it interacts with the $50 cap and belongs in the September
  frontier-arm run rather than now.

## Recommended order

1. **Gap 2** — an hour, pure reliability, no cost change, no architectural risk.
2. **Gap 1** — the real build. Worth doing properly, and worth doing *because it can be
   measured*, not merely because winners do it.
3. **Gap 3** — only as an experiment arm, only if budget and interest survive.

Both 1 and 2 are September work under the recommended path. Neither should compete with
getting live before Aug 28.
