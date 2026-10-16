# Plan — cost-tiered routing, MC debiasing, extremization

> Written 2026-08-27. Implements C1–C4 in [`design-principles.md`](../strategy/design-principles.md).
> Those are **candidates, not findings** — this plan is therefore ordered to *test* them
> cheaply before building on them, not to ship them.

---

## The thesis

> Spend frontier tokens only where they change the answer. The differentiator is the
> routing policy — a human design decision — not the model.

This is a real position and it is defensible on this project's own terms: `vision.md` ranks
**learning and a defensible artifact above raw score**, and a measured routing policy is a
better interview artifact than a better-funded ensemble. It also happens to be the only
identity the $50 cap allows.

**But it is a bet with a specific way of being wrong, and that has to be tested first.**

### The bet that decides everything

Cheap models share architectures, training data and failure modes. So they can agree
*confidently and wrongly*, plausibly on exactly the questions a frontier model would get
right. If that is the case, the router sends the easy questions to the expensive tier and
keeps the hard ones cheap — backwards, while looking like a saving.

**Everything below is sequenced so this is answered before anything is built on it.**

---

## What already exists (so the build is smaller than it sounds)

| Piece | Status |
|---|---|
| Spread metric per question type | ✅ `spread_metrics.compute_spread` |
| Routing on spread, with tuned thresholds | ✅ `stacking_route.route_after_forecasts` (0.15 / 0.20 / 0.15) |
| A free forecaster lineup | ✅ `ablation/forecaster_lineup.FREE_FORECASTER_MODELS` |
| Post-aggregation rail to hang extremization on | ✅ `post_processing.py`, beside Platt and the A3 guard |
| Paired-comparison statistics | ✅ `ablation/scoring.py` (bootstrap, sign test, Wilcoxon) |
| Zero-cost pipeline exercise | ✅ `make backtest_fixture` |
| Per-question cost telemetry | ✅ `credit_telemetry.py` |

What is genuinely missing is a cheap forecaster tier wired into the live path, an escalation
policy, an MC shuffle wrapper, and a fitted extremization parameter.

---

## Phase 0 — the validity test (do this before anything else)

**Question:** does cheap-tier agreement predict frontier-tier agreement?

**Method.** One paired run over ~40 resolved backtest questions. Every question gets *both*
a cheap-tier forecast (3 free/cheap models) and a frontier-tier forecast (the current
3-model roster). Nothing is routed — both arms run on everything, which is what makes the
comparison possible.

Then partition by cheap-tier spread and report, for each bucket:

- how often the frontier tier materially disagrees with the cheap tier
- the paired score difference (cheap − frontier), with a bootstrap CI, per bucket
- the **loss distribution**, not just the mean — this is a tail question by nature

**Draft decision rule, not yet registered:**

| Result | Action |
|---|---|
| Frontier disagrees mostly on **high**-spread questions | C1 is sound. Proceed to Phase 1. |
| Frontier disagrees roughly evenly across buckets | Routing saves money and loses accuracy in proportion. Report it and stop. |
| Frontier disagrees mostly on **low**-spread questions | Potentially harmful route. Compare outcome-based loss before concluding correlated error; kill only under a registered loss rule. |

**Cost:** cheap arm ≈ $0; frontier arm ≈ $0.40/question ⇒ **≈ $16–20 for n=40.**

**This single run pays for itself twice**: it decides C1, *and* it is a clean paired dataset
for the eval harness — which currently has no real data at all.

⚠️ **Power check, honestly.** At n=40 the MDE on a pooled paired difference is roughly
±0.020 (`eval-strategy.md` Rule 2 scaling). That is too coarse to certify a *small*
accuracy loss from routing. It does not establish power for bucket concentration, tail loss, or a proposed routing policy. Do not let this run
be reported as "routing is free."

---

## Phase 1 — cost-tiered escalation (only if Phase 0 passes)

Replace the disabled high-spread action with tier escalation:

```
cheap tier (3 free/cheap models)
    └─ compute_spread()
         ├─ below threshold  → publish the cheap median          (majority of questions)
         └─ above threshold  → re-forecast with the frontier tier, publish that
```

**Build notes**

- Reuse `route_after_forecasts`; add an escalation branch beside the existing stacker branch
  rather than replacing it, so the stacker stays available as a comparison arm.
- Thresholds are **not** inherited. The existing 0.15/0.20/0.15 were tuned for *frontier*
  disagreement; cheap-tier spread has a different distribution and needs its own values,
  taken from the Phase 0 data rather than guessed.
- Emit an `ESCALATION` telemetry marker per question (tier used, spread, threshold) and
  register it in `sibyl/telemetry/markers.py`. **The escalation rate is the headline
  number of this whole design** — without it there is no way to state what the policy bought.
- Ship it **off by default** behind `TIERED_ROUTING_ENABLED`, so the frontier-only arm stays
  available as the control.

**Expected economics** (measured costs: cheap ≈ $0, frontier ≈ $0.40/question):

| Escalation rate | Cost/question | vs. frontier-only |
|---|---|---|
| 100% (control) | $0.40 | — |
| 44% (amigon21's observed rate) | ~$0.18 | −55% |
| 25% | ~$0.10 | −75% |

At a 44% escalation rate, $50 goes from ~60 questions to ~135. **That is the entire argument
for C1** — and note it is a claim about *questions bought*, not about accuracy, which Phase 0
is what bounds.

---

## Phase 2 — extremization (independent of C1; do it in parallel)

Smallest build here and the most-cited technique on the bot-plans page.

- Add `apply_extremization(prediction, question, params)` to `post_processing.py`, beside
  `apply_platt_calibration` and `apply_resolution_state_guard` — that file was reorganised
  for exactly this shape, so MC becomes a branch rather than a new module.
- Order: `aggregate → extremize → Platt → cap → A3 guard`. Extremizing after Platt undoes the
  fit; after the cap it pushes back out of bounds.
- **Fit the factor; do not adopt 1.3–1.5.** `calibration-primer.md` §8 shows the direction is
  not knowable a priori — individual overconfidence pulls `a < 1`, ensemble underconfidence
  pulls `a > 1`. Which wins is a finding, and reporting it *is* the deliverable:
  *"after ensembling, Sibyl required a = 0.85."*
- Until there are ~100 resolved questions to fit on, ship it **disabled**, exactly as Platt
  is. A default-valued extremizer is the failure mode `vision.md` calls non-goal #1.
- **If C1 ships, extremization needs a per-tier parameter or must be off on the cheap path.**
  Extremization amplifies whatever calibration the input has; applying a frontier-fitted
  factor to a cheap-tier median produces a confidently miscalibrated number.

Mechanics are testable at zero cost on `make backtest_fixture` today. The *parameter* is not.

---

## Phase 3 — MC shuffling (independent; smallest scope)

- Wrap the MC runner: K shuffled option orders, average per option, renormalise.
- Add the objectivity check both sources warn about — a mechanically-determined question must
  not be dragged toward a historical base rate.
- Per-bin recalibration waits for held-out resolved MC questions; the shuffle does not.
- **This costs K× on MC questions.** It is an accuracy play, not a budget play. Justified by
  the bot-vs-pro gap being widest on MC, and affordable only because MC is a minority.

**Then test C4:** on the cheap tier, does shuffle-induced disagreement predict frontier
disagreement as well as cross-model spread does? If it does, MC escalation needs only one
cheap model instead of three — which is a further saving and a genuinely novel result.

---

## Sequence and cost

| Phase | Cost | Gate |
|---|---|---|
| 0 — validity test | ~$16–20 | none; do it first |
| 2 — extremization mechanics | $0 (fixture) | none; parallel to Phase 0 |
| 3 — MC shuffle mechanics | $0 (fixture) | none; parallel |
| 1 — tiered routing | ~$0 to build | **Phase 0 verdict** |
| 2b — fit extremization | $0 (offline) | ~100 resolved questions |
| 3b — MC per-bin recalibration | $0 (offline) | held-out resolved MC |

Roughly **$20 of the $50** buys the decision *and* the eval harness's first real dataset.
Phases 2 and 3 mechanics cost nothing and can proceed immediately on the fixture.

---

## How this is reported

Per `eval-strategy.md`, and this is the part that makes it an artifact rather than a config
change:

1. Hypothesis and expected effect size, stated **before** the run.
2. Power check — the n=40 MDE, and what it can and cannot resolve.
3. Paired result: mean difference, 95% CI, p, n.
4. Drop count, both arms.
5. Cohorts: question type, and cheap-tier spread bucket.
6. **Loss distribution** — routing is a tail question; a mean will hide it.
7. Caveats: leakage exposure, and that escalation rate is measured on one question mix.

A null or negative result is publishable and useful here. *"Cheap-tier agreement does not
predict frontier agreement, so cost-tiered routing does not work — here is the measurement"*
is a genuine contribution to a question the field is currently answering by assertion.

## Registration fields still open (2026-10-01)

Phase 0 remains a draft. Before any paid paired run, record the eligible population,
question-family grouping, exact cheap and frontier model configurations, per-type
spread formula, fixed bucket boundaries, and a minimum material cross-tier
prediction difference. Choose one primary per-type score-loss endpoint and loss
margin against always-frontier, a tail-loss statistic, a multiplicity policy, and
how missing or substituted models make a pair ineligible. Compute power for that
bucket or routing contrast from an independent variance source. The earlier pooled
n=40 arithmetic does not establish power for it.

The decision record must distinguish **proceed**, **kill**, and **inconclusive**
using numeric thresholds chosen before outcomes are read. Every eligible question
must receive both tiers. A disagreement shows only that the tiers differ; the
resolved outcome determines which tier was wrong. Any routing threshold selected
from this dataset remains exploratory until tested on held-out questions. The live
router stays disabled.

For the historical-to-forward study, freeze multiple named variants and their
research policies, selection rules, and forecast horizons before collection. Keep
event-level publication and comment identities, each variant's model survivors,
and outcome snapshots. Rank variants on a historical set and test that ranking on
a later, non-overlapping forward set, with a prespecified association statistic,
uncertainty method, sample-size rule, and stop rule. The current latest-forecast
collector lacks an event-level join and cannot certify this study.
