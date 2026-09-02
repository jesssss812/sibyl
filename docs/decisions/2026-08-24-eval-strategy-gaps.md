# Gap audit — `design-principles.md` and `eval-strategy.md` vs. the code

> 2026-08-24, after the M9 + A3 PR. Every verdict below was checked against the code by
> grep/read/execution, not inferred from the docs.
>
> Companion to `improvement-plan.md` (which audited `design-principles.md` pre-PR).
> This one covers what remains there, and audits `eval-strategy.md` — never done before.

---

## Part 1 — `design-principles.md`: what's left

14 of 17 items were already satisfied by the fork; the PR added the two real gaps
(M9, A3). Remaining:

| Item | Status |
|---|---|
| **A6 second half** — elicit `(mean, std)` instead of percentiles | ❌ Open, **deliberately**. The percentile+PCHIP path is ~13 files of sanitizing / tail widening / step enforcement / bound pinning / discrete snapping. Belongs in a paired experiment arm, not a swap. |
| **A3 for multiple-choice** | ❌ **New gap, created by the PR.** The guard is binary-only. MC has the same failure mode through a single dominant option, but its clamp interacts with ballot renormalization (`mc_processing.clamp_and_renormalize_probs`), so it needs its own design rather than a reused threshold. Numeric is unaffected — its analogue is interval width, already tracked by `width_monitor`. |
| **S5 Platt** | ⚠️ Built, disabled, and **not yet fittable**: needs ~100+ resolved questions for two stable parameters. |
| **S4 ensemble size** | ⚠️ 3 members — the floor of the recommended 3–7 band. |

---

## Part 2 — `eval-strategy.md`: rule-by-rule

| Rule | Verdict |
|---|---|
| **1. Every comparison is paired** | ⚠️ **Split.** Fully implemented in `ablation/` (`aggregate_paired`, `bootstrap_mean_ci`, `sign_test`, `wilcoxon_signed_rank`). Absent in `backtest/`, whose `aggregate_scores` returns `bot_mean` / `bot_std` / `bot_minus_community` — a difference of means with no interval. |
| **2. Power-check before you run** | ❌ **Entirely unimplemented.** Zero occurrences of MDE / power / minimum-detectable-effect anywhere in the repo. |
| **3. Never report an absolute backtest score as real** | ❌ **Actively violated** by `generate_backtest_report`, which emits `bot_mean` / `bot_std` / `community_mean` as its headline table. |
| **4. Slice by cohort** | ⚠️ **One of four.** `question_type` ✅ (via `aggregate_paired`'s grouping). `horizon` ❌ — the word appears only in `prompts.py` and `ts_estimators.py`, never in a scoring path. `category` ❌. **Loss distribution / tail concentration ❌.** |
| **5. Uncontaminated holdout** | ⚠️ **Half.** The live tournament record is a genuine holdout and is scored by someone else ✅. But no random backtest slice is reserved from iteration, and nothing marks or enforces one. |
| **6. Degraded runs fail loud, excluded from both arms** | ⚠️ **Mostly.** Both-arms-present is implemented (`ablation/scoring.py:282, 489`). Degradation counters + non-zero process exit are native ✅. **Missing: the drop-rate threshold** (spec default 5%) that must make a run *refuse to report a summary*. No such gate exists. |
| **Non-goals** (frozen corpus, question generation, pastcasting corpus, money) | ✅ **Respected.** Nothing forbidden was built. |
| **Data sources** | ✅ MiniBench is a wired CLI mode; resolved-question backtest exists; live record exists. |

### The experiment log

Spec requires 17 fields per question per run. `QuestionScore` carries five
(`question_id`, `question_type`, `bot_score`, `community_score`, `metric_name`).

**Present elsewhere in the artifact:** outcomes ✅ — `save_backtest_data` serializes
`ground_truths`, and `GroundTruth` carries `resolution`, `resolution_string`,
`actual_resolution_time`, `question_text`, `community_prediction`.

**Missing:** `run_id`, `horizon_days`, `category`, **`forecast`**, `model`, `variant`,
**`git_sha`**, `prompt_version`, `n_llm_calls`, `cost_usd`, **`degraded`**,
`degraded_reason`. The spec marks `git_sha` and `degraded` **mandatory**.

**The raw forecast is the consequential one.** Scores are persisted; the probability
that produced them is not. Without the `(p_raw, outcome)` pair you cannot, from backtest
data:

- build a **reliability diagram** — which `design-principles.md` names as the way to set
  the capping bound empirically;
- **refit Platt** — `fit_platt_cli` works only because it reads `performance_data.json`
  from the *live* path, which does recover per-model forecasts by parsing published
  Metaculus comments;
- **rescore under a different rule** later, so adding a metric means re-running and
  re-paying rather than re-reading.

Note `actual_resolution_time` **is** persisted, so `horizon_days` is derivable in
principle — it is simply never computed or carried.

### The open research question

*"Does a cheap, leaky, hobbyist-grade backtest predict live performance?"* The doc's
design implication is: **preserve the ability to compare backtest-predicted rank against
live-realized rank** — keep run ids, variants, and question sets linkable across both
regimes.

❌ **No such linkage exists.** No shared `run_id` or `variant` spans `backtest/` and
`performance_analysis/`. The nearest thing is `width_monitor`'s era-bucketing by
merge-to-main timestamp, which identifies a *config era*, not an experiment.

Both halves of the comparison exist. The join key does not.

### Structural note

The spec targets a module `sibyl/eval/`. Evaluation code actually lives across
`sibyl/eval/ablation/`, `sibyl/eval/backtest/`, and
`sibyl/eval/performance/`. Minor in itself, but it means the spec's
implementation section points at nothing, and it is why the rigour split in Rule 1 went
unnoticed.

---

## Part 3 — the sharpest consequence

**The A3 guard shipped in this PR cannot currently be evaluated.**

It fires rarely (only on extreme forecasts early in a question's window) and moves each
one by at most **0.03** — a hard ceiling, because every binary prediction is already
clamped to [0.02, 0.98] upstream per forecaster, and a median cannot escape its inputs'
range. It therefore **cannot move a pooled mean** at n=60, where Rule 2's own table puts
the detection threshold at ±0.016.

The bet is a ratio, not a magnitude: in Brier terms a true positive gains ~0.058 and a
false positive costs ~0.0021, so the guard pays whenever more than ~3.5% of early extreme
forecasts are misreads. The quantity to instrument is therefore the **firing rate** (now
emitted as the `RESOLUTION_GUARD` marker), not the score delta.

Measuring it requires exactly the two things that are missing:

1. **Rule 4's loss distribution / tail analysis** — the guard's whole value is truncating
   rare large losses, which is a tail effect by construction. The doc already identifies
   this mechanism twice (Rule 4's *"losses are tail-concentrated, not uniform"*, and the
   note that capping is the strongest within-winners differentiator at r=+0.48) — and
   then never implements the analysis that would show it.
2. **Rule 2's MDE** — to state up front that the pooled-mean test is underpowered for
   this change, which is itself a reportable finding.

So the gap is not academic: the repo currently has no way to tell whether the guard
helped, hurt, or did nothing.

---

## Ranked recommendations

1. **Loss-distribution / tail reporting (Rule 4).** Highest value per line. Unblocks
   evaluating A3 and capping — the two changes whose entire effect is in the tail — and
   the per-question scores it needs are already on disk.
2. **MDE / power check (Rule 2).** ~10 lines. Converts "we didn't measure it" into "we
   declined this as underpowered, here is the arithmetic", which the doc says is a
   finding in its own right.
3. **Persist `forecast`, `git_sha`, `degraded` per question.** Small, and it is what
   makes every later analysis re-runnable off disk instead of re-paid.
4. **Point `backtest/` at `ablation/`'s statistics** rather than maintaining a
   means-only second path. Directly satisfies Rules 1 and 3, and removes the
   "alternative evaluation path" the spec forbids.
5. **Horizon cohort.** `actual_resolution_time` is already persisted; this is derivation,
   not new plumbing.
6. **A drop-rate refusal gate (Rule 6)** and **a reserved backtest holdout (Rule 5).**
7. **A join key across backtest and live** — the only thing standing between the current
   artifacts and the project's headline research question.

Items 1–3 are each small and independently useful. Item 4 is the one that removes a
structural inconsistency rather than adding a feature.
