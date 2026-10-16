# 0005 — Eval consolidation

*2026-09-02. Status: in force.*

## Context

Five offline eval tools reached by five unrelated invocations, with nothing tying them
together — so there was no place to discover that the others existed. The paired
statistics they all needed lived inside `ablation/scoring.py` and were used only there.
And there was no way at all to answer the question the eval layer exists for: *did this
change help?*

## Decision

`python -m sibyl.eval <command>` over backtest, ablation, ensemble, performance,
fit-platt and compare. The dispatcher does **not** re-parse the sub-tool's flags: it takes
the first argument as the command and hands the rest over untouched. Argparse subparsers
would have meant restating every flag of every tool and keeping five copies in sync.
Every old invocation still works and is tested; this is an additional door.

The four paired primitives move to `sibyl/eval/stats.py` as a byte-for-byte move, with
`ablation/scoring.py` re-exporting them so ablation's recorded numbers cannot shift.
`minimum_detectable_effect` and `paired_summary` are added there rather than back-ported.

`eval compare A.json B.json` implements [eval-strategy.md](../strategy/eval-strategy.md):

- **Rule 1, paired.** Questions matched by id and metric; per-question differences
  bootstrapped. Never two standalone summaries side by side — question difficulty is a
  nuisance variable and pairing is what cancels it.
- **Rule 2, power.** Every section prints the MDE at its own n, so a null reads as "no
  effect" or "not enough questions to tell" rather than staying ambiguous.
- **Rule 3, no absolute numbers.** Only deltas, with the leakage caveats attached. A test
  asserts the input scores appear nowhere in the output.
- **Rule 6, degraded questions.** Any question unscored in either arm is dropped from both
  and named, and the summary is **refused** above a 5% drop rate.

## Consequences

- Refusing beats warning. A warned-about number still prints, and still gets quoted.
  `compare` also refuses a file holding more than one bot's results rather than silently
  using its first arm.
- Direction is printed per metric rather than assumed: Brier and CRPS are
  lower-is-better, the log scores higher-is-better, and a sign error there inverts the
  conclusion silently.
- `aggregate_scores` gains a nested `bot_minus_community_ci` with its own mean and n,
  rather than a CI beside the existing scalar. The scalar subtracts two means taken over
  *different* denominators — the bot mean spans every score, the community mean only
  those with a baseline — so on a partly-unpaired group it is not an estimate any
  interval brackets. The golden record shows the divergence: 0.137 scalar against a 0.030
  paired mean.
- `tests/e2e/test_fixture_aa.py` is the A/A null: two identical arms must compare to
  exactly zero. It validates the harness rather than the bot, and it is the check an A/B
  test cannot make — there, a real difference and a harness bug look identical.
- The plan cites "rules 1, 2, 3 and 5"; the drop-rate rule is Rule 6 in the document as
  written. Behaviour follows the document.
