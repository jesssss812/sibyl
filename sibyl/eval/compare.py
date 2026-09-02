"""``python -m sibyl.eval compare A.json B.json`` — the paired comparison, done properly.

Reads two ``backtest_data_*.json`` files written by ``eval/backtest/analysis.py``
:func:`save_backtest_data` and reports what changed, following ``docs/eval-strategy.md``:

* **Rule 1 — every comparison is paired.** Questions are matched by id and the
  per-question difference is bootstrapped. Never two standalone score summaries placed
  side by side: question difficulty is a nuisance variable, and pairing is what cancels
  it. A CI on one system's Brier says more about how hard the question set was than
  about the system.
* **Rule 2 — power check.** Every section reports the MDE at its own n, so a null result
  can be read as "no effect" or "not enough questions to tell" rather than left
  ambiguous.
* **Rule 3 — no absolute numbers.** Only deltas are printed, with the leakage caveat
  attached. Backtests are leaky by construction; leakage inflates both arms roughly
  equally and largely cancels in the difference, which is exactly why the difference is
  reportable and the level is not.
* **Rule 6 — degraded questions are dropped from BOTH arms and counted**, and the summary
  is refused above a 5% drop rate. A paired comparison is only valid if both arms
  answered the same questions.

(The plan that commissioned this cites "rules 1, 2, 3 and 5"; the drop-rate rule is
Rule 6 in the document as written. Behaviour follows the document.)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from sibyl.eval.stats import paired_summary

# Above this share of unusable questions the two arms are no longer answering the same
# question set, and a "paired" summary over what is left is a different experiment from
# the one that was run. Refusing is the point: the number would still print, and someone
# would still quote it.
MAX_DROP_RATE = 0.05

# Lower is better for Brier and CRPS; higher is better for the log scores. Printed rather
# than assumed, because a sign error here inverts the conclusion silently.
LOWER_IS_BETTER = {"brier", "crps"}

CAVEATS = """
CAVEATS (do not report these numbers without them):
  * These are PAIRED DIFFERENCES only. No absolute score is shown, deliberately: a
    backtest is leaky -- the models have seen the outcomes -- so absolute Brier from
    this harness is not a measurement of skill and must never be published or compared
    against anyone else's number.
  * The paired difference is reportable because leakage inflates both arms roughly
    equally and largely cancels. The residual risk is that leakage INTERACTS with the
    change under test, which this design cannot rule out.
  * A confidence interval that contains an effect the size of the MDE has not ruled that
    effect out. Read a null result against the MDE, not on its own.
"""


class CompareError(RuntimeError):
    """A refusal, not a crash: the inputs cannot support a paired comparison."""


def _load(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise CompareError(f"no such backtest data file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CompareError(f"{path} is not valid JSON: {exc}") from exc
    if "results" not in data:
        raise CompareError(f"{path} has no 'results' key; is it a backtest_data_*.json file?")
    return data


def _scores_by_key(data: dict[str, Any], path: Path) -> tuple[dict[tuple[int, str], float], dict[int, str]]:
    """Map (question id, metric) -> bot score, plus question id -> question type.

    A file holding more than one bot's results is refused rather than silently reduced to
    its first arm: pooling or picking would answer a question nobody asked.
    """
    results = data["results"]
    if len(results) != 1:
        names = ", ".join(str(r.get("bot_name", "?")) for r in results) or "none"
        raise CompareError(
            f"{path} holds {len(results)} bot results ({names}); compare needs exactly one per file. "
            "Re-run the backtest with a single bot, or split the file."
        )
    scores: dict[tuple[int, str], float] = {}
    types: dict[int, str] = {}
    for row in results[0].get("scores", []):
        qid, metric = int(row["question_id"]), str(row["metric_name"])
        value = row.get("bot_score")
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue  # unscored in this arm; the pairing step will drop it from both
        scores[(qid, metric)] = float(value)
        types[qid] = str(row.get("question_type", "unknown"))
    return scores, types


def pair_scores(
    a: dict[tuple[int, str], float], b: dict[tuple[int, str], float]
) -> tuple[dict[str, list[tuple[int, float]]], set[int]]:
    """Deltas (B - A) per metric, and the ids dropped because an arm could not score them."""
    all_qids = {qid for qid, _metric in a} | {qid for qid, _metric in b}
    paired_keys = a.keys() & b.keys()
    paired_qids = {qid for qid, _metric in paired_keys}
    deltas: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for qid, metric in sorted(paired_keys):
        deltas[metric].append((qid, b[(qid, metric)] - a[(qid, metric)]))
    return dict(deltas), all_qids - paired_qids


def _format_section(title: str, metric: str, rows: list[tuple[int, float]], seed: int) -> str:
    summary = paired_summary([d for _qid, d in rows], seed=seed)
    direction = "lower is better" if metric in LOWER_IS_BETTER else "higher is better"
    verdict = _verdict(metric, summary)
    return (
        f"  {title}\n"
        f"    n={summary['n']}  ({direction})\n"
        f"    mean Δ   {summary['mean_delta']:+.5f}   95% CI [{summary['mean_ci_low']:+.5f}, "
        f"{summary['mean_ci_high']:+.5f}]\n"
        f"    median Δ {summary['median_delta']:+.5f}   95% CI [{summary['median_ci_low']:+.5f}, "
        f"{summary['median_ci_high']:+.5f}]\n"
        f"    sign p={summary['sign_test_p']:.4f}   "
        f"wilcoxon p={'n/a' if summary['wilcoxon_p'] is None else f'{summary["wilcoxon_p"]:.4f}'}\n"
        f"    MDE at this n: {summary['mde']:.5f}\n"
        f"    -> {verdict}\n"
    )


def _verdict(metric: str, summary: dict[str, Any]) -> str:
    """One plain sentence, so the reader does not have to re-derive the sign convention."""
    lo, hi = summary["mean_ci_low"], summary["mean_ci_high"]
    if summary["n"] == 0 or math.isnan(lo) or math.isnan(hi):
        return "no paired questions"
    if lo <= 0.0 <= hi:
        mde = summary["mde"]
        if not math.isnan(mde) and abs(summary["mean_delta"]) < mde:
            return "no detectable difference, and n is too small to rule out an effect of MDE size"
        return "no detectable difference"
    b_better = (hi < 0.0) if metric in LOWER_IS_BETTER else (lo > 0.0)
    return f"B is {'better' if b_better else 'worse'} than A"


def render_report(path_a: Path, path_b: Path, by_question_type: bool = False, seed: int = 0) -> str:
    data_a, data_b = _load(path_a), _load(path_b)
    scores_a, types_a = _scores_by_key(data_a, path_a)
    scores_b, types_b = _scores_by_key(data_b, path_b)
    types = {**types_a, **types_b}

    deltas, dropped = pair_scores(scores_a, scores_b)
    total = len({qid for qid, _m in scores_a} | {qid for qid, _m in scores_b})
    drop_rate = (len(dropped) / total) if total else 0.0

    header = f"A: {path_a.name}\nB: {path_b.name}\nquestions: {total} seen, {len(dropped)} dropped ({drop_rate:.1%})\n"
    if dropped:
        header += "  dropped (unscored in at least one arm): " + ", ".join(str(q) for q in sorted(dropped)) + "\n"

    if drop_rate > MAX_DROP_RATE:
        raise CompareError(
            f"{header}\nREFUSING to summarise: {drop_rate:.1%} of questions were dropped, above the "
            f"{MAX_DROP_RATE:.0%} limit. Above this the two arms did not answer the same question set, "
            "so a paired summary would describe a different experiment from the one that ran. "
            "Fix the failures and re-run both arms."
        )
    if not deltas:
        raise CompareError(f"{header}\nNo metric is present in both files; nothing to compare.")

    out = [header]
    for metric in sorted(deltas):
        out.append(f"\n{metric}:\n")
        out.append(_format_section("overall", metric, deltas[metric], seed))
        if by_question_type:
            per_type: dict[str, list[tuple[int, float]]] = defaultdict(list)
            for qid, delta in deltas[metric]:
                per_type[types.get(qid, "unknown")].append((qid, delta))
            for qtype in sorted(per_type):
                out.append(_format_section(f"question_type={qtype}", metric, per_type[qtype], seed))
    out.append(CAVEATS)
    return "".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m sibyl.eval compare",
        description="Paired comparison of two backtest_data_*.json files (B minus A).",
    )
    parser.add_argument("file_a", type=Path, help="baseline backtest data (A)")
    parser.add_argument("file_b", type=Path, help="treatment backtest data (B)")
    parser.add_argument(
        "--by",
        choices=["question_type"],
        default=None,
        help="also slice the comparison by question type",
    )
    parser.add_argument("--seed", type=int, default=0, help="bootstrap seed (default: 0)")
    args = parser.parse_args(argv)

    try:
        print(render_report(args.file_a, args.file_b, by_question_type=args.by == "question_type", seed=args.seed))
    except CompareError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0
