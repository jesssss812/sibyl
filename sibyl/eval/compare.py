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
  attached. Backtests are leaky by construction; leakage can interact with treatments even in paired comparisons.
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
from datetime import datetime
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
  * Pairing does not establish that leakage cancels. Leakage may interact with the
    treatment; interpret these retrospective differences with that limitation.
  * Surviving pairs describe complete cases. The drop threshold does not prove
    failures are independent of question difficulty or treatment.
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
    seen = set()
    invalid = {int(qid) for qid, status in results[0].get("question_status", {}).items()
               if status.get("status") != "ok" or status.get("degraded", False)}
    for row in results[0].get("scores", []):
        qid, metric = int(row["question_id"]), str(row["metric_name"])
        key = (qid, metric)
        if key in seen:
            raise CompareError(f"{path}: duplicate score {key}")
        seen.add(key)
        if row.get("degraded") or row.get("errors"):
            invalid.add(qid)
        value = row.get("bot_score")
        if value is None or not math.isfinite(float(value)):
            invalid.add(qid)
            continue  # unscored in this arm; the pairing step will drop it from both
        scores[(qid, metric)] = float(value)
        types[qid] = str(row.get("question_type", "unknown"))
    return {key: val for key, val in scores.items() if key[0] not in invalid}, types


def pair_scores(
    a: dict[tuple[int, str], float], b: dict[tuple[int, str], float]
) -> tuple[dict[str, list[tuple[int, float]]], set[int]]:
    """Deltas (B - A) per metric, and the ids dropped because an arm could not score them."""
    all_qids = {qid for qid, _metric in a} | {qid for qid, _metric in b}
    paired_keys = a.keys() & b.keys()
    incomplete = {qid for qid, _metric in a.keys() ^ b.keys()}
    paired_keys = {key for key in paired_keys if key[0] not in incomplete}
    paired_qids = {qid for qid, _metric in paired_keys}
    deltas: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for qid, metric in sorted(paired_keys):
        deltas[metric].append((qid, b[(qid, metric)] - a[(qid, metric)]))
    return dict(deltas), all_qids - paired_qids


def _format_section(title: str, metric: str, rows: list[tuple[int, float]], seed: int, *, confirmatory: bool = True) -> str:
    summary = paired_summary([d for _qid, d in rows], seed=seed)
    direction = "lower is better" if metric in LOWER_IS_BETTER else "higher is better"
    verdict = _verdict(metric, summary) if confirmatory else "exploratory only; no confirmatory verdict"
    def interval(low: str, high: str) -> str:
        lo, hi = summary[low], summary[high]
        return "unavailable" if not math.isfinite(lo) or not math.isfinite(hi) else f"[{lo:+.5f}, {hi:+.5f}]"
    mde = f"{summary['mde']:.5f}" if math.isfinite(summary["mde"]) else "unavailable"
    return (
        f"  {title}\n"
        f"    n={summary['n']}  ({direction})\n"
        f"    mean Δ   {summary['mean_delta']:+.5f}   95% CI {interval('mean_ci_low', 'mean_ci_high')}\n"
        f"    median Δ {summary['median_delta']:+.5f}   95% CI {interval('median_ci_low', 'median_ci_high')}\n"
        f"    sign p={summary['sign_test_p']:.4f}   "
        f"wilcoxon p={'n/a' if summary['wilcoxon_p'] is None else f'{summary["wilcoxon_p"]:.4f}'}\n"
        f"    Post-run empirical MDE at this n: {mde}\n"
        f"    -> {verdict}\n"
    )


def _verdict(metric: str, summary: dict[str, Any]) -> str:
    """One plain sentence, so the reader does not have to re-derive the sign convention."""
    lo, hi = summary["mean_ci_low"], summary["mean_ci_high"]
    if summary["n"] == 0:
        return "no paired questions"
    if math.isnan(lo) or math.isnan(hi):
        return "insufficient pairs for inference"
    if lo <= 0.0 <= hi:
        return "no detectable difference"
    b_better = (hi < 0.0) if metric in LOWER_IS_BETTER else (lo > 0.0)
    return f"B is {'better' if b_better else 'worse'} than A"


def render_report(path_a: Path, path_b: Path, by_question_type: bool = False, seed: int = 0) -> str:
    data_a, data_b = _load(path_a), _load(path_b)
    scores_a, types_a = _scores_by_key(data_a, path_a)
    scores_b, types_b = _scores_by_key(data_b, path_b)
    for qid in types_a.keys() & types_b.keys():
        if types_a[qid] != types_b[qid]:
            raise CompareError(f"Q{qid}: question type differs between arms")
    types = {**types_a, **types_b}

    deltas, dropped = pair_scores(scores_a, scores_b)
    scheduled_a = {int(q) for q in data_a.get("ground_truths", {})}
    scheduled_b = {int(q) for q in data_b.get("ground_truths", {})}
    for qid in scheduled_a & scheduled_b:
        if data_a["ground_truths"][str(qid)] != data_b["ground_truths"][str(qid)]:
            raise CompareError(f"Q{qid}: ground truth differs between arms")
    all_qids = scheduled_a | scheduled_b | {qid for qid, _ in scores_a} | {qid for qid, _ in scores_b}
    dropped |= all_qids - {qid for rows in deltas.values() for qid, _ in rows}
    total = len(all_qids)
    identity_a = data_a.get("experiment", {})
    identity_b = data_b.get("experiment", {})
    if identity_a.get("manifest_id") or identity_b.get("manifest_id"):
        for key in ("manifest_id", "scoring_version"):
            if not identity_a.get(key) or identity_a.get(key) != identity_b.get(key):
                raise CompareError(f"Experimental {key} differs or is missing")
    if identity_a.get("manifest_id") and scheduled_a != scheduled_b:
        raise CompareError("Frozen scheduled question sets differ")
    fixture = bool(identity_a.get("fixture") and identity_b.get("fixture"))
    plan_a, plan_b = identity_a.get("power_plan"), identity_b.get("power_plan")
    if plan_a and plan_b and plan_a.get("plan_id") == plan_b.get("plan_id") and plan_a != plan_b:
        raise CompareError("Registered power plan payload differs between arms")
    registered = bool(plan_a and plan_b and plan_a == plan_b and identity_a.get("manifest_id"))
    if registered:
        from sibyl.eval.power import assess_plan
        checked = assess_plan(plan_a["plan"])
        if checked["plan_id"] != plan_a["plan_id"] or checked["decision"] != "proceed":
            raise CompareError("Invalid or underpowered registered plan")
        if plan_a["plan"].get("manifest_id") != identity_a.get("manifest_id"):
            raise CompareError("Power plan and frozen manifest differ")
        if not fixture:
            if not identity_a.get("arm_id") or not identity_b.get("arm_id") or identity_a["arm_id"] == identity_b["arm_id"]:
                raise CompareError("Registered real comparison needs two distinct arm IDs")
            try:
                assessed = datetime.fromisoformat(plan_a["assessed_at"])
                if any(datetime.fromisoformat(identity["started_at"]) <= assessed for identity in (identity_a, identity_b)):
                    raise CompareError("Power preflight was not recorded before both runs")
            except (KeyError, TypeError, ValueError) as exc:
                raise CompareError("Registered real comparison lacks valid timing provenance") from exc
        for data, scheduled in ((data_a, scheduled_a), (data_b, scheduled_b)):
            statuses = data["results"][0].get("question_status", {})
            if {int(q) for q in statuses} != scheduled:
                raise CompareError("Registered results require status for every scheduled question")
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
    if registered:
        primary = plan_a["plan"]["endpoint"]
        if len(deltas.get(primary, [])) < plan_a["plan"]["n"]:
            raise CompareError(f"Registered primary {primary} has fewer usable pairs than planned")

    out = [header]
    if fixture:
        out.append("FIXTURE PIPELINE CHECK: not evidence of forecasting skill.\n")
    elif not registered:
        out.append("DESCRIPTIVE/EXPLORATORY DATA: no matched registration; no confirmatory verdict.\n")
    for metric in sorted(deltas):
        out.append(f"\n{metric}:\n")
        out.append(_format_section("overall", metric, deltas[metric], seed, confirmatory=fixture or (registered and metric == plan_a["plan"]["endpoint"])))
        if by_question_type:
            per_type: dict[str, list[tuple[int, float]]] = defaultdict(list)
            for qid, delta in deltas[metric]:
                per_type[types.get(qid, "unknown")].append((qid, delta))
            for qtype in sorted(per_type):
                out.append(_format_section(f"question_type={qtype} (exploratory slice)", metric, per_type[qtype], seed, confirmatory=False))
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
