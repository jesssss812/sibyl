"""Aggregate scores, compare bot vs community, generate reports."""

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from sibyl.eval.backtest.scoring import GroundTruth, QuestionScore
from sibyl.eval.stats import bootstrap_mean_ci

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    bot_name: str
    scores: list[QuestionScore]
    num_questions: int
    num_scored: int
    num_failed: int
    question_status: dict[int, dict[str, Any]] = field(default_factory=dict)


def aggregate_scores(scores: list[QuestionScore]) -> dict[str, dict]:
    """Group scores by metric_name and compute summary statistics for each.

    Returns dict mapping metric_name -> {bot_mean, bot_std, community_mean, bot_minus_community, n}.
    """
    if not scores:
        return {}

    by_metric: dict[str, list[QuestionScore]] = {}
    for s in scores:
        by_metric.setdefault(s.metric_name, []).append(s)

    aggregated: dict[str, dict] = {}
    for metric_name, metric_scores in by_metric.items():
        bot_values = np.array([s.bot_score for s in metric_scores], dtype=float)
        community_values = [s.community_score for s in metric_scores if s.community_score is not None]

        bot_mean = float(np.mean(bot_values))
        bot_std = float(np.std(bot_values))

        if community_values:
            community_mean = float(np.mean(community_values))
            bot_minus_community = bot_mean - community_mean
        else:
            community_mean = None
            bot_minus_community = None

        aggregated[metric_name] = {
            "bot_mean": bot_mean,
            "bot_std": bot_std,
            "community_mean": community_mean,
            "bot_minus_community": bot_minus_community,
            "n": len(metric_scores),
            "bot_minus_community_ci": _bot_minus_community_ci(metric_scores),
        }

    return aggregated


def _bot_minus_community_ci(metric_scores: list[QuestionScore]) -> dict[str, Any] | None:
    """A paired bootstrap CI on bot-minus-community, or None when nothing is paired.

    Nested under its own key with its OWN mean and n, rather than sitting beside the
    legacy ``bot_minus_community`` scalar, because the two are not always the same
    number. The scalar is a difference of two means taken over DIFFERENT denominators --
    the bot mean spans every score in the group while the community mean spans only
    those with a community score -- so when the group is partly unpaired it is not an
    estimate any interval brackets. This one is the mean of the per-question differences
    over the paired subset, which is the quantity Rule 1 of the eval strategy says to
    report, and it equals the scalar exactly when every score is paired.

    Added as a key rather than replacing the scalar: existing consumers and the recorded
    golden aggregate_scores output keep their values unchanged.
    """
    deltas = [s.bot_score - s.community_score for s in metric_scores if s.community_score is not None]
    if not deltas:
        return None
    mean, low, high = bootstrap_mean_ci(deltas, seed=0)
    return {
        "mean": mean,
        "low": low if np.isfinite(low) else None,
        "high": high if np.isfinite(high) else None,
        "n": len(deltas),
    }


def generate_backtest_report(
    results: list[BacktestResult],
    question_set: Any = None,
    output_path: str | None = None,
) -> str:
    """Generate a markdown backtest report string.

    If output_path is provided, writes the report to that file (creating parent dirs).
    Returns the markdown string.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []

    lines.append("# Backtest Report")
    lines.append("Retrospective results are leaky and descriptive; absolute scores do not measure live skill.")
    lines.append(f"Generated: {timestamp}")
    lines.append("")

    lines.append("## Summary")
    lines.append(f"- {len(results)} bots evaluated")

    if question_set is not None and hasattr(question_set, "fetch_metadata"):
        metadata = question_set.fetch_metadata
        total_clean = metadata.get("total_clean", "?")
        lines.append(f"- Questions: {total_clean}")
        type_dist = metadata.get("type_distribution")
        if type_dist:
            dist_parts = [f"{qtype}: {count}" for qtype, count in type_dist.items()]
            lines.append(f"- Type distribution: {', '.join(dist_parts)}")

    lines.append("")

    lines.append("## Results by Bot")
    for result in results:
        lines.append("")
        lines.append(f"### {result.bot_name}")
        lines.append(f"- Questions scored: {result.num_scored}/{result.num_questions} (failed: {result.num_failed})")
        lines.append("")

        agg = aggregate_scores(result.scores)
        if agg:
            lines.append("| Metric | Bot Mean | Bot Std | Community Mean | Bot - Community | N |")
            lines.append("|--------|----------|---------|----------------|-----------------|---|")
            for metric_name, stats in agg.items():
                community_mean_str = f"{stats['community_mean']:.4f}" if stats["community_mean"] is not None else "N/A"
                bot_minus_str = (
                    f"{stats['bot_minus_community']:.4f}" if stats["bot_minus_community"] is not None else "N/A"
                )
                lines.append(
                    f"| {metric_name} "
                    f"| {stats['bot_mean']:.4f} "
                    f"| {stats['bot_std']:.4f} "
                    f"| {community_mean_str} "
                    f"| {bot_minus_str} "
                    f"| {stats['n']} |"
                )
        lines.append("")

    lines.append("## Per-Type Breakdown")
    for result in results:
        by_type: dict[str, list[QuestionScore]] = {}
        for s in result.scores:
            by_type.setdefault(s.question_type, []).append(s)

        for question_type, type_scores in sorted(by_type.items()):
            lines.append("")
            lines.append(f"### {result.bot_name} - {question_type}")
            type_agg = aggregate_scores(type_scores)
            if type_agg:
                lines.append("| Metric | Bot Mean | Bot Std | Community Mean | Bot - Community | N |")
                lines.append("|--------|----------|---------|----------------|-----------------|---|")
                for metric_name, stats in type_agg.items():
                    community_mean_str = (
                        f"{stats['community_mean']:.4f}" if stats["community_mean"] is not None else "N/A"
                    )
                    bot_minus_str = (
                        f"{stats['bot_minus_community']:.4f}" if stats["bot_minus_community"] is not None else "N/A"
                    )
                    lines.append(
                        f"| {metric_name} "
                        f"| {stats['bot_mean']:.4f} "
                        f"| {stats['bot_std']:.4f} "
                        f"| {community_mean_str} "
                        f"| {bot_minus_str} "
                        f"| {stats['n']} |"
                    )

    lines.append("")
    lines.append("## Notes")
    lines.append(
        "- Community prediction is the final CP at resolution time, which benefits from late-stage convergence"
    )
    lines.append("- Numeric community log scores unavailable (Metaculus removed aggregations from list API)")
    lines.append("")

    report = "\n".join(lines)

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(report)
        logger.info(f"Backtest report written to {output_path}")

    return report


def _serialize_ground_truth(gt: GroundTruth) -> dict[str, Any]:
    """Serialize a GroundTruth to a JSON-safe dict."""
    return {
        "question_id": gt.question_id,
        "question_type": gt.question_type,
        "resolution": str(gt.resolution),
        "resolution_string": gt.resolution_string,
        "community_prediction": gt.community_prediction,
        "actual_resolution_time": gt.actual_resolution_time.isoformat() if gt.actual_resolution_time else None,
        "question_text": gt.question_text,
        "page_url": gt.page_url,
    }


def save_backtest_data(
    question_set: Any,
    results: list[BacktestResult],
    output_dir: str,
) -> str:
    """Save ground truths and scores as JSON for offline re-analysis.

    Returns the file path of the saved JSON.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    filepath = output_path / f"backtest_data_{timestamp}.json"

    ground_truths_serialized: dict[str, dict] = {}
    if hasattr(question_set, "ground_truths"):
        for qid, gt in question_set.ground_truths.items():
            ground_truths_serialized[str(qid)] = _serialize_ground_truth(gt)

    results_serialized: list[dict[str, Any]] = []
    for result in results:
        scores_list = [asdict(s) for s in result.scores]
        results_serialized.append(
            {
                "bot_name": result.bot_name,
                "num_questions": result.num_questions,
                "num_scored": result.num_scored,
                "num_failed": result.num_failed,
                "question_status": result.question_status,
                "scores": scores_list,
                "aggregated": aggregate_scores(result.scores),
            }
        )

    data = {
        "timestamp": datetime.now().isoformat(),
        "fetch_metadata": question_set.fetch_metadata if hasattr(question_set, "fetch_metadata") else {},
        "ground_truths": ground_truths_serialized,
        "results": results_serialized,
    }

    from sibyl.eval.artifacts import write_json

    data["experiment"] = data["fetch_metadata"].get("experiment", {})
    write_json(filepath, data)
    logger.info(f"Backtest data saved to {filepath}")

    return str(filepath)
