"""Per-question FORECASTER_SPREAD telemetry: how often the ensemble agrees with itself.

`compute_spread` has always run on every multi-forecaster question, but its value was only
logged when conditional stacking TRIGGERED. Stacking is disabled on all three question
types in production, so the spread has been computed and thrown away on every question this
bot has ever forecast.

That discarded number is the most useful unmeasured quantity in the system:

* **Is 3 forecasters the right count?** If the three agree on nearly everything, the
  ensemble is buying insurance against a rare tail and its cost is mostly premium. If they
  disagree often, the median is doing real work and cutting the roster would cost more than
  the frozen-triple analysis suggests.
* **Would a disagreement-triggered escalation ever fire?** Any policy of the form "spend
  more where the models disagree" needs the base rate of disagreement before it can be
  designed, let alone costed. Designing the policy first and measuring after is how you end
  up with a router tuned to a distribution you assumed.
* **Are the inherited thresholds right?** 0.15 / 0.20 / 0.15 were tuned on a 6-member
  roster. Nothing has checked them against the 3-member triple.

**This module measures and does not decide.** It changes no forecast and takes no routing
action. That separation is deliberate: the spread distribution should be known *before* a
policy is written against it, and a measurement that quietly also acts cannot be trusted as
a measurement.

Marker shape follows `close_margin.py` and `resolution_state_guard.py` — a greppable
`KEY: k=v` line that survives into the `run_logs/` artifact and is parsed by
`sibyl/telemetry/markers.py`. Renaming a key here is a breaking change for that parser.
"""

from __future__ import annotations

# Cohort labels match the names the eval harness already slices on (`aggregate_paired`
# groups by question_type), so spread records join to scored records without translation.
_TYPE_LABELS: dict[str, str] = {
    "BinaryQuestion": "binary",
    "MultipleChoiceQuestion": "mc",
    "NumericQuestion": "numeric",
    # DiscreteQuestion subclasses NumericQuestion and shares its spread metric, so it
    # shares the label — splitting them here would fragment an already-small cohort.
    "DiscreteQuestion": "numeric",
}


def question_type_label(class_name: str) -> str:
    """Map a question class name to its cohort label, or ``unknown``.

    Returns ``unknown`` rather than guessing: a new question type showing up in the
    telemetry as ``unknown`` is a signal worth seeing, whereas silently filing it under
    ``numeric`` would corrupt a cohort.
    """
    return _TYPE_LABELS.get(class_name, "unknown")


def format_spread_marker(
    *,
    question_id: int | None,
    question_type: str,
    n_forecasters: int,
    spread: float,
    threshold: float | None,
    exceeds: bool | None,
) -> str:
    """Build the durable one-line spread marker.

    ``threshold``/``exceeds`` are optional because spread is meaningful under any
    aggregation strategy, while a threshold only exists under conditional stacking. They
    render as ``n/a`` rather than being omitted, so the field count stays fixed and the
    parser regex does not need an optional group.

    Four decimal places on both: binary spreads cluster well below 0.1, and two decimals
    would collapse distinct questions into the same bucket, which is precisely the
    resolution this exists to capture.
    """
    threshold_str = "n/a" if threshold is None else f"{threshold:.4f}"
    exceeds_str = "n/a" if exceeds is None else str(exceeds)
    return (
        f"FORECASTER_SPREAD: question={question_id} qtype={question_type} "
        f"n={n_forecasters} spread={spread:.4f} threshold={threshold_str} exceeds={exceeds_str}"
    )
