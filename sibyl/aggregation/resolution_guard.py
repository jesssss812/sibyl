"""Open-vs-resolved confidence guard (design principle A3).

The failure mode this exists for is a documented, recurring one across bot
tournaments: a forecaster reads a still-OPEN question as already settled and
submits a near-certain probability. The binary prompt actively invites the
shortcut — it tells the model that when the research already shows the
resolution condition met (or impossible), it should assign a near-extreme
probability and skip the reference-class analysis. That shortcut is usually
right and is worth keeping; this module is its backstop for when it isn't.

**The discriminator is the clock.** An extreme forecast issued while most of the
open->close window still remains is far more likely to be a misread than genuine
near-certainty, because the outcome is not yet determined by time. The same
forecast issued near the close is ordinarily earned. So the guard fires early
and abstains late.

**It is a rail, not a neutralization.** An extreme value is pulled back to the
extremity threshold and no further; it is never dragged toward 0.5. Same stance
as the Platt deviation caps: tweak, don't massively deviate.

**Know the size of the effect before reading the rest of this.** Every binary
prediction is already clamped to ``[BINARY_PROB_MIN, BINARY_PROB_MAX]`` = [0.02,
0.98] *upstream*, per forecaster, at ``forecaster_runners.py`` and
``stacking.py`` — and a median or mean cannot escape its inputs' range. So the
worst value this guard can ever see is 0.98, and its **maximum possible
correction is 0.03**. This is a cheap insurance policy against a rare, expensive
misread, not a large intervention.

The bet, in Brier terms: a true positive (0.98 -> 0.95 on a question that
resolves NO) gains ~0.058; a false positive (same move on one that resolves YES)
costs ~0.0021. It pays whenever more than ~3.5% of early extreme forecasts are
misreads. That ratio, not the absolute size, is the argument for it.

A consequence worth stating for anyone planning to measure this: because the
guard fires rarely and moves each firing by at most 0.03, it **cannot** shift a
pooled mean at the sample sizes available here. Evaluating it requires a
pre-specified subgroup analysis restricted to the questions where it fired —
which is valid, because the firing condition depends only on question state and
the raw forecast, both identical across arms.

**Firing is normal operation, not degradation.** The guard emits a telemetry
marker and bumps no alertable counter, matching how the conditional-stacking
counters are deliberately excluded from ``alertable_total``.

Scope: **binary questions only.** Multiple-choice has the same failure mode via
a single dominant option, but its clamp interacts with renormalization across
the whole ballot, so it needs a separate design rather than a reused threshold.
Numeric is unaffected (the analogous failure is interval width, which the width
monitor already tracks).

No I/O, no clock reads, no LLM calls — ``now`` is passed in, so the decision is
determined by its arguments plus exactly one piece of ambient state: the
``RESOLUTION_STATE_GUARD_ENABLED`` env flag, read in ``evaluate_binary``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from forecasting_tools.data_models.questions import MetaculusQuestion

from sibyl.common.time_utils import _as_utc
from sibyl.config.settings import (
    RESOLUTION_GUARD_CAP_HIGH,
    RESOLUTION_GUARD_CAP_LOW,
    RESOLUTION_GUARD_ENABLED_ENV,
    RESOLUTION_GUARD_MIN_WINDOW_REMAINING_FRAC,
    env_flag_enabled,
)

# The single reason string emitted today. Kept as a constant because the marker
# is a parsed telemetry contract (sibyl/telemetry/markers.py stance).
REASON_EXTREME_WITH_OPEN_WINDOW = "extreme_forecast_with_open_window"


@dataclass(frozen=True)
class GuardVerdict:
    """Outcome of evaluating one prediction against its question's state.

    ``adjusted`` is always the value to publish — it equals ``raw`` whenever the
    guard did not fire, so callers can use it unconditionally.
    """

    fired: bool
    raw: float
    adjusted: float
    reason: str | None
    window_remaining_frac: float | None


def _window_remaining_frac(question: MetaculusQuestion, now: datetime) -> float | None:
    """Fraction of the open->close window still remaining, or None if unknowable.

    Returns None when either timestamp is absent or the window is non-positive
    (a close at or before the open would divide by zero or invert the ratio).
    Abstaining is the safe choice: the value has *already* passed the
    ``[BINARY_PROB_MIN, BINARY_PROB_MAX]`` clamp upstream, so it is bounded
    regardless, whereas clamping every question with a missing timestamp would be
    the aggressive one.
    """
    open_time = getattr(question, "open_time", None)
    close_time = getattr(question, "close_time", None)
    if open_time is None or close_time is None:
        return None

    opened = _as_utc(open_time)
    closes = _as_utc(close_time)
    window_seconds = (closes - opened).total_seconds()
    if window_seconds <= 0:
        return None

    return (closes - _as_utc(now)).total_seconds() / window_seconds


def _is_extreme(probability: float) -> bool:
    """True when the probability sits strictly outside the extremity thresholds.

    Strict comparison matters: the thresholds double as clamp targets, so a value
    already at one is in-bounds and must not be reported as adjusted-to-itself.

    NaN returns False here (every comparison against NaN is False), which is the
    right answer — an unrepresentable probability is not something to silently
    "correct" into a plausible-looking one.
    """
    return probability < RESOLUTION_GUARD_CAP_LOW or probability > RESOLUTION_GUARD_CAP_HIGH


def evaluate_binary(question: MetaculusQuestion, probability: float, now: datetime) -> GuardVerdict:
    """Decide whether an aggregated binary probability needs the A3 rail.

    Fires only when all three hold: the guard is enabled, the probability is
    extreme, and at least ``RESOLUTION_GUARD_MIN_WINDOW_REMAINING_FRAC`` of the
    question's window remains. Otherwise the prediction passes through untouched.
    """
    raw = float(probability)
    unfired = GuardVerdict(fired=False, raw=raw, adjusted=raw, reason=None, window_remaining_frac=None)

    if not env_flag_enabled(RESOLUTION_GUARD_ENABLED_ENV, default=True):
        return unfired

    if not _is_extreme(raw):
        return unfired

    remaining = _window_remaining_frac(question, now)
    if remaining is None:
        return unfired

    # Late extremes are ordinarily earned; leave them alone. A negative fraction
    # (already past close) lands here too.
    if remaining < RESOLUTION_GUARD_MIN_WINDOW_REMAINING_FRAC:
        return GuardVerdict(fired=False, raw=raw, adjusted=raw, reason=None, window_remaining_frac=remaining)

    adjusted = min(max(raw, RESOLUTION_GUARD_CAP_LOW), RESOLUTION_GUARD_CAP_HIGH)
    return GuardVerdict(
        fired=True,
        raw=raw,
        adjusted=adjusted,
        reason=REASON_EXTREME_WITH_OPEN_WINDOW,
        window_remaining_frac=remaining,
    )


def format_guard_marker(verdict: GuardVerdict, question_id: int | None) -> str | None:
    """Durable one-line telemetry marker, or None when the guard did not fire.

    Shape mirrors ``CLOSE_MARGIN`` (close_margin.py): a greppable ``KEY: k=v``
    line that survives into the ``run_logs/`` artifact. Renaming a key here is a
    breaking change for any downstream marker parser.
    """
    if not verdict.fired:
        return None

    # `fired=True` is constructed at exactly one site, always with a computed fraction,
    # so this is a total format rather than a defensive one.
    frac = f"{verdict.window_remaining_frac:.3f}" if verdict.window_remaining_frac is not None else "n/a"
    return (
        f"RESOLUTION_GUARD: question={question_id} "
        f"raw={verdict.raw:.4f} adjusted={verdict.adjusted:.4f} "
        f"window_remaining_frac={frac} reason={verdict.reason}"
    )
