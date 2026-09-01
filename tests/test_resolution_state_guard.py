"""Tests for the open-vs-resolved confidence guard (metaculus_bot/resolution_state_guard.py).

Design principle A3: "Don't let the bot confuse open questions with resolved ones …
add an explicit guard that checks question state before allowing extreme confidence."

The failure mode being guarded: a forecaster reads a still-open question as already
settled and submits a near-certain probability. The signal that an extreme forecast is a
MISREAD rather than a legitimate near-certainty is that a substantial fraction of the
question's open->close window still remains — the outcome is not yet determined by the
clock. Late extreme forecasts are left alone precisely because they are usually right.

The guard is a "tweak, don't massively deviate" rail (same stance as Platt's
max_abs_deviation): it pulls an extreme value back to the extremity threshold, it does
not neutralize the forecast toward 0.5.
"""

from datetime import datetime, timedelta, timezone

import pytest
from forecasting_tools.data_models.questions import BinaryQuestion

from metaculus_bot.constants import (
    RESOLUTION_GUARD_CAP_HIGH,
    RESOLUTION_GUARD_CAP_LOW,
    RESOLUTION_GUARD_ENABLED_ENV,
    RESOLUTION_GUARD_MIN_WINDOW_REMAINING_FRAC,
)
from metaculus_bot.resolution_state_guard import (
    GuardVerdict,
    evaluate_binary,
    format_guard_marker,
)

NOW = datetime(2026, 8, 24, 12, 0, 0, tzinfo=timezone.utc)


def _question(
    *,
    open_offset_days: float = -10.0,
    close_offset_days: float = 10.0,
    qid: int = 44620,
    naive: bool = False,
) -> BinaryQuestion:
    """A binary question whose window straddles NOW by the given day offsets."""
    open_time = NOW + timedelta(days=open_offset_days)
    close_time = NOW + timedelta(days=close_offset_days)
    if naive:
        open_time = open_time.replace(tzinfo=None)
        close_time = close_time.replace(tzinfo=None)
    return BinaryQuestion(
        question_text="Will X happen by the deadline?",
        id_of_question=qid,
        open_time=open_time,
        close_time=close_time,
    )


class TestNonExtremeIsUntouched:
    """The guard only ever looks at extreme forecasts. Everything else passes through."""

    @pytest.mark.parametrize("p", [0.5, 0.06, 0.94, 0.2, 0.8])
    def test_mid_range_never_fires(self, p: float) -> None:
        verdict = evaluate_binary(_question(), p, NOW)
        assert verdict.fired is False
        assert verdict.adjusted == p
        assert verdict.raw == p

    def test_exactly_at_threshold_is_not_extreme(self) -> None:
        # The thresholds are the clamp targets, so a value already AT one is
        # in-bounds and must not be "adjusted" to itself with fired=True.
        assert evaluate_binary(_question(), RESOLUTION_GUARD_CAP_HIGH, NOW).fired is False
        assert evaluate_binary(_question(), RESOLUTION_GUARD_CAP_LOW, NOW).fired is False


class TestFiresOnEarlyExtremes:
    """An extreme forecast with most of the window still to run is the A3 failure mode."""

    def test_extreme_high_early_is_clamped_down(self) -> None:
        # Half the window remains — well above the min-remaining floor.
        verdict = evaluate_binary(_question(open_offset_days=-10, close_offset_days=10), 0.98, NOW)
        assert verdict.fired is True
        assert verdict.raw == 0.98
        assert verdict.adjusted == RESOLUTION_GUARD_CAP_HIGH
        assert verdict.window_remaining_frac == pytest.approx(0.5)
        assert verdict.reason is not None

    def test_extreme_low_early_is_clamped_up(self) -> None:
        verdict = evaluate_binary(_question(open_offset_days=-10, close_offset_days=10), 0.01, NOW)
        assert verdict.fired is True
        assert verdict.adjusted == RESOLUTION_GUARD_CAP_LOW

    def test_clamp_is_a_rail_not_a_neutralization(self) -> None:
        """The guard must not drag the forecast toward 0.5 — it pulls to the threshold."""
        verdict = evaluate_binary(_question(), 0.999, NOW)
        assert verdict.adjusted == RESOLUTION_GUARD_CAP_HIGH
        assert verdict.adjusted > 0.9, "clamping toward 0.5 would destroy real signal"

    def test_naive_datetimes_are_treated_as_utc(self) -> None:
        """Metaculus timestamps arrive naive from some call sites (see time_utils._as_utc)."""
        verdict = evaluate_binary(_question(naive=True), 0.98, NOW)
        assert verdict.fired is True
        assert verdict.window_remaining_frac == pytest.approx(0.5)


class TestDoesNotFireLate:
    """Near the close, an extreme forecast is usually legitimate — leave it alone."""

    def test_extreme_near_close_is_untouched(self) -> None:
        # 1 day left of a 20-day window = 5% remaining, below the floor.
        verdict = evaluate_binary(_question(open_offset_days=-19, close_offset_days=1), 0.98, NOW)
        assert verdict.fired is False
        assert verdict.adjusted == 0.98

    def test_boundary_at_min_remaining_frac_fires(self) -> None:
        """At exactly the floor the guard fires: the comparison is >=."""
        # Construct a window where remaining fraction is exactly the floor.
        total_days = 100.0
        remaining_days = total_days * RESOLUTION_GUARD_MIN_WINDOW_REMAINING_FRAC
        verdict = evaluate_binary(
            _question(open_offset_days=-(total_days - remaining_days), close_offset_days=remaining_days),
            0.99,
            NOW,
        )
        assert verdict.window_remaining_frac == pytest.approx(RESOLUTION_GUARD_MIN_WINDOW_REMAINING_FRAC)
        assert verdict.fired is True

    def test_after_close_does_not_fire(self) -> None:
        """Past the close time the window fraction is negative; nothing to guard."""
        verdict = evaluate_binary(_question(open_offset_days=-20, close_offset_days=-1), 0.99, NOW)
        assert verdict.fired is False


class TestUnknowableWindowsAreLeftAlone:
    """When the window can't be computed the guard abstains rather than guessing.

    The standard [0.02, 0.98] cap still applies downstream, so abstaining is safe;
    clamping on every question with a missing timestamp would be the aggressive choice.
    """

    def test_missing_close_time_does_not_fire(self) -> None:
        q = _question()
        q.close_time = None  # type: ignore[assignment]
        verdict = evaluate_binary(q, 0.99, NOW)
        assert verdict.fired is False
        assert verdict.window_remaining_frac is None

    def test_missing_open_time_does_not_fire(self) -> None:
        q = _question()
        q.open_time = None  # type: ignore[assignment]
        verdict = evaluate_binary(q, 0.99, NOW)
        assert verdict.fired is False
        assert verdict.window_remaining_frac is None

    def test_non_positive_window_does_not_fire(self) -> None:
        """close_time <= open_time would divide by zero or invert the fraction."""
        q = _question(open_offset_days=5, close_offset_days=5)
        verdict = evaluate_binary(q, 0.99, NOW)
        assert verdict.fired is False
        assert verdict.window_remaining_frac is None


class TestEnvGate:
    def test_disabled_via_env_never_fires(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(RESOLUTION_GUARD_ENABLED_ENV, "false")
        verdict = evaluate_binary(_question(), 0.99, NOW)
        assert verdict.fired is False
        assert verdict.adjusted == 0.99

    def test_enabled_by_default_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A3 is a reliability guard, so it is on unless explicitly turned off.

        Turning it OFF is how it becomes the control arm of a paired experiment.
        """
        monkeypatch.delenv(RESOLUTION_GUARD_ENABLED_ENV, raising=False)
        assert evaluate_binary(_question(), 0.99, NOW).fired is True


class TestMarker:
    """The marker is a parsed telemetry contract, same stance as CLOSE_MARGIN."""

    def test_marker_shape_when_fired(self) -> None:
        verdict = evaluate_binary(_question(), 0.98, NOW)
        marker = format_guard_marker(verdict, question_id=44620)
        assert marker == (
            "RESOLUTION_GUARD: question=44620 raw=0.9800 adjusted=0.9500 "
            "window_remaining_frac=0.500 reason=extreme_forecast_with_open_window"
        )

    def test_no_marker_when_not_fired(self) -> None:
        verdict = evaluate_binary(_question(), 0.5, NOW)
        assert format_guard_marker(verdict, question_id=44620) is None


class TestVerdictIsImmutable:
    def test_frozen(self) -> None:
        verdict = evaluate_binary(_question(), 0.5, NOW)
        assert isinstance(verdict, GuardVerdict)
        with pytest.raises(Exception):
            verdict.fired = True  # type: ignore[misc]
