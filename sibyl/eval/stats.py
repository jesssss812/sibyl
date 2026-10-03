"""Paired statistics, shared across the eval layer.

These four primitives were defined inside ``eval/ablation/scoring.py`` and used only
there. They are not ablation-specific: a paired bootstrap over a list of deltas is the
same operation whether the pairs come from two stacker arms, two backtest runs, or two
model rosters, and ``eval compare`` needs exactly the same machinery. Moved here so the
second consumer is a call rather than a copy.

Samples smaller than five retain point estimates but have unavailable intervals.
Non-finite observations must be accounted for as explicit exclusions by the caller.

Seeded throughout. A bootstrap CI that changes between runs is not a measurement, and an
eval layer whose own numbers wobble cannot be used to judge anything else.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import stats as scipy_stats

__all__ = [
    "BOOTSTRAP_MIN_N",
    "WILCOXON_MIN_N",
    "bootstrap_mean_ci",
    "bootstrap_median_ci",
    "minimum_detectable_effect",
    "paired_summary",
    "sign_test",
    "wilcoxon_signed_rank",
]

# Below this many pairs report descriptive centres only; uncertainty is unavailable.
BOOTSTRAP_MIN_N = 5
# scipy's Wilcoxon is not meaningful below this n (its normal approximation and exact
# tables both need more), so it returns None rather than a number nobody should read.
WILCOXON_MIN_N = 6


def _validate_bootstrap(deltas: Sequence[float], n_bootstrap: int, alpha: float) -> None:
    if not isinstance(n_bootstrap, int) or n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be a positive integer")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    if any(not math.isfinite(d) for d in deltas):
        raise ValueError("deltas must be finite; record invalid scores as explicit drops")


def bootstrap_mean_ci(
    deltas: list[float], n_bootstrap: int = 5000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Paired bootstrap on the deltas: resample with replacement, recompute mean, take CI percentiles.

    For n < 5, returns (mean, NaN, NaN) — too few samples for a meaningful CI.
    """
    _validate_bootstrap(deltas, n_bootstrap, alpha)
    n = len(deltas)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))

    arr = np.asarray(deltas, dtype=float)
    mean = float(arr.mean())

    if n < BOOTSTRAP_MIN_N:
        return (mean, float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = arr[indices].mean(axis=1)

    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return mean, lo, hi


def sign_test(deltas: list[float]) -> float:
    """Two-sided sign test on Δ vs 0. Ties (delta == 0) excluded.

    Returns p-value. If all values are ties, returns 1.0.
    """
    nonzero = [d for d in deltas if d != 0.0]
    n = len(nonzero)
    if n == 0:
        return 1.0

    n_pos = sum(1 for d in nonzero if d > 0)
    result = scipy_stats.binomtest(n_pos, n, p=0.5, alternative="two-sided")
    return float(result.pvalue)


def wilcoxon_signed_rank(deltas: list[float]) -> float | None:
    """Wilcoxon signed-rank, two-sided. Returns p-value, or None if n < WILCOXON_MIN_N.

    All-zero or all-equal arrays raise inside scipy; we return None in that case as well.
    """
    if len(deltas) < WILCOXON_MIN_N:
        return None

    if all(d == 0.0 for d in deltas):
        return None

    try:
        result = scipy_stats.wilcoxon(deltas, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        return None
    return float(result.pvalue)


def bootstrap_median_ci(
    deltas: list[float], n_bootstrap: int = 5000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Paired bootstrap on deltas — resample, recompute MEDIAN, take CI percentiles.

    For n < BOOTSTRAP_MIN_N, returns (median, NaN, NaN).
    """
    _validate_bootstrap(deltas, n_bootstrap, alpha)
    n = len(deltas)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))

    arr = np.asarray(deltas, dtype=float)
    median = float(np.median(arr))

    if n < BOOTSTRAP_MIN_N:
        return (median, float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_medians = np.median(arr[indices], axis=1)

    lo = float(np.percentile(boot_medians, 100 * alpha / 2))
    hi = float(np.percentile(boot_medians, 100 * (1 - alpha / 2)))
    return median, lo, hi


def minimum_detectable_effect(sd: float, n: int, power: float = 0.8, alpha: float = 0.05) -> float:
    """The smallest true effect this many pairs could reliably detect.

    ``(z_{1-alpha/2} + z_{power}) * sd / sqrt(n)`` -- the standard two-sided paired-test
    MDE. Reported alongside every comparison because a null result is ambiguous without
    it: "no significant difference at n=40" and "no difference" are very different
    claims, and only the MDE says which one the data supports. A confidence interval
    that comfortably contains an effect the size of the MDE has not ruled that effect
    out.

    Returns NaN for a non-positive n rather than dividing by zero, so a caller that
    dropped every pair gets a missing number instead of a crash.
    """
    if not 0 < alpha < 1 or not 0.5 < power < 1:
        raise ValueError("alpha must be in (0, 1), power in (0.5, 1)")
    if not math.isfinite(sd) or sd < 0:
        raise ValueError("sd must be finite and nonnegative")
    if n <= 0:
        return float("nan")
    z_alpha = float(scipy_stats.norm.ppf(1 - alpha / 2))
    z_power = float(scipy_stats.norm.ppf(power))
    return (z_alpha + z_power) * sd / math.sqrt(n)


def paired_summary(deltas: Sequence[float], n_bootstrap: int = 5000, seed: int = 0) -> dict[str, Any]:
    """Everything worth saying about one set of paired deltas, in one call.

    Mean and median with bootstrap CIs, the two non-parametric tests, and the MDE at the
    surviving n. Both centres because they disagree informatively: a mean that moves
    while the median does not is a handful of questions, not a systematic shift, and
    reporting only one hides which case you are in.

    Non-finite deltas are refused. Callers must account for invalid scores in their
    scheduled-question denominator before computing this summary.
    """
    _validate_bootstrap(deltas, n_bootstrap, 0.05)
    clean = list(deltas)
    n = len(clean)
    mean_delta, mean_lo, mean_hi = bootstrap_mean_ci(clean, n_bootstrap=n_bootstrap, seed=seed)
    median_delta, median_lo, median_hi = bootstrap_median_ci(clean, n_bootstrap=n_bootstrap, seed=seed)
    sd = float(np.std(clean, ddof=1)) if n > 1 else float("nan")
    return {
        "n": n,
        "mean_delta": mean_delta,
        "median_delta": median_delta,
        "mean_ci_low": mean_lo,
        "mean_ci_high": mean_hi,
        "median_ci_low": median_lo,
        "median_ci_high": median_hi,
        "sign_test_p": sign_test(clean),
        "wilcoxon_p": wilcoxon_signed_rank(clean),
        "mde": minimum_detectable_effect(sd=sd, n=n) if n > 1 else float("nan"),
        "inference_status": "available" if n >= BOOTSTRAP_MIN_N else "insufficient_pairs",
    }
