"""Paired statistics, shared across the eval layer.

These four primitives were defined inside ``eval/ablation/scoring.py`` and used only
there. They are not ablation-specific: a paired bootstrap over a list of deltas is the
same operation whether the pairs come from two stacker arms, two backtest runs, or two
model rosters, and ``eval compare`` needs exactly the same machinery. Moved here so the
second consumer is a call rather than a copy.

The move is byte-for-byte -- ``ablation/scoring.py`` re-exports them, so ablation and its
tests are untouched and the recorded ablation numbers cannot shift. Anything new (the
MDE, ``paired_summary``) is added here rather than back-ported into ablation.

Seeded throughout. A bootstrap CI that changes between runs is not a measurement, and an
eval layer whose own numbers wobble cannot be used to judge anything else.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as scipy_stats

__all__ = [
    "BOOTSTRAP_MIN_N",
    "WILCOXON_MIN_N",
    "bootstrap_mean_ci",
    "bootstrap_median_ci",
    "sign_test",
    "wilcoxon_signed_rank",
]

# Below this many pairs a bootstrap percentile interval is resampling noise rather than
# a sampling distribution, so the CI collapses to the point estimate instead of
# pretending to a width it has not earned.
BOOTSTRAP_MIN_N = 5
# scipy's Wilcoxon is not meaningful below this n (its normal approximation and exact
# tables both need more), so it returns None rather than a number nobody should read.
WILCOXON_MIN_N = 6


def bootstrap_mean_ci(
    deltas: list[float], n_bootstrap: int = 5000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Paired bootstrap on the deltas: resample with replacement, recompute mean, take CI percentiles.

    For n < 5, returns (mean, mean, mean) — too few samples for a meaningful CI.
    """
    n = len(deltas)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))

    arr = np.asarray(deltas, dtype=float)
    mean = float(arr.mean())

    if n < BOOTSTRAP_MIN_N:
        return (mean, mean, mean)

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

    For n < BOOTSTRAP_MIN_N, returns (median, median, median).
    """
    n = len(deltas)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))

    arr = np.asarray(deltas, dtype=float)
    median = float(np.median(arr))

    if n < BOOTSTRAP_MIN_N:
        return (median, median, median)

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_medians = np.median(arr[indices], axis=1)

    lo = float(np.percentile(boot_medians, 100 * alpha / 2))
    hi = float(np.percentile(boot_medians, 100 * (1 - alpha / 2)))
    return median, lo, hi
