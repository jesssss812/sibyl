"""Known-answer tests for the shared paired statistics.

Every case here has an answer derivable without running the code -- all-zero deltas, a
constant positive shift, a symmetric set. That is deliberate: these functions are what
the eval layer uses to decide whether a change helped, so a test that merely records
what they currently return would certify a bug as readily as a fix.
"""

from __future__ import annotations

import math

import pytest

from sibyl.eval.stats import (
    BOOTSTRAP_MIN_N,
    WILCOXON_MIN_N,
    bootstrap_mean_ci,
    bootstrap_median_ci,
    minimum_detectable_effect,
    paired_summary,
    sign_test,
    wilcoxon_signed_rank,
)


class TestBootstrapMeanCi:
    def test_all_zero_deltas_give_a_zero_ci(self) -> None:
        point, lo, hi = bootstrap_mean_ci([0.0] * 20, n_bootstrap=200, seed=0)
        assert (point, lo, hi) == (0.0, 0.0, 0.0)

    def test_a_constant_shift_is_recovered_exactly(self) -> None:
        # Every resample of a constant is that constant, so the CI has zero width.
        point, lo, hi = bootstrap_mean_ci([0.25] * 20, n_bootstrap=200, seed=0)
        assert point == pytest.approx(0.25)
        assert lo == pytest.approx(0.25) and hi == pytest.approx(0.25)

    def test_ci_contains_zero_for_a_symmetric_sample(self) -> None:
        point, lo, hi = bootstrap_mean_ci([-0.2, -0.1, 0.0, 0.1, 0.2] * 4, n_bootstrap=500, seed=0)
        assert point == pytest.approx(0.0, abs=1e-12)
        assert lo <= 0.0 <= hi

    def test_ci_excludes_zero_for_a_clearly_positive_sample(self) -> None:
        deltas = [0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19]
        _point, lo, hi = bootstrap_mean_ci(deltas, n_bootstrap=1000, seed=0)
        assert lo > 0.0 and hi > 0.0

    def test_empty_is_nan_not_an_exception(self) -> None:
        assert all(math.isnan(v) for v in bootstrap_mean_ci([]))

    def test_below_the_minimum_the_ci_is_unavailable(self) -> None:
        # Small samples have a point estimate, but do not earn an interval.
        deltas = [0.1, 0.2, 0.3, 0.4][: BOOTSTRAP_MIN_N - 1]
        point, lo, hi = bootstrap_mean_ci(deltas, seed=0)
        assert point == pytest.approx(sum(deltas) / len(deltas))
        assert math.isnan(lo) and math.isnan(hi)

    def test_below_the_minimum_the_seed_cannot_matter(self) -> None:
        deltas = [0.1, 0.2, 0.3, 0.4]
        assert bootstrap_mean_ci(deltas, seed=0)[0] == bootstrap_mean_ci(deltas, seed=99)[0]

    def test_same_seed_reproduces_across_calls(self) -> None:
        deltas = [0.05, -0.02, 0.01, 0.0, -0.04, 0.03, 0.02, -0.01]
        assert bootstrap_mean_ci(deltas, n_bootstrap=300, seed=7) == bootstrap_mean_ci(deltas, n_bootstrap=300, seed=7)


class TestBootstrapMedianCi:
    def test_all_zero_deltas_give_a_zero_ci(self) -> None:
        assert bootstrap_median_ci([0.0] * 20, n_bootstrap=200, seed=0) == (0.0, 0.0, 0.0)

    def test_median_ignores_an_extreme_outlier_the_mean_would_follow(self) -> None:
        deltas = [0.1] * 20 + [100.0]
        median, _lo, _hi = bootstrap_median_ci(deltas, n_bootstrap=300, seed=0)
        mean, _, _ = bootstrap_mean_ci(deltas, n_bootstrap=300, seed=0)
        assert median == pytest.approx(0.1)
        assert mean > 4.0

    def test_empty_is_nan(self) -> None:
        assert all(math.isnan(v) for v in bootstrap_median_ci([]))


class TestSignTest:
    def test_all_ties_give_p_of_one(self) -> None:
        assert sign_test([0.0] * 10) == 1.0

    def test_empty_gives_p_of_one(self) -> None:
        assert sign_test([]) == 1.0

    def test_a_perfect_split_is_not_significant(self) -> None:
        assert sign_test([1.0, -1.0, 1.0, -1.0]) == pytest.approx(1.0)

    def test_all_positive_is_the_binomial_tail(self) -> None:
        # 10 of 10 positive, two-sided: 2 * 0.5**10 = 1/512.
        assert sign_test([0.5] * 10) == pytest.approx(2 / 1024)

    def test_ties_are_excluded_not_counted_as_negative(self) -> None:
        # Six positives and four ties must read as 6/6, not 6/10.
        assert sign_test([1.0] * 6 + [0.0] * 4) == pytest.approx(sign_test([1.0] * 6))


class TestWilcoxon:
    def test_returns_none_below_the_minimum_n(self) -> None:
        assert wilcoxon_signed_rank([0.1] * (WILCOXON_MIN_N - 1)) is None

    def test_returns_none_for_all_zero_deltas(self) -> None:
        assert wilcoxon_signed_rank([0.0] * 10) is None

    def test_a_clear_shift_is_significant(self) -> None:
        p = wilcoxon_signed_rank([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        assert p is not None and p < 0.05


class TestMinimumDetectableEffect:
    def test_shrinks_as_n_grows(self) -> None:
        # MDE goes as 1/sqrt(n): quadrupling n halves it.
        assert minimum_detectable_effect(sd=1.0, n=100) == pytest.approx(2 * minimum_detectable_effect(sd=1.0, n=400))

    def test_scales_linearly_with_sd(self) -> None:
        assert minimum_detectable_effect(sd=2.0, n=50) == pytest.approx(2 * minimum_detectable_effect(sd=1.0, n=50))

    def test_more_power_demands_a_bigger_effect(self) -> None:
        assert minimum_detectable_effect(sd=1.0, n=50, power=0.9) > minimum_detectable_effect(sd=1.0, n=50, power=0.8)

    def test_a_stricter_alpha_demands_a_bigger_effect(self) -> None:
        assert minimum_detectable_effect(sd=1.0, n=50, alpha=0.01) > minimum_detectable_effect(sd=1.0, n=50, alpha=0.05)

    def test_known_value_at_the_standard_settings(self) -> None:
        # (1.959964 + 0.841621) * 1 / sqrt(100) = 0.2801585
        assert minimum_detectable_effect(sd=1.0, n=100) == pytest.approx(0.2801585, abs=1e-6)

    @pytest.mark.parametrize("n", [0, -1])
    def test_non_positive_n_is_nan_rather_than_a_zero_division(self, n: int) -> None:
        assert math.isnan(minimum_detectable_effect(sd=1.0, n=n))

    def test_zero_sd_means_any_effect_is_detectable(self) -> None:
        assert minimum_detectable_effect(sd=0.0, n=50) == 0.0


class TestPairedSummary:
    def test_all_zero_deltas_report_no_effect(self) -> None:
        summary = paired_summary([0.0] * 20, n_bootstrap=200, seed=0)
        assert summary["n"] == 20
        assert summary["mean_delta"] == 0.0
        assert summary["median_delta"] == 0.0
        assert summary["mean_ci_low"] <= 0.0 <= summary["mean_ci_high"]
        assert summary["sign_test_p"] == 1.0

    def test_a_constant_positive_delta_excludes_zero(self) -> None:
        summary = paired_summary([0.3] * 20, n_bootstrap=200, seed=0)
        assert summary["mean_delta"] == pytest.approx(0.3)
        assert summary["mean_ci_low"] > 0.0
        assert summary["sign_test_p"] < 0.001

    def test_reports_every_documented_key(self) -> None:
        summary = paired_summary([0.1, -0.2, 0.3, 0.0, 0.15, -0.05, 0.2], n_bootstrap=200, seed=0)
        assert set(summary) == {
            "n",
            "mean_delta",
            "median_delta",
            "mean_ci_low",
            "mean_ci_high",
            "median_ci_low",
            "median_ci_high",
            "sign_test_p",
            "wilcoxon_p",
            "mde",
            "inference_status",
        }

    def test_nonfinite_deltas_require_explicit_upstream_exclusion(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            paired_summary([0.2] * 10 + [float("nan")], seed=0)

    def test_empty_input_is_reported_not_raised(self) -> None:
        summary = paired_summary([], n_bootstrap=200, seed=0)
        assert summary["n"] == 0
        assert math.isnan(summary["mean_delta"])
        assert summary["wilcoxon_p"] is None

    def test_is_reproducible_for_a_fixed_seed(self) -> None:
        deltas = [0.05, -0.02, 0.01, 0.0, -0.04, 0.03, 0.02, -0.01, 0.06, -0.03]
        assert paired_summary(deltas, n_bootstrap=300, seed=1) == paired_summary(deltas, n_bootstrap=300, seed=1)

    def test_the_mde_uses_the_surviving_pair_count(self) -> None:
        deltas = [0.2] * 10
        summary = paired_summary(deltas, n_bootstrap=200, seed=0)
        assert summary["mde"] == pytest.approx(minimum_detectable_effect(sd=0.0, n=10), abs=1e-12)
