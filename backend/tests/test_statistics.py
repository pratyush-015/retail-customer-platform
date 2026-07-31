"""Tests for the hypothesis-testing module.

These target the pure statistical functions with hand-constructed inputs whose
correct answers are known in advance. Two properties matter most and are
asserted repeatedly:

1. A test must not claim significance where none exists (no false positives on
   identical groups), and must find it where the separation is obvious.
2. Effect size must be reported and labelled correctly, because that is what
   stops the dashboard from over-claiming on a p-value alone.

The warehouse-backed wrappers (``aov_by_country`` and friends) are exercised
against the real database by ``scripts/run_stats.py`` and are covered again in
the API test phase, where a full star-schema fixture exists.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from app.core.exceptions import InsufficientDataError, StatisticalTestError
from app.services.statistics import (
    ALPHA,
    AVAILABLE_TESTS,
    MIN_GROUP_SIZE,
    TestResult,
    chi_square_independence,
    independent_ttest,
    one_way_anova,
)


# ---------------------------------------------------------------------------
# Deterministic sample builders - no RNG, so failures are always reproducible
# ---------------------------------------------------------------------------
def sample(mean: float, spread: float = 1.0, n: int = 60) -> np.ndarray:
    """n values centred exactly on ``mean``, alternating +/- ``spread``.

    Mean is exact and the standard deviation is known, which makes Cohen's d
    analytically predictable rather than something the test has to trust.
    """
    half = n // 2
    return np.array([mean - spread, mean + spread] * half, dtype=float)


# ---------------------------------------------------------------------------
# t-test: no difference
# ---------------------------------------------------------------------------
def test_identical_groups_are_not_significant():
    """The most important negative case: same data must not look different."""
    values = sample(10.0)

    result = independent_ttest(values, values.copy(), label_a="A", label_b="B")

    assert result.statistic == 0.0
    assert result.p_value == pytest.approx(1.0)
    assert result.significant is False
    assert result.effect_size == 0.0
    assert result.effect_size_label == "negligible"


def test_non_significant_interpretation_avoids_claiming_equality():
    """Failing to reject H0 is not proof of no difference; the text must say so."""
    values = sample(10.0)
    result = independent_ttest(values, values.copy(), label_a="A", label_b="B")

    assert "no statistically significant difference" in result.interpretation
    assert "not proof they are equal" in result.interpretation


# ---------------------------------------------------------------------------
# t-test: real difference, exact effect size
# ---------------------------------------------------------------------------
def test_clear_difference_is_detected_with_exact_cohens_d():
    """A two-unit shift with SD ~1.008 gives d = -2/1.008 = -1.983.

    Computed by hand: both groups have deviations of exactly +/-1 around their
    mean, so var = n/(n-1) = 60/59 and pooled SD = sqrt(60/59) = 1.00844.
    """
    a = sample(10.0)  # mean 10
    b = sample(12.0)  # mean 12

    result = independent_ttest(a, b, label_a="Low", label_b="High")

    expected_pooled_sd = np.sqrt(60 / 59)
    expected_d = (10.0 - 12.0) / expected_pooled_sd

    assert result.effect_size == pytest.approx(expected_d, rel=1e-3)
    assert result.effect_size == pytest.approx(-1.9833, abs=1e-3)
    assert result.effect_size_label == "large"
    assert result.significant is True
    assert result.p_value < 0.001


def test_direction_is_reported_correctly():
    """Group A lower than B must read "lower", not "higher"."""
    low = independent_ttest(sample(10.0), sample(20.0), label_a="A", label_b="B")
    high = independent_ttest(sample(20.0), sample(10.0), label_a="A", label_b="B")

    assert "significantly lower" in low.interpretation
    assert "significantly higher" in high.interpretation
    # Sign of d must flip with the comparison order.
    assert low.effect_size < 0 < high.effect_size


def test_group_means_appear_in_the_summary():
    result = independent_ttest(sample(10.0), sample(12.0), label_a="A", label_b="B")

    by_name = {g["name"]: g for g in result.groups}
    assert by_name["A"]["mean"] == 10.0
    assert by_name["B"]["mean"] == 12.0
    assert by_name["A"]["n"] == 60


# ---------------------------------------------------------------------------
# t-test: the significant-but-trivial case this module exists to catch
# ---------------------------------------------------------------------------
def test_significant_but_negligible_effect_is_flagged_as_weak():
    """Large n + tiny difference = significant p, meaningless effect.

    This is the Germany-vs-UK situation in the real data. The interpretation
    must actively warn the reader instead of reporting "significant" and
    letting them conclude the difference matters.
    """
    # 4000 points per group, means differ by 0.02 with SD ~1.
    a = sample(10.00, spread=1.0, n=4000)
    b = sample(10.05, spread=1.0, n=4000)

    result = independent_ttest(a, b, label_a="Germany", label_b="UK")

    assert result.significant is True, "expected significance from the large n"
    assert result.effect_size_label in {"negligible", "small"}
    assert "small in practical terms" in result.interpretation
    assert "weak signal" in result.interpretation


def test_large_effect_is_described_as_commercially_meaningful():
    result = independent_ttest(
        sample(10.0), sample(20.0), label_a="A", label_b="B"
    )
    assert result.effect_size_label == "large"
    assert "commercially" in result.interpretation


# ---------------------------------------------------------------------------
# t-test: guards and assumptions
# ---------------------------------------------------------------------------
def test_small_groups_raise_rather_than_return_a_meaningless_p_value():
    tiny = sample(10.0, n=10)

    with pytest.raises(InsufficientDataError) as exc_info:
        independent_ttest(tiny, sample(12.0), label_a="Tiny", label_b="Big")

    error = exc_info.value
    assert error.details["n_a"] == 10
    assert error.details["minimum_required"] == MIN_GROUP_SIZE
    # The message must name the groups so the caller knows which one failed.
    assert "Tiny" in error.message


@pytest.mark.parametrize("n", [MIN_GROUP_SIZE - 1, 0])
def test_group_below_threshold_always_raises(n):
    with pytest.raises(InsufficientDataError):
        independent_ttest(
            sample(10.0, n=n) if n else np.array([]),
            sample(12.0),
            label_a="A",
            label_b="B",
        )


def test_welch_correction_is_recorded_in_assumptions():
    """Unequal variances are the norm here, so Welch must be used and stated."""
    narrow = sample(10.0, spread=1.0)
    wide = sample(10.0, spread=20.0)

    result = independent_ttest(narrow, wide, label_a="Narrow", label_b="Wide")

    assert result.test_type == "welch_ttest"
    assert result.assumptions["welch_correction_applied"] is True
    assert result.assumptions["equal_variance_assumed"] is False
    # Variance ratio should be roughly 20^2 = 400.
    assert result.assumptions["variance_ratio"] == pytest.approx(400, rel=0.01)


def test_nan_values_are_excluded_not_propagated():
    """A NaN must not turn the whole statistic into NaN."""
    a = np.concatenate([sample(10.0), [np.nan, np.nan]])
    b = sample(12.0)

    result = independent_ttest(a, b, label_a="A", label_b="B")

    assert np.isfinite(result.statistic)
    assert result.groups[0]["n"] == 60, "NaNs should not be counted"


def test_alpha_is_respected():
    """A borderline result must flip when the threshold moves."""
    a = sample(10.00, spread=1.0, n=200)
    b = sample(10.28, spread=1.0, n=200)

    lenient = independent_ttest(a, b, label_a="A", label_b="B", alpha=0.05)
    strict = independent_ttest(a, b, label_a="A", label_b="B", alpha=0.000001)

    assert lenient.p_value == strict.p_value  # same data, same p
    assert lenient.significant is True
    assert strict.significant is False, "a stricter alpha must be harder to pass"


# ---------------------------------------------------------------------------
# ANOVA
# ---------------------------------------------------------------------------
def test_anova_on_identical_groups_is_not_significant():
    groups = {name: sample(10.0) for name in ("A", "B", "C")}

    result = one_way_anova(groups)

    assert result.statistic == pytest.approx(0.0)
    assert result.significant is False
    assert result.effect_size == pytest.approx(0.0)


def test_anova_detects_separated_groups():
    groups = {"Low": sample(10.0), "Mid": sample(20.0), "High": sample(30.0)}

    result = one_way_anova(groups, factor="tier")

    assert result.significant is True
    assert result.effect_size_label == "large"
    assert result.details["highest_group"] == "High"
    assert result.details["lowest_group"] == "Low"
    assert result.degrees_of_freedom == 2.0  # k - 1
    assert result.details["df_within"] == 177  # 180 - 3


def test_anova_eta_squared_is_variance_explained():
    """eta-squared must equal between-group SS / total SS."""
    groups = {"A": sample(10.0), "B": sample(20.0), "C": sample(30.0)}
    result = one_way_anova(groups)

    # Between-group variance dominates here: spread of means (10/20/30) is far
    # larger than the +/-1 within-group spread, so eta-sq should be near 1.
    assert 0.9 < result.effect_size <= 1.0
    assert result.details["variance_explained_pct"] == pytest.approx(
        result.effect_size * 100, rel=1e-6
    )


def test_anova_groups_are_sorted_by_mean_descending():
    """The dashboard renders this list directly, so order must be meaningful."""
    groups = {"Mid": sample(20.0), "High": sample(30.0), "Low": sample(10.0)}
    result = one_way_anova(groups)

    means = [g["mean"] for g in result.groups]
    assert means == sorted(means, reverse=True)
    assert result.groups[0]["name"] == "High"


def test_anova_excludes_small_groups_and_says_which():
    """Dropping a group silently would change the test without explanation."""
    groups = {
        "A": sample(10.0),
        "B": sample(20.0),
        "C": sample(30.0),
        "Tiny": sample(15.0, n=4),
    }

    result = one_way_anova(groups)

    assert result.assumptions["excluded_groups"] == ["Tiny"]
    assert str(MIN_GROUP_SIZE) in result.assumptions["exclusion_reason"]
    assert {g["name"] for g in result.groups} == {"A", "B", "C"}


def test_anova_needs_three_qualifying_groups():
    with pytest.raises(InsufficientDataError) as exc_info:
        one_way_anova({"A": sample(10.0), "B": sample(20.0)})

    assert "t-test" in exc_info.value.message  # points at the right alternative
    assert exc_info.value.details["qualifying_groups"] == ["A", "B"]


def test_anova_rejects_when_groups_are_too_small_even_if_numerous():
    groups = {f"G{i}": sample(10.0 + i, n=5) for i in range(6)}

    with pytest.raises(InsufficientDataError):
        one_way_anova(groups)


# ---------------------------------------------------------------------------
# Chi-square
# ---------------------------------------------------------------------------
def test_perfectly_independent_table_is_not_significant():
    """Identical proportions in every row means no association at all."""
    table = pd.DataFrame(
        {"Active": [50, 50], "Churned": [50, 50]}, index=["UK", "Germany"]
    )

    result = chi_square_independence(table, row_label="Country", column_label="Churn")

    assert result.statistic == pytest.approx(0.0)
    assert result.p_value == pytest.approx(1.0)
    assert result.significant is False
    assert "appear independent" in result.interpretation


def test_strong_association_is_detected():
    """A near-diagonal table is a textbook strong association."""
    table = pd.DataFrame(
        {"Active": [90, 10], "Churned": [10, 90]}, index=["Loyal", "At Risk"]
    )

    result = chi_square_independence(
        table, row_label="Segment", column_label="Churn status"
    )

    assert result.significant is True
    # scipy applies the Yates continuity correction to 2x2 tables, which pulls
    # chi2 from 128.0 down to ~124.8, so V lands near 0.79 rather than 0.80.
    assert result.effect_size == pytest.approx(0.79, abs=0.01)
    assert result.effect_size_label == "large"
    assert result.degrees_of_freedom == 1.0


def test_chi_square_reports_the_most_deviant_cell():
    table = pd.DataFrame(
        {"Active": [90, 10], "Churned": [10, 90]}, index=["Loyal", "At Risk"]
    )
    result = chi_square_independence(
        table, row_label="Segment", column_label="Churn status"
    )

    # The interpretation must point somewhere concrete.
    assert any(name in result.interpretation for name in ("Loyal", "At Risk"))
    assert "more cases" in result.interpretation or "fewer cases" in result.interpretation


def test_chi_square_returns_observed_and_expected_counts():
    table = pd.DataFrame(
        {"Active": [80, 20], "Churned": [20, 80]}, index=["A", "B"]
    )
    result = chi_square_independence(table, row_label="R", column_label="C")

    assert result.details["observed"]["A"]["Active"] == 80
    # Under independence every cell expects 50 here (balanced margins).
    assert result.details["expected"]["A"]["Active"] == pytest.approx(50.0)


def test_sparse_table_raises_instead_of_reporting_an_invalid_p_value():
    """Expected counts under 5 invalidate the approximation - must refuse."""
    table = pd.DataFrame({"Active": [2, 1], "Churned": [1, 2]}, index=["A", "B"])

    with pytest.raises(StatisticalTestError) as exc_info:
        chi_square_independence(table, row_label="R", column_label="C")

    error = exc_info.value
    assert error.details["cells_below_5"] == 4
    assert "unreliable" in error.message
    # Must suggest a remedy, not just refuse.
    assert "Combine" in error.message


@pytest.mark.parametrize(
    "table",
    [
        pd.DataFrame({"Active": [10]}, index=["A"]),          # 1x1
        pd.DataFrame({"Active": [10], "Churned": [20]}, index=["A"]),  # 1x2
        pd.DataFrame(),                                        # empty
    ],
)
def test_degenerate_tables_raise(table):
    with pytest.raises(StatisticalTestError):
        chi_square_independence(table, row_label="R", column_label="C")


def test_chi_square_handles_larger_tables():
    """Segment x churn in the real data is 11x2, so k>2 rows must work."""
    table = pd.DataFrame(
        {
            "Active": [100, 80, 60, 40, 20],
            "Churned": [20, 40, 60, 80, 100],
        },
        index=["Champions", "Loyal", "Promising", "At Risk", "Lost"],
    )

    result = chi_square_independence(
        table, row_label="Segment", column_label="Churn status"
    )

    assert result.degrees_of_freedom == 4.0  # (5-1)(2-1)
    assert result.significant is True
    assert len(result.groups) == 5


# ---------------------------------------------------------------------------
# Result serialisation
# ---------------------------------------------------------------------------
def test_result_is_json_serialisable():
    """The API returns this dict directly, so it must survive json.dumps."""
    result = independent_ttest(sample(10.0), sample(12.0), label_a="A", label_b="B")

    payload = json.dumps(result.to_dict())
    restored = json.loads(payload)

    assert restored["test_type"] == "welch_ttest"
    assert restored["significant"] is True
    assert isinstance(restored["groups"], list)


def test_non_finite_floats_become_null_not_nan():
    """json.dumps emits bare NaN, which is invalid JSON and breaks the client."""
    result = TestResult(
        test_type="t",
        question="q",
        null_hypothesis="h",
        statistic=float("nan"),
        p_value=float("inf"),
        significant=False,
        alpha=ALPHA,
        interpretation="i",
        effect_size=float("-inf"),
    )

    payload = result.to_dict()
    assert payload["statistic"] is None
    assert payload["p_value"] is None
    assert payload["effect_size"] is None

    # Strict mode rejects NaN/Infinity, proving the output is valid JSON.
    assert json.dumps(payload, allow_nan=False)


def test_every_result_carries_a_question_and_hypothesis():
    """The Statistical Insights page renders these as plain-language headings."""
    results = [
        independent_ttest(sample(10.0), sample(12.0), label_a="A", label_b="B"),
        one_way_anova({"A": sample(10.0), "B": sample(20.0), "C": sample(30.0)}),
        chi_square_independence(
            pd.DataFrame({"Active": [90, 10], "Churned": [10, 90]}, index=["X", "Y"]),
            row_label="R",
            column_label="C",
        ),
    ]

    for result in results:
        assert result.question.endswith("?")
        assert result.null_hypothesis
        assert len(result.interpretation) > 40, "interpretation is too thin to be useful"
        assert result.effect_size is not None
        assert result.effect_size_name


# ---------------------------------------------------------------------------
# Test catalogue
# ---------------------------------------------------------------------------
def test_available_tests_registry_is_complete():
    """The API exposes this catalogue, so every entry needs full metadata."""
    assert len(AVAILABLE_TESTS) == 5

    for name, meta in AVAILABLE_TESTS.items():
        assert meta["test"] in {
            "welch_ttest", "one_way_anova", "chi_square_independence",
        }, f"{name} names an unknown test type"
        assert meta["title"]
        assert meta["description"]
        assert "parameters" in meta
