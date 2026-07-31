"""Hypothesis testing over the warehouse.

Why this module exists
----------------------
A dashboard that says "Germany's AOV is 32.10 and the UK's is 28.40" invites the
reader to conclude Germany is better. With n=95 vs n=4,000 and a long right
tail, that gap may be indistinguishable from noise. These tests attach a p-value
and, more importantly, an **effect size** and a **plain-language verdict** to
every such comparison, so the dashboard states what the data supports rather
than what it merely suggests.

Statistical decisions worth knowing
-----------------------------------
1. **Welch's t-test, not Student's.** Country order values have wildly unequal
   variances and group sizes. Student's t-test assumes equal variance and is
   badly anti-conservative when that fails; Welch's does not, and costs nothing
   when variances happen to be equal. It is the correct default.

2. **Effect size is always reported.** With ~800k order lines almost any
   difference reaches p < 0.05. Cohen's d (t-test) and eta-squared (ANOVA)
   answer the question that actually matters: is the difference *large enough
   to act on*? A result can be highly significant and commercially irrelevant,
   and the interpretation string says so explicitly.

3. **Assumptions are checked and reported, not assumed.** Normality (via
   skewness) and variance ratios are computed and surfaced. The tests are
   reasonably robust at these sample sizes by the CLT, but hiding a violation
   is how misleading conclusions get published.

4. **Analysis is at the order level, not the line level.** AOV means "value of
   an order", so lines are aggregated to invoices first. Testing raw lines
   would inflate n by ~7x and shrink every p-value toward zero - a textbook
   pseudo-replication error.

5. **Cancellations are excluded from AOV.** A returned order has negative value
   and is not a purchase decision. They stay in the warehouse and in revenue
   totals, but they are not observations of "what a customer chose to spend".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import func, select
from sqlalchemy.engine import Engine

from app.core.exceptions import InsufficientDataError, StatisticalTestError
from app.core.logging_config import get_logger
from app.models.star_schema import DimCountry, DimCustomer, FactSales

logger = get_logger(__name__)

ALPHA = 0.05

# Below this, group means are too unstable to compare and the t-test's
# normality assumption stops being rescued by the CLT.
MIN_GROUP_SIZE = 30


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class TestResult:
    """One hypothesis test, in a shape the API can return directly."""

    # pytest collects any class named Test*; this is a result container, not a
    # test case, and without this it emits a PytestCollectionWarning.
    __test__ = False

    test_type: str
    question: str
    null_hypothesis: str
    statistic: float
    p_value: float
    significant: bool
    alpha: float
    interpretation: str
    effect_size: float | None = None
    effect_size_name: str | None = None
    effect_size_label: str | None = None
    degrees_of_freedom: float | None = None
    groups: list[dict[str, Any]] = field(default_factory=list)
    assumptions: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        # JSON has no NaN/Infinity. scipy returns NaN for degenerate inputs, and
        # emitting a bare NaN produces invalid JSON that breaks the frontend
        # parser rather than the chart.
        for key, value in payload.items():
            if isinstance(value, float) and not np.isfinite(value):
                payload[key] = None
        return payload


# ---------------------------------------------------------------------------
# Effect-size interpretation
# ---------------------------------------------------------------------------
def _label_cohens_d(d: float) -> str:
    """Cohen's conventional thresholds for |d|."""
    magnitude = abs(d)
    if magnitude < 0.2:
        return "negligible"
    if magnitude < 0.5:
        return "small"
    if magnitude < 0.8:
        return "medium"
    return "large"


def _label_eta_squared(eta_sq: float) -> str:
    if eta_sq < 0.01:
        return "negligible"
    if eta_sq < 0.06:
        return "small"
    if eta_sq < 0.14:
        return "medium"
    return "large"


def _label_cramers_v(v: float) -> str:
    if v < 0.1:
        return "negligible"
    if v < 0.3:
        return "small"
    if v < 0.5:
        return "medium"
    return "large"


def _describe(values: np.ndarray, name: str) -> dict[str, Any]:
    """Summary statistics for one group, for the API and the chart."""
    return {
        "name": name,
        "n": int(values.size),
        "mean": round(float(np.mean(values)), 2),
        "median": round(float(np.median(values)), 2),
        "std": round(float(np.std(values, ddof=1)), 2) if values.size > 1 else 0.0,
        "min": round(float(np.min(values)), 2),
        "max": round(float(np.max(values)), 2),
    }


# ---------------------------------------------------------------------------
# Two-sample comparison
# ---------------------------------------------------------------------------
def independent_ttest(
    group_a: np.ndarray | pd.Series,
    group_b: np.ndarray | pd.Series,
    *,
    label_a: str,
    label_b: str,
    metric: str = "average order value",
    alpha: float = ALPHA,
) -> TestResult:
    """Welch's two-sample t-test with Cohen's d.

    Raises
    ------
    InsufficientDataError
        If either group has fewer than ``MIN_GROUP_SIZE`` observations. Saying
        "not enough data" is far more useful than returning a p-value computed
        from eleven orders.
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if a.size < MIN_GROUP_SIZE or b.size < MIN_GROUP_SIZE:
        raise InsufficientDataError(
            f"Comparing {label_a!r} and {label_b!r} needs at least "
            f"{MIN_GROUP_SIZE} observations each; got {a.size} and {b.size}. "
            "With samples this small the test would not be meaningful.",
            details={
                "group_a": label_a, "n_a": int(a.size),
                "group_b": label_b, "n_b": int(b.size),
                "minimum_required": MIN_GROUP_SIZE,
            },
        )

    # equal_var=False -> Welch. See module docstring.
    statistic, p_value = stats.ttest_ind(a, b, equal_var=False)

    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    var_a, var_b = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))

    # Pooled SD for Cohen's d. Guarded because two constant groups give 0.
    pooled_sd = np.sqrt(((a.size - 1) * var_a + (b.size - 1) * var_b)
                        / (a.size + b.size - 2))
    cohens_d = (mean_a - mean_b) / pooled_sd if pooled_sd > 0 else 0.0

    # Welch-Satterthwaite degrees of freedom.
    df = ((var_a / a.size + var_b / b.size) ** 2 / (
        (var_a / a.size) ** 2 / (a.size - 1) + (var_b / b.size) ** 2 / (b.size - 1)
    )) if a.size > 1 and b.size > 1 else float("nan")

    significant = bool(p_value < alpha)
    difference = mean_a - mean_b
    direction = "higher" if difference > 0 else "lower"
    effect_label = _label_cohens_d(cohens_d)

    if significant:
        interpretation = (
            f"{label_a} has a significantly {direction} {metric} than {label_b} "
            f"({mean_a:,.2f} vs {mean_b:,.2f}, a difference of "
            f"{abs(difference):,.2f}; p={p_value:.4f}). "
        )
        if effect_label in {"negligible", "small"}:
            # The single most important sentence in this module: at n in the
            # thousands, statistical significance is cheap. Without this caveat
            # the dashboard would overstate a difference no one should act on.
            interpretation += (
                f"However, the effect size is {effect_label} (Cohen's d="
                f"{cohens_d:.2f}), so although the difference is unlikely to be "
                "chance, it is small in practical terms - treat it as a weak "
                "signal rather than a basis for reallocating budget."
            )
        else:
            interpretation += (
                f"The effect size is {effect_label} (Cohen's d={cohens_d:.2f}), "
                "so the difference is both statistically and commercially "
                "meaningful."
            )
    else:
        interpretation = (
            f"There is no statistically significant difference in {metric} "
            f"between {label_a} ({mean_a:,.2f}) and {label_b} ({mean_b:,.2f}); "
            f"p={p_value:.4f}, above the {alpha} threshold. The observed gap of "
            f"{abs(difference):,.2f} is consistent with random variation. Note "
            "that this is not proof they are equal - only that the data does "
            "not demonstrate a difference."
        )

    # Skew is reported because a heavily skewed distribution is where a
    # t-test's normality assumption is doing the most work.
    skew_a, skew_b = float(stats.skew(a)), float(stats.skew(b))
    variance_ratio = max(var_a, var_b) / min(var_a, var_b) if min(var_a, var_b) > 0 else float("inf")

    return TestResult(
        test_type="welch_ttest",
        question=f"Does {metric} differ between {label_a} and {label_b}?",
        null_hypothesis=f"Mean {metric} is the same for {label_a} and {label_b}.",
        statistic=round(float(statistic), 4),
        p_value=float(p_value),
        significant=significant,
        alpha=alpha,
        interpretation=interpretation,
        effect_size=round(float(cohens_d), 4),
        effect_size_name="cohens_d",
        effect_size_label=effect_label,
        degrees_of_freedom=round(float(df), 2) if np.isfinite(df) else None,
        groups=[_describe(a, label_a), _describe(b, label_b)],
        assumptions={
            "variance_ratio": round(variance_ratio, 2) if np.isfinite(variance_ratio) else None,
            "equal_variance_assumed": False,
            "welch_correction_applied": True,
            "skewness": {label_a: round(skew_a, 2), label_b: round(skew_b, 2)},
            "note": (
                "Welch's t-test does not assume equal variances. Group sizes "
                "exceed 30, so the Central Limit Theorem makes the test robust "
                "to the right-skew typical of order values."
            ),
        },
        details={"difference_in_means": round(difference, 2)},
    )


# ---------------------------------------------------------------------------
# Multi-group comparison
# ---------------------------------------------------------------------------
def one_way_anova(
    groups: dict[str, np.ndarray | pd.Series],
    *,
    metric: str = "average order value",
    factor: str = "group",
    alpha: float = ALPHA,
) -> TestResult:
    """One-way ANOVA across 3+ groups, with eta-squared.

    ANOVA answers "are these all the same?" - not "which one differs". When it
    is significant, the largest and smallest group means are reported so the
    reader has somewhere concrete to look, but that is descriptive, not a
    post-hoc test.
    """
    usable = {
        name: np.asarray(values, dtype=float)[np.isfinite(np.asarray(values, dtype=float))]
        for name, values in groups.items()
    }
    usable = {name: values for name, values in usable.items() if values.size >= MIN_GROUP_SIZE}

    if len(usable) < 3:
        raise InsufficientDataError(
            f"ANOVA needs at least 3 groups with {MIN_GROUP_SIZE}+ observations "
            f"each; only {len(usable)} qualified. Use a t-test for two groups.",
            details={
                "qualifying_groups": list(usable),
                "supplied_groups": list(groups),
                "minimum_group_size": MIN_GROUP_SIZE,
            },
        )

    samples = list(usable.values())
    statistic, p_value = stats.f_oneway(*samples)

    # eta-squared = between-group SS / total SS.
    all_values = np.concatenate(samples)
    grand_mean = float(np.mean(all_values))
    ss_between = sum(s.size * (float(np.mean(s)) - grand_mean) ** 2 for s in samples)
    ss_total = float(np.sum((all_values - grand_mean) ** 2))
    eta_squared = (ss_between / ss_total) if ss_total > 0 else 0.0

    df_between = len(samples) - 1
    df_within = all_values.size - len(samples)

    significant = bool(p_value < alpha)
    effect_label = _label_eta_squared(eta_squared)

    means = {name: float(np.mean(values)) for name, values in usable.items()}
    highest = max(means, key=means.get)
    lowest = min(means, key=means.get)

    if significant:
        interpretation = (
            f"{metric.capitalize()} differs significantly across {factor}s "
            f"(F={statistic:.2f}, p={p_value:.4f}). {highest} is highest at "
            f"{means[highest]:,.2f} and {lowest} is lowest at {means[lowest]:,.2f}. "
            f"The {factor} explains {eta_squared * 100:.1f}% of the variation in "
            f"{metric} ({effect_label} effect)."
        )
        if effect_label in {"negligible", "small"}:
            interpretation += (
                " Most of the variation therefore comes from differences between "
                "individual customers rather than between groups, so segment-level "
                "targeting on this metric alone would have limited impact."
            )
    else:
        interpretation = (
            f"There is no statistically significant difference in {metric} across "
            f"{factor}s (F={statistic:.2f}, p={p_value:.4f}). The observed spread "
            f"from {means[lowest]:,.2f} ({lowest}) to {means[highest]:,.2f} "
            f"({highest}) is consistent with random variation."
        )

    variances = [float(np.var(s, ddof=1)) for s in samples]
    variance_ratio = max(variances) / min(variances) if min(variances) > 0 else float("inf")

    return TestResult(
        test_type="one_way_anova",
        question=f"Does {metric} differ across {factor}s?",
        null_hypothesis=f"Mean {metric} is equal across all {factor}s.",
        statistic=round(float(statistic), 4),
        p_value=float(p_value),
        significant=significant,
        alpha=alpha,
        interpretation=interpretation,
        effect_size=round(float(eta_squared), 4),
        effect_size_name="eta_squared",
        effect_size_label=effect_label,
        degrees_of_freedom=float(df_between),
        groups=[_describe(values, name) for name, values in
                sorted(usable.items(), key=lambda kv: float(np.mean(kv[1])), reverse=True)],
        assumptions={
            "variance_ratio": round(variance_ratio, 2) if np.isfinite(variance_ratio) else None,
            "homogeneity_of_variance_note": (
                "ANOVA assumes similar variances across groups. A ratio above ~4 "
                "means the F-statistic should be read with caution."
            ),
            "excluded_groups": [
                name for name in groups if name not in usable
            ],
            "exclusion_reason": f"fewer than {MIN_GROUP_SIZE} observations",
        },
        details={
            "df_between": df_between,
            "df_within": int(df_within),
            "highest_group": highest,
            "lowest_group": lowest,
            "variance_explained_pct": round(eta_squared * 100, 2),
        },
    )


# ---------------------------------------------------------------------------
# Categorical association
# ---------------------------------------------------------------------------
def chi_square_independence(
    contingency: pd.DataFrame,
    *,
    row_label: str,
    column_label: str,
    alpha: float = ALPHA,
) -> TestResult:
    """Chi-square test of independence with Cramer's V.

    Raises
    ------
    StatisticalTestError
        If more than 20% of expected cell counts fall below 5, which invalidates
        the chi-square approximation. Returning a p-value anyway would be
        reporting a number the test cannot support.
    """
    if contingency.empty or contingency.shape[0] < 2 or contingency.shape[1] < 2:
        raise StatisticalTestError(
            "A chi-square test needs a contingency table of at least 2x2; got "
            f"{contingency.shape[0]}x{contingency.shape[1]}.",
            details={"shape": list(contingency.shape)},
        )

    observed = contingency.to_numpy(dtype=float)
    statistic, p_value, dof, expected = stats.chi2_contingency(observed)

    small_cells = int((expected < 5).sum())
    total_cells = expected.size
    small_cell_pct = 100 * small_cells / total_cells
    if small_cell_pct > 20:
        raise StatisticalTestError(
            f"{small_cells} of {total_cells} expected cell counts ({small_cell_pct:.0f}%) "
            "are below 5, so the chi-square approximation is unreliable. Combine "
            "sparse categories or collect more data before testing.",
            details={
                "cells_below_5": small_cells,
                "total_cells": total_cells,
                "pct_below_5": round(small_cell_pct, 1),
            },
        )

    n = observed.sum()
    min_dim = min(observed.shape) - 1
    cramers_v = np.sqrt(statistic / (n * min_dim)) if n > 0 and min_dim > 0 else 0.0

    significant = bool(p_value < alpha)
    effect_label = _label_cramers_v(cramers_v)

    if significant:
        # Point at the cell that deviates most from independence, so the
        # reader knows where the association actually lives.
        residuals = (observed - expected) / np.sqrt(expected)
        idx = np.unravel_index(np.argmax(np.abs(residuals)), residuals.shape)
        row_name = str(contingency.index[idx[0]])
        col_name = str(contingency.columns[idx[1]])
        over_under = "more" if residuals[idx] > 0 else "fewer"

        interpretation = (
            f"{row_label} and {column_label} are not independent "
            f"(chi2={statistic:.2f}, df={dof}, p={p_value:.4f}). The strongest "
            f"deviation is {row_name} / {col_name}, which has {over_under} cases "
            f"than independence would predict. The association is {effect_label} "
            f"(Cramer's V={cramers_v:.3f})."
        )
        if effect_label in {"negligible", "small"}:
            interpretation += (
                " With a large sample, even a weak association reaches "
                "significance, so this is a real but weak relationship - not a "
                "reliable basis for targeting on its own."
            )
    else:
        interpretation = (
            f"{row_label} and {column_label} appear independent "
            f"(chi2={statistic:.2f}, df={dof}, p={p_value:.4f}). Knowing a "
            f"customer's {column_label.lower()} does not help predict their "
            f"{row_label.lower()}."
        )

    return TestResult(
        test_type="chi_square_independence",
        question=f"Are {row_label} and {column_label} related?",
        null_hypothesis=f"{row_label} is independent of {column_label}.",
        statistic=round(float(statistic), 4),
        p_value=float(p_value),
        significant=significant,
        alpha=alpha,
        interpretation=interpretation,
        effect_size=round(float(cramers_v), 4),
        effect_size_name="cramers_v",
        effect_size_label=effect_label,
        degrees_of_freedom=float(dof),
        groups=[
            {"name": str(idx), **{str(col): int(contingency.loc[idx, col])
                                  for col in contingency.columns}}
            for idx in contingency.index
        ],
        assumptions={
            "expected_cells_below_5": small_cells,
            "total_cells": total_cells,
            "note": (
                "The chi-square approximation requires most expected cell counts "
                "to be at least 5; this table satisfies that."
            ),
        },
        details={
            "observed": contingency.to_dict(orient="index"),
            "expected": {
                str(contingency.index[i]): {
                    str(contingency.columns[j]): round(float(expected[i, j]), 2)
                    for j in range(expected.shape[1])
                }
                for i in range(expected.shape[0])
            },
        },
    )


# ---------------------------------------------------------------------------
# Warehouse-backed analyses
# ---------------------------------------------------------------------------
def load_order_values(engine: Engine) -> pd.DataFrame:
    """One row per invoice: its total value, country and customer.

    Aggregating to invoice level here is what makes "average order value" mean
    what it says. Cancelled lines are excluded - see the module docstring.
    """
    stmt = (
        select(
            FactSales.invoice_no.label("invoice_no"),
            DimCountry.country_name.label("country"),
            FactSales.customer_id.label("customer_id"),
            func.sum(FactSales.revenue).label("order_value"),
        )
        .join(DimCountry, FactSales.country_id == DimCountry.country_id)
        .where(FactSales.is_cancelled.is_(False))
        .group_by(FactSales.invoice_no, DimCountry.country_name, FactSales.customer_id)
    )
    with engine.connect() as conn:
        df = pd.read_sql(stmt, conn)

    df["order_value"] = df["order_value"].astype(float)
    # A non-positive invoice total is a fully-offset order, not a purchase.
    df = df[df["order_value"] > 0]
    logger.info("  Loaded %s orders for statistical analysis.", f"{len(df):,}")
    return df


def aov_by_country(
    engine: Engine, country_a: str, country_b: str, *, alpha: float = ALPHA
) -> TestResult:
    """Compare average order value between two named countries."""
    orders = load_order_values(engine)
    available = set(orders["country"].unique())

    for country in (country_a, country_b):
        if country not in available:
            raise StatisticalTestError(
                f"Country {country!r} is not present in the dataset.",
                details={
                    "requested": country,
                    "available_sample": sorted(available)[:20],
                    "total_countries": len(available),
                },
            )

    return independent_ttest(
        orders.loc[orders["country"] == country_a, "order_value"],
        orders.loc[orders["country"] == country_b, "order_value"],
        label_a=country_a,
        label_b=country_b,
        metric="average order value",
        alpha=alpha,
    )


def aov_across_countries(
    engine: Engine, *, top_n: int = 8, alpha: float = ALPHA
) -> TestResult:
    """ANOVA on order value across the busiest countries.

    Restricted to the top N by order count: including all 43 would add dozens
    of tiny groups that contribute noise and inflate the family-wise error.
    """
    orders = load_order_values(engine)
    top_countries = orders["country"].value_counts().head(top_n).index.tolist()
    groups = {
        country: orders.loc[orders["country"] == country, "order_value"].to_numpy()
        for country in top_countries
    }
    return one_way_anova(groups, metric="order value", factor="country", alpha=alpha)


def aov_across_segments(engine: Engine, *, alpha: float = ALPHA) -> TestResult:
    """ANOVA on order value across RFM segments.

    A significant result is the statistical justification for the segmentation
    itself: it shows the segments separate customers who behave differently,
    rather than slicing one homogeneous population into arbitrary buckets.
    """
    stmt = (
        select(
            DimCustomer.rfm_segment.label("segment"),
            DimCustomer.avg_order_value.label("avg_order_value"),
        )
        .where(
            DimCustomer.rfm_segment.is_not(None),
            DimCustomer.avg_order_value.is_not(None),
            DimCustomer.avg_order_value > 0,
        )
    )
    with engine.connect() as conn:
        df = pd.read_sql(stmt, conn)

    if df.empty:
        raise InsufficientDataError(
            "No customers carry an RFM segment yet. Run the segmentation first: "
            "`python scripts/run_rfm.py`.",
        )

    groups = {
        segment: frame["avg_order_value"].astype(float).to_numpy()
        for segment, frame in df.groupby("segment")
    }
    return one_way_anova(
        groups, metric="average order value", factor="RFM segment", alpha=alpha
    )


def churn_by_country(
    engine: Engine, *, top_n: int = 8, churn_days: int = 90, alpha: float = ALPHA
) -> TestResult:
    """Test whether churn rate is independent of country.

    Churn is defined here the same way the model defines it - no purchase in
    ``churn_days`` - so the dashboard cannot show two different churn numbers.
    """
    stmt = select(
        DimCustomer.customer_id,
        DimCustomer.country,
        DimCustomer.rfm_recency,
    ).where(
        DimCustomer.rfm_recency.is_not(None),
        DimCustomer.country.is_not(None),
    )
    with engine.connect() as conn:
        df = pd.read_sql(stmt, conn)

    if df.empty:
        raise InsufficientDataError(
            "No customers have RFM recency yet. Run `python scripts/run_rfm.py` first."
        )

    df["churned"] = np.where(df["rfm_recency"] >= churn_days, "Churned", "Active")
    top_countries = df["country"].value_counts().head(top_n).index.tolist()
    subset = df[df["country"].isin(top_countries)]

    contingency = pd.crosstab(subset["country"], subset["churned"])
    # Guarantee a stable column order so the API response is deterministic.
    for column in ("Active", "Churned"):
        if column not in contingency.columns:
            contingency[column] = 0
    contingency = contingency[["Active", "Churned"]]

    return chi_square_independence(
        contingency, row_label="Country", column_label="Churn status", alpha=alpha
    )


def churn_by_segment(
    engine: Engine, *, churn_days: int = 90, alpha: float = ALPHA
) -> TestResult:
    """Test whether churn is independent of RFM segment.

    This should come out strongly significant - recency drives both. It serves
    as a sanity check that the segmentation and churn definition agree, and a
    null result here would mean something is wrong upstream.
    """
    stmt = select(
        DimCustomer.rfm_segment,
        DimCustomer.rfm_recency,
    ).where(
        DimCustomer.rfm_recency.is_not(None),
        DimCustomer.rfm_segment.is_not(None),
    )
    with engine.connect() as conn:
        df = pd.read_sql(stmt, conn)

    if df.empty:
        raise InsufficientDataError(
            "No segmented customers found. Run `python scripts/run_rfm.py` first."
        )

    df["churned"] = np.where(df["rfm_recency"] >= churn_days, "Churned", "Active")
    contingency = pd.crosstab(df["rfm_segment"], df["churned"])
    for column in ("Active", "Churned"):
        if column not in contingency.columns:
            contingency[column] = 0
    contingency = contingency[["Active", "Churned"]]

    return chi_square_independence(
        contingency, row_label="RFM segment", column_label="Churn status", alpha=alpha
    )


TestName = Literal[
    "aov_by_country", "aov_across_countries", "aov_across_segments",
    "churn_by_country", "churn_by_segment",
]

# Registry so the API can expose a discoverable catalogue rather than a
# hard-coded if/elif chain in the router.
AVAILABLE_TESTS: dict[str, dict[str, str]] = {
    "aov_by_country": {
        "test": "welch_ttest",
        "title": "Order value: two countries compared",
        "description": "Welch's t-test on order value between any two countries.",
        "parameters": "country_a, country_b",
    },
    "aov_across_countries": {
        "test": "one_way_anova",
        "title": "Order value across countries",
        "description": "ANOVA across the busiest countries by order count.",
        "parameters": "top_n",
    },
    "aov_across_segments": {
        "test": "one_way_anova",
        "title": "Order value across RFM segments",
        "description": "Tests whether the segmentation separates spending behaviour.",
        "parameters": "-",
    },
    "churn_by_country": {
        "test": "chi_square_independence",
        "title": "Churn vs country",
        "description": "Chi-square test of churn status against country.",
        "parameters": "top_n, churn_days",
    },
    "churn_by_segment": {
        "test": "chi_square_independence",
        "title": "Churn vs RFM segment",
        "description": "Chi-square test of churn status against RFM segment.",
        "parameters": "churn_days",
    },
}


__all__ = [
    "TestResult",
    "independent_ttest",
    "one_way_anova",
    "chi_square_independence",
    "load_order_values",
    "aov_by_country",
    "aov_across_countries",
    "aov_across_segments",
    "churn_by_country",
    "churn_by_segment",
    "AVAILABLE_TESTS",
    "ALPHA",
    "MIN_GROUP_SIZE",
]
