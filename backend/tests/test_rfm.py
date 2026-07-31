"""RFM correctness tests.

These assert against hand-computed values from ``synthetic_transactions``
rather than against whatever the code happens to produce. A test that merely
snapshots current output would pass just as happily with recency inverted or
frequency counting line items, which are the two failure modes that actually
matter here.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.core.exceptions import InsufficientDataError
from app.services import rfm as rfm_service
from app.services.rfm import (
    SEGMENT_ORDER,
    SEGMENT_RULES,
    UNCLASSIFIED_SEGMENT,
    assign_segment,
    compute_rfm,
    score_quantile,
    summarise_segments,
)
from tests.conftest import REFERENCE_DATE


@pytest.fixture
def rfm_small(monkeypatch, synthetic_transactions):
    """RFM for the 5-customer fixture, with the quantile floor lowered.

    ``compute_rfm`` normally refuses to score fewer than 25 customers because
    quintiles over a handful of people are meaningless. That guard is correct
    and separately tested below; here it is lowered so the *arithmetic* can be
    verified against hand-computed values on a set small enough to reason about.
    """
    monkeypatch.setattr(rfm_service, "MIN_CUSTOMERS_FOR_QUANTILES", 5)
    return compute_rfm(synthetic_transactions, reference_date=REFERENCE_DATE)


# ---------------------------------------------------------------------------
# Raw metric correctness
# ---------------------------------------------------------------------------
def test_recency_frequency_monetary_match_hand_computed(rfm_small, expected_rfm):
    """Every R, F and M value matches the value computed by hand."""
    assert set(rfm_small.index) == set(expected_rfm)

    for customer_id, expected in expected_rfm.items():
        row = rfm_small.loc[customer_id]
        assert row["recency"] == expected["recency"], (
            f"customer {customer_id}: recency {row['recency']} != "
            f"{expected['recency']}"
        )
        assert row["frequency"] == expected["frequency"], (
            f"customer {customer_id}: frequency {row['frequency']} != "
            f"{expected['frequency']}"
        )
        assert row["monetary"] == pytest.approx(expected["monetary"]), (
            f"customer {customer_id}: monetary {row['monetary']} != "
            f"{expected['monetary']}"
        )


def test_frequency_counts_invoices_not_line_items(rfm_small):
    """Customer 1001's 30-line invoice counts once, not thirty times.

    This is the assertion that catches the most common RFM implementation bug.
    """
    assert rfm_small.loc[1001, "frequency"] == 3
    assert rfm_small.loc[1001, "total_lines"] == 32  # 30 + 1 + 1


def test_guest_checkouts_are_excluded(rfm_small, synthetic_transactions):
    """Rows without a customer id are dropped from RFM but exist in the input."""
    assert synthetic_transactions["customer_id"].isna().sum() == 1
    assert rfm_small["frequency"].sum() == 9  # 3+2+1+1+2, guest order excluded


def test_avg_order_value_is_monetary_over_frequency(rfm_small):
    """AOV divides by order count, not line count."""
    assert rfm_small.loc[1001, "avg_order_value"] == pytest.approx(200.00)  # 600/3
    assert rfm_small.loc[1003, "avg_order_value"] == pytest.approx(150.00)  # 150/1


def test_one_time_buyers_have_no_interorder_gap(rfm_small):
    """Average days between orders is undefined for a single purchase.

    Left as NaN rather than 0, which would falsely imply same-day reordering
    and would drag down any average computed over the column.
    """
    assert pd.isna(rfm_small.loc[1003, "avg_days_between_orders"])
    assert pd.isna(rfm_small.loc[1004, "avg_days_between_orders"])
    assert not pd.isna(rfm_small.loc[1001, "avg_days_between_orders"])


# ---------------------------------------------------------------------------
# Score direction - the inversion bug
# ---------------------------------------------------------------------------
def test_recency_scoring_is_inverted(rfm_small):
    """The most recent customer scores highest on R, the most dormant lowest.

    If the reverse flag were dropped, 1001 (1 day) and 1004 (300 days) would
    swap and every segment would silently invert.
    """
    assert rfm_small.loc[1001, "r_score"] > rfm_small.loc[1004, "r_score"]
    assert rfm_small.loc[1001, "r_score"] == 5  # 1 day since purchase - best
    assert rfm_small.loc[1004, "r_score"] == 1  # 300 days - worst


def test_frequency_and_monetary_scoring_are_not_inverted(rfm_small):
    """Higher order counts and higher spend earn higher scores."""
    assert rfm_small.loc[1001, "f_score"] == 5  # 3 orders - most
    assert rfm_small.loc[1001, "m_score"] == 5  # 600.00 - highest
    assert rfm_small.loc[1004, "m_score"] == 1  # 50.00 - lowest


def test_score_quantile_reverse_flag_flips_order():
    """Directly verify the reverse flag on a clean, evenly spread series."""
    series = pd.Series(range(100), name="metric")

    ascending = score_quantile(series)
    assert ascending.iloc[0] == 1 and ascending.iloc[-1] == 5

    descending = score_quantile(series, reverse=True)
    assert descending.iloc[0] == 5 and descending.iloc[-1] == 1


def test_score_quantile_survives_duplicate_edges(caplog):
    """A dominant value (as with one-time buyers) must not crash scoring.

    ``pd.qcut`` raises on duplicate quantile edges. Real frequency
    distributions always trigger it, so the rank fallback is load-bearing
    rather than defensive.
    """
    # 90% of customers ordered exactly once - the shape retail data always has.
    skewed = pd.Series([1] * 90 + [2, 3, 4, 5, 6, 7, 8, 9, 10, 11], name="frequency")

    scores = score_quantile(skewed)

    assert len(scores) == len(skewed)
    assert scores.between(1, 5).all()
    assert scores.notna().all()
    # The heavy tail must still be scored above the dominant value.
    assert scores.iloc[-1] > scores.iloc[0]


def test_rfm_score_string_concatenates_all_three(rfm_small):
    for customer_id, row in rfm_small.iterrows():
        assert row["rfm_score"] == f"{row.r_score}{row.f_score}{row.m_score}"
        assert len(row["rfm_score"]) == 3


# ---------------------------------------------------------------------------
# Segment assignment
# ---------------------------------------------------------------------------
def test_best_customer_is_a_champion(rfm_small):
    """R=5, F=5 must map to Champions."""
    assert rfm_small.loc[1001, "segment"] == "Champions"


def test_lapsed_high_spender_is_flagged_at_risk(rfm_small):
    """Customer 1005 spent well but has not returned in 200 days.

    The business value of RFM rests on this row landing somewhere actionable
    rather than in a generic bucket.
    """
    segment = rfm_small.loc[1005, "segment"]
    assert segment in {"At Risk", "Cannot Lose Them", "Hibernating", "About To Sleep"}
    assert rfm_small.loc[1005, "r_score"] <= 2  # genuinely dormant


@pytest.mark.parametrize(
    ("r_score", "f_score", "expected"),
    [
        (5, 5, "Champions"),
        (4, 4, "Champions"),
        (5, 1, "New Customers"),
        (4, 1, "Promising"),
        (1, 5, "Cannot Lose Them"),
        (1, 1, "Lost"),
        (2, 5, "At Risk"),
        (3, 3, "Loyal Customers"),
    ],
)
def test_segment_assignment_is_deterministic(r_score, f_score, expected):
    """Overlapping rules resolve by priority, so the mapping is stable."""
    assert assign_segment(r_score, f_score) == expected


def test_every_score_combination_maps_to_a_named_segment():
    """All 25 (R, F) cells resolve to a name - never NaN or an empty string."""
    for r in range(1, 6):
        for f in range(1, 6):
            segment = assign_segment(r, f)
            assert segment in SEGMENT_ORDER, f"({r},{f}) produced {segment!r}"
            assert segment


def test_segment_rule_priorities_are_unique():
    """Duplicate priorities would make first-match-wins ordering ambiguous."""
    priorities = [rule.priority for rule in SEGMENT_RULES]
    assert len(priorities) == len(set(priorities))


def test_every_segment_rule_is_reachable():
    """No rule may be fully shadowed by a broader, higher-priority rule.

    This is the test that caught the original ordering bug: "Cannot Lose Them"
    and "Hibernating" were both defined but sat behind broader rules that
    matched all their cells first, so no customer could ever be assigned to
    them. A defined-but-unreachable segment is a silent hole in the analysis -
    the most valuable win-back group simply never appeared.
    """
    produced = {assign_segment(r, f) for r in range(1, 6) for f in range(1, 6)}
    defined = {rule.name for rule in SEGMENT_RULES}
    unreachable = defined - produced
    assert not unreachable, (
        f"these segments are defined but no (R,F) cell reaches them: "
        f"{sorted(unreachable)}"
    )



# ---------------------------------------------------------------------------
# Segment summary
# ---------------------------------------------------------------------------
def test_segment_summary_percentages_total_100(rfm_small):
    summary = summarise_segments(rfm_small)

    assert summary["customers"].sum() == len(rfm_small)
    assert summary["pct_customers"].sum() == pytest.approx(100.0, abs=0.05)
    assert summary["pct_revenue"].sum() == pytest.approx(100.0, abs=0.05)
    assert summary["total_revenue"].sum() == pytest.approx(
        rfm_small["monetary"].sum(), abs=0.01
    )


def test_segment_summary_uses_canonical_ordering(rfm_small):
    """Chart legends must not reshuffle between runs."""
    summary = summarise_segments(rfm_small)
    positions = [SEGMENT_ORDER.index(name) for name in summary.index]
    assert positions == sorted(positions)


# ---------------------------------------------------------------------------
# Guard rails
# ---------------------------------------------------------------------------
def test_too_few_customers_raises_rather_than_producing_junk(synthetic_transactions):
    """Five customers cannot support quintiles, and saying so beats guessing.

    Silently emitting 5 segments of one person each would look like output but
    mean nothing.
    """
    with pytest.raises(InsufficientDataError) as exc_info:
        compute_rfm(synthetic_transactions, reference_date=REFERENCE_DATE)

    error = exc_info.value
    assert error.status == 422
    assert error.details["customers"] == 5
    assert error.details["minimum_required"] == 25
    assert "quintile" in error.message


def test_empty_input_raises_insufficient_data():
    empty = pd.DataFrame(
        columns=[
            "customer_id", "invoice_no", "invoice_date",
            "revenue", "quantity", "is_cancelled",
        ]
    )
    with pytest.raises(InsufficientDataError):
        compute_rfm(empty)


def test_all_guest_checkouts_raises_with_a_useful_message(synthetic_transactions):
    """If nothing is identifiable, the error must say why - not return empty."""
    anonymous = synthetic_transactions.copy()
    anonymous["customer_id"] = None

    with pytest.raises(InsufficientDataError) as exc_info:
        compute_rfm(anonymous, reference_date=REFERENCE_DATE)

    assert exc_info.value.details["rows_with_customer"] == 0
    assert "customer id" in exc_info.value.message


def test_reference_date_defaults_to_dataset_max(monkeypatch, synthetic_transactions):
    """Recency is measured against the data, not the wall clock.

    Defaulting to today would report every customer as ~14 years dormant and
    collapse the entire base into "Lost".
    """
    monkeypatch.setattr(rfm_service, "MIN_CUSTOMERS_FOR_QUANTILES", 5)
    result = compute_rfm(synthetic_transactions)  # no reference_date

    # Last transaction is 2011-12-09; default reference is the following day.
    assert result.loc[1001, "recency"] == 1
    assert result["recency"].max() < 400  # nothing looks like a decade old


def test_return_rate_reflects_cancelled_lines(monkeypatch, synthetic_transactions):
    """A customer with no cancellations has a return rate of exactly zero."""
    monkeypatch.setattr(rfm_service, "MIN_CUSTOMERS_FOR_QUANTILES", 5)
    result = compute_rfm(synthetic_transactions, reference_date=REFERENCE_DATE)

    assert (result["return_rate"] == 0).all()
    assert (result["cancelled_lines"] == 0).all()


def test_unclassified_segment_is_never_silently_produced(rfm_small):
    """The fixture is ordinary data; nothing should fall through the rules."""
    assert UNCLASSIFIED_SEGMENT not in set(rfm_small["segment"])
