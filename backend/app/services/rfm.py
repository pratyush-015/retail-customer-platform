"""RFM (Recency, Frequency, Monetary) segmentation.

What RFM is
-----------
Three behavioural measures per customer, taken from the transaction history:

* **Recency**   - days since their last purchase (lower is better)
* **Frequency** - how many distinct orders they placed (higher is better)
* **Monetary**  - total net revenue they generated (higher is better)

Each is scored 1-5 by quintile, and the (R, F) pair is mapped to a named
segment such as "Champions" or "At Risk" that a marketer can actually act on.

Design decisions worth knowing
------------------------------
1. **Reference date is the last date in the dataset, not today.** This data ends
   in Dec 2011. Measuring recency against the real current date would make
   every customer look ~14 years dormant and every segment collapse into
   "Lost". The reference date is therefore configurable and defaults to the
   dataset maximum, which is the standard treatment for a static extract.

2. **Recency is scored in reverse.** For F and M, a bigger raw value earns a
   bigger score. For R, a *smaller* number of days is better, so the quintile
   labels are inverted. Getting this backwards is the single most common RFM
   bug and it silently inverts every segment.

3. **Frequency counts distinct invoices, not line items.** A basket of 30
   products is one order, not 30. Counting rows would rank bulk buyers as
   loyal repeat purchasers.

4. **Monetary is net of returns.** Cancelled lines carry negative revenue and
   are summed in, so a customer who bought heavily and returned everything is
   not credited as high-value.

5. **Quantile edges are deduplicated.** Frequency is heavily skewed - a large
   share of customers ordered exactly once - so several quintile boundaries
   land on the same value. ``pd.qcut`` raises on duplicate edges, so scoring
   falls back to a dense rank. Without this the pipeline crashes on precisely
   the distribution retail data always has.

6. **Customers with no identity are excluded.** ~22% of rows are guest
   checkouts. They belong in revenue totals but cannot be segmented, since
   there is no identity to track across orders.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import bindparam, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from app.core.exceptions import InsufficientDataError
from app.core.logging_config import get_logger
from app.models.star_schema import DimCustomer, FactSales

logger = get_logger(__name__)

# Minimum customers before quintile scoring is meaningful. Below this, a
# 5-bucket split produces buckets of one or two people and the segment names
# become noise rather than insight.
MIN_CUSTOMERS_FOR_QUANTILES = 25

SCORE_LABELS = [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Segment definitions
# ---------------------------------------------------------------------------
# Segments are assigned from the (Recency, Frequency) pair. Monetary is
# deliberately not part of the rule: it correlates strongly with frequency, and
# folding it in produces 125 cells that no one can act on. M is retained as a
# separate score for prioritising within a segment.
@dataclass(frozen=True)
class SegmentRule:
    """A named segment plus the R/F score ranges that define it."""

    name: str
    recency_scores: frozenset[int]
    frequency_scores: frozenset[int]
    description: str
    # Lower wins when several rules match. Narrower, more actionable rules MUST
    # sort before broader ones that overlap them, or the broad rule shadows the
    # narrow one into unreachability. A reachability test guards this.
    priority: int

    def matches(self, r_score: int, f_score: int) -> bool:
        return r_score in self.recency_scores and f_score in self.frequency_scores


SEGMENT_RULES: tuple[SegmentRule, ...] = (
    SegmentRule(
        "Champions", frozenset({4, 5}), frozenset({4, 5}),
        "Bought recently, buy often and spend the most. Reward them; they are "
        "the most likely to advocate for the brand.",
        priority=1,
    ),
    SegmentRule(
        "Loyal Customers", frozenset({3, 4, 5}), frozenset({3, 4, 5}),
        "Buy regularly and responsive to promotions. Upsell higher-value "
        "products.",
        priority=2,
    ),
    SegmentRule(
        "Potential Loyalists", frozenset({4, 5}), frozenset({2, 3}),
        "Recent customers with moderate frequency. Membership or loyalty "
        "programmes convert these into Loyal Customers.",
        priority=3,
    ),
    SegmentRule(
        "New Customers", frozenset({5}), frozenset({1}),
        "Bought very recently but only once. Onboarding and early support "
        "determine whether they return.",
        priority=4,
    ),
    SegmentRule(
        "Promising", frozenset({4}), frozenset({1}),
        "Recent one-time buyers showing interest. Build awareness and offer "
        "a reason for a second purchase.",
        priority=5,
    ),
    SegmentRule(
        "Need Attention", frozenset({3}), frozenset({2, 3}),
        "Above-average history but slipping. Time-limited offers can "
        "reactivate them before they lapse.",
        priority=6,
    ),
    # Hibernating (R=2, F=2-3) is narrower than both About To Sleep and At Risk,
    # which overlap it, so it must be evaluated before them.
    SegmentRule(
        "Hibernating", frozenset({2}), frozenset({2, 3}),
        "Last purchase was long ago, with low spend and few orders. Low-cost "
        "reactivation only.",
        priority=7,
    ),
    # Cannot Lose Them (R=1, F=4-5) is the narrower, more urgent case inside At
    # Risk's range: the very best customers, now completely gone. Ranked after
    # At Risk it was unreachable, silently erasing the single highest-value
    # win-back group from the dashboard.
    SegmentRule(
        "Cannot Lose Them", frozenset({1}), frozenset({4, 5}),
        "Formerly the best customers, now long dormant. Win back aggressively; "
        "they have proven high lifetime value.",
        priority=8,
    ),
    SegmentRule(
        "About To Sleep", frozenset({2, 3}), frozenset({1, 2}),
        "Below-average recency and frequency. Will be lost without "
        "intervention.",
        priority=9,
    ),
    SegmentRule(
        "At Risk", frozenset({1, 2}), frozenset({3, 4, 5}),
        "Spent a lot and bought often, but not recently. The highest-value "
        "group to win back - losing them costs real revenue.",
        priority=10,
    ),
    SegmentRule(
        "Lost", frozenset({1}), frozenset({1, 2}),
        "Lowest recency and frequency. Unlikely to return; exclude from "
        "high-cost campaigns.",
        priority=11,
    ),
)

# Fallback for the handful of (R, F) cells no rule covers. Naming it explicitly
# beats silently emitting NaN or forcing a wrong segment.
UNCLASSIFIED_SEGMENT = "Others"

SEGMENT_DESCRIPTIONS: dict[str, str] = {
    **{rule.name: rule.description for rule in SEGMENT_RULES},
    UNCLASSIFIED_SEGMENT: (
        "Behaviour does not match a standard segment profile. Review "
        "individually rather than campaigning to the group."
    ),
}

# Stable display order: best to worst, for consistent chart legends.
SEGMENT_ORDER: tuple[str, ...] = (
    "Champions",
    "Loyal Customers",
    "Potential Loyalists",
    "New Customers",
    "Promising",
    "Need Attention",
    "About To Sleep",
    "At Risk",
    "Cannot Lose Them",
    "Hibernating",
    "Lost",
    UNCLASSIFIED_SEGMENT,
)


def assign_segment(r_score: int, f_score: int) -> str:
    """Map an (R, F) score pair to a named segment.

    Rules are evaluated in priority order and the first match wins, so
    overlapping definitions resolve deterministically rather than depending on
    dict ordering.
    """
    for rule in sorted(SEGMENT_RULES, key=lambda r: r.priority):
        if rule.matches(r_score, f_score):
            return rule.name
    return UNCLASSIFIED_SEGMENT


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_quantile(series: pd.Series, *, reverse: bool = False) -> pd.Series:
    """Score a series 1-5 by quintile, robust to skewed distributions.

    Parameters
    ----------
    series
        The raw metric (recency days, order count, revenue).
    reverse
        When True, *low* values earn *high* scores. Used for recency, where
        fewer days since purchase is better.

    Notes
    -----
    ``pd.qcut`` raises ``ValueError`` when quantile edges are not unique, which
    happens whenever a value dominates the distribution - and in retail data,
    "ordered exactly once" always does. In that case we fall back to ranking
    the values first, which spreads ties evenly across buckets instead of
    failing. The fallback is logged so the choice is visible, not silent.
    """
    labels = list(reversed(SCORE_LABELS)) if reverse else SCORE_LABELS

    try:
        scores = pd.qcut(series, q=5, labels=labels, duplicates="raise")
    except ValueError:
        logger.debug(
            "  qcut found duplicate quantile edges for %r (%s distinct values "
            "across %s rows); falling back to rank-based scoring.",
            series.name, series.nunique(), len(series),
        )
        ranked = series.rank(method="first", pct=True)
        scores = pd.qcut(ranked, q=5, labels=labels, duplicates="drop")

    return scores.astype(int)


def compute_rfm(
    transactions: pd.DataFrame,
    *,
    reference_date: date | None = None,
) -> pd.DataFrame:
    """Compute per-customer RFM metrics, scores and segments.

    Parameters
    ----------
    transactions
        Frame with ``customer_id``, ``invoice_no``, ``invoice_date``,
        ``revenue``, ``quantity`` and ``is_cancelled`` columns.
    reference_date
        The "today" against which recency is measured. Defaults to the day
        after the last transaction, so the most recent buyer has recency 1
        rather than 0 (a zero would make log transforms in the churn model
        undefined).

    Returns
    -------
    DataFrame indexed by ``customer_id`` with raw metrics, 1-5 scores, the
    combined score string and the segment name.
    """
    if transactions.empty:
        raise InsufficientDataError(
            "No transactions were supplied, so RFM cannot be computed.",
            details={"rows": 0},
        )

    df = transactions[transactions["customer_id"].notna()].copy()
    if df.empty:
        raise InsufficientDataError(
            "No transactions carry a customer id, so no customer can be "
            "segmented. RFM requires an identity to track across orders.",
            details={"rows_supplied": len(transactions), "rows_with_customer": 0},
        )

    df["invoice_date"] = pd.to_datetime(df["invoice_date"])

    if reference_date is None:
        reference_date = (df["invoice_date"].max() + timedelta(days=1)).date()
    reference_ts = pd.Timestamp(reference_date)

    logger.info(
        "Computing RFM for %s transactions (reference date %s)...",
        f"{len(df):,}", reference_date,
    )

    # Recency is measured in whole calendar days, so the last purchase is
    # normalised to midnight before subtracting. Without this, an order placed
    # at 09:00 the previous day is a 15-hour delta, and `.dt.days` truncates
    # that to 0 - reporting "purchased today" for a customer who did not.
    df["invoice_day"] = df["invoice_date"].dt.normalize()

    # Frequency counts distinct invoices; monetary nets returns out.
    grouped = df.groupby("customer_id")
    rfm = pd.DataFrame({
        "recency": (reference_ts - grouped["invoice_day"].max()).dt.days,
        "frequency": grouped["invoice_no"].nunique(),
        "monetary": grouped["revenue"].sum().round(2),
        "first_purchase": grouped["invoice_date"].min().dt.date,
        "last_purchase": grouped["invoice_date"].max().dt.date,
        "total_items": grouped["quantity"].sum(),
        "total_lines": grouped.size(),
    })

    # Returns rate: share of a customer's lines that were cancellations. A
    # useful churn feature and a sanity check on the monetary figure.
    cancelled_lines = df[df["is_cancelled"]].groupby("customer_id").size()
    rfm["cancelled_lines"] = cancelled_lines.reindex(rfm.index).fillna(0).astype(int)
    rfm["return_rate"] = (rfm["cancelled_lines"] / rfm["total_lines"]).round(4)

    rfm["tenure_days"] = (
        pd.to_datetime(rfm["last_purchase"]) - pd.to_datetime(rfm["first_purchase"])
    ).dt.days

    # Average gap between orders. Undefined for one-time buyers, so left NaN
    # rather than filled with a fake zero that would imply daily ordering.
    repeat = rfm["frequency"] > 1
    rfm["avg_days_between_orders"] = np.where(
        repeat, rfm["tenure_days"] / (rfm["frequency"] - 1).clip(lower=1), np.nan
    ).round(2)

    rfm["avg_order_value"] = (rfm["monetary"] / rfm["frequency"]).round(2)

    if len(rfm) < MIN_CUSTOMERS_FOR_QUANTILES:
        raise InsufficientDataError(
            f"Only {len(rfm)} customers available; at least "
            f"{MIN_CUSTOMERS_FOR_QUANTILES} are needed for quintile scoring to "
            "produce meaningful segments.",
            details={
                "customers": len(rfm),
                "minimum_required": MIN_CUSTOMERS_FOR_QUANTILES,
            },
        )

    # Recency is reversed: fewer days since purchase earns a higher score.
    rfm["r_score"] = score_quantile(rfm["recency"], reverse=True)
    rfm["f_score"] = score_quantile(rfm["frequency"])
    rfm["m_score"] = score_quantile(rfm["monetary"])

    rfm["rfm_score"] = (
        rfm["r_score"].astype(str) + rfm["f_score"].astype(str) + rfm["m_score"].astype(str)
    )
    rfm["segment"] = [
        assign_segment(r, f) for r, f in zip(rfm["r_score"], rfm["f_score"])
    ]

    unclassified = int((rfm["segment"] == UNCLASSIFIED_SEGMENT).sum())
    if unclassified:
        logger.warning(
            "  %s customer(s) (%.2f%%) did not match any segment rule and were "
            "labelled %r.",
            f"{unclassified:,}", 100 * unclassified / len(rfm), UNCLASSIFIED_SEGMENT,
        )

    logger.info(
        "  RFM computed for %s customers across %s segments.",
        f"{len(rfm):,}", rfm["segment"].nunique(),
    )
    return rfm


def summarise_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    """Aggregate an RFM frame into a per-segment summary for the dashboard."""
    summary = rfm.groupby("segment").agg(
        customers=("monetary", "size"),
        total_revenue=("monetary", "sum"),
        avg_revenue=("monetary", "mean"),
        avg_recency=("recency", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_order_value=("avg_order_value", "mean"),
    )

    total_revenue = summary["total_revenue"].sum()
    total_customers = summary["customers"].sum()
    summary["pct_customers"] = (100 * summary["customers"] / total_customers).round(2)
    summary["pct_revenue"] = (
        (100 * summary["total_revenue"] / total_revenue).round(2)
        if total_revenue
        else 0.0
    )
    summary = summary.round(2)

    # Order by the canonical best-to-worst sequence so charts stay stable
    # between runs, with any unexpected segment appended rather than dropped.
    known = [s for s in SEGMENT_ORDER if s in summary.index]
    extra = [s for s in summary.index if s not in SEGMENT_ORDER]
    return summary.loc[known + extra]


# ---------------------------------------------------------------------------
# Warehouse integration
# ---------------------------------------------------------------------------
def load_transactions(engine: Engine) -> pd.DataFrame:
    """Read the identified-customer transactions needed for RFM."""
    stmt = (
        select(
            FactSales.customer_id,
            FactSales.invoice_no,
            FactSales.invoice_timestamp.label("invoice_date"),
            FactSales.revenue,
            FactSales.quantity,
            FactSales.is_cancelled,
        )
        .where(FactSales.customer_id.is_not(None))
    )
    with engine.connect() as conn:
        df = pd.read_sql(stmt, conn)

    df["revenue"] = df["revenue"].astype(float)
    logger.info("  Read %s identified transactions from the warehouse.", f"{len(df):,}")
    return df


def dataset_max_date(engine: Engine) -> date:
    """The latest transaction date, used as the recency reference point."""
    with engine.connect() as conn:
        value = conn.execute(select(func.max(FactSales.invoice_timestamp))).scalar()
    if value is None:
        raise InsufficientDataError(
            "The warehouse contains no transactions. Run the ETL first: "
            "`python scripts/run_etl.py`."
        )
    return pd.Timestamp(value).date()


def persist_rfm(engine: Engine, rfm: pd.DataFrame) -> int:
    """Write RFM results back onto ``dim_customer``.

    Uses a bulk UPDATE keyed on customer_id rather than delete-and-reinsert:
    the dimension is referenced by a million fact rows, so replacing it would
    either violate the foreign key or require dropping the constraint.

    The bind parameter is named ``_customer_id`` rather than ``customer_id``
    so it cannot collide with the SET clause: SQLAlchemy would otherwise treat
    the WHERE value and a same-named column update as one parameter.
    """
    payload = [
        {
            "_customer_id": int(customer_id),
            "rfm_recency": int(row.recency),
            "rfm_frequency": int(row.frequency),
            "rfm_monetary": float(row.monetary),
            "r_score": int(row.r_score),
            "f_score": int(row.f_score),
            "m_score": int(row.m_score),
            "rfm_score": str(row.rfm_score),
            "rfm_segment": str(row.segment),
            "total_orders": int(row.frequency),
            "total_revenue": float(row.monetary),
            "total_items": int(row.total_items),
            "avg_order_value": float(row.avg_order_value),
            "return_rate": float(row.return_rate),
            "tenure_days": int(row.tenure_days),
            "avg_days_between_orders": (
                None if pd.isna(row.avg_days_between_orders)
                else float(row.avg_days_between_orders)
            ),
        }
        for customer_id, row in rfm.iterrows()
    ]

    stmt = DimCustomer.__table__.update().where(
        DimCustomer.customer_id == bindparam("_customer_id")
    )
    with Session(engine) as session:
        session.connection().execute(stmt, payload)
        session.commit()

    logger.info("  Persisted RFM for %s customers to dim_customer.", f"{len(payload):,}")
    return len(payload)


__all__ = [
    "compute_rfm",
    "summarise_segments",
    "assign_segment",
    "score_quantile",
    "load_transactions",
    "dataset_max_date",
    "persist_rfm",
    "SEGMENT_RULES",
    "SEGMENT_ORDER",
    "SEGMENT_DESCRIPTIONS",
    "UNCLASSIFIED_SEGMENT",
    "MIN_CUSTOMERS_FOR_QUANTILES",
]
