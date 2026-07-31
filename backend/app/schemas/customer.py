"""Response models for the customer, segment, and churn-risk endpoints.

These mirror the dicts returned by ``app.services.customer`` field for field.
Because ``APIModel`` sets ``extra="forbid"``, any drift between a service and its
schema fails loudly the first time a test hits the endpoint, rather than silently
dropping a field the dashboard was relying on.

Optionality here is deliberate and load-bearing. Most numeric fields are
``| None`` because the analytics pipeline may not have run yet, or because a
customer can legitimately lack the history a metric needs (a one-order customer
has no average gap between orders). Defaulting those to ``0`` would be a lie the
frontend cannot detect, so the API returns null and the UI renders "not
computed".
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import Field

from app.schemas.common import APIModel

# Coarse health grouping the frontend colours by. Sent from the server so the
# palette mapping lives in exactly one place.
SegmentHealth = Literal["healthy", "neutral", "watch", "at_risk", "lost"]
RiskBand = Literal["high", "medium", "low"]


# ---------------------------------------------------------------------------
# Segment distribution
# ---------------------------------------------------------------------------


class SegmentStats(APIModel):
    """One RFM segment, with both its size and its commercial weight.

    Share of customers and share of revenue are both returned because the gap
    between them is the insight: Champions are typically a small fraction of the
    customer base and a large fraction of revenue.
    """

    segment: str = Field(description="Segment name, e.g. 'Champions'.")
    health: SegmentHealth = Field(description="Coarse grouping for UI colouring.")
    customer_count: int = Field(description="Customers in this segment.")
    customer_pct: float = Field(description="Share of all segmented customers (%).")
    revenue: float = Field(description="Lifetime revenue attributed to this segment.")
    revenue_pct: float = Field(description="Share of customer-attributed revenue (%).")
    orders: int = Field(description="Total orders placed by this segment.")
    avg_revenue_per_customer: float = Field(
        description="Revenue per customer — the figure that justifies retention spend."
    )
    avg_recency_days: float = Field(description="Mean days since last purchase.")
    avg_frequency: float = Field(description="Mean distinct orders per customer.")
    avg_monetary: float = Field(description="Mean lifetime spend per customer.")
    avg_churn_risk: float | None = Field(
        default=None,
        description="Mean churn probability, or null if the model has not scored anyone.",
    )


class SegmentDistribution(APIModel):
    """Segment breakdown for the Customer Segments page."""

    segments: list[SegmentStats] = Field(
        description="Ordered healthiest-first; empty when RFM has not been computed."
    )
    total_customers: int = Field(description="Customers carrying a segment label.")
    attributed_revenue: float = Field(
        description=(
            "Revenue attributable to identified customers. Intentionally lower "
            "than total revenue from /kpis/summary — invoices without a customer "
            "id cannot belong to a segment."
        )
    )
    computed: bool = Field(
        description="False when the analytics step has not run; the UI shows a prompt, not zeros."
    )


# ---------------------------------------------------------------------------
# Customer profile
# ---------------------------------------------------------------------------


class RFMDetail(APIModel):
    """Raw RFM values alongside their 1–5 quantile scores.

    Both are exposed because they answer different questions: the raw recency
    ("92 days ago") is what a human understands, while the score ("R=2") is what
    placed the customer in their segment.
    """

    recency_days: int | None = Field(default=None, description="Days since last purchase.")
    frequency: int | None = Field(default=None, description="Distinct orders placed.")
    monetary: float | None = Field(default=None, description="Lifetime spend.")
    r_score: int | None = Field(default=None, ge=1, le=5)
    f_score: int | None = Field(default=None, ge=1, le=5)
    m_score: int | None = Field(default=None, ge=1, le=5)
    rfm_score: str | None = Field(default=None, description="Concatenated scores, e.g. '545'.")
    segment: str | None = Field(default=None)
    health: SegmentHealth | None = Field(default=None)


class OrderHistoryEntry(APIModel):
    """One invoice, aggregated from its line items for the timeline chart."""

    invoice_no: str
    order_date: date
    revenue: float
    items: int = Field(description="Net units; can be negative on a return.")
    distinct_products: int
    is_cancelled: bool


class TopProduct(APIModel):
    """A product this customer spent the most on. Cancellations excluded."""

    stock_code: str
    description: str | None = Field(default=None)
    category: str | None = Field(default=None)
    revenue: float
    quantity: int


class CustomerProfile(APIModel):
    """Everything the drill-down page needs for one customer, in one response.

    Deliberately a single call rather than four: the page renders as a unit, and
    four round trips would mean four independent loading and error states for
    what the user experiences as one view.
    """

    customer_id: int
    country: str | None = Field(default=None)
    first_purchase_date: date | None = Field(default=None)
    last_purchase_date: date | None = Field(default=None)
    tenure_days: int | None = Field(
        default=None, description="Days between first and last purchase."
    )

    rfm: RFMDetail
    rfm_computed: bool = Field(
        description="False means this customer exists but has not been scored yet."
    )

    total_orders: int | None = Field(default=None)
    total_revenue: float
    total_items: int | None = Field(default=None)
    avg_order_value: float | None = Field(default=None)
    distinct_products: int | None = Field(default=None)
    return_rate: float | None = Field(
        default=None, description="Fraction of orders cancelled, 0–1."
    )
    avg_days_between_orders: float | None = Field(
        default=None, description="Null for single-order customers — no gap exists to average."
    )
    revenue_percentile: float | None = Field(
        default=None,
        description="Percentile rank by lifetime revenue; turns a raw total into context.",
    )

    churn_risk_score: float | None = Field(default=None, ge=0.0, le=1.0)
    churn_risk_band: RiskBand | None = Field(
        default=None, description="Server-side banding so API and UI cannot disagree."
    )
    is_churned: bool | None = Field(
        default=None, description="Observed label: no purchase within the churn window."
    )

    order_history: list[OrderHistoryEntry] = Field(
        default_factory=list, description="Oldest first, so the timeline reads left to right."
    )
    top_products: list[TopProduct] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# At-risk customers
# ---------------------------------------------------------------------------


class AtRiskCustomer(APIModel):
    """A customer above the churn threshold, with the money attached.

    ``expected_revenue_loss`` (probability × lifetime revenue) is the field the
    table sorts by. Ranking by probability alone puts trivial customers at the
    top of a retention worklist.
    """

    customer_id: int
    country: str | None = Field(default=None)
    segment: str | None = Field(default=None)
    recency_days: int | None = Field(default=None)
    frequency: int | None = Field(default=None)
    total_revenue: float
    avg_order_value: float | None = Field(default=None)
    last_purchase_date: date | None = Field(default=None)
    churn_risk_score: float = Field(ge=0.0, le=1.0)
    risk_band: RiskBand
    expected_revenue_loss: float = Field(description="churn_risk_score × total_revenue.")


class AtRiskResponse(APIModel):
    """Paginated at-risk list plus the totals for the summary cards.

    The two revenue figures answer different questions and are both returned so
    the UI does not have to pick one and mislabel it:
    ``revenue_at_risk`` is the worst case where every flagged customer leaves;
    ``expected_revenue_loss`` is the probability-weighted forecast.

    Totals cover the whole filtered set, not just the current page — a summary
    card that changed when you paged through the table would be wrong.
    """

    items: list[AtRiskCustomer]
    total: int = Field(description="Customers matching the filter, across all pages.")
    limit: int
    offset: int
    threshold: float = Field(ge=0.0, le=1.0, description="Threshold actually applied.")
    revenue_at_risk: float = Field(description="Worst case: all flagged customers churn.")
    expected_revenue_loss: float = Field(description="Probability-weighted expected loss.")
    scored_customers: int = Field(
        description="Customers with any churn score; 0 means the model has not been run."
    )
