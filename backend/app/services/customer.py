"""Customer-facing aggregations: RFM segments, profiles, and churn risk lists.

These read the enrichment columns on ``dim_customer`` that the analytics
pipeline writes (RFM scores, churn probability) rather than recomputing them
per request. Recomputing RFM on every API call would mean a full scan of
``fact_sales`` for a value that only changes when the ETL runs.

A consequence worth stating plainly: if the analytics step has not run, these
columns are NULL. Each function below treats that as "not yet computed" and
says so, instead of reporting zeros that look like real answers.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import and_, case, distinct, func, select
from sqlalchemy.orm import Session

from app.core.exceptions import CustomerNotFoundError
from app.core.risk import risk_band
from app.models.star_schema import DimCustomer, DimDate, DimProduct, FactSales


from app.services.kpi import _to_float, _to_int

logger = logging.getLogger(__name__)

# Display order for segments: healthiest first, most disengaged last. The API
# returns segments in this order so the frontend never has to hardcode it.
SEGMENT_ORDER: list[str] = [
    "Champions",
    "Loyal Customers",
    "Potential Loyalist",
    "New Customers",
    "Promising",
    "Need Attention",
    "About To Sleep",
    "At Risk",
    "Cannot Lose Them",
    "Hibernating",
    "Lost",
]

# Coarse health grouping used by the segment endpoint, so the UI can colour by
# intent without duplicating this mapping in JavaScript.
SEGMENT_HEALTH: dict[str, str] = {
    "Champions": "healthy",
    "Loyal Customers": "healthy",
    "Potential Loyalist": "healthy",
    "New Customers": "neutral",
    "Promising": "neutral",
    "Need Attention": "watch",
    "About To Sleep": "watch",
    "At Risk": "at_risk",
    "Cannot Lose Them": "at_risk",
    "Hibernating": "lost",
    "Lost": "lost",
}


def _segment_sort_key(name: str) -> tuple[int, str]:
    """Order known segments canonically, then anything unexpected alphabetically."""
    try:
        return (SEGMENT_ORDER.index(name), "")
    except ValueError:
        return (len(SEGMENT_ORDER), name)


# ---------------------------------------------------------------------------
# Segment distribution
# ---------------------------------------------------------------------------


def get_segment_distribution(db: Session) -> dict[str, Any]:
    """Customer count and revenue per RFM segment.

    Revenue comes from the precomputed ``total_revenue`` column, so the
    per-segment figures sum to the customer-attributed revenue total, not to
    overall revenue: guest/NULL-customer invoices are excluded by definition.
    That gap is surfaced as ``attributed_revenue`` so the number can be
    reconciled against the KPI endpoint instead of looking like a discrepancy.
    """
    stmt = (
        select(
            DimCustomer.rfm_segment.label("segment"),
            func.count(DimCustomer.customer_id).label("customer_count"),
            func.sum(DimCustomer.total_revenue).label("revenue"),
            func.sum(DimCustomer.total_orders).label("orders"),
            func.avg(DimCustomer.rfm_recency).label("avg_recency"),
            func.avg(DimCustomer.rfm_frequency).label("avg_frequency"),
            func.avg(DimCustomer.rfm_monetary).label("avg_monetary"),
            func.avg(DimCustomer.churn_risk_score).label("avg_churn_risk"),
        )
        .where(DimCustomer.rfm_segment.is_not(None))
        .group_by(DimCustomer.rfm_segment)
    )

    rows = db.execute(stmt).all()

    if not rows:
        logger.warning(
            "No RFM segments found in dim_customer; the analytics step has "
            "likely not been run."
        )
        return {
            "segments": [],
            "total_customers": 0,
            "attributed_revenue": 0.0,
            "computed": False,
        }

    total_customers = sum(_to_int(r.customer_count) for r in rows)
    total_revenue = sum(_to_float(r.revenue) for r in rows)

    segments = []
    for row in rows:
        count = _to_int(row.customer_count)
        revenue = _to_float(row.revenue)
        orders = _to_int(row.orders)
        segments.append(
            {
                "segment": row.segment,
                "health": SEGMENT_HEALTH.get(row.segment, "neutral"),
                "customer_count": count,
                "customer_pct": round(count / total_customers * 100, 2) if total_customers else 0.0,
                "revenue": round(revenue, 2),
                "revenue_pct": round(revenue / total_revenue * 100, 2) if total_revenue else 0.0,
                "orders": orders,
                # Revenue per customer is the number that actually justifies
                # spending retention budget on a segment.
                "avg_revenue_per_customer": round(revenue / count, 2) if count else 0.0,
                "avg_recency_days": round(_to_float(row.avg_recency), 1),
                "avg_frequency": round(_to_float(row.avg_frequency), 2),
                "avg_monetary": round(_to_float(row.avg_monetary), 2),
                "avg_churn_risk": (
                    round(_to_float(row.avg_churn_risk), 4)
                    if row.avg_churn_risk is not None
                    else None
                ),
            }
        )

    segments.sort(key=lambda s: _segment_sort_key(s["segment"]))

    return {
        "segments": segments,
        "total_customers": total_customers,
        "attributed_revenue": round(total_revenue, 2),
        "computed": True,
    }


# ---------------------------------------------------------------------------
# Individual customer profile
# ---------------------------------------------------------------------------


def _order_history(db: Session, customer_id: int, limit: int) -> list[dict[str, Any]]:
    """Invoice-level history for the drill-down timeline.

    Aggregated per invoice rather than per line item: a customer with 400 line
    items has perhaps 12 orders, and the timeline chart wants the latter.
    """
    stmt = (
        select(
            FactSales.invoice_no,
            DimDate.full_date.label("order_date"),
            func.sum(FactSales.revenue).label("revenue"),
            func.sum(FactSales.quantity).label("items"),
            func.count(distinct(FactSales.product_id)).label("distinct_products"),
            # An invoice is a cancellation as a whole, so max() over the flag
            # reflects the invoice without needing a second query.
            func.max(case((FactSales.is_cancelled, 1), else_=0)).label("is_cancelled"),
        )
        .join(DimDate, FactSales.date_id == DimDate.date_id)
        .where(FactSales.customer_id == customer_id)
        .group_by(FactSales.invoice_no, DimDate.full_date)
        .order_by(DimDate.full_date.desc(), FactSales.invoice_no.desc())
        .limit(limit)
    )

    rows = db.execute(stmt).all()
    history = [
        {
            "invoice_no": r.invoice_no,
            "order_date": r.order_date,
            "revenue": round(_to_float(r.revenue), 2),
            "items": _to_int(r.items),
            "distinct_products": _to_int(r.distinct_products),
            "is_cancelled": bool(r.is_cancelled),
        }
        for r in rows
    ]
    # Query descending to get the most recent N, present ascending so the
    # timeline reads left-to-right.
    history.reverse()
    return history


def _top_products(db: Session, customer_id: int, limit: int = 5) -> list[dict[str, Any]]:
    """Highest-revenue products for this customer, cancellations excluded."""
    stmt = (
        select(
            DimProduct.stock_code,
            DimProduct.description,
            DimProduct.category,
            func.sum(FactSales.revenue).label("revenue"),
            func.sum(FactSales.quantity).label("quantity"),
        )
        .join(DimProduct, FactSales.product_id == DimProduct.product_id)
        .where(
            and_(
                FactSales.customer_id == customer_id,
                FactSales.is_cancelled.is_(False),
            )
        )
        .group_by(DimProduct.stock_code, DimProduct.description, DimProduct.category)
        .order_by(func.sum(FactSales.revenue).desc())
        .limit(limit)
    )
    return [
        {
            "stock_code": r.stock_code,
            "description": r.description,
            "category": r.category,
            "revenue": round(_to_float(r.revenue), 2),
            "quantity": _to_int(r.quantity),
        }
        for r in db.execute(stmt).all()
    ]


def get_customer_profile(
    db: Session,
    customer_id: int,
    *,
    history_limit: int = 100,
) -> dict[str, Any]:
    """Full drill-down profile for one customer.

    Raises
    ------
    CustomerNotFoundError
        If the id is absent from ``dim_customer``. Surfaced as a 404 with code
        ``CUSTOMER_NOT_FOUND`` by the global handler — deliberately distinct
        from "exists but has no RFM yet", which returns a profile with null
        scores and ``rfm_computed=False``.
    """
    customer = db.get(DimCustomer, customer_id)
    if customer is None:
        raise CustomerNotFoundError(customer_id)

    history = _order_history(db, customer_id, history_limit)

    # Percentile rank contextualises a customer's value: "£4,200 spent" means
    # little on its own, "top 3% of customers" is immediately actionable.
    revenue_percentile: float | None = None
    if customer.total_revenue:
        below = db.execute(
            select(func.count(DimCustomer.customer_id)).where(
                DimCustomer.total_revenue < customer.total_revenue
            )
        ).scalar_one()
        total = db.execute(select(func.count(DimCustomer.customer_id))).scalar_one()
        if total:
            revenue_percentile = round(_to_int(below) / _to_int(total) * 100, 1)

    return {
        "customer_id": customer.customer_id,
        "country": customer.country,
        "first_purchase_date": customer.first_purchase_date,
        "last_purchase_date": customer.last_purchase_date,
        "tenure_days": customer.tenure_days,
        # ---- RFM ----
        "rfm": {
            "recency_days": customer.rfm_recency,
            "frequency": customer.rfm_frequency,
            "monetary": _to_float(customer.rfm_monetary) if customer.rfm_monetary else None,
            "r_score": customer.r_score,
            "f_score": customer.f_score,
            "m_score": customer.m_score,
            "rfm_score": customer.rfm_score,
            "segment": customer.rfm_segment,
            "health": (
                SEGMENT_HEALTH.get(customer.rfm_segment, "neutral")
                if customer.rfm_segment
                else None
            ),
        },
        "rfm_computed": customer.rfm_segment is not None,
        # ---- Behaviour ----
        "total_orders": customer.total_orders,
        "total_revenue": round(_to_float(customer.total_revenue), 2),
        "total_items": customer.total_items,
        "avg_order_value": (
            round(_to_float(customer.avg_order_value), 2)
            if customer.avg_order_value is not None
            else None
        ),
        "distinct_products": customer.distinct_products,
        "return_rate": (
            round(_to_float(customer.return_rate), 4)
            if customer.return_rate is not None
            else None
        ),
        "avg_days_between_orders": (
            round(_to_float(customer.avg_days_between_orders), 1)
            if customer.avg_days_between_orders is not None
            else None
        ),
        "revenue_percentile": revenue_percentile,
        # ---- Churn ----
        "churn_risk_score": (
            round(_to_float(customer.churn_risk_score), 4)
            if customer.churn_risk_score is not None
            else None
        ),
        "churn_risk_band": risk_band(customer.churn_risk_score),
        "is_churned": customer.is_churned,
        "order_history": history,
        "top_products": _top_products(db, customer_id),
    }


# ---------------------------------------------------------------------------
# Churn risk
# ---------------------------------------------------------------------------


# `risk_band` is re-exported from core.risk rather than defined here. It was
# originally defined in this module, which meant the batch scorer in app.ml had
# no way to reach it without importing a service, and duplicated the thresholds
# instead — with different values. See app/core/risk.py.



def get_at_risk_customers(
    db: Session,
    *,
    threshold: float = 0.5,
    limit: int = 50,
    offset: int = 0,
    segment: str | None = None,
    country: str | None = None,
) -> dict[str, Any]:
    """Customers whose churn probability is at or above ``threshold``.

    Ordered by revenue at stake (probability × historical revenue) rather than
    probability alone. A 95%-likely churner worth £40 matters less than a
    60%-likely churner worth £12,000, and a retention team needs the list
    sorted the way they would actually work it.
    """
    filters = [
        DimCustomer.churn_risk_score.is_not(None),
        DimCustomer.churn_risk_score >= threshold,
    ]
    if segment:
        filters.append(DimCustomer.rfm_segment == segment)
    if country:
        filters.append(DimCustomer.country == country)

    where = and_(*filters)
    expected_loss = DimCustomer.churn_risk_score * DimCustomer.total_revenue

    totals = db.execute(
        select(
            func.count(DimCustomer.customer_id).label("total"),
            func.sum(DimCustomer.total_revenue).label("revenue_at_risk"),
            func.sum(expected_loss).label("expected_revenue_loss"),
        ).where(where)
    ).one()

    total = _to_int(totals.total)

    rows = db.execute(
        select(
            DimCustomer.customer_id,
            DimCustomer.country,
            DimCustomer.rfm_segment,
            DimCustomer.rfm_recency,
            DimCustomer.rfm_frequency,
            DimCustomer.total_revenue,
            DimCustomer.avg_order_value,
            DimCustomer.last_purchase_date,
            DimCustomer.churn_risk_score,
            expected_loss.label("expected_revenue_loss"),
        )
        .where(where)
        .order_by(expected_loss.desc().nullslast())
        .limit(limit)
        .offset(offset)
    ).all()

    items = [
        {
            "customer_id": r.customer_id,
            "country": r.country,
            "segment": r.rfm_segment,
            "recency_days": r.rfm_recency,
            "frequency": r.rfm_frequency,
            "total_revenue": round(_to_float(r.total_revenue), 2),
            "avg_order_value": (
                round(_to_float(r.avg_order_value), 2)
                if r.avg_order_value is not None
                else None
            ),
            "last_purchase_date": r.last_purchase_date,
            "churn_risk_score": round(_to_float(r.churn_risk_score), 4),
            "risk_band": risk_band(r.churn_risk_score),
            "expected_revenue_loss": round(_to_float(r.expected_revenue_loss), 2),
        }
        for r in rows
    ]

    return {
        "items": items,
        "total": total,
        "limit": limit,
        "offset": offset,
        "threshold": threshold,
        "revenue_at_risk": round(_to_float(totals.revenue_at_risk), 2),
        # The probability-weighted figure is the honest forecast; the raw sum
        # above is the worst case where every flagged customer leaves.
        "expected_revenue_loss": round(_to_float(totals.expected_revenue_loss), 2),
        "scored_customers": _to_int(
            db.execute(
                select(func.count(DimCustomer.customer_id)).where(
                    DimCustomer.churn_risk_score.is_not(None)
                )
            ).scalar_one()
        ),
    }


def count_scored_customers(db: Session) -> int:
    """How many customers carry a churn score. Zero means analytics never ran."""
    return _to_int(
        db.execute(
            select(func.count(DimCustomer.customer_id)).where(
                DimCustomer.churn_risk_score.is_not(None)
            )
        ).scalar_one()
    )
