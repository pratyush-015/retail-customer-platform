"""Response models for the KPI and sales endpoints.

Every monetary field is a ``float`` rather than ``Decimal``: the warehouse
columns are ``Numeric`` so SQLAlchemy hands back ``Decimal``, which Pydantic
would serialise as a JSON string and break arithmetic in the frontend.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import Field

from app.schemas.common import APIModel

Granularity = Literal["month", "week"]


class MoMChange(APIModel):
    """Month-over-month deltas that sit under each KPI card.

    Percentages are ``None`` when there is no prior month to compare against,
    or when the prior month was zero. A card showing "+∞%" is worse than one
    that shows no delta at all.
    """

    current_period: str = Field(description="Most recent complete period, e.g. '2011-11'.")
    previous_period: str | None = Field(default=None, description="The period compared against.")
    revenue_pct: float | None = Field(default=None, description="Revenue change, percent.")
    orders_pct: float | None = None
    customers_pct: float | None = None
    aov_pct: float | None = None


class KPISummary(APIModel):
    """Headline numbers for the Executive Overview page."""

    total_revenue: float = Field(description="Net revenue, cancellations excluded.")
    total_orders: int = Field(description="Distinct non-cancelled invoices.")
    total_customers: int = Field(description="Distinct customers who purchased.")
    avg_order_value: float = Field(description="Revenue divided by orders.")
    total_units: int = Field(description="Units sold, net of returns.")
    return_rate_pct: float = Field(
        description="Share of invoices that were cancellations or returns."
    )
    period_start: date | None = Field(default=None, description="Earliest date in scope.")
    period_end: date | None = Field(default=None, description="Latest date in scope.")
    mom: MoMChange | None = Field(
        default=None, description="Null when the window spans fewer than two months."
    )


class TrendPoint(APIModel):
    period: str = Field(description="'2011-03' for month, '2011-W09' for week.")
    period_start: date = Field(description="First calendar day in the bucket, for the x-axis.")
    revenue: float
    orders: int
    customers: int
    units: int


class SalesTrend(APIModel):
    granularity: Granularity
    points: list[TrendPoint]
    total_revenue: float = Field(description="Sum over the returned points.")


class CountryRevenue(APIModel):
    country_id: int
    country: str
    region: str | None = None
    revenue: float
    orders: int
    customers: int
    avg_order_value: float
    revenue_share_pct: float = Field(description="Percent of total revenue in scope.")


class SalesByCountry(APIModel):
    items: list[CountryRevenue]
    total_revenue: float
    country_count: int
