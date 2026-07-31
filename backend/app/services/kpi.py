"""Aggregation queries behind the KPI and sales endpoints.

All aggregation happens in SQL. Pulling ~1M fact rows into pandas to sum a
column would work on a laptop and fall over on a small dyno, so the service
returns already-reduced rows.

Two rules are applied consistently and are worth stating once:

* **Cancellations are excluded from revenue.** The ETL keeps them
  (``is_cancelled``) because returns are a real business signal, but they must
  not net against sales in headline figures. They are reported separately as
  ``return_rate_pct``.
* **Orders are distinct invoices, not fact rows.** ``fact_sales`` is one row
  per line item, so ``COUNT(*)`` would overstate order counts several-fold and
  quietly deflate AOV.
"""

from __future__ import annotations

import logging
from datetime import date
from decimal import Decimal
from typing import Any, Literal, Sequence

from sqlalchemy import Select, and_, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.exceptions import DatabaseQueryError
from app.models.star_schema import DimCountry, DimDate, FactSales

logger = logging.getLogger(__name__)

Granularity = Literal["month", "week"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_float(value: Any) -> float:
    """Normalise ``Decimal``/``None`` aggregates to a plain float.

    ``SUM`` over zero rows returns ``NULL``, which must become ``0.0`` rather
    than propagating a ``None`` into a non-optional response field.
    """
    if value is None:
        return 0.0
    if isinstance(value, Decimal):
        return float(value)
    return float(value)


def _to_int(value: Any) -> int:
    return 0 if value is None else int(value)


def _date_predicates(
    start_date: date | None, end_date: date | None
) -> list[Any]:
    """Build date filters against ``dim_date.full_date``.

    Filtering the dimension rather than ``fact_sales.invoice_timestamp`` keeps
    the boundaries inclusive whole days; a timestamp comparison would silently
    drop everything after midnight on ``end_date``.
    """
    predicates: list[Any] = []
    if start_date is not None:
        predicates.append(DimDate.full_date >= start_date)
    if end_date is not None:
        predicates.append(DimDate.full_date <= end_date)
    return predicates


def _base_query(
    *columns: Any,
    start_date: date | None = None,
    end_date: date | None = None,
    include_cancelled: bool = False,
    join_date: bool = False,
) -> Select:
    """Select over ``fact_sales``, joining ``dim_date`` only when needed.

    The join is conditional because the summary endpoint's unfiltered path is
    the hot one, and a needless dimension join there costs real time on a
    million rows.

    ``select_from(FactSales)`` is explicit rather than inferred. Left to itself
    SQLAlchemy derives the FROM clause from the selected columns, so a caller
    selecting only dimension aggregates (e.g. ``min(DimDate.full_date)``) would
    anchor on ``dim_date`` and then fail to join it to itself. The fact table is
    always the anchor of a star-schema query, so state that once here.
    """
    predicates = _date_predicates(start_date, end_date)
    needs_date = join_date or bool(predicates)

    query = select(*columns).select_from(FactSales)
    if needs_date:
        query = query.join(DimDate, FactSales.date_id == DimDate.date_id)
    if not include_cancelled:
        predicates.append(FactSales.is_cancelled.is_(False))
    if predicates:
        query = query.where(and_(*predicates))
    return query


_METRIC_COLUMNS = (
    func.coalesce(func.sum(FactSales.revenue), 0).label("revenue"),
    func.count(func.distinct(FactSales.invoice_no)).label("orders"),
    func.count(func.distinct(FactSales.customer_id)).label("customers"),
    func.coalesce(func.sum(FactSales.quantity), 0).label("units"),
)


def _pct_change(current: float, previous: float | None) -> float | None:
    """Percent change, or ``None`` when the baseline makes it meaningless."""
    if previous is None or previous == 0:
        return None
    return round((current - previous) / abs(previous) * 100, 2)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def get_kpi_summary(
    db: Session,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
) -> dict[str, Any]:
    """Headline KPIs plus month-over-month deltas.

    Returns zeros (not an error) for an empty window: an empty date filter is a
    legitimate question with the answer "nothing happened", and the UI renders
    that as an empty state.
    """
    try:
        totals = db.execute(
            _base_query(*_METRIC_COLUMNS, start_date=start_date, end_date=end_date)
        ).one()

        revenue = _to_float(totals.revenue)
        orders = _to_int(totals.orders)

        span = db.execute(
            _base_query(
                func.min(DimDate.full_date).label("first_day"),
                func.max(DimDate.full_date).label("last_day"),
                start_date=start_date,
                end_date=end_date,
                join_date=True,
            )
        ).one()

        # Cancelled invoices are counted against all invoices, so the rate
        # reflects order-level returns rather than line-level noise.
        cancelled_orders = _to_int(
            db.execute(
                _base_query(
                    func.count(func.distinct(FactSales.invoice_no)),
                    start_date=start_date,
                    end_date=end_date,
                    include_cancelled=True,
                ).where(FactSales.is_cancelled.is_(True))
            ).scalar()
        )
        all_orders = orders + cancelled_orders

        return {
            "total_revenue": round(revenue, 2),
            "total_orders": orders,
            "total_customers": _to_int(totals.customers),
            "avg_order_value": round(revenue / orders, 2) if orders else 0.0,
            "total_units": _to_int(totals.units),
            "return_rate_pct": (
                round(cancelled_orders / all_orders * 100, 2) if all_orders else 0.0
            ),
            "period_start": span.first_day,
            "period_end": span.last_day,
            "mom": _month_over_month(db, start_date=start_date, end_date=end_date),
        }
    except SQLAlchemyError as exc:
        logger.exception("KPI summary aggregation failed")
        raise DatabaseQueryError(
            "Could not compute KPI summary.", details={"operation": "kpi_summary"}
        ) from exc


def _month_over_month(
    db: Session,
    *,
    start_date: date | None,
    end_date: date | None,
) -> dict[str, Any] | None:
    """Compare the two most recent months present in the window.

    The comparison uses the last two months *with data* rather than the
    calendar's previous month, so a gap in the data produces an honest
    comparison instead of a fabricated -100%.
    """
    rows = db.execute(
        _base_query(
            DimDate.year_month.label("period"),
            *_METRIC_COLUMNS,
            start_date=start_date,
            end_date=end_date,
            join_date=True,
        )
        .group_by(DimDate.year_month)
        .order_by(DimDate.year_month.desc())
        .limit(2)
    ).all()

    if not rows:
        return None

    current = rows[0]
    previous = rows[1] if len(rows) > 1 else None

    def _aov(row: Any) -> float | None:
        orders = _to_int(row.orders)
        return _to_float(row.revenue) / orders if orders else None

    current_aov = _aov(current)
    previous_aov = _aov(previous) if previous is not None else None

    return {
        "current_period": current.period,
        "previous_period": previous.period if previous is not None else None,
        "revenue_pct": _pct_change(
            _to_float(current.revenue),
            _to_float(previous.revenue) if previous is not None else None,
        ),
        "orders_pct": _pct_change(
            _to_int(current.orders),
            _to_int(previous.orders) if previous is not None else None,
        ),
        "customers_pct": _pct_change(
            _to_int(current.customers),
            _to_int(previous.customers) if previous is not None else None,
        ),
        "aov_pct": (
            _pct_change(current_aov, previous_aov) if current_aov is not None else None
        ),
    }


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------


def get_sales_trend(
    db: Session,
    *,
    granularity: Granularity = "month",
    start_date: date | None = None,
    end_date: date | None = None,
) -> dict[str, Any]:
    """Revenue time series bucketed by month or ISO week.

    Bucket labels are assembled in Python from integer year/week columns
    instead of via SQL string concatenation, which is spelled differently in
    SQLite and PostgreSQL. The service has to behave identically on both:
    tests run on SQLite, deployments run on Postgres.
    """
    if granularity == "month":
        group_columns: Sequence[Any] = (DimDate.year_month,)
        label_columns: Sequence[Any] = (DimDate.year_month.label("year_month"),)
    else:
        group_columns = (DimDate.year, DimDate.week_of_year)
        label_columns = (
            DimDate.year.label("year"),
            DimDate.week_of_year.label("week_of_year"),
        )

    try:
        rows = db.execute(
            _base_query(
                *label_columns,
                func.min(DimDate.full_date).label("period_start"),
                *_METRIC_COLUMNS,
                start_date=start_date,
                end_date=end_date,
                join_date=True,
            )
            .group_by(*group_columns)
            .order_by(func.min(DimDate.full_date))
        ).all()
    except SQLAlchemyError as exc:
        logger.exception("Sales trend aggregation failed (granularity=%s)", granularity)
        raise DatabaseQueryError(
            "Could not compute the sales trend.",
            details={"operation": "sales_trend", "granularity": granularity},
        ) from exc

    points = [
        {
            "period": (
                row.year_month
                if granularity == "month"
                else f"{row.year}-W{int(row.week_of_year):02d}"
            ),
            "period_start": row.period_start,
            "revenue": round(_to_float(row.revenue), 2),
            "orders": _to_int(row.orders),
            "customers": _to_int(row.customers),
            "units": _to_int(row.units),
        }
        for row in rows
    ]

    return {
        "granularity": granularity,
        "points": points,
        "total_revenue": round(sum(point["revenue"] for point in points), 2),
    }


# ---------------------------------------------------------------------------
# By country
# ---------------------------------------------------------------------------


def get_sales_by_country(
    db: Session,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Revenue split by country, ordered by revenue descending.

    Share percentages are computed against the total of the returned scope so
    the chart's segments always sum to 100% even when ``limit`` truncates the
    long tail.
    """
    try:
        rows = db.execute(
            _base_query(
                DimCountry.country_id.label("country_id"),
                DimCountry.country_name.label("country"),
                DimCountry.region.label("region"),
                *_METRIC_COLUMNS,
                start_date=start_date,
                end_date=end_date,
            )
            .join(DimCountry, FactSales.country_id == DimCountry.country_id)
            .group_by(DimCountry.country_id, DimCountry.country_name, DimCountry.region)
            .order_by(func.sum(FactSales.revenue).desc())
        ).all()
    except SQLAlchemyError as exc:
        logger.exception("Country revenue aggregation failed")
        raise DatabaseQueryError(
            "Could not compute revenue by country.",
            details={"operation": "sales_by_country"},
        ) from exc

    total_revenue = sum(_to_float(row.revenue) for row in rows)
    selected = rows[:limit] if limit else rows

    items = []
    for row in selected:
        revenue = _to_float(row.revenue)
        orders = _to_int(row.orders)
        items.append(
            {
                "country_id": row.country_id,
                "country": row.country,
                "region": row.region,
                "revenue": round(revenue, 2),
                "orders": orders,
                "customers": _to_int(row.customers),
                "avg_order_value": round(revenue / orders, 2) if orders else 0.0,
                "revenue_share_pct": (
                    round(revenue / total_revenue * 100, 2) if total_revenue else 0.0
                ),
            }
        )

    return {
        "items": items,
        "total_revenue": round(total_revenue, 2),
        "country_count": len(rows),
    }
