"""Post-load integrity checks against the warehouse.

Run after an ETL load to confirm the star schema is internally consistent:

    python scripts/verify_warehouse.py

The checks assert properties the ETL is *supposed* to guarantee, so a failure
here means a real regression rather than merely surprising data:

* no duplicate invoice lines (the idempotency contract)
* no orphaned foreign keys (referential integrity across the star)
* revenue reconciles with quantity x unit_price
* the date dimension is a gapless calendar

Checks marked *informational* report a number without asserting anything: they
cover known quirks of the source data, where a non-zero count is expected and
failing the run over it would be wrong.

Every query is expressed with SQLAlchemy Core rather than hand-written SQL, so
the same checks run unmodified against SQLite (the zero-setup default) and
Postgres (the docker-compose/production target). Raw SQL here previously used
``JULIANDAY()`` and compared a boolean to ``1``; both are SQLite-only spellings
that would have failed silently or loudly on Postgres.

Exit code is non-zero when any asserted check fails, so CI can gate on it.
"""

from __future__ import annotations

import operator
import sys
from typing import Callable

import _bootstrap  # noqa: F401  - import side-effect puts `app` on sys.path
from sqlalchemy import Select, func, select
from sqlalchemy.orm import aliased

from app.core.database import get_engine  # noqa: E402
from app.core.logging_config import configure_logging, get_logger  # noqa: E402
from app.models.star_schema import (  # noqa: E402
    DimCountry,
    DimCustomer,
    DimDate,
    DimProduct,
    FactSales,
)

logger = get_logger(__name__)

# Comparators are looked up rather than eval'd: a check is data, and evaluating
# strings to compare two numbers is needless risk for zero benefit.
Predicate = tuple[str, Callable[[float, float], bool], float]
IS_ZERO: Predicate = ("== 0", operator.eq, 0.0)
IS_POSITIVE: Predicate = ("> 0", operator.gt, 0.0)
# Reports a number without asserting on it; never contributes a failure.
IS_INFO: Predicate = ("no assertion", lambda _value, _expected: True, 0.0)

REVENUE_TOLERANCE = 0.01


def _orphan_count(fk_column, dim_table, dim_pk, *, nullable: bool = False) -> Select:
    """Count fact rows whose foreign key has no matching dimension row."""
    stmt = (
        select(func.count())
        .select_from(FactSales)
        .outerjoin(dim_table, fk_column == dim_pk)
        .where(dim_pk.is_(None))
    )
    if nullable:
        # A NULL FK is legitimate (guest checkouts have no customer); only a
        # non-NULL key pointing at nothing is an orphan.
        stmt = stmt.where(fk_column.is_not(None))
    return stmt


def _duplicate_invoice_lines() -> Select:
    """Count (invoice, product, line) triples appearing more than once."""
    grouped = (
        select(FactSales.invoice_no, FactSales.product_id, FactSales.line_number)
        .group_by(FactSales.invoice_no, FactSales.product_id, FactSales.line_number)
        .having(func.count() > 1)
        .subquery()
    )
    return select(func.count()).select_from(grouped)


def _revenue_mismatch() -> Select:
    """Count rows where stored revenue disagrees with quantity x unit_price."""
    computed = FactSales.quantity * FactSales.unit_price
    return (
        select(func.count())
        .select_from(FactSales)
        .where(func.abs(FactSales.revenue - computed) > REVENUE_TOLERANCE)
    )


def _dates_without_facts() -> Select:
    """Count calendar days carrying no sales (expected: non-trading days)."""
    fact = aliased(FactSales)
    return (
        select(func.count())
        .select_from(DimDate)
        .outerjoin(fact, fact.date_id == DimDate.date_id)
        .where(fact.date_id.is_(None))
    )


# (label, statement, expected predicate)
CHECKS: list[tuple[str, Select, Predicate]] = [
    (
        "fact rows loaded",
        select(func.count()).select_from(FactSales),
        IS_POSITIVE,
    ),
    (
        "duplicate invoice lines",
        _duplicate_invoice_lines(),
        IS_ZERO,
    ),
    (
        "orphaned product_id",
        _orphan_count(FactSales.product_id, DimProduct, DimProduct.product_id),
        IS_ZERO,
    ),
    (
        "orphaned date_id",
        _orphan_count(FactSales.date_id, DimDate, DimDate.date_id),
        IS_ZERO,
    ),
    (
        "orphaned country_id",
        _orphan_count(FactSales.country_id, DimCountry, DimCountry.country_id),
        IS_ZERO,
    ),
    (
        "orphaned customer_id",
        _orphan_count(
            FactSales.customer_id, DimCustomer, DimCustomer.customer_id, nullable=True
        ),
        IS_ZERO,
    ),
    (
        "revenue != qty * price (>0.01 tolerance)",
        _revenue_mismatch(),
        IS_ZERO,
    ),
    # Not asserted as zero: one C-prefixed invoice in this dataset carries a
    # positive quantity. That contradiction is real source data, recorded under
    # `cancelled_invoice_with_positive_qty` in the quality report, so failing
    # the load over it would be wrong. It is surfaced here for visibility.
    (
        "cancelled rows with positive revenue",
        select(func.count())
        .select_from(FactSales)
        .where(FactSales.is_cancelled.is_(True), FactSales.revenue > 0),
        IS_INFO,
    ),
    # Dates with no sales are expected (the retailer does not trade daily), so
    # this confirms the calendar is gapless rather than that every day traded.
    (
        "dim_date rows with no matching facts",
        _dates_without_facts(),
        IS_INFO,
    ),
]


def _check_date_continuity(conn) -> tuple[str, float, Predicate]:
    """Assert dim_date has one row per day between its min and max date.

    Computed in Python from the min/max dates rather than in SQL: date
    arithmetic is where dialects diverge most (``JULIANDAY`` vs ``AGE`` vs
    ``DATEDIFF``), and the subtraction is trivial to do here correctly.
    """
    bounds = conn.execute(
        select(func.min(DimDate.full_date), func.max(DimDate.full_date), func.count())
    ).one()
    first, last, row_count = bounds
    if first is None or last is None:
        return ("dim_date gaps (days missing between min and max)", 0.0, IS_ZERO)
    expected_days = (last - first).days + 1
    return (
        "dim_date gaps (days missing between min and max)",
        float(expected_days - row_count),
        IS_ZERO,
    )


SUMMARY_LABELS = [
    ("fact_sales rows", "fact_rows", "{:>14,}"),
    ("customers", "customers", "{:>14,}"),
    ("products", "products", "{:>14,}"),
    ("countries", "countries", "{:>14,}"),
    ("dates", "dates", "{:>14,}"),
    ("gross revenue", "gross_revenue", "{:>14,.2f}"),
    ("net revenue", "net_revenue", "{:>14,.2f}"),
]


def _summary(conn) -> dict[str, float]:
    """Collect headline warehouse numbers for the printout."""
    scalar = lambda stmt: conn.execute(stmt).scalar() or 0  # noqa: E731

    fact_rows = scalar(select(func.count()).select_from(FactSales))
    net_revenue = scalar(select(func.sum(FactSales.revenue)))
    gross_revenue = scalar(
        select(func.sum(FactSales.revenue)).where(FactSales.is_cancelled.is_(False))
    )
    no_customer = scalar(
        select(func.count()).select_from(FactSales).where(FactSales.customer_id.is_(None))
    )

    return {
        "fact_rows": fact_rows,
        "customers": scalar(select(func.count()).select_from(DimCustomer)),
        "products": scalar(select(func.count()).select_from(DimProduct)),
        "countries": scalar(select(func.count()).select_from(DimCountry)),
        "dates": scalar(select(func.count()).select_from(DimDate)),
        "gross_revenue": float(gross_revenue),
        "net_revenue": float(net_revenue),
        "pct_no_customer": (100.0 * no_customer / fact_rows) if fact_rows else 0.0,
    }


TOP_COUNTRIES = (
    select(DimCountry.country_name, func.sum(FactSales.revenue).label("revenue"))
    .join(DimCountry, FactSales.country_id == DimCountry.country_id)
    .where(FactSales.is_cancelled.is_(False))
    .group_by(DimCountry.country_name)
    .order_by(func.sum(FactSales.revenue).desc())
    .limit(5)
)


def main() -> int:
    configure_logging()
    engine = get_engine()

    print("=" * 74)
    print(f"WAREHOUSE INTEGRITY CHECKS  ({engine.dialect.name})")
    print("=" * 74)

    failures = 0
    with engine.connect() as conn:
        results: list[tuple[str, float, Predicate]] = []
        for label, stmt, predicate in CHECKS:
            value = conn.execute(stmt).scalar()
            results.append((label, float(value or 0), predicate))

        # Date continuity needs a round-trip before it can be evaluated, so it
        # is appended rather than declared inline with the static checks.
        results.append(_check_date_continuity(conn))

        for label, value, (description, compare, expected) in results:
            informational = compare is IS_INFO[1]
            ok = compare(value, expected)
            status = "INFO" if informational else ("PASS" if ok else "FAIL")
            if not ok and not informational:
                failures += 1
            print(f"  [{status}] {label:<48} {value:>12,.0f}  (expect {description})")

        print("-" * 74)
        row = _summary(conn)
        print("WAREHOUSE SUMMARY")
        for label, key, fmt in SUMMARY_LABELS:
            print(f"  {label:<22}: {fmt.format(row[key])}")
        print(f"  {'rows w/o customer id':<22}: {row['pct_no_customer']:>13,.2f}%")

        print("-" * 74)
        print("TOP COUNTRIES BY REVENUE (excluding cancellations)")
        for name, revenue in conn.execute(TOP_COUNTRIES):
            print(f"  {name:<24} {float(revenue):>16,.2f}")

    print("=" * 74)
    if failures:
        print(f"RESULT: {failures} check(s) FAILED")
        return 1
    print("RESULT: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
