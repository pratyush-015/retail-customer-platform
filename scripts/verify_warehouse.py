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

Exit code is non-zero when any asserted check fails, so CI can gate on it.
"""

from __future__ import annotations

import operator
import sys
from typing import Callable

import _bootstrap  # noqa: F401  - import side-effect puts `app` on sys.path
from sqlalchemy import text

from app.core.database import get_engine  # noqa: E402
from app.core.logging_config import configure_logging, get_logger  # noqa: E402

logger = get_logger(__name__)

# Comparators are looked up rather than eval'd: a check is data, and evaluating
# strings to compare two numbers is needless risk for zero benefit.
Predicate = tuple[str, Callable[[float, float], bool], float]
IS_ZERO: Predicate = ("== 0", operator.eq, 0.0)
IS_POSITIVE: Predicate = ("> 0", operator.gt, 0.0)
# Reports a number without asserting on it; never contributes a failure.
IS_INFO: Predicate = ("no assertion", lambda _value, _expected: True, 0.0)


# (label, SQL returning a single number, expected predicate)
CHECKS: list[tuple[str, str, Predicate]] = [
    (
        "fact rows loaded",
        "SELECT COUNT(*) FROM fact_sales",
        IS_POSITIVE,
    ),
    (
        "duplicate invoice lines",
        """SELECT COUNT(*) FROM (
               SELECT invoice_no, product_id, line_number
               FROM fact_sales
               GROUP BY 1, 2, 3
               HAVING COUNT(*) > 1
           ) d""",
        IS_ZERO,
    ),
    (
        "orphaned product_id",
        """SELECT COUNT(*) FROM fact_sales f
           LEFT JOIN dim_product p ON f.product_id = p.product_id
           WHERE p.product_id IS NULL""",
        IS_ZERO,
    ),
    (
        "orphaned date_id",
        """SELECT COUNT(*) FROM fact_sales f
           LEFT JOIN dim_date d ON f.date_id = d.date_id
           WHERE d.date_id IS NULL""",
        IS_ZERO,
    ),
    (
        "orphaned country_id",
        """SELECT COUNT(*) FROM fact_sales f
           LEFT JOIN dim_country c ON f.country_id = c.country_id
           WHERE c.country_id IS NULL""",
        IS_ZERO,
    ),
    (
        "orphaned customer_id",
        """SELECT COUNT(*) FROM fact_sales f
           LEFT JOIN dim_customer c ON f.customer_id = c.customer_id
           WHERE f.customer_id IS NOT NULL AND c.customer_id IS NULL""",
        IS_ZERO,
    ),
    (
        "revenue != qty * price (>0.01 tolerance)",
        """SELECT COUNT(*) FROM fact_sales
           WHERE ABS(revenue - (quantity * unit_price)) > 0.01""",
        IS_ZERO,
    ),
    # Not asserted as zero: one C-prefixed invoice in this dataset carries a
    # positive quantity. That contradiction is real source data, recorded under
    # `cancelled_invoice_with_positive_qty` in the quality report, so failing
    # the load over it would be wrong. It is surfaced here for visibility.
    (
        "cancelled rows with positive revenue",
        "SELECT COUNT(*) FROM fact_sales WHERE is_cancelled = 1 AND revenue > 0",
        IS_INFO,
    ),
    (
        "dim_date gaps (days missing between min and max)",
        """SELECT (JULIANDAY(MAX(full_date)) - JULIANDAY(MIN(full_date)) + 1) - COUNT(*)
           FROM dim_date""",
        IS_ZERO,
    ),
    # Dates with no sales are expected (the retailer does not trade daily), so
    # this confirms the calendar is gapless rather than that every day traded.
    (
        "dim_date rows with no matching facts",
        """SELECT COUNT(*) FROM dim_date d
           LEFT JOIN fact_sales f ON f.date_id = d.date_id
           WHERE f.date_id IS NULL""",
        IS_INFO,
    ),
]

SUMMARY = """
SELECT
    (SELECT COUNT(*) FROM fact_sales)                                   AS fact_rows,
    (SELECT COUNT(*) FROM dim_customer)                                 AS customers,
    (SELECT COUNT(*) FROM dim_product)                                  AS products,
    (SELECT COUNT(*) FROM dim_country)                                  AS countries,
    (SELECT COUNT(*) FROM dim_date)                                     AS dates,
    (SELECT ROUND(SUM(revenue), 2) FROM fact_sales)                     AS net_revenue,
    (SELECT ROUND(SUM(revenue), 2) FROM fact_sales WHERE is_cancelled=0) AS gross_revenue,
    (SELECT ROUND(100.0 * SUM(CASE WHEN customer_id IS NULL THEN 1 ELSE 0 END)
                  / COUNT(*), 2) FROM fact_sales)                       AS pct_no_customer
"""

TOP_COUNTRIES = """
SELECT c.country_name, ROUND(SUM(f.revenue), 2) AS revenue
FROM fact_sales f
JOIN dim_country c ON f.country_id = c.country_id
WHERE f.is_cancelled = 0
GROUP BY c.country_name
ORDER BY revenue DESC
LIMIT 5
"""


def main() -> int:
    configure_logging()
    engine = get_engine()

    print("=" * 74)
    print("WAREHOUSE INTEGRITY CHECKS")
    print("=" * 74)

    failures = 0
    with engine.connect() as conn:
        for label, sql, (description, compare, expected) in CHECKS:
            value = conn.execute(text(sql)).scalar_one()
            value = 0.0 if value is None else float(value)
            informational = compare is IS_INFO[1]
            ok = compare(value, expected)
            if informational:
                status = "INFO"
            else:
                status = "PASS" if ok else "FAIL"
            if not ok and not informational:
                failures += 1
            print(f"  [{status}] {label:<48} {value:>12,.0f}  (expect {description})")

        print("-" * 74)
        row = conn.execute(text(SUMMARY)).mappings().one()
        print("WAREHOUSE SUMMARY")
        print(f"  fact_sales rows       : {row['fact_rows']:>14,}")
        print(f"  customers             : {row['customers']:>14,}")
        print(f"  products              : {row['products']:>14,}")
        print(f"  countries             : {row['countries']:>14,}")
        print(f"  dates                 : {row['dates']:>14,}")
        print(f"  gross revenue         : {row['gross_revenue']:>14,.2f}")
        print(f"  net revenue           : {row['net_revenue']:>14,.2f}")
        print(f"  rows w/o customer id  : {row['pct_no_customer']:>13,.2f}%")

        print("-" * 74)
        print("TOP COUNTRIES BY REVENUE (excluding cancellations)")
        for name, revenue in conn.execute(text(TOP_COUNTRIES)):
            print(f"  {name:<24} {revenue:>16,.2f}")

    print("=" * 74)
    if failures:
        print(f"RESULT: {failures} check(s) FAILED")
        return 1
    print("RESULT: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
