"""Compute RFM segmentation over the loaded warehouse and persist it.

Usage
-----
    python scripts/run_rfm.py              # compute, print summary, persist
    python scripts/run_rfm.py --dry-run    # compute and print, write nothing
    python scripts/run_rfm.py --reference-date 2011-12-10

The reference date defaults to the latest transaction in the warehouse. It is
exposed as a flag so the segmentation can be reproduced as-of an earlier date,
which is also how the churn model builds its training window.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

import _bootstrap  # noqa: F401  (sys.path setup)

from app.core.database import get_engine, wait_for_database
from app.core.exceptions import DatabaseConnectionError, RetailInsightsError
from app.core.logging_config import configure_logging, get_logger
from app.services.rfm import (
    SEGMENT_DESCRIPTIONS,
    compute_rfm,
    dataset_max_date,
    load_transactions,
    persist_rfm,
    summarise_segments,
)

logger = get_logger("scripts.run_rfm")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute and persist RFM segmentation.")
    parser.add_argument("--reference-date", default=None,
                        help="ISO date to measure recency against "
                             "(defaults to the latest transaction in the warehouse).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute and print the summary without writing to dim_customer.")
    parser.add_argument("--log-level", default=None,
                        help="Override the configured log level (DEBUG, INFO, ...).")
    args = parser.parse_args()

    if args.log_level:
        configure_logging(args.log_level, force=True)

    reference_date: date | None = None
    if args.reference_date:
        try:
            reference_date = date.fromisoformat(args.reference_date)
        except ValueError:
            logger.error(
                "--reference-date must be an ISO date (YYYY-MM-DD), got %r.",
                args.reference_date,
            )
            return 2

    try:
        # get_engine() is lazy - it does not open a socket - so without this
        # the first query fails with a raw SQLAlchemy traceback instead of the
        # retried, human-readable DatabaseConnectionError below.
        wait_for_database()
        engine = get_engine()
        transactions = load_transactions(engine)
        if reference_date is None:
            reference_date = dataset_max_date(engine)
        rfm = compute_rfm(transactions, reference_date=reference_date)
        summary = summarise_segments(rfm)
        persisted = 0 if args.dry_run else persist_rfm(engine, rfm)
    except DatabaseConnectionError as exc:
        logger.error("Database problem: %s", exc.message)
        logger.error(
            "  Is the database running? For a zero-dependency run use:\n"
            "    $env:DATABASE_URL='sqlite:///./data/retail_insights.db'"
        )
        return 3
    except RetailInsightsError as exc:
        logger.error("[%s] %s", exc.code, exc.message)
        if exc.details:
            logger.error("  details: %s", exc.details)
        if exc.code == "INSUFFICIENT_DATA":
            logger.error("  Has the ETL run? Try: python scripts/run_etl.py")
        return 4
    except KeyboardInterrupt:  # pragma: no cover
        logger.warning("Interrupted by user.")
        return 130

    total_customers = len(rfm)
    total_revenue = float(rfm["monetary"].sum())

    print()
    print("=" * 96)
    print("RFM SEGMENTATION SUMMARY")
    print("=" * 96)
    print(f"  Reference date (recency measured against) : {reference_date}")
    print(f"  Customers segmented                       : {total_customers:>12,}")
    print(f"  Net revenue attributed                    : {total_revenue:>12,.2f}")
    print(f"  Median recency / frequency / monetary     : "
          f"{rfm['recency'].median():.0f}d / "
          f"{rfm['frequency'].median():.0f} orders / "
          f"{rfm['monetary'].median():,.2f}")
    print("-" * 96)
    header = (
        f"  {'Segment':<21}{'Custs':>8}{'Cust%':>8}{'Revenue':>15}{'Rev%':>8}"
        f"{'AvgRec':>9}{'AvgFreq':>9}{'AvgAOV':>10}"
    )
    print(header)
    print("-" * 96)
    for name, row in summary.iterrows():
        print(
            f"  {name:<21}{int(row['customers']):>8,}{row['pct_customers']:>7.1f}%"
            f"{row['total_revenue']:>15,.0f}{row['pct_revenue']:>7.1f}%"
            f"{row['avg_recency']:>8.0f}d{row['avg_frequency']:>9.1f}"
            f"{row['avg_order_value']:>10,.0f}"
        )
    print("-" * 96)

    # The headline business read: a small share of customers usually carries a
    # disproportionate share of revenue, and that concentration is the whole
    # argument for segment-targeted retention spend.
    top_segments = summary.nlargest(3, "pct_revenue")
    print("  Revenue concentration:")
    for name, row in top_segments.iterrows():
        print(
            f"    {name:<21} {row['pct_customers']:>5.1f}% of customers "
            f"-> {row['pct_revenue']:>5.1f}% of revenue"
        )
    print("-" * 96)

    at_risk_names = ["At Risk", "Cannot Lose Them", "About To Sleep", "Hibernating"]
    at_risk = summary[summary.index.isin(at_risk_names)]
    if not at_risk.empty:
        print(
            f"  Lapsing/lapsed segments: {int(at_risk['customers'].sum()):,} customers "
            f"({at_risk['pct_customers'].sum():.1f}%) holding "
            f"{at_risk['total_revenue'].sum():,.0f} "
            f"({at_risk['pct_revenue'].sum():.1f}%) of historical revenue."
        )
        for name in at_risk.index:
            print(f"    - {name}: {SEGMENT_DESCRIPTIONS[name].split('.')[0]}.")
    print("-" * 96)
    if args.dry_run:
        print("  Dry run: dim_customer was NOT modified.")
    else:
        print(f"  Persisted RFM to dim_customer for {persisted:,} customers.")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    sys.exit(main())
