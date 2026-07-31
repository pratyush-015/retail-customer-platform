"""Run the ETL pipeline.

Usage
-----
    python scripts/run_etl.py                    # download (or reuse cache) + load
    python scripts/run_etl.py --source path.xlsx # load a specific file
    python scripts/run_etl.py --create-tables    # skip Alembic, create via metadata
    python scripts/run_etl.py --append           # append instead of full refresh
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401  (sys.path setup)

from app.core.exceptions import DataSourceError, DatabaseConnectionError, RetailInsightsError
from app.core.logging_config import configure_logging, get_logger
from app.etl.pipeline import run_etl

logger = get_logger("scripts.run_etl")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the retail ETL pipeline.")
    parser.add_argument("--source", default=None,
                        help="Path to the dataset file (defaults to the cached UCI download).")
    parser.add_argument("--create-tables", action="store_true",
                        help="Create tables from SQLAlchemy metadata instead of requiring Alembic.")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing data instead of a full refresh.")
    parser.add_argument("--log-level", default=None,
                        help="Override the configured log level (DEBUG, INFO, ...).")
    args = parser.parse_args()

    if args.log_level:
        configure_logging(args.log_level, force=True)

    try:
        result = run_etl(
            source=args.source,
            create_tables=args.create_tables,
            replace=not args.append,
        )
    except DataSourceError as exc:
        logger.error("Dataset problem: %s", exc.message)
        if exc.details:
            logger.error("  details: %s", exc.details)
        return 2
    except DatabaseConnectionError as exc:
        logger.error("Database problem: %s", exc.message)
        logger.error(
            "  Is Postgres running? For a zero-dependency run use:\n"
            "    $env:DATABASE_URL='sqlite:///./data/retail_insights.db'"
        )
        return 3
    except RetailInsightsError as exc:
        logger.error("[%s] %s", exc.code, exc.message)
        if exc.details:
            logger.error("  details: %s", exc.details)
        return 4
    except KeyboardInterrupt:  # pragma: no cover
        logger.warning("Interrupted by user.")
        return 130

    # Concise, reviewable summary of what actually landed in the warehouse.
    print()
    print("=" * 74)
    print("ETL SUMMARY")
    print("=" * 74)
    print(f"  Rows read from source : {result.rows_in:>12,}")
    print(f"  Rows after cleaning   : {result.rows_out:>12,}")
    print(f"  Retention             : {result.rows_out / max(result.rows_in, 1) * 100:>11.2f}%")
    print("-" * 74)
    for table, count in result.table_counts.items():
        print(f"  {table:<22}: {count:>12,}")
    print("-" * 74)
    metrics = result.metrics
    if metrics:
        print(f"  Date range            : {metrics.get('date_range_start')} .. {metrics.get('date_range_end')}")
        print(f"  Gross revenue         : {metrics.get('gross_revenue', 0):>12,.2f}")
        print(f"  Returns value         : {metrics.get('returns_value', 0):>12,.2f}")
        print(f"  Net revenue           : {metrics.get('net_revenue', 0):>12,.2f}")
        print(f"  Rows w/o customer id  : {metrics.get('pct_rows_without_customer', 0):>11.2f}%")
    print("-" * 74)
    print(f"  Duration              : {result.duration_seconds:>11.1f}s")
    print(f"  Quality report        : {result.report_path}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
