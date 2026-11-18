"""ETL orchestration: extract -> clean -> transform -> load.

This module wires the stages together and is the single entry point used by
both ``scripts/run_etl.py`` and the test-suite. Keeping the orchestration here
(rather than in the script) means the tests exercise exactly the same code path
a reviewer runs from the command line.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.config import settings
from app.core.database import get_engine, session_scope, wait_for_database
from app.core.exceptions import DataSourceError
from app.core.logging_config import get_logger, log_section
from app.etl.clean import clean_transactions
from app.etl.download import resolve_dataset_path
from app.etl.load import load_star_schema, verify_load
from app.etl.quality import DataQualityReport
from app.etl.transform import transform_all
from app.models import Base

logger = get_logger(__name__)


@dataclass
class ETLResult:
    """Outcome of a pipeline run, returned for logging and assertions."""

    rows_in: int = 0
    rows_out: int = 0
    table_counts: dict[str, int] = field(default_factory=dict)
    report_path: Path | None = None
    duration_seconds: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)


def extract(source: str | Path | None = None) -> pd.DataFrame:
    """Read the source workbook into a single frame.

    The workbook ships two sheets (2009-2010, 2010-2011) which are concatenated
    into one continuous transaction history.
    """
    path = resolve_dataset_path(source)
    logger.info("Reading %s ...", path.name)

    started = time.perf_counter()
    try:
        if path.suffix.lower() == ".csv":
            frames = [pd.read_csv(path, encoding="ISO-8859-1")]
        else:
            workbook = pd.ExcelFile(path)
            logger.info("  Sheets found: %s", workbook.sheet_names)
            frames = []
            for sheet in workbook.sheet_names:
                sheet_df = workbook.parse(sheet)
                logger.info("  %-18s %s rows", sheet, f"{len(sheet_df):,}")
                frames.append(sheet_df)
    except DataSourceError:
        raise
    except Exception as exc:  # malformed workbook, unreadable file
        raise DataSourceError(
            f"Could not read the dataset at {path}: {exc}",
            details={"path": str(path), "error_type": type(exc).__name__},
        ) from exc

    df = pd.concat(frames, ignore_index=True)
    logger.info(
        "  Extracted %s rows in %.1fs",
        f"{len(df):,}",
        time.perf_counter() - started,
    )
    return df


def run_etl(
    source: str | Path | None = None,
    *,
    create_tables: bool = False,
    replace: bool = True,
    report_path: Path | None = None,
) -> ETLResult:
    """Run the full pipeline end to end.

    Parameters
    ----------
    source:
        Explicit dataset path. Defaults to the cached/downloaded UCI file.
    create_tables:
        Create tables via metadata instead of requiring Alembic to have run.
        Used by tests and the throwaway SQLite path.
    replace:
        Full refresh (default) versus append.
    """
    started = time.perf_counter()
    settings.ensure_directories()

    log_section(
        logger,
        "ETL pipeline",
        database=settings.effective_database_url.split("@")[-1],
        mode="full refresh" if replace else "append",
    )

    # ---------------------------------------------------------- 1. extract --
    df_raw = extract(source)

    # ------------------------------------------------------------ 2. clean --
    log_section(logger, "Cleaning & data quality audit")
    report = DataQualityReport(source_file=str(source or "UCI Online Retail II"))
    df_clean = clean_transactions(df_raw, report)
    report.log_summary()

    # -------------------------------------------------------- 3. transform --
    log_section(logger, "Transform to star schema")
    frames = transform_all(df_clean)

    # ------------------------------------------------------------- 4. load --
    log_section(logger, "Load")
    wait_for_database()
    engine = get_engine()

    if create_tables:
        logger.info("Ensuring tables exist (metadata.create_all)...")
        Base.metadata.create_all(engine)

    with session_scope() as session:
        load_star_schema(frames, session, replace=replace)

    with session_scope() as session:
        counts = verify_load(session)

    # ----------------------------------------------------------- 5. report --
    for table, count in counts.items():
        report.add_metric(f"loaded_{table}", count)

    path = report_path or settings.data_quality_report_file
    report.save(path)

    duration = time.perf_counter() - started
    result = ETLResult(
        rows_in=report.rows_in,
        rows_out=report.rows_out,
        table_counts=counts,
        report_path=path,
        duration_seconds=duration,
        metrics=report.metrics,
    )

    log_section(
        logger,
        "ETL complete",
        duration=f"{duration:.1f}s",
        rows_in=f"{result.rows_in:,}",
        rows_loaded=f"{counts.get('fact_sales', 0):,}",
        report=str(path),
    )
    return result


__all__ = ["run_etl", "extract", "ETLResult"]
