"""Idempotent loading of star-schema frames into the warehouse.

Idempotency strategy
--------------------
Running the ETL twice must not double the warehouse. Two mechanisms enforce
that:

1. **Truncate-and-reload inside one transaction** (the default). The source is
   a static historical extract, so a full refresh is both correct and simple:
   either the whole warehouse is replaced or nothing changes. There is no
   window where a reader sees half-loaded data.

2. **The ``uq_fact_sales_line`` unique constraint** guarantees that even an
   append-mode load cannot create duplicate invoice lines.

Bulk inserts use pandas ``to_sql`` with explicit chunking, on the same
connection as the deletes.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import delete, func, select
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.orm import Session

from app.core.logging_config import get_logger
from app.models import DimCountry, DimCustomer, DimDate, DimProduct, FactSales

logger = get_logger(__name__)

# Multi-row INSERTs bind one parameter *per column per row*, and both engines
# cap the number of bound parameters in a single statement. A fixed row-count
# chunk therefore silently overflows as soon as a table is wide enough, so the
# chunk size is derived from the column count at run time instead.
#
#   SQLite   : 999 for builds < 3.32, 32 766 after. 999 is assumed for safety.
#   Postgres : 32 767 (int16 parameter index).
MAX_BIND_PARAMS = {"sqlite": 900, "postgresql": 30_000}
DEFAULT_MAX_BIND_PARAMS = 900

# Upper bound on rows per statement, so a narrow table doesn't produce an
# enormous single INSERT.
MAX_ROWS_PER_CHUNK = 5_000


# Children before parents - FK constraints are enforced, so ordering matters.
TRUNCATE_ORDER = ["fact_sales", "dim_customer", "dim_product", "dim_country", "dim_date"]
LOAD_ORDER = ["dim_date", "dim_country", "dim_product", "dim_customer", "fact_sales"]

_MODEL_BY_TABLE = {
    "dim_date": DimDate,
    "dim_country": DimCountry,
    "dim_product": DimProduct,
    "dim_customer": DimCustomer,
    "fact_sales": FactSales,
}


def truncate_warehouse(session: Session) -> dict[str, int]:
    """Delete all warehouse rows, children first.

    ``DELETE`` is used rather than ``TRUNCATE`` because it is portable across
    SQLite and Postgres and participates in the surrounding transaction, which
    is what makes the reload atomic.
    """
    deleted: dict[str, int] = {}
    for table in TRUNCATE_ORDER:
        model = _MODEL_BY_TABLE[table]
        count = session.execute(select(func.count()).select_from(model)).scalar_one()
        if count:
            session.execute(delete(model))
            logger.info("  Cleared %-14s %s existing rows", table, f"{count:,}")
        deleted[table] = int(count)
    return deleted


def chunk_size_for(engine: Engine, n_columns: int) -> int:
    """Largest safe row count per multi-row INSERT for this engine and width.

    ``method="multi"`` binds ``n_columns`` parameters per row, so the row limit
    is the dialect's parameter cap divided by the width. Without this, wide
    tables raise "too many SQL variables" on SQLite once the chunk exceeds the
    cap - a failure that only appears at load time, not in unit tests using
    narrow fixtures.
    """
    cap = MAX_BIND_PARAMS.get(engine.dialect.name, DEFAULT_MAX_BIND_PARAMS)
    # max(1, ...) keeps a pathologically wide table loadable one row at a time
    # rather than producing chunksize=0, which pandas rejects.
    return max(1, min(MAX_ROWS_PER_CHUNK, cap // max(1, n_columns)))


def _bulk_insert(connection: Connection, table: str, frame: pd.DataFrame) -> int:
    """Insert a frame in batches within the warehouse transaction."""
    if frame.empty:
        logger.warning("  %s: nothing to insert (empty frame)", table)
        return 0

    uses_sqlite = connection.dialect.name == "sqlite"
    # SQLite executemany binds a single row per execution, so the chunk
    # limits Python memory without increasing the bound parameters per SQL statement.
    chunk = MAX_ROWS_PER_CHUNK if uses_sqlite else chunk_size_for(connection.engine, len(frame.columns))
    frame.to_sql(
        table,
        connection,
        if_exists="append",
        index=False,
        chunksize=chunk,
        method=None if uses_sqlite else "multi",
    )

    logger.info("  Loaded  %-14s %s rows", table, f"{len(frame):,}")
    return len(frame)


def load_star_schema(
    frames: dict[str, pd.DataFrame],
    session: Session,
    *,
    replace: bool = True,
) -> dict[str, int]:
    """Load every frame into the warehouse.

    Parameters
    ----------
    replace:
        When True (default) the warehouse is emptied first, making the load a
        full refresh and therefore idempotent.

    Returns
    -------
    Row counts actually written, per table.
    """
    if replace:
        logger.info("Clearing existing warehouse data (full refresh)...")
        truncate_warehouse(session)

    logger.info("Loading star schema into the database...")
    counts: dict[str, int] = {}
    connection = session.connection()
    for table in LOAD_ORDER:
        frame = frames.get(table)
        if frame is None:
            logger.warning("  %s: no frame supplied, skipping", table)
            counts[table] = 0
            continue
        counts[table] = _bulk_insert(connection, table, _prepare(table, frame))

    return counts


def _prepare(table: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Align a frame's columns with its table, dropping ETL-only helpers.

    Frames carry working columns (``has_customer``, ``stock_code`` on the fact)
    that have no column in the target table. Passing them to ``to_sql`` would
    raise, so they are dropped explicitly here rather than being silently
    tolerated somewhere upstream.
    """
    model = _MODEL_BY_TABLE[table]
    table_columns = {c.name for c in model.__table__.columns}
    keep = [c for c in frame.columns if c in table_columns]
    dropped = [c for c in frame.columns if c not in table_columns]
    if dropped:
        logger.debug("  %s: dropping non-column fields %s", table, dropped)
    return frame[keep]


def verify_load(session: Session) -> dict[str, int]:
    """Count rows per table after a load, for the run summary."""
    counts: dict[str, int] = {}
    for table, model in _MODEL_BY_TABLE.items():
        counts[table] = int(
            session.execute(select(func.count()).select_from(model)).scalar_one()
        )
    return counts


__all__ = [
    "load_star_schema",
    "truncate_warehouse",
    "verify_load",
    "chunk_size_for",
]
