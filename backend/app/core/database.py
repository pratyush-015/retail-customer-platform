"""Engine/session management with startup retry and exponential backoff.

The retry logic exists for a concrete reason: under ``docker-compose up`` the
API container reliably wins the race against Postgres' first-boot
initialisation. Without a backoff the API crash-loops and the reviewer sees a
wall of stack traces. With it, the API waits, logs each attempt, and comes up
cleanly.
"""

from __future__ import annotations

import time
from collections.abc import Generator, Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.core.exceptions import DatabaseConnectionError
from app.core.logging_config import get_logger

logger = get_logger(__name__)

_engine: Engine | None = None
_SessionFactory: sessionmaker[Session] | None = None


def _engine_kwargs() -> dict:
    """Dialect-appropriate engine options.

    SQLite rejects the pool sizing arguments that Postgres wants, so the two
    cases are configured separately instead of passing a lowest-common
    denominator set that tunes neither well.
    """
    if settings.is_sqlite:
        return {
            # Required because the FastAPI threadpool touches sessions from
            # worker threads.
            "connect_args": {"check_same_thread": False},
            "echo": settings.db_echo,
        }
    return {
        "pool_size": settings.db_pool_size,
        "max_overflow": settings.db_max_overflow,
        # Recycle before typical cloud-provider idle timeouts (Render/Railway
        # drop idle connections at ~5 min) and verify liveness on checkout.
        "pool_pre_ping": True,
        "pool_recycle": 280,
        "echo": settings.db_echo,
    }


def _enable_sqlite_foreign_keys(engine: Engine) -> None:
    """Turn on FK enforcement for SQLite.

    SQLite ignores foreign keys unless explicitly enabled per connection. The
    star schema depends on referential integrity, and tests would otherwise
    pass against SQLite while failing against Postgres.
    """

    @event.listens_for(engine, "connect")
    def _set_pragma(dbapi_connection, _connection_record):  # type: ignore[no-untyped-def]
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def get_engine(force_new: bool = False) -> Engine:
    """Return the process-wide engine, creating it on first use."""
    global _engine
    if _engine is not None and not force_new:
        return _engine

    url = settings.effective_database_url
    safe_url = _redact(url)
    logger.debug("Creating SQLAlchemy engine for %s", safe_url)

    _engine = create_engine(url, **_engine_kwargs())
    if settings.is_sqlite:
        _enable_sqlite_foreign_keys(_engine)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """Return the process-wide session factory."""
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(
            bind=get_engine(),
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )
    return _SessionFactory


def _redact(url: str) -> str:
    """Strip the password from a DB URL so it is safe to log."""
    if "@" not in url or "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    creds, host = rest.rsplit("@", 1)
    user = creds.split(":", 1)[0]
    return f"{scheme}://{user}:***@{host}"


def wait_for_database(
    max_retries: int | None = None,
    base_delay: float | None = None,
    max_delay: float | None = None,
) -> None:
    """Block until the database answers ``SELECT 1``, with exponential backoff.

    Raises
    ------
    DatabaseConnectionError
        If the database is still unreachable after ``max_retries`` attempts.
        The caller decides whether that is fatal (ETL) or degraded (API).
    """
    max_retries = max_retries if max_retries is not None else settings.db_connect_max_retries
    base_delay = base_delay if base_delay is not None else settings.db_connect_base_delay
    max_delay = max_delay if max_delay is not None else settings.db_connect_max_delay

    engine = get_engine()
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            if attempt > 1:
                logger.info("Database connection established on attempt %d.", attempt)
            else:
                logger.debug("Database connection established.")
            return
        except OperationalError as exc:
            last_error = exc
            if attempt == max_retries:
                break
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            logger.warning(
                "Database unreachable (attempt %d/%d): %s. Retrying in %.1fs...",
                attempt,
                max_retries,
                str(exc.orig or exc).strip().splitlines()[0][:160],
                delay,
            )
            time.sleep(delay)
        except SQLAlchemyError as exc:  # pragma: no cover - config-level failure
            # Not a transient connectivity issue (bad driver, malformed URL);
            # retrying cannot help, so fail immediately with a clear message.
            raise DatabaseConnectionError(
                f"Database configuration error: {exc}",
                details={"url": _redact(settings.effective_database_url)},
            ) from exc

    raise DatabaseConnectionError(
        f"Could not connect to the database after {max_retries} attempts.",
        details={
            "url": _redact(settings.effective_database_url),
            "attempts": max_retries,
            "last_error": str(last_error)[:300] if last_error else None,
        },
    )


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope for scripts: commit on success, rollback on error."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a request-scoped session.

    Connectivity failures are translated into ``DatabaseConnectionError`` so the
    global handler returns a clean 503 envelope rather than leaking a SQLAlchemy
    traceback to the client.
    """
    try:
        session = get_session_factory()()
    except SQLAlchemyError as exc:  # pragma: no cover - defensive
        raise DatabaseConnectionError(details={"reason": str(exc)[:200]}) from exc

    try:
        yield session
    except OperationalError as exc:
        session.rollback()
        raise DatabaseConnectionError(
            "Lost connection to the database while serving this request.",
            details={"reason": str(exc.orig or exc)[:200]},
        ) from exc
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engine() -> None:
    """Dispose the engine and clear cached factories (used by tests)."""
    global _engine, _SessionFactory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionFactory = None


__all__ = [
    "get_engine",
    "get_session_factory",
    "wait_for_database",
    "session_scope",
    "get_db",
    "reset_engine",
]
