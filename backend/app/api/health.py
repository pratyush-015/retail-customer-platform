"""Health check.

Deliberately never raises. A health endpoint that 500s when the database is
down is useless for diagnosis — the caller cannot tell "API is dead" from
"API is up, dependency is dead". Instead each component is probed
independently and reported, with the overall status downgraded to
``degraded``.

The endpoint always returns HTTP 200 so container orchestrators keep the
instance alive while a dependency recovers; consumers should read
``status``/``components`` rather than the status code.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import get_db
from app.ml import registry
from app.models.star_schema import FactSales
from app.schemas.common import ComponentHealth, HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

APP_VERSION = "1.0.0"


def _check_database(db: Session) -> ComponentHealth:
    """Probe with a real query against a real table.

    ``SELECT 1`` proves only that a socket is open; counting fact rows proves
    the schema was migrated and the warehouse was actually loaded, which is
    the failure a reviewer running the project fresh is most likely to hit.
    """
    try:
        row_count = db.execute(select(FactSales.sale_id).limit(1)).first()
    except SQLAlchemyError as exc:
        logger.warning("Health check: database unreachable (%s)", exc.__class__.__name__)
        return ComponentHealth(
            name="database",
            status="unavailable",
            detail="Database is unreachable. Check DATABASE_URL and that Postgres is running.",
        )

    if row_count is None:
        return ComponentHealth(
            name="database",
            status="degraded",
            detail="Connected, but fact_sales is empty. Run: python scripts/seed_all.py",
        )
    return ComponentHealth(name="database", status="ok", detail="Connected, warehouse populated.")


def _check_model() -> ComponentHealth:
    """Report model availability without importing it into the request path."""
    if registry.is_available():
        metadata = registry.get_metadata()
        version = metadata.get("model_version", "unknown")
        return ComponentHealth(
            name="churn_model", status="ok", detail=f"Model v{version} loaded."
        )
    return ComponentHealth(
        name="churn_model",
        status="unavailable",
        detail=(
            "Churn model artifact missing or unreadable; churn endpoints will "
            "return 503. Run: python scripts/train_churn_model.py"
        ),
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service and dependency health",
    description=(
        "Always returns 200. Inspect `status` and `components` to distinguish a "
        "healthy service from one running with a degraded dependency."
    ),
)
def health(db: Session = Depends(get_db)) -> HealthResponse:
    settings = get_settings()
    components = [_check_database(db), _check_model()]
    healthy = all(component.status == "ok" for component in components)

    return HealthResponse(
        status="ok" if healthy else "degraded",
        version=APP_VERSION,
        environment=settings.environment,
        components=components,
    )
