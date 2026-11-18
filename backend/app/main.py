"""FastAPI application factory.

Startup is intentionally tolerant. Neither a missing database nor a missing
model artifact aborts boot:

* The DB connection is retried with exponential backoff, then the app starts
  anyway. A crash-looping container tells a reviewer nothing; a running app
  whose ``/health`` says "database unreachable" tells them exactly what to fix.
* The model is warmed eagerly so the first prediction request does not pay the
  joblib load, but a failure only logs — churn endpoints then return a clean
  503 from the registry.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.health import health
from app.api.router import api_router
from app.core.config import get_settings
from app.core.database import reset_engine, wait_for_database
from app.core.error_handlers import register_exception_handlers
from app.core.exceptions import DatabaseConnectionError
from app.core.logging_config import configure_logging
from app.ml import registry
from app.schemas.common import HealthResponse

logger = logging.getLogger(__name__)

APP_VERSION = "1.0.0"

DESCRIPTION = """
Business intelligence API over the **Online Retail II** dataset: KPIs, RFM
segmentation, hypothesis testing and churn prediction served from a
PostgreSQL star schema.

**Errors** use one shape everywhere:

```json
{"error": {"code": "CUSTOMER_NOT_FOUND", "message": "...", "status": 404}}
```

**Degraded operation** is explicit rather than fatal: if the warehouse is empty
or the churn model artifact is missing, `/api/health` reports which component
is down and the affected endpoints return 503 with instructions instead of
crashing.
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Probe dependencies on boot, release the engine on shutdown."""
    settings = get_settings()
    configure_logging(settings.log_level)

    logger.info(
        "Starting %s v%s (environment=%s)",
        settings.app_name,
        APP_VERSION,
        settings.environment,
    )

    # wait_for_database() signals failure by raising, not by returning a flag —
    # batch scripts want that, since an ETL run against a dead database should
    # stop loudly. The API is the one caller that must survive it, so the
    # exception is downgraded to a log here and reported via /api/health.
    try:
        wait_for_database()
        logger.info("Database connection established.")
    except DatabaseConnectionError as exc:
        logger.error(
            "Database unreachable after retries (%s). Serving anyway; "
            "/api/health will report the failure and data endpoints will "
            "return 503.",
            exc.message,
        )

    # is_available() loads the artifact lazily and swallows load errors, so this
    # both warms the cache and reports status without a second code path.
    if registry.is_available():
        metadata = registry.get_metadata()
        logger.info(
            "Churn model loaded (version=%s, roc_auc=%s)",
            metadata.get("model_version", "unknown"),
            metadata.get("metrics", {}).get("roc_auc", "n/a"),
        )
    else:
        logger.warning(
            "Churn model not available. Churn endpoints will return 503 until "
            "'python scripts/train_churn_model.py' has been run."
        )

    yield

    logger.info("Shutting down; disposing database engine.")
    reset_engine()


def create_app() -> FastAPI:
    """Build the application.

    A factory rather than a module-level singleton so tests can construct an
    isolated instance with overridden settings and dependencies.
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=APP_VERSION,
        description=DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Explicit origin list, not "*": the wildcard cannot be combined with
    # credentialed requests, and locking this down now avoids a surprise later.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)
    app.include_router(api_router, prefix=settings.api_v1_prefix)

    @app.get("/readyz", include_in_schema=False)
    def readiness(result: HealthResponse = Depends(health)) -> JSONResponse:
        return JSONResponse(
            content=result.model_dump(mode="json"),
            status_code=200 if result.status == "ok" else 503,
        )

    if settings.frontend_dist is not None:
        dist = settings.frontend_dist
        index = dist / "index.html"
        if not index.is_file():
            raise RuntimeError(f"Dashboard build is missing: {index}")
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{route:path}", include_in_schema=False)
        def dashboard(route: str) -> FileResponse:
            api_prefix = settings.api_v1_prefix.strip("/")
            if route == api_prefix or route.startswith(f"{api_prefix}/"):
                raise HTTPException(status_code=404, detail="Not Found")
            return FileResponse(index, media_type="text/html", headers={"Cache-Control": "no-cache"})

        return app

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        """Point stray browser traffic at the docs."""
        return {
            "service": settings.app_name,
            "version": APP_VERSION,
            "docs": "/docs",
            "health": f"{settings.api_v1_prefix}/health",
        }

    return app


app = create_app()
