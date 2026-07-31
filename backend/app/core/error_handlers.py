"""Global exception handlers producing one consistent JSON error envelope.

Every error leaving this API - domain error, validation failure, router 404, or
an unanticipated crash - is rendered as:

    {"error": {"code": "...", "message": "...", "status": 4xx, "details": {...}}}

That uniformity is what lets the frontend keep exactly one error-parsing path
instead of guessing at each response shape. Registered in ``app.main.create_app``.

Two deliberate choices:

* **Validation errors keep field-level detail.** FastAPI's default 422 body is a
  bare ``detail`` array whose ``loc`` tuples are awkward to render. We reshape it
  into ``details.fields``, each naming the field, the reason, and the rejected
  value, so the UI can show the message next to the offending input.
* **Unhandled exceptions never leak internals.** The client gets a generic
  message; the stack trace goes to the log with the request path attached.
  Anything reaching that handler is a bug, so it is logged at exception level
  rather than swallowed.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.exceptions import (
    DatabaseConnectionError,
    DatabaseQueryError,
    RetailInsightsError,
)
from app.core.logging_config import get_logger

logger = get_logger(__name__)


def _envelope(
    code: str,
    message: str,
    http_status: int,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    """Build the canonical error response."""
    payload: dict[str, Any] = {"code": code, "message": message, "status": http_status}
    if details:
        payload["details"] = details
    return JSONResponse(status_code=http_status, content={"error": payload})


# Maps Starlette's bare status codes onto our vocabulary, so a router-raised 404
# looks identical to a domain-raised one from the client's perspective.
_HTTP_STATUS_CODES: dict[int, str] = {
    status.HTTP_400_BAD_REQUEST: "BAD_REQUEST",
    status.HTTP_401_UNAUTHORIZED: "UNAUTHORIZED",
    status.HTTP_403_FORBIDDEN: "FORBIDDEN",
    status.HTTP_404_NOT_FOUND: "NOT_FOUND",
    status.HTTP_405_METHOD_NOT_ALLOWED: "METHOD_NOT_ALLOWED",
    status.HTTP_409_CONFLICT: "CONFLICT",
    status.HTTP_422_UNPROCESSABLE_ENTITY: "VALIDATION_ERROR",
    status.HTTP_429_TOO_MANY_REQUESTS: "RATE_LIMITED",
    status.HTTP_500_INTERNAL_SERVER_ERROR: "INTERNAL_ERROR",
    status.HTTP_503_SERVICE_UNAVAILABLE: "SERVICE_UNAVAILABLE",
}


def _friendly_field(loc: tuple[Any, ...]) -> str:
    """Turn a Pydantic ``loc`` tuple into a dotted field path.

    ``("query", "threshold")`` becomes ``"threshold"``; ``("body", "rfm", 0,
    "recency")`` becomes ``"rfm.0.recency"``. The leading request-part marker is
    dropped because the client already knows where it put the value.
    """
    parts = [str(p) for p in loc]
    if parts and parts[0] in {"query", "path", "body", "header", "cookie"}:
        parts = parts[1:]
    return ".".join(parts) if parts else "request"


async def retail_insights_error_handler(
    request: Request, exc: RetailInsightsError
) -> JSONResponse:
    """Handle every deliberately-raised domain error.

    These carry their own code/status/message, so this mostly renders them. 5xx
    domain errors are logged as errors (something is broken); 4xx at info, since
    they are normal client mistakes and would otherwise fill the log with noise.
    """
    if exc.status >= 500:
        logger.error(
            "%s on %s %s: %s",
            exc.code, request.method, request.url.path, exc.message,
        )
    else:
        logger.info(
            "%s on %s %s: %s",
            exc.code, request.method, request.url.path, exc.message,
        )
    return _envelope(exc.code, exc.message, exc.status, exc.details)


async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Reshape Pydantic's 422 into a field-addressable envelope.

    The response names each rejected field, why it was rejected, and what was
    submitted - which is what makes a bad request debuggable from the client
    side alone.
    """
    fields: list[dict[str, Any]] = []
    for error in exc.errors():
        entry: dict[str, Any] = {
            "field": _friendly_field(tuple(error.get("loc", ()))),
            "message": error.get("msg", "Invalid value."),
            "type": error.get("type", "value_error"),
        }
        # ``input`` is absent for some error types; include it only when present
        # and JSON-safe, since e.g. a bytes body is not serialisable.
        if "input" in error:
            value = error["input"]
            if isinstance(value, (str, int, float, bool, type(None))):
                entry["received"] = value
            else:
                entry["received"] = repr(value)[:200]
        fields.append(entry)

    logger.info(
        "Validation failed on %s %s: %s",
        request.method, request.url.path,
        ", ".join(f["field"] for f in fields),
    )

    summary = (
        f"Invalid value for '{fields[0]['field']}'."
        if len(fields) == 1
        else f"{len(fields)} request parameter(s) failed validation."
    )
    return _envelope(
        "VALIDATION_ERROR",
        summary,
        status.HTTP_422_UNPROCESSABLE_ENTITY,
        {"fields": fields},
    )


async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Render framework-raised HTTPExceptions in our envelope."""
    code = _HTTP_STATUS_CODES.get(exc.status_code, f"HTTP_{exc.status_code}")
    detail = exc.detail if isinstance(exc.detail, str) else "Request failed."
    return _envelope(code, detail, exc.status_code)


async def sqlalchemy_error_handler(
    request: Request, exc: SQLAlchemyError
) -> JSONResponse:
    """Translate raw driver errors into domain errors.

    A dropped connection is a transient 503 the client should retry; anything
    else is a genuine 500. Either way the SQL detail stays in the log rather than
    going out over the wire, since it can expose schema and query internals.
    """
    if isinstance(exc, OperationalError):
        logger.error(
            "Database unavailable on %s %s: %s",
            request.method, request.url.path, exc,
        )
        domain: RetailInsightsError = DatabaseConnectionError()
    else:
        logger.exception("Query failed on %s %s", request.method, request.url.path)
        domain = DatabaseQueryError()
    return _envelope(domain.code, domain.message, domain.status, domain.details)


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Last resort: log the full trace, return a generic 500.

    Reaching this handler always means a bug, so the trace is logged at exception
    level with the request path. The client gets no internals.
    """
    logger.exception(
        "Unhandled %s on %s %s",
        type(exc).__name__, request.method, request.url.path,
    )
    return _envelope(
        "INTERNAL_ERROR",
        "An unexpected error occurred. The incident has been logged.",
        status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach every handler to the app.

    Order is irrelevant to Starlette (it dispatches on exception class, most
    specific first), but the list reads narrowest-to-widest.
    """
    app.add_exception_handler(RetailInsightsError, retail_insights_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(SQLAlchemyError, sqlalchemy_error_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)


__all__ = ["register_exception_handlers"]
