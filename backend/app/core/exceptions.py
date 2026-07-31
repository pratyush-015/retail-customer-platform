"""Domain-specific exception hierarchy.

Rather than sprinkling bare ``try/except Exception`` blocks around the codebase,
each *known* failure mode gets its own exception type carrying:

* a stable, machine-readable ``code`` (e.g. ``CUSTOMER_NOT_FOUND``) that the
  frontend can branch on without string-matching prose,
* the correct HTTP ``status`` code,
* a human-readable ``message``,
* optional structured ``details`` for debugging.

A single global handler (see ``app.core.error_handlers``) renders every one of
these into the same JSON envelope:

    {"error": {"code": "...", "message": "...", "status": 404, "details": {...}}}

Unknown/unexpected exceptions are the only ones that become a generic 500, and
they are logged with a full stack trace so nothing fails silently.
"""

from __future__ import annotations

from typing import Any


class RetailInsightsError(Exception):
    """Base class for every error this application raises deliberately."""

    code: str = "INTERNAL_ERROR"
    status: int = 500
    message: str = "An unexpected error occurred."

    def __init__(
        self,
        message: str | None = None,
        *,
        details: dict[str, Any] | None = None,
        code: str | None = None,
        status: int | None = None,
    ) -> None:
        self.message = message or self.__class__.message
        self.details = details or {}
        if code is not None:
            self.code = code
        if status is not None:
            self.status = status
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        """Render the error into the canonical API envelope."""
        payload: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "status": self.status,
        }
        if self.details:
            payload["details"] = self.details
        return {"error": payload}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"{self.__class__.__name__}(code={self.code!r}, "
            f"status={self.status}, message={self.message!r})"
        )


# ---------------------------------------------------------------------------
# Infrastructure errors
# ---------------------------------------------------------------------------
class DatabaseConnectionError(RetailInsightsError):
    """Raised when the database is unreachable after all retries.

    Surfaces as 503 (not 500) because it is a transient dependency failure -
    the correct client behaviour is to retry, which the dashboard's retry
    button does.
    """

    code = "DATABASE_CONNECTION_ERROR"
    status = 503
    message = (
        "Could not connect to the database. The service may still be starting "
        "up - please retry in a few seconds."
    )


class DatabaseQueryError(RetailInsightsError):
    """Raised when a query fails for a reason that is not the caller's fault."""

    code = "DATABASE_QUERY_ERROR"
    status = 500
    message = "A database query failed while serving this request."


# ---------------------------------------------------------------------------
# Request / validation errors
# ---------------------------------------------------------------------------
class InvalidDateRangeError(RetailInsightsError):
    """Raised when a caller supplies an incoherent date window."""

    code = "INVALID_DATE_RANGE"
    status = 422
    message = "The supplied date range is invalid."

    @classmethod
    def start_after_end(cls, start: Any, end: Any) -> "InvalidDateRangeError":
        return cls(
            f"start_date ({start}) must be on or before end_date ({end}).",
            details={"start_date": str(start), "end_date": str(end)},
        )


class CustomerNotFoundError(RetailInsightsError):
    """Raised when a customer id does not exist in ``dim_customer``."""

    code = "CUSTOMER_NOT_FOUND"
    status = 404
    message = "The requested customer does not exist."

    def __init__(self, customer_id: Any = None, **kwargs: Any) -> None:
        message = kwargs.pop("message", None)
        if message is None and customer_id is not None:
            message = f"Customer {customer_id} was not found."
        details = kwargs.pop("details", None) or {}
        if customer_id is not None:
            details.setdefault("customer_id", customer_id)
        super().__init__(message, details=details, **kwargs)


class SegmentNotFoundError(RetailInsightsError):
    """Raised when an unknown RFM segment name is requested."""

    code = "SEGMENT_NOT_FOUND"
    status = 404
    message = "The requested customer segment does not exist."


class InvalidParameterError(RetailInsightsError):
    """Raised for semantically invalid params Pydantic cannot express alone."""

    code = "INVALID_PARAMETER"
    status = 422
    message = "One or more request parameters are invalid."


# ---------------------------------------------------------------------------
# Analytics / ML errors
# ---------------------------------------------------------------------------
class ModelNotLoadedError(RetailInsightsError):
    """Raised when the churn model artifact is missing or corrupted.

    Deliberately a 503 rather than a 500: the API itself is healthy, but this
    particular capability is unavailable until ``scripts/train_model.py`` has
    been run. Endpoints that need the model degrade instead of taking down the
    whole server.
    """

    code = "MODEL_NOT_LOADED"
    status = 503
    message = (
        "The churn model is not available. Run `python scripts/train_model.py` "
        "to train and persist the model artifact."
    )


class InsufficientDataError(RetailInsightsError):
    """Raised when a computation cannot run because the dataset is too small."""

    code = "INSUFFICIENT_DATA"
    status = 422
    message = "There is not enough data to perform this computation."


class StatisticalTestError(RetailInsightsError):
    """Raised when a hypothesis test cannot be computed as requested."""

    code = "STATISTICAL_TEST_ERROR"
    status = 422
    message = "The requested statistical test could not be computed."


# ---------------------------------------------------------------------------
# ETL errors
# ---------------------------------------------------------------------------
class DataSourceError(RetailInsightsError):
    """Raised when the raw dataset cannot be downloaded or read."""

    code = "DATA_SOURCE_ERROR"
    status = 500
    message = "The source dataset could not be downloaded or read."


class ETLValidationError(RetailInsightsError):
    """Raised when incoming data fails a structural precondition.

    Row-level dirt (bad dates, missing ids, negative quantities) is *not* an
    error - it is cleaned and audited. This is reserved for structural
    problems, e.g. the source file is missing required columns entirely.
    """

    code = "ETL_VALIDATION_ERROR"
    status = 500
    message = "The source data failed structural validation."


__all__ = [
    "RetailInsightsError",
    "DatabaseConnectionError",
    "DatabaseQueryError",
    "InvalidDateRangeError",
    "CustomerNotFoundError",
    "SegmentNotFoundError",
    "InvalidParameterError",
    "ModelNotLoadedError",
    "InsufficientDataError",
    "StatisticalTestError",
    "DataSourceError",
    "ETLValidationError",
]
