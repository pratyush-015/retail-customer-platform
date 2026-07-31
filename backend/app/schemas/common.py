"""Shared Pydantic models: query parameters and the standard error envelope.

Validation lives here rather than in the routers so that the same rules apply
everywhere a date range or pagination window is accepted, and so the OpenAPI
schema documents the constraints instead of hiding them in handler code.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.exceptions import InvalidDateRangeError

T = TypeVar("T")


class APIModel(BaseModel):
    """Base for every response model.

    ``from_attributes`` lets routers return SQLAlchemy rows directly, and
    forbidding extras means a typo in a service dict fails loudly in tests
    rather than quietly dropping a field the frontend expects.
    """

    model_config = ConfigDict(from_attributes=True, extra="forbid")


# ---------------------------------------------------------------------------
# Error envelope
# ---------------------------------------------------------------------------


class FieldError(APIModel):
    """One failed field in a 422, flattened out of Pydantic's nested report."""

    field: str = Field(description="Dotted path, e.g. 'body.recency'.")
    message: str = Field(description="Why this field was rejected.")
    type: str | None = Field(default=None, description="Pydantic error type.")


class ErrorDetail(APIModel):
    model_config = ConfigDict(from_attributes=True, extra="allow")

    code: str = Field(description="Stable machine-readable code, e.g. CUSTOMER_NOT_FOUND.")
    message: str = Field(description="Human-readable explanation.")
    status: int = Field(description="HTTP status code, mirrored for convenience.")
    details: dict[str, Any] | None = Field(
        default=None, description="Context such as the offending field list."
    )


class ErrorResponse(APIModel):
    """The single shape every non-2xx response uses.

    Declared so it shows up in ``/docs`` — an error contract the frontend can
    code against is only useful if it is documented.
    """

    error: ErrorDetail


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class ComponentHealth(APIModel):
    name: str
    status: Literal["ok", "degraded", "unavailable"]
    detail: str | None = None


class HealthResponse(APIModel):
    status: Literal["ok", "degraded"] = Field(
        description="'degraded' when a dependency is down but the API still serves."
    )
    version: str
    environment: str
    components: list[ComponentHealth]


# ---------------------------------------------------------------------------
# Query parameter models
# ---------------------------------------------------------------------------


class DateRangeParams(BaseModel):
    """An optional, half-open-friendly date window shared by the sales endpoints."""

    model_config = ConfigDict(extra="forbid")

    start_date: date | None = Field(
        default=None, description="Inclusive lower bound (YYYY-MM-DD)."
    )
    end_date: date | None = Field(
        default=None, description="Inclusive upper bound (YYYY-MM-DD)."
    )

    @model_validator(mode="after")
    def _validate_order(self) -> "DateRangeParams":
        """Reject inverted ranges.

        Raising the domain exception (not ``ValueError``) keeps the response
        code ``INVALID_DATE_RANGE`` instead of a generic validation failure:
        the caller gets a message about the range, not about a field.
        """
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise InvalidDateRangeError.start_after_end(self.start_date, self.end_date)
        return self


class PaginationParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    limit: int = Field(default=50, ge=1, le=500, description="Max rows to return.")
    offset: int = Field(default=0, ge=0, description="Rows to skip.")


class Page(APIModel, Generic[T]):
    """Envelope for list endpoints so the UI can paginate without guessing."""

    items: list[T]
    total: int = Field(description="Total matching rows, ignoring limit/offset.")
    limit: int
    offset: int

    @property
    def has_more(self) -> bool:
        return self.offset + len(self.items) < self.total
