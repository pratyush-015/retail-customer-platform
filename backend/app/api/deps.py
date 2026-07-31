"""Shared FastAPI dependencies.

Query-parameter models are exposed as dependencies so validation rules are
declared once and reused, and so ``/docs`` renders each parameter with its
constraints.
"""

from __future__ import annotations

from datetime import date

from fastapi import Depends, Query

from app.schemas.common import DateRangeParams, PaginationParams


def date_range_params(
    start_date: date | None = Query(
        default=None, description="Inclusive start date (YYYY-MM-DD)."
    ),
    end_date: date | None = Query(
        default=None, description="Inclusive end date (YYYY-MM-DD)."
    ),
) -> DateRangeParams:
    """Validate an optional date window.

    Construction is what triggers the ordering check, which raises
    ``InvalidDateRangeError`` and is rendered by the global handler as a 422
    with code ``INVALID_DATE_RANGE``.
    """
    return DateRangeParams(start_date=start_date, end_date=end_date)


def pagination_params(
    limit: int = Query(default=50, ge=1, le=500, description="Max rows to return."),
    offset: int = Query(default=0, ge=0, description="Rows to skip."),
) -> PaginationParams:
    return PaginationParams(limit=limit, offset=offset)


DateRangeDep = Depends(date_range_params)
PaginationDep = Depends(pagination_params)
