"""Sales trend and country breakdown endpoints."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import date_range_params
from app.core.database import get_db
from app.schemas.common import DateRangeParams, ErrorResponse
from app.schemas.kpi import SalesByCountry, SalesTrend
from app.services import kpi as kpi_service

router = APIRouter(prefix="/sales", tags=["sales"])


@router.get(
    "/trend",
    response_model=SalesTrend,
    summary="Revenue time series",
    description=(
        "Revenue, orders, customers and units bucketed by month or ISO week. "
        "An unknown `granularity` is rejected with 422 by the enum constraint."
    ),
    responses={422: {"model": ErrorResponse, "description": "Invalid granularity or date range."}},
)
def sales_trend(
    granularity: Literal["month", "week"] = Query(
        default="month", description="Bucket size for the series."
    ),
    date_range: DateRangeParams = Depends(date_range_params),
    db: Session = Depends(get_db),
) -> SalesTrend:
    trend = kpi_service.get_sales_trend(
        db,
        granularity=granularity,
        start_date=date_range.start_date,
        end_date=date_range.end_date,
    )
    return SalesTrend.model_validate(trend)


@router.get(
    "/by-country",
    response_model=SalesByCountry,
    summary="Revenue breakdown by country",
    description=(
        "Countries ordered by revenue descending. `limit` truncates the long "
        "tail for charting; `country_count` always reflects the full total."
    ),
    responses={422: {"model": ErrorResponse, "description": "Invalid limit or date range."}},
)
def sales_by_country(
    limit: int | None = Query(
        default=None, ge=1, le=200, description="Return only the top N countries."
    ),
    date_range: DateRangeParams = Depends(date_range_params),
    db: Session = Depends(get_db),
) -> SalesByCountry:
    breakdown = kpi_service.get_sales_by_country(
        db,
        start_date=date_range.start_date,
        end_date=date_range.end_date,
        limit=limit,
    )
    return SalesByCountry.model_validate(breakdown)
