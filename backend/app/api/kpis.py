"""KPI summary endpoint (Executive Overview header cards)."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import date_range_params
from app.core.database import get_db
from app.schemas.common import DateRangeParams, ErrorResponse
from app.schemas.kpi import KPISummary
from app.services import kpi as kpi_service

router = APIRouter(prefix="/kpis", tags=["kpis"])


@router.get(
    "/summary",
    response_model=KPISummary,
    summary="Headline KPIs with month-over-month deltas",
    description=(
        "Total revenue, orders, customers, AOV and units for the requested "
        "window, plus month-over-month change. Cancelled invoices are excluded "
        "from revenue and reported separately as `return_rate_pct`."
    ),
    responses={
        422: {"model": ErrorResponse, "description": "start_date is after end_date."},
        503: {"model": ErrorResponse, "description": "Database unavailable."},
    },
)
def kpi_summary(
    date_range: DateRangeParams = Depends(date_range_params),
    db: Session = Depends(get_db),
) -> KPISummary:
    summary = kpi_service.get_kpi_summary(
        db, start_date=date_range.start_date, end_date=date_range.end_date
    )
    return KPISummary.model_validate(summary)
