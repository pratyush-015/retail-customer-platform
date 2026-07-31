"""Customer endpoints: RFM segment distribution and individual drill-down.

Route ordering note: ``/segments`` is declared before ``/{customer_id}``.
FastAPI matches in declaration order, so with the reverse order a request for
``/customers/segments`` would be matched against the path parameter and rejected
as a malformed integer. Declaring the literal path first keeps that from being an
accident waiting to happen if the parameter type ever loosens to a string.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Path, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.common import ErrorResponse

from app.schemas.customer import CustomerProfile, SegmentDistribution
from app.services import customer as customer_service

router = APIRouter(prefix="/customers", tags=["customers"])


@router.get(
    "/segments",
    response_model=SegmentDistribution,
    summary="RFM segment distribution",
    response_description="Per-segment customer counts and revenue contribution.",
)
def get_segments(db: Session = Depends(get_db)) -> SegmentDistribution:
    """Customer count and revenue for every RFM segment.

    Returns ``computed: false`` with an empty list when the analytics step has
    not run yet, rather than an array of zeros. The distinction matters: zeros
    would render as a real (and wrong) chart, while the flag lets the UI tell the
    reviewer to run the pipeline.
    """
    data = customer_service.get_segment_distribution(db)
    return SegmentDistribution.model_validate(data)


@router.get(
    "/{customer_id}",
    response_model=CustomerProfile,
    summary="Individual customer profile",
    responses={404: {"model": ErrorResponse, "description": "No such customer."}},
)
def get_customer(
    customer_id: int = Path(
        ...,
        ge=1,
        description="Customer id as loaded into dim_customer.",
        examples=[17850],
    ),
    history_limit: int = Query(
        100,
        ge=1,
        le=500,
        description="Most recent invoices to include in the timeline.",
    ),
    db: Session = Depends(get_db),
) -> CustomerProfile:
    """Full profile for one customer: RFM, behaviour, churn score, and history.

    Raises ``CustomerNotFoundError`` (404, ``CUSTOMER_NOT_FOUND``) for an unknown
    id. A customer who exists but has not been scored is *not* an error — the
    response carries ``rfm_computed: false`` with null scores, because "we have
    no score for this real customer" and "this customer does not exist" call for
    different handling in the UI.

    ``history_limit`` is capped so a pathological customer with thousands of
    invoices cannot generate an unbounded response.
    """
    data = customer_service.get_customer_profile(
        db, customer_id, history_limit=history_limit
    )
    return CustomerProfile.model_validate(data)
