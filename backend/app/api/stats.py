"""Hypothesis-testing endpoints.

Exposes the tests in ``app.services.statistics`` over HTTP so the findings are
part of the product rather than a notebook cell. Each result carries its
statistic, p-value, effect size and a plain-language interpretation, because a
p-value without an effect size invites the classic error of calling a trivial
difference "significant" on the strength of a large n.

These read via ``Engine`` rather than ``Session``: the service layer uses pandas
``read_sql`` for the group extractions, which wants a connectable.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Query
from sqlalchemy.engine import Engine

from app.core.database import get_engine
from app.core.exceptions import InsufficientDataError, InvalidParameterError, StatisticalTestError
from app.core.logging_config import get_logger
from app.schemas.common import ErrorResponse
from app.schemas.stats import HypothesisTestResult, InsightsResponse
from app.services import statistics as stats_service

logger = get_logger(__name__)

router = APIRouter(prefix="/stats", tags=["statistics"])

TestType = Literal["ttest", "anova_country", "anova_segment", "chi2_country", "chi2_segment"]

ERROR_RESPONSES = {
    422: {"model": ErrorResponse, "description": "Test could not be computed as requested."}
}


@router.get(
    "/hypothesis-test",
    response_model=HypothesisTestResult,
    summary="Run a single hypothesis test",
    responses=ERROR_RESPONSES,
)
def hypothesis_test(
    type: TestType = Query(
        "ttest",
        description=(
            "ttest: AOV between two countries. anova_country / anova_segment: AOV "
            "across groups. chi2_country / chi2_segment: churn independence."
        ),
    ),
    country_a: str | None = Query(None, max_length=100, description="Required for ttest."),
    country_b: str | None = Query(None, max_length=100, description="Required for ttest."),
    top_n: int = Query(8, ge=2, le=20, description="Groups to include in ANOVA/chi-square."),
    alpha: float = Query(0.05, gt=0.0, lt=1.0, description="Significance level."),
) -> HypothesisTestResult:
    """Dispatch to the requested test.

    ``type`` is a ``Literal``, so an unknown value is rejected by FastAPI as a
    422 listing the permitted options rather than reaching this function.

    Underlying failures are already typed: too few observations raises
    ``InsufficientDataError`` (422, ``INSUFFICIENT_DATA``) and a degenerate test
    raises ``StatisticalTestError``. Both are rendered by the global handler, so
    there is no blanket try/except here - an unexpected error should surface as a
    500 with a log entry, not be flattened into a misleading 422.
    """
    engine: Engine = get_engine()

    if type == "ttest":
        if not country_a or not country_b:
            raise InvalidParameterError(
                "A t-test compares two groups, so country_a and country_b are both required.",
                details={"missing": [
                    name for name, value in
                    (("country_a", country_a), ("country_b", country_b)) if not value
                ]},
            )
        if country_a.strip().lower() == country_b.strip().lower():
            raise InvalidParameterError(
                "country_a and country_b must differ; comparing a group with itself "
                "always yields p=1.",
                details={"country_a": country_a, "country_b": country_b},
            )
        result = stats_service.aov_by_country(engine, country_a, country_b, alpha=alpha)
    elif type == "anova_country":
        result = stats_service.aov_across_countries(engine, top_n=top_n, alpha=alpha)
    elif type == "anova_segment":
        result = stats_service.aov_across_segments(engine, alpha=alpha)
    elif type == "chi2_country":
        result = stats_service.churn_by_country(engine, top_n=top_n, alpha=alpha)
    else:  # chi2_segment
        result = stats_service.churn_by_segment(engine, alpha=alpha)

    return HypothesisTestResult.model_validate(result.to_dict())


@router.get(
    "/insights",
    response_model=InsightsResponse,
    summary="All pre-defined tests for the Statistical Insights page",
)
def insights(
    alpha: float = Query(0.05, gt=0.0, lt=1.0),
    top_n: int = Query(8, ge=2, le=20),
) -> InsightsResponse:
    """Run the standard battery in one call.

    A test that cannot run (e.g. RFM segments not computed yet, or a country with
    fewer than 30 orders) is reported in ``unavailable`` with its reason while the
    others still return. Failing the whole page because one test lacks data would
    hide the four that worked.

    Only the two domain-level data errors are caught per-test; anything else
    propagates, so a genuine bug is not silently listed as "unavailable".
    """
    engine: Engine = get_engine()

    planned = [
        ("aov_across_countries", lambda: stats_service.aov_across_countries(engine, top_n=top_n, alpha=alpha)),
        ("aov_across_segments", lambda: stats_service.aov_across_segments(engine, alpha=alpha)),
        ("churn_by_country", lambda: stats_service.churn_by_country(engine, top_n=top_n, alpha=alpha)),
        ("churn_by_segment", lambda: stats_service.churn_by_segment(engine, alpha=alpha)),
        ("aov_uk_vs_germany", lambda: stats_service.aov_by_country(
            engine, "United Kingdom", "Germany", alpha=alpha)),
    ]

    tests: list[HypothesisTestResult] = []
    unavailable: list[dict[str, str]] = []

    for name, run in planned:
        try:
            tests.append(HypothesisTestResult.model_validate(run().to_dict()))
        except (InsufficientDataError, StatisticalTestError) as exc:
            logger.info("Skipping test %s: %s", name, exc)
            unavailable.append({"test": name, "reason": str(exc)})

    return InsightsResponse(tests=tests, unavailable=unavailable, alpha=alpha)


__all__ = ["router"]
