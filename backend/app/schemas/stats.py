"""Response models for the hypothesis-testing endpoints.

Mirrors ``app.services.statistics.TestResult``. Several numeric fields are
optional because ``TestResult.to_dict`` replaces non-finite values with ``None``:
scipy returns NaN for degenerate inputs, and emitting a bare NaN would produce
invalid JSON that breaks the frontend's parser rather than just the chart.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from app.schemas.common import APIModel


class GroupSummary(APIModel):
    """One group's summary, used for the accompanying chart.

    Every field except ``name`` is optional and ``extra`` is allowed, because the
    tests emit genuinely different group shapes: a t-test or ANOVA group carries
    ``n``/``mean``/``std``, while a chi-square contingency row carries a count per
    category (``Active``, ``Churned``). Forcing one shape onto both would mean
    either dropping the contingency counts the chart needs or inventing a
    meaningless ``n`` for them.
    """

    model_config = APIModel.model_config | {"extra": "allow"}

    name: str
    n: int | None = None
    mean: float | None = None
    median: float | None = None
    std: float | None = None



class HypothesisTestResult(APIModel):
    """One statistical test, with its interpretation in plain language."""

    model_config = APIModel.model_config | {"extra": "allow"}

    test_type: str = Field(description="e.g. 'independent t-test', 'one-way ANOVA'.")
    question: str = Field(description="The business question in plain English.")
    null_hypothesis: str
    statistic: float | None = Field(default=None, description="Test statistic.")
    p_value: float | None = None
    significant: bool = Field(description="Whether p < alpha.")
    alpha: float
    interpretation: str = Field(
        description="Plain-language reading, safe to render directly in the UI."
    )
    effect_size: float | None = None
    effect_size_name: str | None = Field(
        default=None, description="e.g. \"Cohen's d\", 'eta-squared', \"Cramer's V\"."
    )
    effect_size_label: str | None = Field(
        default=None, description="negligible | small | medium | large."
    )
    degrees_of_freedom: float | None = None
    groups: list[GroupSummary] = Field(default_factory=list)
    assumptions: dict[str, Any] = Field(
        default_factory=dict,
        description="Assumption checks (normality, equal variance) so the reader can judge validity.",
    )
    details: dict[str, Any] = Field(default_factory=dict)


class InsightsResponse(APIModel):
    """Every pre-defined test, for the Statistical Insights page in one call.

    Individual tests can fail on insufficient data without invalidating the
    others, so failures are reported per-test in ``unavailable`` rather than
    failing the whole response.
    """

    tests: list[HypothesisTestResult]
    unavailable: list[dict[str, str]] = Field(
        default_factory=list,
        description="Tests that could not run, each with a reason.",
    )
    alpha: float = Field(description="Significance level applied to all tests.")


__all__ = ["GroupSummary", "HypothesisTestResult", "InsightsResponse"]
