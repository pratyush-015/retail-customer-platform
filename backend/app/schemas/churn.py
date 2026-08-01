"""Response and request models for the churn endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from app.core.risk import RiskBandLiteral
from app.schemas.common import APIModel


class ConfusionMatrix(APIModel):
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int


class FeatureImportance(APIModel):
    feature: str
    coefficient: float
    abs_coefficient: float
    odds_ratio: float = Field(description="exp(coefficient): effect on odds.")
    direction: str
    description: str | None = None


class ROCPoint(APIModel):
    fpr: float
    tpr: float


class ThresholdRow(APIModel):
    """Precision/recall at each candidate cut-off, for the threshold slider."""

    threshold: float
    customers_flagged: int
    pct_flagged: float
    precision: float
    recall: float
    f1: float


class ModelPerformance(APIModel):
    """The model card, reshaped for the performance panel."""

    # Pydantic reserves the ``model_`` prefix for its own methods and warns on
    # any field using it. ``model_version``/``model_type`` are the names the
    # metadata file and the UI already use, so the guard is disabled here rather
    # than renaming the public contract to satisfy an internal convention.
    model_config = APIModel.model_config | {"protected_namespaces": ()}

    model_version: str

    model_type: str
    trained_at: str
    churn_definition: str
    features: list[str]
    roc_auc: float
    pr_auc: float | None = None
    accuracy: float
    precision: float
    recall: float
    f1: float
    brier_score: float | None = None
    baseline_accuracy: float | None = Field(
        default=None,
        description="Majority-class accuracy; without it 74% looks good in a vacuum.",
    )
    confusion_matrix: ConfusionMatrix
    feature_importance: list[FeatureImportance]
    roc_curve: list[ROCPoint] = Field(default_factory=list)
    threshold_analysis: list[ThresholdRow] = Field(default_factory=list)
    cross_validation: dict[str, Any] | None = None
    dataset: dict[str, Any] | None = None


class ChurnPredictionRequest(APIModel):
    """What-if input for POST /api/predict/churn.

    Only R/F/M are required; the rest are derived, because the point is to let a
    user drag three sliders, not to demand ten behavioural features.
    """

    recency: float = Field(ge=0, le=5000, description="Days since last purchase.")
    frequency: float = Field(ge=1, le=10000, description="Distinct orders placed.")
    monetary: float = Field(ge=0, description="Lifetime spend.")

    avg_order_value: float | None = Field(default=None, ge=0)
    tenure_days: float | None = Field(default=None, ge=0, le=5000)
    avg_days_between_orders: float | None = Field(default=None, ge=0, le=5000)
    distinct_products: float | None = Field(default=None, ge=0)
    total_items: float | None = Field(default=None, ge=0)
    return_rate: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def _check_consistency(self) -> "ChurnPredictionRequest":
        """Reject an AOV that contradicts monetary/frequency.

        Scoring an impossible customer silently is worse than a 422.
        """
        if self.avg_order_value is not None and self.frequency:
            implied = self.monetary / self.frequency
            if implied > 0 and not (0.5 * implied <= self.avg_order_value <= 2.0 * implied):
                raise ValueError(
                    f"avg_order_value ({self.avg_order_value:.2f}) is inconsistent "
                    f"with monetary/frequency ({implied:.2f}); omit it to derive it."
                )
        return self


class ChurnPredictionResponse(APIModel):
    churn_probability: float = Field(ge=0.0, le=1.0)
    risk_band: RiskBandLiteral
    model_version: str
    features_used: dict[str, float] = Field(
        description="The full vector actually scored, including derived defaults."
    )
    derived_features: list[str] = Field(default_factory=list)
    interpretation: str


__all__ = [
    "ConfusionMatrix",
    "FeatureImportance",
    "ROCPoint",
    "ThresholdRow",
    "ModelPerformance",
    "ChurnPredictionRequest",
    "ChurnPredictionResponse",
]
