"""Churn endpoints: at-risk worklist, model performance, and ad-hoc scoring.

The at-risk list reads pre-computed scores out of ``dim_customer`` rather than
invoking the model per request. Scoring 5,000 customers on every page load would
be slow and would also make the table inconsistent with the rest of the
dashboard, which reads the same stored column.

``/model-performance`` and ``/predict`` do need the artifact, so both degrade to
a typed 503 (``MODEL_NOT_LOADED``) when it is missing - the global handler turns
:class:`ModelNotLoadedError` into that shape. A reviewer who has not run the
training script still gets working KPI, sales and segment pages.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.logging_config import get_logger
from app.core.risk import risk_band
from app.ml import registry
from app.ml.features import FEATURE_NAMES
from app.schemas.churn import (
    ChurnPredictionRequest,
    ChurnPredictionResponse,
    ModelPerformance,
)
from app.schemas.common import ErrorResponse
from app.schemas.customer import AtRiskResponse
from app.services import customer as customer_service

logger = get_logger(__name__)

router = APIRouter(tags=["churn"])

MODEL_UNAVAILABLE = {
    503: {"model": ErrorResponse, "description": "Churn model artifact unavailable."}
}


@router.get(
    "/churn/at-risk",
    response_model=AtRiskResponse,
    summary="Customers above a churn-probability threshold",
)
def get_at_risk(
    threshold: float = Query(
        0.5,
        ge=0.0,
        le=1.0,
        description="Minimum churn probability. Bounded so a negative or >1 value is a 422.",
    ),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    segment: str | None = Query(None, max_length=50),
    country: str | None = Query(None, max_length=100),
    db: Session = Depends(get_db),
) -> AtRiskResponse:
    """Retention worklist, ordered by revenue at stake.

    Sorted by ``expected_revenue_loss`` (probability x lifetime revenue) rather
    than probability: a 95%-likely churner worth £40 matters less than a
    60%-likely churner worth £12,000.

    An empty list is a valid answer, not an error - ``scored_customers: 0``
    distinguishes "the model has never run" from "nobody is above this
    threshold", which the UI renders as two different empty states.
    """
    data = customer_service.get_at_risk_customers(
        db, threshold=threshold, limit=limit, offset=offset,
        segment=segment, country=country,
    )
    return AtRiskResponse.model_validate(data)


@router.get(
    "/churn/model-performance",
    response_model=ModelPerformance,
    summary="Evaluation metrics for the trained churn model",
    responses=MODEL_UNAVAILABLE,
)
def get_model_performance() -> ModelPerformance:
    """Confusion matrix, ROC curve, and feature importance from the model card.

    Served from ``model_metadata.json`` written at training time rather than
    recomputed: these are hold-out test-set numbers, and recomputing them here
    would either need the test split (gone) or silently report training-set
    metrics, which flatter the model.
    """
    meta = registry.get_metadata()
    metrics: dict[str, Any] = meta.get("metrics", {})

    return ModelPerformance.model_validate(
        {
            "model_version": meta.get("model_version", "unknown"),
            "model_type": meta.get("model_type", "unknown"),
            "trained_at": meta.get("trained_at", "unknown"),
            "churn_definition": meta.get("churn_definition", ""),
            "features": meta.get("features", []),
            "roc_auc": metrics.get("roc_auc", 0.0),
            "pr_auc": metrics.get("pr_auc"),
            "accuracy": metrics.get("accuracy", 0.0),
            "precision": metrics.get("precision", 0.0),
            "recall": metrics.get("recall", 0.0),
            "f1": metrics.get("f1", 0.0),
            "brier_score": metrics.get("brier_score"),
            "baseline_accuracy": metrics.get("majority_class_baseline_accuracy"),
            "confusion_matrix": metrics.get("confusion_matrix", {}),
            "feature_importance": meta.get("feature_importance", []),
            "roc_curve": meta.get("roc_curve", []),
            "threshold_analysis": meta.get("threshold_analysis", []),
            "cross_validation": metrics.get("cross_validation"),
            "dataset": meta.get("dataset"),
        }
    )


def _build_feature_vector(
    payload: ChurnPredictionRequest,
) -> tuple[dict[str, float], list[str]]:
    """Expand a partial R/F/M payload into the full ten-feature vector.

    Derivations are arithmetic identities where one exists (AOV is monetary over
    frequency; the mean inter-order gap is tenure over the number of gaps, which
    is ``frequency - 1``) and conservative constants otherwise. Every inferred
    field is reported back in ``derived_features`` so a caller can see the score
    rests partly on assumptions rather than on their input.
    """
    derived: list[str] = []

    def fill(name: str, supplied: float | None, fallback: float) -> float:
        if supplied is not None:
            return float(supplied)
        derived.append(name)
        return float(fallback)

    frequency = float(payload.frequency)
    monetary = float(payload.monetary)
    recency = float(payload.recency)

    avg_order_value = fill(
        "avg_order_value", payload.avg_order_value, monetary / frequency if frequency else monetary
    )
    # A single-order customer has no elapsed history to measure; recency is the
    # only time signal available, so tenure falls back to it.
    tenure_days = fill("tenure_days", payload.tenure_days, recency if frequency <= 1 else recency * 2)

    # With f orders there are f-1 gaps. For a one-order customer no gap exists,
    # so recency itself is the most honest stand-in for "their usual cadence".
    gaps = max(frequency - 1.0, 1.0)
    avg_gap = fill(
        "avg_days_between_orders",
        payload.avg_days_between_orders,
        max(tenure_days / gaps, 1.0),
    )

    distinct_products = fill("distinct_products", payload.distinct_products, frequency * 8)
    total_items = fill("total_items", payload.total_items, frequency * 20)
    return_rate = fill("return_rate", payload.return_rate, 0.0)

    # recency_ratio is always computed, never supplied: it is a function of two
    # other features and letting a caller contradict it would be meaningless.
    recency_ratio = recency / max(avg_gap, 1.0)

    vector = {
        "recency": recency,
        "frequency": frequency,
        "monetary": monetary,
        "avg_order_value": avg_order_value,
        "tenure_days": tenure_days,
        "avg_days_between_orders": avg_gap,
        "distinct_products": distinct_products,
        "total_items": total_items,
        "return_rate": return_rate,
        "recency_ratio": recency_ratio,
    }
    # Guard the positional contract: if FEATURE_NAMES gains a feature and this
    # function is not updated, fail loudly here rather than mis-score.
    missing = [name for name in FEATURE_NAMES if name not in vector]
    if missing:  # pragma: no cover - defensive
        raise RuntimeError(f"Feature vector is incomplete: {missing}")
    return vector, derived


def _interpret(probability: float, band: str) -> str:
    """One sentence a non-technical reader can act on."""
    pct = probability * 100
    if band == "high":
        return (
            f"{pct:.0f}% likely to churn - high risk. This customer warrants a "
            "retention offer now."
        )
    if band == "medium":
        return f"{pct:.0f}% likely to churn - worth monitoring, but not yet urgent."
    return f"{pct:.0f}% likely to churn - low risk. Engagement looks healthy."


@router.post(
    "/predict/churn",
    response_model=ChurnPredictionResponse,
    summary="Score a hypothetical customer",
    responses=MODEL_UNAVAILABLE,
)
def predict_churn(payload: ChurnPredictionRequest) -> ChurnPredictionResponse:
    """Return P(churn) for supplied RFM values.

    Stateless: nothing is read from or written to the database, so this can be
    used as a what-if tool for a customer who does not exist yet.
    """
    vector, derived = _build_feature_vector(payload)
    probability = float(registry.predict_proba(vector)[0])
    band = risk_band(probability) or "low"

    return ChurnPredictionResponse(
        churn_probability=round(probability, 4),
        risk_band=band,
        model_version=registry.load_model().version,
        features_used={k: round(v, 4) for k, v in vector.items()},
        derived_features=derived,
        interpretation=_interpret(probability, band),
    )


__all__ = ["router"]
