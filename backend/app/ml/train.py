"""Churn model training.

Model choice
------------
Logistic regression, deliberately. A gradient-boosted tree would likely add a
point or two of AUC, but this model's output has to be *explained* to a
commercial stakeholder on a dashboard - "recency contributes +2.1 to the log-odds
of churning" is a defensible sentence, and signed coefficients give the
Statistical Insights page something honest to render. Interpretability is worth
more than a marginal lift here.

Three decisions that materially affect the numbers
--------------------------------------------------
1. **Features are scaled.** Monetary runs to six figures while return_rate is
   bounded by 1. Unscaled, the L2 penalty would effectively only regularise the
   large-magnitude features, and the coefficients would be incomparable. The
   scaler is fitted on the training fold only and saved inside the pipeline, so
   inference applies exactly the same transformation.

2. **class_weight="balanced".** Roughly 60% of customers churn under a 90-day
   definition, and the business cost is asymmetric: missing a churner who was
   about to leave costs their future revenue, while a false positive costs one
   unnecessary discount email. Balanced weighting pushes the model toward
   recall on the churn class.

3. **Stratified split.** The class ratio is preserved across train and test so
   the reported metrics are not an artefact of an unlucky partition.

Threshold
---------
0.5 is reported as the primary operating point because it is the conventional
default, but the metadata also records precision/recall across a sweep of
thresholds. The at-risk endpoint takes a threshold parameter for exactly this
reason: the right cutoff is a commercial decision about campaign budget, not a
property of the model.
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy.engine import Engine

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.ml.features import (
    FEATURE_DESCRIPTIONS,
    FEATURE_NAMES,
    ChurnDataset,
    build_training_dataset,
)

logger = get_logger(__name__)

# Model contract version. Bumped when the feature list or preprocessing
# changes, so a stale artifact can be detected rather than silently misused.
MODEL_VERSION = "1.0.0"

THRESHOLD_SWEEP = [0.3, 0.4, 0.5, 0.6, 0.7]


@dataclass
class TrainingResult:
    """Everything a reviewer needs to judge whether the model works."""

    pipeline: Pipeline
    metrics: dict[str, Any]
    feature_importance: list[dict[str, Any]]
    roc_curve_points: list[dict[str, float]]
    threshold_analysis: list[dict[str, Any]]
    dataset_summary: dict[str, Any]
    trained_at: datetime
    cutoff_date: date

    metadata: dict[str, Any] = field(default_factory=dict)


def _feature_importance(pipeline: Pipeline) -> list[dict[str, Any]]:
    """Signed coefficients, ranked by absolute influence.

    Because inputs are standardised, a coefficient is the change in log-odds per
    one standard deviation of that feature, which makes them directly
    comparable. The sign carries the story: positive pushes toward churn.
    """
    model: LogisticRegression = pipeline.named_steps["model"]
    coefficients = model.coef_[0]

    rows = [
        {
            "feature": name,
            "coefficient": round(float(coef), 4),
            "abs_coefficient": round(abs(float(coef)), 4),
            # exp(coef) is the odds multiplier per standard deviation - the
            # form a non-technical reader can actually act on.
            "odds_ratio": round(float(np.exp(coef)), 4),
            "direction": "increases churn risk" if coef > 0 else "reduces churn risk",
            "description": FEATURE_DESCRIPTIONS.get(name, ""),
        }
        for name, coef in zip(FEATURE_NAMES, coefficients)
    ]
    rows.sort(key=lambda row: row["abs_coefficient"], reverse=True)
    return rows


def _threshold_analysis(y_true: np.ndarray, probabilities: np.ndarray) -> list[dict[str, Any]]:
    """Precision/recall trade-off across candidate operating points."""
    rows = []
    for threshold in THRESHOLD_SWEEP:
        predictions = (probabilities >= threshold).astype(int)
        flagged = int(predictions.sum())
        rows.append({
            "threshold": threshold,
            "customers_flagged": flagged,
            "pct_flagged": round(100 * flagged / len(predictions), 2),
            "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
        })
    return rows


def _roc_points(y_true: np.ndarray, probabilities: np.ndarray, max_points: int = 60) -> list[dict[str, float]]:
    """Downsampled ROC curve for the frontend chart.

    The full curve has one point per unique probability - thousands of them,
    which would bloat the API response for a chart 400px wide.
    """
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    if len(fpr) > max_points:
        indices = np.linspace(0, len(fpr) - 1, max_points).astype(int)
        fpr, tpr = fpr[indices], tpr[indices]
    return [
        {"fpr": round(float(f), 4), "tpr": round(float(t), 4)}
        for f, t in zip(fpr, tpr)
    ]


def train_churn_model(
    engine: Engine,
    *,
    observation_days: int | None = None,
    test_size: float | None = None,
    random_state: int | None = None,
) -> TrainingResult:
    """Train, evaluate and package the churn model."""
    settings = get_settings()
    observation_days = observation_days or settings.churn_threshold_days
    test_size = test_size or settings.model_test_size
    random_state = random_state if random_state is not None else settings.random_seed

    dataset: ChurnDataset = build_training_dataset(
        engine, observation_days=observation_days
    )

    X = dataset.features
    y = dataset.labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,  # preserve the churn ratio in both folds
    )
    logger.info(
        "  Split: %s train / %s test (stratified, churn rate %.1f%% vs %.1f%%).",
        f"{len(X_train):,}", f"{len(X_test):,}",
        y_train.mean() * 100, y_test.mean() * 100,
    )

    # Scaler inside the pipeline so inference cannot forget to apply it.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,          # generous: converges well before this
            random_state=random_state,
            solver="lbfgs",
        )),
    ])

    logger.info("  Fitting logistic regression...")
    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()

    # Cross-validation on the training fold guards against a lucky split: a
    # large gap between CV and test AUC would mean the test score is not
    # reproducible.
    cv_scores = cross_val_score(
        pipeline, X_train, y_train, cv=5, scoring="roc_auc"
    )

    metrics: dict[str, Any] = {
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "pr_auc": round(float(average_precision_score(y_test, probabilities)), 4),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        # Calibration matters: the dashboard shows probabilities as percentages,
        # so they need to mean something. Lower Brier is better.
        "brier_score": round(float(brier_score_loss(y_test, probabilities)), 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "cross_validation": {
            "folds": 5,
            "roc_auc_mean": round(float(cv_scores.mean()), 4),
            "roc_auc_std": round(float(cv_scores.std()), 4),
            "scores": [round(float(s), 4) for s in cv_scores],
        },
        "test_set_size": int(len(y_test)),
        "train_set_size": int(len(y_train)),
        "positive_class_rate": round(float(y_test.mean()), 4),
        # The baseline a coin-flip-by-prevalence model would achieve. Without
        # it, "70% accuracy" sounds good when the majority class is already 60%.
        "majority_class_baseline_accuracy": round(
            float(max(y_test.mean(), 1 - y_test.mean())), 4
        ),
        "decision_threshold": 0.5,
    }

    result = TrainingResult(
        pipeline=pipeline,
        metrics=metrics,
        feature_importance=_feature_importance(pipeline),
        roc_curve_points=_roc_points(y_test.to_numpy(), probabilities),
        threshold_analysis=_threshold_analysis(y_test.to_numpy(), probabilities),
        dataset_summary=dataset.summary(),
        trained_at=datetime.now(timezone.utc),
        cutoff_date=dataset.cutoff_date,
    )
    result.metadata = _build_metadata(result, observation_days, test_size, random_state)
    return result


def _build_metadata(
    result: TrainingResult,
    observation_days: int,
    test_size: float,
    random_state: int,
) -> dict[str, Any]:
    """The model card written alongside the artifact."""
    return {
        "model_version": MODEL_VERSION,
        "model_type": "LogisticRegression",
        "trained_at": result.trained_at.isoformat(),
        "churn_definition": (
            f"A customer is churned if they made no purchase in the "
            f"{observation_days} days following the training cutoff "
            f"({result.cutoff_date.isoformat()})."
        ),
        "leakage_prevention": (
            "Features are computed exclusively from transactions before the "
            "cutoff date; the label is derived exclusively from transactions "
            "after it. Recency is therefore a valid predictor rather than a "
            "restatement of the target, which is why ROC-AUC is realistic "
            "instead of 1.00."
        ),
        "features": FEATURE_NAMES,
        "feature_descriptions": FEATURE_DESCRIPTIONS,
        "feature_count": len(FEATURE_NAMES),
        "preprocessing": "StandardScaler fitted on the training fold only.",
        "hyperparameters": {
            "class_weight": "balanced",
            "solver": "lbfgs",
            "max_iter": 2000,
            "penalty": "l2",
            "random_state": random_state,
        },
        "training_config": {
            "observation_days": observation_days,
            "test_size": test_size,
            "stratified": True,
        },
        "dataset": result.dataset_summary,
        "metrics": result.metrics,
        "feature_importance": result.feature_importance,
        "threshold_analysis": result.threshold_analysis,
        "roc_curve": result.roc_curve_points,
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }


def save_artifacts(result: TrainingResult) -> tuple[Path, Path]:
    """Persist the pipeline and its model card.

    Written atomically: a partially-written joblib is a corrupt model that would
    fail at request time rather than at training time.
    """
    settings = get_settings()
    settings.ensure_directories()

    model_path = settings.model_file
    metadata_path = settings.model_metadata_file
    model_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_model = model_path.with_suffix(".tmp")
    joblib.dump(
        {
            "pipeline": result.pipeline,
            "feature_names": FEATURE_NAMES,
            "model_version": MODEL_VERSION,
            "trained_at": result.trained_at.isoformat(),
        },
        tmp_model,
    )
    tmp_model.replace(model_path)

    tmp_meta = metadata_path.with_suffix(".tmp")
    with tmp_meta.open("w", encoding="utf-8") as handle:
        json.dump(result.metadata, handle, indent=2, ensure_ascii=False)
    tmp_meta.replace(metadata_path)

    logger.info("  Model artifact  -> %s", model_path)
    logger.info("  Model metadata  -> %s", metadata_path)
    return model_path, metadata_path


__all__ = [
    "MODEL_VERSION",
    "TrainingResult",
    "train_churn_model",
    "save_artifacts",
]
