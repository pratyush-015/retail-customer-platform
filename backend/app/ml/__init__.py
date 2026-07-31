"""Machine learning module: churn prediction with temporal split training."""

from app.ml.features import (
    FEATURE_DESCRIPTIONS,
    FEATURE_NAMES,
    ChurnDataset,
    build_scoring_features,
    build_training_dataset,
    compute_features,
    latest_warehouse_date,
    load_transactions,
)
from app.ml.train import MODEL_VERSION, TrainingResult, save_artifacts, train_churn_model

__all__ = [
    "FEATURE_NAMES",
    "FEATURE_DESCRIPTIONS",
    "ChurnDataset",
    "load_transactions",
    "compute_features",
    "build_training_dataset",
    "build_scoring_features",
    "latest_warehouse_date",
    "MODEL_VERSION",
    "TrainingResult",
    "train_churn_model",
    "save_artifacts",
]
