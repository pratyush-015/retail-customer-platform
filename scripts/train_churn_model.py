"""Train the churn prediction model and save artifacts.

    python scripts/train_churn_model.py

This is the reviewable proof that the model training works. It prints the
temporal split dates, the train/test metrics, the confusion matrix, the top
feature importance coefficients, and the cross-validation AUC before saving
the .joblib artifact and its JSON metadata.
"""

from __future__ import annotations

import sys

import _bootstrap  # noqa: F401

from app.core.database import get_engine, wait_for_database
from app.core.logging_config import configure_logging, get_logger
from app.ml import TrainingResult, save_artifacts, train_churn_model

logger = get_logger(__name__)

RULE = "=" * 78


def main() -> int:
    configure_logging("INFO", force=True)
    wait_for_database()
    engine = get_engine()

    print(f"\n{RULE}")
    print("  CHURN MODEL TRAINING")
    print(RULE)

    result: TrainingResult = train_churn_model(engine)

    print(f"\n{RULE}")
    print("  TEMPORAL SPLIT")
    print(RULE)
    print(f"  Feature window : {result.dataset_summary['feature_window_days']} days")
    print(f"  Cutoff date    : {result.cutoff_date.isoformat()}")
    print(f"  Max date       : {result.dataset_summary['max_date']}")
    print(f"  Observation    : {result.dataset_summary['observation_days']} days")
    print(f"  Training set   : {result.metrics['train_set_size']:,} customers")
    print(f"  Test set       : {result.metrics['test_set_size']:,} customers")
    print(f"  Churn rate     : {result.dataset_summary['churn_rate_pct']:.1f}%")
    print(f"  Excluded (new) : {result.dataset_summary['excluded_new_customers']:,} customers")

    print(f"\n{RULE}")
    print("  TEST METRICS")
    print(RULE)
    print(f"  ROC-AUC                : {result.metrics['roc_auc']:.4f}")
    print(f"  PR-AUC                 : {result.metrics['pr_auc']:.4f}")
    print(f"  Accuracy               : {result.metrics['accuracy']:.4f}")
    print(f"  Precision              : {result.metrics['precision']:.4f}")
    print(f"  Recall                 : {result.metrics['recall']:.4f}")
    print(f"  F1                     : {result.metrics['f1']:.4f}")
    print(f"  Brier score            : {result.metrics['brier_score']:.4f}")
    print(f"  Decision threshold     : {result.metrics['decision_threshold']}")
    print(f"  Baseline (majority)    : {result.metrics['majority_class_baseline_accuracy']:.4f}")

    print(f"\n{RULE}")
    print("  CONFUSION MATRIX")
    print(RULE)
    cm = result.metrics["confusion_matrix"]
    tn, fp = cm["true_negatives"], cm["false_positives"]
    fn, tp = cm["false_negatives"], cm["true_positives"]

    print("                  Predicted")
    print("                  Retained    Churned")
    print(f"  Actual Retained  {tn:>6,}     {fp:>6,}")
    print(f"         Churned   {fn:>6,}     {tp:>6,}")
    print()
    print(f"  False positive rate: {100 * fp / (fp + tn):.1f}%")
    print(f"  False negative rate: {100 * fn / (fn + tp):.1f}%")

    print(f"\n{RULE}")
    print("  CROSS-VALIDATION (5-fold, train set only)")
    print(RULE)
    cv = result.metrics["cross_validation"]
    print(f"  ROC-AUC mean : {cv['roc_auc_mean']:.4f}")
    print(f"  ROC-AUC std  : {cv['roc_auc_std']:.4f}")
    print(f"  Folds        : {', '.join(f'{s:.4f}' for s in cv['scores'])}")

    print(f"\n{RULE}")
    print("  FEATURE IMPORTANCE (top 5 by absolute coefficient)")
    print(RULE)
    print(f"  {'feature':<28} {'coef':>8} {'odds ratio':>12} {'direction'}")
    print(f"  {'-' * 28} {'-' * 8} {'-' * 12} {'-' * 20}")
    for row in result.feature_importance[:5]:
        print(
            f"  {row['feature']:<28} {row['coefficient']:>8.4f} "
            f"{row['odds_ratio']:>12.4f} {row['direction']}"
        )

    print(f"\n{RULE}")
    print("  THRESHOLD ANALYSIS")
    print(RULE)
    print(f"  {'threshold':>10} {'flagged':>10} {'%':>6} {'precision':>10} {'recall':>10} {'F1':>10}")
    print(f"  {'-' * 10} {'-' * 10} {'-' * 6} {'-' * 10} {'-' * 10} {'-' * 10}")
    for row in result.threshold_analysis:
        print(
            f"  {row['threshold']:>10.1f} {row['customers_flagged']:>10,} "
            f"{row['pct_flagged']:>6.1f} {row['precision']:>10.4f} "
            f"{row['recall']:>10.4f} {row['f1']:>10.4f}"
        )

    print(f"\n{RULE}")
    print("  SAVING ARTIFACTS")
    print(RULE)
    model_path, metadata_path = save_artifacts(result)
    print(f"  ✓ {model_path.relative_to(model_path.parents[4])}")
    print(f"  ✓ {metadata_path.relative_to(metadata_path.parents[4])}")

    print(f"\n{RULE}")
    print(f"  Training complete. ROC-AUC: {result.metrics['roc_auc']:.4f}")
    print(RULE + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
