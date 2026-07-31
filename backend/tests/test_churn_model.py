"""Tests for churn feature engineering and model training.

The centrepiece is ``test_temporal_split_prevents_label_leakage``, which
demonstrates the failure this design exists to avoid: training on features
computed *after* the cutoff yields a perfect AUC, while the correct temporal
split yields a realistic one. Everything else verifies that each feature means
what its name claims on data whose answers are computed by hand.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.core.exceptions import InsufficientDataError
from app.ml.features import (
    FEATURE_DESCRIPTIONS,
    FEATURE_NAMES,
    compute_features,
)
from app.ml.train import MODEL_VERSION, THRESHOLD_SWEEP, _feature_importance

REFERENCE = date(2011, 12, 10)


def make_transactions(rows: list[dict]) -> pd.DataFrame:
    """Build a transaction frame in the shape ``load_transactions`` returns."""
    frame = pd.DataFrame(rows)
    frame["invoice_date"] = pd.to_datetime(frame["invoice_date"])
    frame["revenue"] = frame["revenue"].astype(float)
    frame["quantity"] = frame["quantity"].astype(int)
    if "is_cancelled" not in frame:
        frame["is_cancelled"] = False
    return frame


def line(
    customer: int,
    invoice: str,
    days_before: int,
    revenue: float = 100.0,
    quantity: int = 10,
    product: int = 1,
    cancelled: bool = False,
) -> dict:
    """One invoice line, dated relative to the reference date."""
    return {
        "customer_id": customer,
        "invoice_no": invoice,
        "product_id": product,
        "quantity": quantity,
        "revenue": revenue,
        "is_cancelled": cancelled,
        "invoice_date": REFERENCE - timedelta(days=days_before),
        "country": "United Kingdom",
    }


# ---------------------------------------------------------------------------
# Feature correctness on hand-computed data
# ---------------------------------------------------------------------------
def test_recency_is_days_since_last_purchase():
    transactions = make_transactions([
        line(1, "A1", days_before=30),
        line(1, "A2", days_before=10),   # most recent
    ])

    features = compute_features(transactions, as_of=REFERENCE)

    assert features.loc[1, "recency"] == 10


def test_frequency_counts_distinct_invoices_not_lines():
    """Three lines on one invoice is one order, not three."""
    transactions = make_transactions([
        line(1, "A1", days_before=10, product=1),
        line(1, "A1", days_before=10, product=2),
        line(1, "A1", days_before=10, product=3),
        line(1, "A2", days_before=5),
    ])

    features = compute_features(transactions, as_of=REFERENCE)

    assert features.loc[1, "frequency"] == 2
    assert features.loc[1, "distinct_products"] == 3


def test_monetary_and_aov_are_consistent():
    transactions = make_transactions([
        line(1, "A1", days_before=20, revenue=300.0),
        line(1, "A2", days_before=10, revenue=100.0),
    ])

    features = compute_features(transactions, as_of=REFERENCE)

    assert features.loc[1, "monetary"] == 400.0
    assert features.loc[1, "avg_order_value"] == 200.0  # 400 / 2 orders


def test_tenure_is_span_between_first_and_last_purchase():
    transactions = make_transactions([
        line(1, "A1", days_before=100),
        line(1, "A2", days_before=40),
    ])

    features = compute_features(transactions, as_of=REFERENCE)

    assert features.loc[1, "tenure_days"] == 60


def test_avg_days_between_orders_uses_gaps_not_order_count():
    """Three orders have two gaps, so tenure is divided by 2, not 3."""
    transactions = make_transactions([
        line(1, "A1", days_before=100),
        line(1, "A2", days_before=55),
        line(1, "A3", days_before=10),
    ])

    features = compute_features(transactions, as_of=REFERENCE)

    assert features.loc[1, "frequency"] == 3
    assert features.loc[1, "tenure_days"] == 90
    assert features.loc[1, "avg_days_between_orders"] == 45.0  # 90 / 2 gaps


def test_as_of_boundary_is_exclusive():
    """A transaction dated exactly on ``as_of`` is outside the window.

    This is not a detail: the label in ``build_training_dataset`` covers
    ``invoice_date >= cutoff``, so if features used ``<=`` the cutoff day's
    orders would land on both sides of the split and leak the outcome.
    """
    transactions = make_transactions([
        line(1, "BEFORE", days_before=1),
        line(1, "ON_THE_DAY", days_before=0),   # exactly as_of
    ])

    features = compute_features(transactions, as_of=REFERENCE)

    assert features.loc[1, "frequency"] == 1, "the as_of day must be excluded"
    assert features.loc[1, "recency"] == 1


def test_single_order_customer_gets_tenure_as_gap_not_zero():
    """A one-off buyer's tenure is 0, which would imply "buys daily" if used
    directly. The fallback must instead express how long they have been silent.
    """
    transactions = make_transactions([line(1, "A1", days_before=200)])

    features = compute_features(transactions, as_of=REFERENCE)

    assert features.loc[1, "frequency"] == 1
    assert features.loc[1, "tenure_days"] == 0
    # 200 days known without a reorder - not 0.
    assert features.loc[1, "avg_days_between_orders"] == 200
    assert features.loc[1, "recency_ratio"] == pytest.approx(1.0)


def test_recency_ratio_flags_overdue_relative_to_own_habit():
    """The point of this feature: same recency, different meaning."""
    transactions = make_transactions([
        # Frequent buyer: ~10-day rhythm, silent 60 days = very overdue.
        line(1, "F1", days_before=90),
        line(1, "F2", days_before=80),
        line(1, "F3", days_before=70),
        line(1, "F4", days_before=60),
        # Occasional buyer: ~180-day rhythm, silent 60 days = normal.
        line(2, "S1", days_before=420),
        line(2, "S2", days_before=240),
        line(2, "S3", days_before=60),
    ])

    features = compute_features(transactions, as_of=REFERENCE)

    assert features.loc[1, "recency"] == features.loc[2, "recency"] == 60
    # Identical recency, but the ratio separates them.
    assert features.loc[1, "recency_ratio"] > 5.0
    assert features.loc[2, "recency_ratio"] < 0.5


def test_return_rate_is_share_of_cancelled_orders():
    transactions = make_transactions([
        line(1, "A1", days_before=30),
        line(1, "A2", days_before=20),
        line(1, "A3", days_before=25),
        line(1, "C1", days_before=10, revenue=-100.0, cancelled=True),
    ])

    features = compute_features(transactions, as_of=REFERENCE)

    # 1 cancelled of 4 total orders.
    assert features.loc[1, "return_rate"] == pytest.approx(0.25)


def test_cancellations_excluded_from_revenue_but_kept_as_signal():
    transactions = make_transactions([
        line(1, "A1", days_before=30, revenue=500.0),
        line(1, "C1", days_before=10, revenue=-500.0, cancelled=True),
    ])

    features = compute_features(transactions, as_of=REFERENCE)

    # Monetary counts the sale only; a netted 0 would erase the customer.
    assert features.loc[1, "monetary"] == 500.0
    assert features.loc[1, "return_rate"] == pytest.approx(0.5)
    # Recency comes from the purchase, not the refund.
    assert features.loc[1, "recency"] == 30


def test_features_exclude_transactions_on_or_after_cutoff():
    """The whole temporal-split guarantee rests on this boundary being strict."""
    transactions = make_transactions([
        line(1, "BEFORE", days_before=100, revenue=100.0),
        line(1, "AFTER", days_before=5, revenue=999.0),
    ])
    cutoff = REFERENCE - timedelta(days=50)

    features = compute_features(transactions, as_of=cutoff)

    # Only the pre-cutoff order may contribute.
    assert features.loc[1, "frequency"] == 1
    assert features.loc[1, "monetary"] == 100.0
    assert features.loc[1, "recency"] == 50  # 100 days before ref = 50 before cutoff


def test_feature_matrix_has_exactly_the_contract_columns():
    """Column order is the model contract - the scaler is positional."""
    transactions = make_transactions([
        line(1, "A1", days_before=20),
        line(2, "B1", days_before=10),
    ])

    features = compute_features(transactions, as_of=REFERENCE)

    assert list(features.columns) == FEATURE_NAMES
    assert features.index.name == "customer_id"


def test_no_non_finite_values_reach_the_model():
    """sklearn raises on NaN/inf, so the matrix must be clean by construction."""
    transactions = make_transactions([
        line(1, "A1", days_before=0, revenue=0.0, quantity=0),   # same-day, zero value
        line(2, "C1", days_before=5, revenue=-50.0, cancelled=True),  # only a refund
        line(3, "B1", days_before=10),
    ])

    features = compute_features(transactions, as_of=REFERENCE)

    assert np.isfinite(features.to_numpy(dtype=float)).all()
    assert not features.isna().any().any()


def test_empty_feature_window_raises_clearly():
    transactions = make_transactions([line(1, "A1", days_before=10)])
    # Cutoff earlier than every transaction.
    early = REFERENCE - timedelta(days=500)

    with pytest.raises(InsufficientDataError) as exc_info:
        compute_features(transactions, as_of=early)

    assert "No transactions exist before" in exc_info.value.message


def test_window_of_only_cancellations_raises():
    transactions = make_transactions([
        line(1, "C1", days_before=10, revenue=-100.0, cancelled=True),
    ])

    with pytest.raises(InsufficientDataError) as exc_info:
        compute_features(transactions, as_of=REFERENCE)

    assert "cancellation" in exc_info.value.message


# ---------------------------------------------------------------------------
# The leakage demonstration
# ---------------------------------------------------------------------------
def _synthetic_history(n_customers: int = 400) -> pd.DataFrame:
    """Customers whose behaviour is predictable but genuinely overlapping.

    Each customer buys on their own rhythm until a personal stop date. The
    rhythms are drawn wide (10-100 days) on purpose, so the two groups blur at
    the cutoff exactly as they do in the real dataset:

    * a slow-but-active buyer (100-day rhythm) can look lapsed at the cutoff
      and still return inside the outcome window
    * a fast buyer who quit (15-day rhythm) can look recent at the cutoff and
      never come back

    That overlap is what keeps the honest model well under a perfect score. An
    earlier version of this fixture gave every group a clean, non-overlapping
    rhythm, and *both* models scored 1.0 - which said more about the fixture
    than about the code.
    """
    rng = np.random.default_rng(42)
    rows: list[dict] = []
    span = 540  # days of history before the reference date

    for customer in range(1, n_customers + 1):
        interval = int(rng.integers(10, 101))     # personal buying rhythm
        # Roughly half keep buying to the end; the rest quit at a random point.
        if rng.random() < 0.5:
            stop = 0
        else:
            stop = int(rng.integers(30, 300))

        order = 0
        day = span
        while day > stop:
            rows.append(line(
                customer, f"{customer}-{order}",
                days_before=max(1, day),
                revenue=float(rng.integers(50, 300)),
                product=int(rng.integers(1, 50)),
            ))
            # Jitter each gap so rhythms are irregular, as real ones are.
            day -= max(1, interval + int(rng.integers(-5, 6)))
            order += 1

        # Guarantee at least one order so the customer exists in the matrix.
        if order == 0:
            rows.append(line(customer, f"{customer}-0", days_before=span))

    return make_transactions(rows)


def _fit_and_score(features: pd.DataFrame, labels: pd.Series) -> float:
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
    ])
    pipeline.fit(features, labels)
    probabilities = pipeline.predict_proba(features)[:, 1]
    return float(roc_auc_score(labels, probabilities))


def test_temporal_split_prevents_label_leakage():
    """The reason this module is structured the way it is.

    Two models are trained on the same customers and the same labels. The only
    difference is *when* the features are measured:

    * leaky   - features measured at the reference date, i.e. after the outcome
                window, so recency directly encodes the label
    * correct - features measured at the cutoff, before the outcome window

    The leaky model scores near-perfectly and the correct one does not. That gap
    is the leakage this design removes.
    """
    transactions = _synthetic_history()
    cutoff = REFERENCE - timedelta(days=90)
    cutoff_ts = pd.Timestamp(cutoff)

    # Labels: did the customer buy in the 90 days after the cutoff?
    returned = set(
        transactions.loc[transactions["invoice_date"] >= cutoff_ts, "customer_id"]
    )

    leaky_features = compute_features(transactions, as_of=REFERENCE)
    correct_features = compute_features(transactions, as_of=cutoff)

    leaky_labels = pd.Series(
        [0 if c in returned else 1 for c in leaky_features.index],
        index=leaky_features.index, dtype=int,
    )
    correct_labels = pd.Series(
        [0 if c in returned else 1 for c in correct_features.index],
        index=correct_features.index, dtype=int,
    )

    # Sanity: both classes present, otherwise AUC is undefined.
    assert 0 < leaky_labels.sum() < len(leaky_labels)
    assert 0 < correct_labels.sum() < len(correct_labels)

    leaky_auc = _fit_and_score(leaky_features, leaky_labels)
    correct_auc = _fit_and_score(correct_features, correct_labels)

    # The leaky model is essentially perfect because post-cutoff recency IS
    # the label restated.
    assert leaky_auc > 0.99, f"expected leakage to give a perfect score, got {leaky_auc}"

    # The honest model still learns the real pattern...
    assert correct_auc > 0.75, f"expected real signal, got {correct_auc}"
    # ...but is measurably worse than the cheating one, which is the point.
    assert correct_auc < leaky_auc


def test_post_cutoff_recency_would_encode_the_label_exactly():
    """Shows *why* the leaky model wins: the threshold is the definition.

    Measured at the reference date, `recency >= 90` reproduces the churn label
    almost perfectly with no model at all.
    """
    transactions = _synthetic_history()
    cutoff = REFERENCE - timedelta(days=90)
    returned = set(
        transactions.loc[
            transactions["invoice_date"] >= pd.Timestamp(cutoff), "customer_id"
        ]
    )

    leaky = compute_features(transactions, as_of=REFERENCE)
    labels = np.array([0 if c in returned else 1 for c in leaky.index])

    # A bare threshold on the leaked feature, no learning involved.
    naive_prediction = (leaky["recency"] >= 90).astype(int).to_numpy()

    agreement = float((naive_prediction == labels).mean())
    assert agreement > 0.99, (
        "post-cutoff recency should be an algebraic restatement of the label; "
        f"agreement was {agreement:.3f}"
    )


# ---------------------------------------------------------------------------
# Training artifacts and reporting
# ---------------------------------------------------------------------------
def test_feature_importance_covers_every_feature_and_is_ranked():
    features = compute_features(_synthetic_history(), as_of=REFERENCE - timedelta(days=90))
    labels = pd.Series(
        (features["recency"] > features["recency"].median()).astype(int),
        index=features.index,
    )
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
    ])
    pipeline.fit(features, labels)

    importance = _feature_importance(pipeline)

    assert len(importance) == len(FEATURE_NAMES)
    assert {row["feature"] for row in importance} == set(FEATURE_NAMES)

    # Ranked by absolute influence, descending.
    magnitudes = [row["abs_coefficient"] for row in importance]
    assert magnitudes == sorted(magnitudes, reverse=True)

    for row in importance:
        # odds_ratio must be exp(coefficient) so the dashboard can quote it.
        assert row["odds_ratio"] == pytest.approx(np.exp(row["coefficient"]), rel=1e-3)
        expected = "increases churn risk" if row["coefficient"] > 0 else "reduces churn risk"
        assert row["direction"] == expected
        assert row["description"], f"{row['feature']} has no description"


def test_every_feature_is_documented():
    """The API surfaces these strings; a missing one becomes a blank tooltip."""
    assert set(FEATURE_DESCRIPTIONS) == set(FEATURE_NAMES)
    for name, description in FEATURE_DESCRIPTIONS.items():
        assert len(description) > 15, f"{name} description is too thin"


def test_threshold_sweep_is_ordered_and_within_range():
    assert THRESHOLD_SWEEP == sorted(THRESHOLD_SWEEP)
    assert all(0 < t < 1 for t in THRESHOLD_SWEEP)
    assert 0.5 in THRESHOLD_SWEEP, "the reported default must appear in the sweep"


def test_model_version_is_semver():
    parts = MODEL_VERSION.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)


# ---------------------------------------------------------------------------
# Trained artifact on disk (skipped when training has not been run)
# ---------------------------------------------------------------------------
@pytest.fixture
def metadata() -> dict:
    from app.core.config import get_settings

    path = get_settings().model_metadata_file
    if not path.exists():
        pytest.skip("model not trained yet; run scripts/train_churn_model.py")
    return json.loads(path.read_text(encoding="utf-8"))


def test_saved_metadata_documents_the_model(metadata: dict):
    assert metadata["model_type"] == "LogisticRegression"
    assert metadata["features"] == FEATURE_NAMES
    assert metadata["feature_count"] == len(FEATURE_NAMES)
    assert metadata["trained_at"]
    # The two claims a reviewer will check first.
    assert "churn_definition" in metadata
    assert "leakage_prevention" in metadata


def test_saved_metrics_are_realistic_not_perfect(metadata: dict):
    """A saved AUC of 1.0 would mean leakage crept back in."""
    metrics = metadata["metrics"]

    assert 0.6 < metrics["roc_auc"] < 0.98, (
        f"ROC-AUC of {metrics['roc_auc']} is outside the believable range - "
        "suspect label leakage or a broken split"
    )
    # Must beat the trivial always-predict-majority model.
    assert metrics["accuracy"] > metrics["majority_class_baseline_accuracy"]

    # Test performance should be reproducible: close to the CV mean.
    cv_mean = metrics["cross_validation"]["roc_auc_mean"]
    assert abs(metrics["roc_auc"] - cv_mean) < 0.1, (
        "test AUC diverges from cross-validated AUC; the split may be lucky"
    )


def test_saved_confusion_matrix_totals_the_test_set(metadata: dict):
    metrics = metadata["metrics"]
    cm = metrics["confusion_matrix"]

    assert sum(cm.values()) == metrics["test_set_size"]

    # Recall must be recomputable from the matrix, or one of them is wrong.
    tp, fn = cm["true_positives"], cm["false_negatives"]
    assert metrics["recall"] == pytest.approx(tp / (tp + fn), abs=1e-3)


def test_saved_artifact_loads_and_scores(metadata: dict):
    """A model that cannot be loaded and used is not a deliverable."""
    import joblib

    from app.core.config import get_settings

    path = get_settings().model_file
    if not path.exists():
        pytest.skip("model artifact missing")

    bundle = joblib.load(path)
    assert bundle["feature_names"] == FEATURE_NAMES
    assert bundle["model_version"] == MODEL_VERSION

    # Score a plausible customer: recent, frequent, valuable.
    sample = pd.DataFrame([[10, 20, 5000.0, 250.0, 400, 20.0, 60, 500, 0.0, 0.5]],
                          columns=FEATURE_NAMES)
    probability = bundle["pipeline"].predict_proba(sample)[0, 1]

    assert 0.0 <= probability <= 1.0
    # A model that called this customer high-risk would be inverted.
    assert probability < 0.5, f"loyal customer scored {probability:.3f}"


def test_roc_curve_is_downsampled_and_monotonic(metadata: dict):
    points = metadata["roc_curve"]

    assert 2 <= len(points) <= 60, "curve should be downsampled for the chart"
    fprs = [p["fpr"] for p in points]
    tprs = [p["tpr"] for p in points]
    assert fprs == sorted(fprs)
    assert tprs == sorted(tprs)
    assert all(0.0 <= v <= 1.0 for v in fprs + tprs)


def test_threshold_analysis_shows_the_expected_tradeoff(metadata: dict):
    """Raising the threshold must trade recall for precision."""
    rows = metadata["threshold_analysis"]
    assert len(rows) == len(THRESHOLD_SWEEP)

    recalls = [row["recall"] for row in rows]
    flagged = [row["customers_flagged"] for row in rows]

    # Stricter threshold flags fewer customers and catches fewer churners.
    assert recalls == sorted(recalls, reverse=True)
    assert flagged == sorted(flagged, reverse=True)
