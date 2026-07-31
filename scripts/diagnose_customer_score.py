"""Decompose a single customer's churn score into per-feature log-odds.

Why this exists
---------------
Written to explain a specific contradiction: customer 17850 was observably
churned (373 days since their last order) yet model v1.0.0 scored them
P(churn)=0.001. Aggregate metrics looked healthy - ROC-AUC 0.80, mean score
tracking the base rate - because ranking in the bulk of the distribution was
fine and only the tail was wrong. No summary statistic was going to surface that.

The decomposition it prints:

    compressed_i   = log1p(x_i) for the skewed features
    z_i            = (compressed_i - mean_i) / scale_i     <- StandardScaler
    contribution_i = coef_i * z_i                          <- log-odds pushed
    logit          = intercept + sum(contributions)

which found frequency at z=14.5 contributing -7.2 to the log-odds while recency
could contribute at most +0.67, because z-scoring a variable with skew 12.7
leaves its tail exactly as far out in sigma units as it started. That motivated
the log1p step now in the pipeline (see ``ml.features.SKEWED_FEATURES``).

Kept as a permanent tool rather than deleted: "why is this one customer scored
like that?" is a question a stakeholder will ask again, and it is the only view
that catches a per-feature problem an AUC cannot see. It cross-checks its own
arithmetic against ``pipeline.predict_proba`` so a wrong decomposition announces
itself instead of implicating the model.

Run:  python scripts/diagnose_customer_score.py [customer_id]
"""

from __future__ import annotations

import sys

import _bootstrap  # noqa: F401  (puts backend/ on sys.path)

import numpy as np

from app.core.database import get_engine
from app.core.logging_config import configure_logging
from app.ml import registry
from app.ml.features import (
    FEATURE_NAMES,
    SKEWED_FEATURES,
    build_scoring_features,
    compress_skew,
    latest_warehouse_date,
)
from datetime import timedelta


def main() -> int:
    configure_logging()
    customer_id = int(sys.argv[1]) if len(sys.argv) > 1 else 17850
    engine = get_engine()

    reference = latest_warehouse_date(engine) + timedelta(days=1)
    features = build_scoring_features(engine, as_of=reference)

    if customer_id not in features.index:
        print(f"Customer {customer_id} has no scoring features.")
        return 1

    bundle = registry.load_model()
    pipeline = bundle.pipeline
    scaler = pipeline.named_steps["scaler"]
    model = pipeline.named_steps["model"]

    row = features.loc[customer_id, FEATURE_NAMES].astype(float)

    # The scaler is fitted *after* compression, so its mean_/scale_ live in
    # log1p space for the skewed columns. Comparing raw values against those
    # statistics produces nonsense (z in the tens of thousands), so the same
    # transform the pipeline applies has to be applied here.
    compressed = compress_skew(row.to_numpy().reshape(1, -1))[0]
    z = (compressed - scaler.mean_) / scaler.scale_
    coefs = model.coef_[0]
    contributions = coefs * z
    logit = float(model.intercept_[0] + contributions.sum())
    probability = 1.0 / (1.0 + np.exp(-logit))

    # Cross-check the hand-computed arithmetic against the pipeline itself. If
    # these disagree, this script is misrepresenting the model and its output
    # cannot be trusted to diagnose anything.
    pipeline_probability = float(
        pipeline.predict_proba(row.to_numpy().reshape(1, -1))[0, 1]
    )

    print("=" * 96)
    print(f"CHURN SCORE DECOMPOSITION — customer {customer_id}")
    print("=" * 96)
    print("'log' marks features transformed with log1p before scaling; their")
    print("'xform' and 'pop mean' columns are on that compressed scale.")
    print("-" * 96)
    print(
        f"{'feature':<26}{'raw':>14}{'xform':>10}{'pop mean':>10}"
        f"{'z':>8}{'coef':>9}{'logit':>10}{'':>4}"
    )
    print("-" * 96)
    order = np.argsort(-np.abs(contributions))
    for i in order:
        name = FEATURE_NAMES[i]
        tag = "log" if name in SKEWED_FEATURES else ""
        print(
            f"{name:<26}{row.iloc[i]:>14,.2f}{compressed[i]:>10.2f}"
            f"{scaler.mean_[i]:>10.2f}{z[i]:>8.2f}{coefs[i]:>9.3f}"
            f"{contributions[i]:>10.3f}{tag:>4}"
        )
    pad = f"{'':>14}{'':>10}{'':>10}{'':>8}{'':>9}"
    print("-" * 96)
    print(f"{'intercept':<26}{pad}{model.intercept_[0]:>10.3f}")
    print(f"{'TOTAL logit':<26}{pad}{logit:>10.3f}")
    print(f"\nP(churn) = {probability:.6f}")
    print(f"P(churn) via pipeline = {pipeline_probability:.6f}", end="  ")
    if abs(probability - pipeline_probability) < 1e-9:
        print("[match]")
    else:
        print("[MISMATCH - this decomposition is wrong, not the model]")

    dominant = FEATURE_NAMES[int(order[0])]
    share = abs(contributions[order[0]]) / np.abs(contributions).sum() * 100
    print(f"Dominant term: {dominant} ({share:.1f}% of total absolute log-odds)")

    print("\nPopulation z-score ranges, post-compression (max z near ~3-5 is healthy)")
    z_all = (
        compress_skew(features[FEATURE_NAMES].to_numpy()) - scaler.mean_
    ) / scaler.scale_
    print(f"{'feature':<26}{'max z':>10}{'p99 z':>10}{'skew':>10}")
    for i, name in enumerate(FEATURE_NAMES):
        col = z_all[:, i]
        skew = float(((col - col.mean()) ** 3).mean() / (col.std() ** 3 + 1e-12))
        print(f"{name:<26}{col.max():>10.1f}{np.percentile(col, 99):>10.1f}{skew:>10.1f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
