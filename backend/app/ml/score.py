"""Batch scoring: write churn predictions back onto ``dim_customer``.

Why a separate step exists
--------------------------
``train.py`` produces a model; ``registry.predict_proba`` scores a payload on
demand. Neither one fills in ``dim_customer.churn_risk_score``, and without that
column populated the churn dashboard has nothing to rank: ``/api/churn/at-risk``
would correctly return an empty list, which looks identical to "no customers are
at risk" while actually meaning "nobody ever ran inference".

Scoring every customer once and persisting the result is also the right shape for
the read path. Ranking thousands of customers by risk is a sort over an indexed
column, not thousands of pickle invocations per request.

Two columns, two different meanings
-----------------------------------
* ``is_churned`` is **observed fact**: this customer's recency, measured at the
  warehouse's last date, exceeds the churn window. No model involved.
* ``churn_risk_score`` is the **model's forecast**: P(no purchase in the next 90
  days) given behaviour up to now.

Keeping both is what makes the dashboard honest. Already-churned customers score
near 1.0 and would otherwise drown out the interesting cases, so the API can
filter on ``is_churned = false`` to surface *active* customers who are drifting -
the ones where intervention is still worth something.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd
from sqlalchemy import bindparam, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from app.core.exceptions import InsufficientDataError
from app.core.logging_config import get_logger
from app.core.risk import RISK_BAND_LABELS, risk_band
from app.ml import registry
from app.ml.features import build_scoring_features, latest_warehouse_date
from app.models.star_schema import DimCustomer


logger = get_logger(__name__)

# Must match the training definition in features.build_training_dataset.
CHURN_WINDOW_DAYS = 90

# Band thresholds live in app.core.risk, imported above. They were duplicated
# here with different cut-points (0.75/0.50/0.25 and a fourth "critical" band),
# which would have written a label the API schema rejects.


@dataclass
class ScoringResult:
    """Outcome of a scoring run, for the CLI summary and the seed pipeline."""

    scored: int
    updated: int
    as_of: date
    churn_window_days: int
    already_churned: int
    mean_score: float
    band_counts: dict[str, int] = field(default_factory=dict)
    revenue_at_risk: float = 0.0

    def summary(self) -> dict:
        return {
            "customers_scored": self.scored,
            "rows_updated": self.updated,
            "as_of": self.as_of.isoformat(),
            "churn_window_days": self.churn_window_days,
            "already_churned": self.already_churned,
            "mean_churn_probability": round(self.mean_score, 4),
            "risk_bands": self.band_counts,
            "active_revenue_at_risk": round(self.revenue_at_risk, 2),
        }


def score_all_customers(engine: Engine, *, as_of: date | None = None) -> pd.DataFrame:
    """Compute churn probability for every customer with purchase history.

    Returns a frame indexed by ``customer_id`` with the probability, the observed
    churn flag and the recency it was derived from.

    Raises:
        ModelNotLoadedError: if the artifact is missing or corrupt (surfaced from
            the registry - scoring cannot be faked).
        InsufficientDataError: if the warehouse has no customer-attributed sales.
    """
    reference = as_of or (latest_warehouse_date(engine) + timedelta(days=1))

    # Features are computed as of `reference`, so recency here is measured from
    # the same instant the model was asked about. Mixing reference dates between
    # features and labels is exactly the bug the temporal split guards against.
    features = build_scoring_features(engine, as_of=reference)
    if features.empty:  # pragma: no cover - build_scoring_features raises first
        raise InsufficientDataError("No customers available to score.")

    probabilities = registry.predict_proba(features)

    result = pd.DataFrame(
        {
            "churn_risk_score": probabilities.round(4),
            "recency": features["recency"].astype(int),
            "monetary": features["monetary"].astype(float),
        },
        index=features.index,
    )
    result["is_churned"] = result["recency"] >= CHURN_WINDOW_DAYS

    logger.info(
        "  Scored %s customers as of %s (mean P(churn)=%.3f, %s already churned).",
        f"{len(result):,}",
        reference.isoformat(),
        float(result["churn_risk_score"].mean()),
        f"{int(result['is_churned'].sum()):,}",
    )
    return result


def persist_churn_scores(engine: Engine, scores: pd.DataFrame) -> int:
    """Write ``churn_risk_score`` and ``is_churned`` onto ``dim_customer``.

    Uses a single executemany UPDATE keyed on the primary key, mirroring
    ``services.rfm.persist_rfm``. Idempotent: re-running overwrites in place
    rather than accumulating, so the seed pipeline can be run repeatedly.
    """
    payload = [
        {
            "_customer_id": int(customer_id),
            "churn_risk_score": float(row.churn_risk_score),
            "is_churned": bool(row.is_churned),
        }
        for customer_id, row in scores.iterrows()
    ]
    if not payload:
        logger.warning("  No churn scores to persist.")
        return 0

    stmt = (
        DimCustomer.__table__.update()
        .where(DimCustomer.customer_id == bindparam("_customer_id"))
        .values(
            churn_risk_score=bindparam("churn_risk_score"),
            is_churned=bindparam("is_churned"),
        )
    )
    with Session(engine) as session:
        session.connection().execute(stmt, payload)
        session.commit()

    logger.info("  Persisted churn scores for %s customers.", f"{len(payload):,}")
    return len(payload)


def run_scoring(engine: Engine, *, as_of: date | None = None) -> ScoringResult:
    """Score every customer and persist the results. Returns a run summary.

    The reference date is resolved *here* and passed down, so the date reported
    in the summary is provably the one the features were computed against.
    Re-deriving it independently in two places is how a run ends up reporting a
    date it did not actually use.
    """
    reference = as_of or (latest_warehouse_date(engine) + timedelta(days=1))
    scores = score_all_customers(engine, as_of=reference)
    updated = persist_churn_scores(engine, scores)

    bands = scores["churn_risk_score"].map(risk_band)

    band_counts = {label: int((bands == label).sum()) for label in RISK_BAND_LABELS}

    # Revenue at stake from customers who are still active: the actionable slice.
    active_high_risk = scores[(~scores["is_churned"]) & (scores["churn_risk_score"] >= 0.5)]

    return ScoringResult(
        scored=len(scores),
        updated=updated,
        as_of=reference,
        churn_window_days=CHURN_WINDOW_DAYS,
        already_churned=int(scores["is_churned"].sum()),
        mean_score=float(scores["churn_risk_score"].mean()),
        band_counts=band_counts,
        revenue_at_risk=float(active_high_risk["monetary"].sum()),
    )


def count_scored(engine: Engine) -> int:
    """How many customers currently carry a churn score."""
    with engine.connect() as conn:
        return int(
            conn.execute(
                select(func.count())
                .select_from(DimCustomer)
                .where(DimCustomer.churn_risk_score.is_not(None))
            ).scalar_one()
        )


__all__ = [
    "CHURN_WINDOW_DAYS",
    "ScoringResult",
    "risk_band",
    "score_all_customers",
    "persist_churn_scores",
    "run_scoring",
    "count_scored",
]
