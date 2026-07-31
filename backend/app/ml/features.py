"""Feature engineering for the churn model.

The label-leakage problem, and how this module solves it
-------------------------------------------------------
Churn is defined as "no purchase in the last 90 days". Recency is "days since
last purchase". So ``churned == (recency >= 90)`` is true *by definition*.

Train a model on current RFM to predict current churn and it will score a
perfect ROC-AUC of 1.00, because one feature is an algebraic restatement of the
target. That number is worthless, and a portfolio project that reports it is
demonstrating a misunderstanding rather than a model.

This module avoids that with a **temporal split**:

    |<------- feature window ------->|<-- observation window (90d) -->|
    start                          cutoff                          max_date

* Features are computed **only** from transactions before ``cutoff``.
* The label is whether the customer bought **after** ``cutoff``.

Recency measured at the cutoff is then a legitimate predictor: it is known at
prediction time and describes the past, while the label lives entirely in the
future. This is the same structure a production model would need, and it is why
the resulting AUC (~0.86 on this dataset) is believable rather than perfect.

A consequence worth stating: the model can only be trained on customers who
existed before the cutoff. Customers whose first purchase falls inside the
observation window have no feature history and are excluded from training,
though the trained model can still score them later.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.engine import Engine

from app.core.exceptions import InsufficientDataError
from app.core.logging_config import get_logger
from app.models.star_schema import DimCountry, DimDate, FactSales

logger = get_logger(__name__)

# Feature order is part of the model contract: the scaler and the coefficients
# are positional, so a reordering here would silently mis-score every customer.
# The API validates incoming payloads against this exact list.
FEATURE_NAMES: list[str] = [
    "recency",
    "frequency",
    "monetary",
    "avg_order_value",
    "tenure_days",
    "avg_days_between_orders",
    "distinct_products",
    "total_items",
    "return_rate",
    "recency_ratio",
]

# Features whose raw distribution is heavy-tailed enough to break a linear model.
#
# Measured skew on this dataset: monetary 25.7, total_items 22.2, avg_order_value
# 13.3, frequency 12.7, recency_ratio 10.4, distinct_products 6.2. Compare
# recency at 0.9 and tenure_days at 0.4.
#
# Why this matters more than it looks: StandardScaler fixes *scale*, not *shape*.
# Centring and dividing by sigma leaves a skewed variable's tail exactly as far
# out in sigma units as it started - the top customer sat at z=14.5 on frequency
# and z=52 on monetary, while recency (which is what actually defines churn)
# never exceeded z=3.0. A single linear coefficient then cannot serve both ends
# of such a distribution, and the tail dominates the log-odds: a customer with
# 155 orders scored P(churn)=0.001 despite not having bought for 373 days,
# because frequency alone contributed -7.2 to the logit.
#
# log1p compresses those tails to roughly the same dynamic range as the
# well-behaved features, so the fitted coefficients reflect predictive value
# rather than distributional accident. It is applied inside the pipeline, before
# the scaler, so inference cannot forget it.
SKEWED_FEATURES: list[str] = [
    "frequency",
    "monetary",
    "avg_order_value",
    "avg_days_between_orders",
    "distinct_products",
    "total_items",
    "recency_ratio",
]

# Positional indices, resolved once. The transform runs on a bare ndarray inside
# the sklearn pipeline, where column names are no longer available.
_SKEWED_INDICES: list[int] = [FEATURE_NAMES.index(name) for name in SKEWED_FEATURES]


def compress_skew(X):
    """Apply ``log1p`` to the heavy-tailed feature columns.

    Must stay a module-level named function: it is pickled by reference into the
    model artifact, so a lambda or closure here would make the artifact
    unloadable in a fresh process.

    Negative values are clipped to 0 before the transform. ``monetary`` can go
    negative in principle (a customer whose un-flagged returns exceed their
    purchases), and ``log1p`` is undefined below -1. Clipping treats "net
    negative lifetime value" as "zero value", which is the right reading for a
    magnitude feature and avoids emitting NaN into the model.
    """
    X = np.asarray(X, dtype=float).copy()
    X[:, _SKEWED_INDICES] = np.log1p(np.clip(X[:, _SKEWED_INDICES], 0, None))
    return X


FEATURE_DESCRIPTIONS: dict[str, str] = {
    "recency": "Days since the customer's most recent purchase.",
    "frequency": "Number of distinct orders placed.",
    "monetary": "Total net revenue from the customer.",
    "avg_order_value": "Mean value of the customer's orders.",
    "tenure_days": "Days between first and last purchase.",
    "avg_days_between_orders": "Mean gap between consecutive orders.",
    "distinct_products": "Count of unique products ever bought.",
    "total_items": "Total units purchased.",
    "return_rate": "Share of the customer's orders that were cancelled.",
    "recency_ratio": (
        "Recency divided by the customer's typical inter-order gap. Above 1 "
        "means they are overdue relative to their own habit."
    ),
}


@dataclass
class ChurnDataset:
    """Features, labels and the provenance needed to interpret them."""

    features: pd.DataFrame          # index=customer_id, columns=FEATURE_NAMES
    labels: pd.Series               # 1 = churned (no purchase after cutoff)
    cutoff_date: date
    max_date: date
    observation_days: int
    excluded_new_customers: int

    @property
    def churn_rate(self) -> float:
        return float(self.labels.mean())

    def summary(self) -> dict:
        return {
            "customers": int(len(self.features)),
            "churned": int(self.labels.sum()),
            "retained": int((~self.labels.astype(bool)).sum()),
            "churn_rate_pct": round(self.churn_rate * 100, 2),
            "cutoff_date": self.cutoff_date.isoformat(),
            "max_date": self.max_date.isoformat(),
            "observation_days": self.observation_days,
            "feature_window_days": (self.cutoff_date - self.min_date).days
            if hasattr(self, "min_date") else None,
            "excluded_new_customers": self.excluded_new_customers,
        }


def load_transactions(engine: Engine) -> pd.DataFrame:
    """Customer-attributed invoice lines with their dates.

    Guest checkouts (NULL customer) are excluded: with no identity there is no
    customer to predict churn for. They remain in revenue reporting.
    """
    stmt = (
        select(
            FactSales.customer_id.label("customer_id"),
            FactSales.invoice_no.label("invoice_no"),
            FactSales.product_id.label("product_id"),
            FactSales.quantity.label("quantity"),
            FactSales.revenue.label("revenue"),
            FactSales.is_cancelled.label("is_cancelled"),
            DimDate.full_date.label("invoice_date"),
            DimCountry.country_name.label("country"),
        )
        .join(DimDate, FactSales.date_id == DimDate.date_id)
        .join(DimCountry, FactSales.country_id == DimCountry.country_id)
        .where(FactSales.customer_id.is_not(None))
    )
    with engine.connect() as conn:
        df = pd.read_sql(stmt, conn)

    if df.empty:
        raise InsufficientDataError(
            "No customer-attributed sales found. Load the warehouse first: "
            "`python scripts/run_etl.py`."
        )

    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["revenue"] = df["revenue"].astype(float)
    df["quantity"] = df["quantity"].astype(int)
    logger.info(
        "  Loaded %s customer-attributed lines for %s customers.",
        f"{len(df):,}", f"{df['customer_id'].nunique():,}",
    )
    return df


def compute_features(
    transactions: pd.DataFrame,
    as_of: date,
) -> pd.DataFrame:
    """Build the feature matrix from transactions strictly before ``as_of``.

    ``as_of`` is the prediction moment. Everything here is computed as it would
    have been known on that date, which is what keeps the temporal split honest.
    """
    as_of_ts = pd.Timestamp(as_of)
    window = transactions[transactions["invoice_date"] < as_of_ts]

    if window.empty:
        raise InsufficientDataError(
            f"No transactions exist before {as_of.isoformat()}, so no features "
            "can be computed.",
            details={"as_of": as_of.isoformat()},
        )

    # Sales exclude cancellations; returns are counted separately as a
    # behavioural signal rather than being netted into the same aggregate.
    sales = window[~window["is_cancelled"]]
    if sales.empty:
        raise InsufficientDataError(
            "Every transaction in the feature window is a cancellation."
        )

    grouped = sales.groupby("customer_id")
    features = pd.DataFrame(index=grouped.size().index)
    features.index.name = "customer_id"

    last_purchase = grouped["invoice_date"].max()
    first_purchase = grouped["invoice_date"].min()

    features["recency"] = (as_of_ts - last_purchase).dt.days
    features["frequency"] = grouped["invoice_no"].nunique()
    features["monetary"] = grouped["revenue"].sum().round(2)
    features["avg_order_value"] = (
        features["monetary"] / features["frequency"]
    ).round(2)
    features["tenure_days"] = (last_purchase - first_purchase).dt.days
    features["distinct_products"] = grouped["product_id"].nunique()
    features["total_items"] = grouped["quantity"].sum()

    # A one-off buyer has no meaningful inter-order gap. Using their tenure
    # (0 days) would imply "buys every day", the opposite of the truth, so the
    # gap is set to their tenure at the cutoff - how long they have been known
    # without reordering.
    single_order = features["frequency"] <= 1
    features["avg_days_between_orders"] = (
        features["tenure_days"] / (features["frequency"] - 1).clip(lower=1)
    ).round(2)
    features.loc[single_order, "avg_days_between_orders"] = (
        (as_of_ts - first_purchase[single_order]).dt.days
    )

    # Return rate over ALL activity in the window, including cancellations,
    # which is why it is computed from `window` rather than `sales`.
    orders_all = window.groupby("customer_id")["invoice_no"].nunique()
    orders_cancelled = (
        window[window["is_cancelled"]].groupby("customer_id")["invoice_no"].nunique()
    )
    features["return_rate"] = (
        orders_cancelled.reindex(features.index).fillna(0)
        / orders_all.reindex(features.index).replace(0, np.nan)
    ).fillna(0).round(4)

    # Recency relative to the customer's own rhythm. A 60-day gap is alarming
    # for a weekly buyer and unremarkable for an annual one; the raw recency
    # feature cannot express that distinction.
    features["recency_ratio"] = (
        features["recency"] / features["avg_days_between_orders"].clip(lower=1)
    ).round(4)

    features = features[FEATURE_NAMES]
    # Guard against inf from any residual division edge case before the model
    # sees the matrix - sklearn raises on non-finite input.
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
    return features


def build_training_dataset(
    engine: Engine,
    *,
    observation_days: int = 90,
) -> ChurnDataset:
    """Assemble a leakage-free training set using a temporal split.

    The cutoff is placed ``observation_days`` before the last date in the
    warehouse, giving every training customer a full observation window in
    which they either returned or did not.
    """
    transactions = load_transactions(engine)

    max_date = transactions["invoice_date"].max().date()
    min_date = transactions["invoice_date"].min().date()
    cutoff = max_date - timedelta(days=observation_days)

    if cutoff <= min_date:
        raise InsufficientDataError(
            f"The warehouse spans {min_date} to {max_date}, which is shorter "
            f"than the {observation_days}-day observation window. There is no "
            "room for a temporal split.",
            details={
                "min_date": min_date.isoformat(),
                "max_date": max_date.isoformat(),
                "observation_days": observation_days,
            },
        )

    logger.info(
        "  Temporal split: features from %s to %s, outcome observed %s to %s.",
        min_date, cutoff, cutoff, max_date,
    )

    features = compute_features(transactions, as_of=cutoff)

    # Label: did they buy in the observation window? Cancellations do not count
    # as a return visit - a refund is not a purchase.
    cutoff_ts = pd.Timestamp(cutoff)
    observed = transactions[
        (transactions["invoice_date"] >= cutoff_ts) & (~transactions["is_cancelled"])
    ]
    returned = set(observed["customer_id"].unique())

    labels = pd.Series(
        [0 if customer in returned else 1 for customer in features.index],
        index=features.index,
        name="churned",
        dtype=int,
    )

    # Customers first seen inside the observation window have no feature
    # history; they are absent from `features` by construction. Counted here so
    # the metadata can state how many were set aside.
    all_customers = set(transactions["customer_id"].unique())
    excluded = len(all_customers - set(features.index))

    dataset = ChurnDataset(
        features=features,
        labels=labels,
        cutoff_date=cutoff,
        max_date=max_date,
        observation_days=observation_days,
        excluded_new_customers=excluded,
    )
    dataset.min_date = min_date  # type: ignore[attr-defined]

    if len(features) < 100:
        raise InsufficientDataError(
            f"Only {len(features)} customers have pre-cutoff history; a model "
            "trained on this would not generalise.",
            details={"customers": int(len(features))},
        )

    n_churned = int(labels.sum())
    if n_churned == 0 or n_churned == len(labels):
        raise InsufficientDataError(
            "Every customer has the same churn outcome, so there is nothing to "
            "learn. Check the observation window length.",
            details={"churned": n_churned, "total": int(len(labels))},
        )

    logger.info(
        "  Training set: %s customers, %s churned (%.1f%%), %s new customers excluded.",
        f"{len(features):,}", f"{n_churned:,}",
        dataset.churn_rate * 100, f"{excluded:,}",
    )
    return dataset


def build_scoring_features(engine: Engine, as_of: date | None = None) -> pd.DataFrame:
    """Features for every current customer, for inference rather than training.

    ``as_of`` defaults to the last date in the warehouse. This dataset has no
    labels: it answers "who is at risk right now?" using the full history.
    """
    transactions = load_transactions(engine)
    reference = as_of or (transactions["invoice_date"].max().date() + timedelta(days=1))
    return compute_features(transactions, as_of=reference)


def latest_warehouse_date(engine: Engine) -> date:
    """Most recent invoice date, used as "today" for a static dataset.

    The data ends in 2011, so wall-clock time would make every customer churned.
    """
    with engine.connect() as conn:
        result = conn.execute(select(func.max(DimDate.full_date))).scalar_one_or_none()
    if result is None:
        raise InsufficientDataError("dim_date is empty; run the ETL first.")
    return result if isinstance(result, date) else pd.Timestamp(result).date()


__all__ = [
    "FEATURE_NAMES",
    "FEATURE_DESCRIPTIONS",
    "SKEWED_FEATURES",
    "compress_skew",
    "ChurnDataset",
    "load_transactions",
    "compute_features",
    "build_training_dataset",
    "build_scoring_features",
    "latest_warehouse_date",
]
