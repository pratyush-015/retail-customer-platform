"""Canonical churn-risk banding.

This module exists because the mapping from probability to severity label was
briefly defined twice — once in ``services.customer`` (three bands at 0.7/0.4)
and once in ``ml.score`` (four bands at 0.75/0.50/0.25) — with both call sites
documenting themselves as the single source of truth. The batch scorer would
have written ``"critical"`` into a summary the API schema declares as
``Literal["high", "medium", "low"]``, so any response carrying that label would
have failed validation and surfaced as a 500.

It lives in ``core`` rather than in either caller so the dependency direction
stays honest: ``services`` and ``ml`` both depend on ``core``, and neither has to
import the other to agree on what "high risk" means.

Thresholds are a product decision, not a statistical one. 0.7 is the point where
the retention team is willing to spend money on an intervention, and 0.4 is where
a customer is worth watching. They are deliberately *not* derived from the
model's decision threshold: recalibrating the model should not silently redefine
the labels on the dashboard.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

# Ordered high to low. `risk_band` relies on this ordering, so the first
# threshold a score clears wins.
RISK_BAND_THRESHOLDS: tuple[tuple[float, str], ...] = (
    (0.7, "high"),
    (0.4, "medium"),
    (0.0, "low"),
)

# Severity order, for iterating band counts in a stable, meaningful sequence.
# `schemas.customer.RiskBand` must stay in sync with this; a test asserts it.
RISK_BAND_LABELS: tuple[str, ...] = tuple(label for _, label in RISK_BAND_THRESHOLDS)

RiskBandLiteral = Literal["high", "medium", "low"]


def risk_band(score: float | Decimal | None) -> str | None:
    """Bucket a churn probability into the band the UI colours by.

    Returns ``None`` for an unscored customer. That is distinct from ``"low"``:
    "we have not run the model" and "the model says this customer is fine" look
    the same on a dashboard that collapses them, and only one of them is a reason
    to go run the pipeline.
    """
    if score is None:
        return None
    value = float(score)
    for threshold, label in RISK_BAND_THRESHOLDS:
        if value >= threshold:
            return label
    return "low"  # pragma: no cover - the 0.0 threshold catches every real score


__all__ = ["RISK_BAND_THRESHOLDS", "RISK_BAND_LABELS", "RiskBandLiteral", "risk_band"]
