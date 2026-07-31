"""Score every customer for churn risk and persist the results.

Run after `train_churn_model.py`: training produces the artifact, this applies it
to the whole customer base and fills in `dim_customer.churn_risk_score` /
`is_churned`, which is what the churn dashboard reads.

Run:  python scripts/score_customers.py
"""

from __future__ import annotations

import sys

import _bootstrap  # noqa: F401  (puts backend/ on sys.path)

from app.core.database import get_engine
from app.core.exceptions import InsufficientDataError, ModelNotLoadedError
from app.core.logging_config import configure_logging
from app.core.risk import RISK_BAND_LABELS
from app.ml.score import run_scoring


def main() -> int:
    configure_logging()
    engine = get_engine()

    print("=" * 78)
    print("CHURN SCORING")
    print("=" * 78)

    try:
        result = run_scoring(engine)
    except ModelNotLoadedError as exc:
        # The actionable case: the reviewer skipped training.
        print(f"\nMODEL UNAVAILABLE: {exc.message}")
        print("Run `python scripts/train_churn_model.py` first.")
        return 1
    except InsufficientDataError as exc:
        print(f"\nNOT ENOUGH DATA: {exc.message}")
        print("Run `python scripts/run_etl.py` first.")
        return 1

    print(f"\nScored          : {result.scored:,} customers")
    print(f"Rows updated    : {result.updated:,}")
    print(f"As of           : {result.as_of.isoformat()} (churn window {result.churn_window_days}d)")
    print(f"Already churned : {result.already_churned:,} "
          f"({result.already_churned / max(result.scored, 1) * 100:.1f}%)")
    print(f"Mean P(churn)   : {result.mean_score:.3f}")

    print("\nRisk band distribution")
    for band in RISK_BAND_LABELS:
        count = result.band_counts.get(band, 0)
        share = count / max(result.scored, 1) * 100
        bar = "#" * int(share / 2)
        print(f"  {band:<9}{count:>7,}  {share:>5.1f}%  {bar}")

    print(f"\nRevenue at risk (active, P>=0.5): £{result.revenue_at_risk:,.2f}")
    print("\nDone. The churn endpoints can now rank customers by risk.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
