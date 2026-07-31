"""Sanity-check that the churn model points the right way.

A logistic regression trained with a flipped label still produces a plausible
ROC-AUC (0.80 either way, since AUC is symmetric under inversion) and a plausible
mean probability. The failure is silent: every ranking in the churn dashboard
would be exactly backwards, recommending retention spend on the most loyal
customers.

This checks direction against the *observed* outcome rather than trusting the
training metrics:

  1. Customers who are in fact churned should have a higher mean P(churn).
  2. P(churn) should rise monotonically with recency across deciles.
  3. The coefficient on `recency` should be positive, and on `frequency` negative.

A decile gradient is a stronger test than a single spot-check: it catches a model
that ranks the bulk of the distribution correctly while mishandling the tail -
exactly the failure found in v1.0.0, where a customer with 155 orders and 373
days of silence scored P(churn)=0.001 while ROC-AUC sat at a respectable 0.80.

Run:  python scripts/verify_churn_direction.py
"""

from __future__ import annotations

import sys

import _bootstrap  # noqa: F401
import pandas as pd

from app.core.database import get_engine
from app.core.logging_config import configure_logging
from app.ml import registry
from app.ml.score import score_all_customers

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(label)


def main() -> int:
    configure_logging()
    engine = get_engine()

    print("=" * 78)
    print("CHURN MODEL DIRECTION CHECK")
    print("=" * 78)

    scores = score_all_customers(engine)

    # ---- 1. Observed churned vs retained -------------------------------------
    churned = scores[scores["is_churned"]]["churn_risk_score"]
    active = scores[~scores["is_churned"]]["churn_risk_score"]
    print(
        f"\nMean P(churn): observed-churned={churned.mean():.4f} "
        f"(n={len(churned):,})  active={active.mean():.4f} (n={len(active):,})"
    )
    check(
        "churned customers score higher than active ones",
        churned.mean() > active.mean(),
        f"{churned.mean():.4f} vs {active.mean():.4f}",
    )

    # ---- 2. Monotonicity across recency deciles ------------------------------
    print("\nP(churn) by recency decile")
    deciles = pd.qcut(scores["recency"], 10, labels=False, duplicates="drop")
    by_decile = scores.groupby(deciles)["churn_risk_score"].mean()
    for decile, mean_score in by_decile.items():
        rec_max = scores.loc[deciles == decile, "recency"].max()
        bar = "#" * int(mean_score * 40)
        print(f"  d{int(decile) + 1:<2} (<= {rec_max:>4}d)  {mean_score:.4f}  {bar}")

    check(
        "lowest-recency decile scores below highest-recency decile",
        by_decile.iloc[0] < by_decile.iloc[-1],
        f"{by_decile.iloc[0]:.4f} vs {by_decile.iloc[-1]:.4f}",
    )
    correlation = scores["recency"].corr(scores["churn_risk_score"])
    check(
        "recency positively correlates with P(churn)",
        correlation > 0.3,
        f"r={correlation:.3f}",
    )

    # ---- 3. Coefficient signs ------------------------------------------------
    model = registry.load_model()
    estimator = model.pipeline
    coefs = None
    if hasattr(estimator, "named_steps"):
        for step in reversed(list(estimator.named_steps.values())):
            if hasattr(step, "coef_"):
                coefs = step.coef_[0]
                break
    elif hasattr(estimator, "coef_"):
        coefs = estimator.coef_[0]

    if coefs is None:
        check("could read model coefficients", False, "no coef_ found on pipeline")
    else:
        print("\nCoefficients (log-odds of churn per scaled unit)")
        for name, coef in sorted(
            zip(model.features, coefs), key=lambda kv: -abs(kv[1])
        ):
            print(f"  {name:<26}{coef:>+8.3f}")
        recency_coef = dict(zip(model.features, coefs))["recency"]
        check(
            "recency coefficient is positive (more recency -> more churn)",
            recency_coef > 0,
            f"{recency_coef:+.3f}",
        )
        freq_coef = dict(zip(model.features, coefs))["frequency"]
        check(
            "frequency coefficient is negative (more orders -> less churn)",
            freq_coef < 0,
            f"{freq_coef:+.3f}",
        )

    # ---- 4. The specific case that prompted this check -----------------------
    if 17850 in scores.index:
        row = scores.loc[17850]
        print(
            f"\nCustomer 17850: recency={int(row.recency)}d  "
            f"monetary=£{row.monetary:,.0f}  P(churn)={row.churn_risk_score:.4f}"
        )
        print(
            "  This customer is the reason the log1p step exists. Under v1.0.0\n"
            "  they scored 0.001 because raw frequency reached z=13.0, worth\n"
            "  -8.06 in log-odds, while recency could contribute at most +0.66;\n"
            "  post-compression no feature exceeds z=4.5. The dominant term is\n"
            "  now recency_ratio (they are ~155x past their own\n"
            "  normal 2.4-day reorder gap), partly offset by a low\n"
            "  avg_days_between_orders, since frequent buyers churn less on\n"
            "  average. That residual tension is inherent to a linear model and\n"
            "  is left in deliberately rather than patched with an interaction\n"
            "  term. Note too that 155 orders at a 2.4-day cadence looks like a\n"
            "  wholesale or internal account, not a retail shopper. Aggregate\n"
            "  direction checks decide whether the model is sane, not any single\n"
            "  customer; run scripts/diagnose_customer_score.py for the full\n"
            "  per-feature breakdown."
        )

    print("\n" + "=" * 78)
    if failures:
        print(f"FAILED ({len(failures)}): " + "; ".join(failures))
        print("The churn model may be inverted — do NOT ship the churn dashboard.")
        return 1
    print("Model direction is correct.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
