"""Run every hypothesis test against the warehouse and print the results.

    python scripts/run_stats.py

This is the reviewable proof that the statistics layer works before any of it
is wired to an API. Each test prints its question, the numbers, and the
plain-language verdict the dashboard will show.
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401  (puts backend/ on sys.path)

from app.core.database import get_engine, wait_for_database
from app.core.exceptions import InsufficientDataError, StatisticalTestError
from app.core.logging_config import configure_logging, get_logger
from app.services.statistics import (
    TestResult,
    aov_across_countries,
    aov_across_segments,
    aov_by_country,
    churn_by_country,
    churn_by_segment,
)

logger = get_logger(__name__)

RULE = "=" * 78


def _format_p(p: float) -> str:
    """Very small p-values print as 0.0000, which reads as exactly zero."""
    return "< 0.0001" if p < 0.0001 else f"{p:.4f}"


def _print_result(result: TestResult) -> None:
    print(f"\n{RULE}")
    print(f"  {result.question}")
    print(RULE)
    print(f"  Test          : {result.test_type}")
    print(f"  H0            : {result.null_hypothesis}")
    print(f"  Statistic     : {result.statistic:,.4f}")
    print(f"  p-value       : {_format_p(result.p_value)}")
    if result.degrees_of_freedom is not None:
        print(f"  df            : {result.degrees_of_freedom:,.2f}")
    if result.effect_size is not None:
        print(
            f"  Effect size   : {result.effect_size:.4f} "
            f"({result.effect_size_name}, {result.effect_size_label})"
        )
    verdict = "REJECT H0" if result.significant else "FAIL TO REJECT H0"
    print(f"  Verdict       : {verdict} at alpha={result.alpha}")

    if result.groups:
        print("\n  Groups:")
        first = result.groups[0]
        if "mean" in first:
            print(f"    {'group':<22} {'n':>8} {'mean':>12} {'median':>12} {'std':>12}")
            print(f"    {'-' * 22} {'-' * 8} {'-' * 12} {'-' * 12} {'-' * 12}")
            for group in result.groups:
                print(
                    f"    {group['name'][:22]:<22} {group['n']:>8,} "
                    f"{group['mean']:>12,.2f} {group['median']:>12,.2f} "
                    f"{group['std']:>12,.2f}"
                )
        else:
            # Contingency table shape.
            columns = [k for k in first if k != "name"]
            header = "".join(f"{c:>12}" for c in columns)
            print(f"    {'category':<22}{header}{'churn %':>12}")
            print(f"    {'-' * 22}{'-' * (12 * len(columns))}{'-' * 12}")
            for group in result.groups:
                counts = [group[c] for c in columns]
                total = sum(counts)
                churn_pct = (group.get("Churned", 0) / total * 100) if total else 0
                cells = "".join(f"{v:>12,}" for v in counts)
                print(f"    {group['name'][:22]:<22}{cells}{churn_pct:>11.1f}%")

    print(f"\n  Interpretation:\n    {result.interpretation}")

    excluded = result.assumptions.get("excluded_groups")
    if excluded:
        print(f"\n  Note: excluded {len(excluded)} group(s) - "
              f"{result.assumptions.get('exclusion_reason')}: {', '.join(map(str, excluded))}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country-a", default="Germany")
    parser.add_argument("--country-b", default="United Kingdom")
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--churn-days", type=int, default=90)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args()

    if args.log_level:
        configure_logging(args.log_level, force=True)
    wait_for_database()
    engine = get_engine()

    print(f"\n{RULE}")
    print("  STATISTICAL ANALYSIS")
    print(RULE)

    # Each test is run independently: one failing on data conditions should not
    # prevent the others from reporting.
    tests = [
        ("AOV: two countries",
         lambda: aov_by_country(engine, args.country_a, args.country_b, alpha=args.alpha)),
        ("AOV across countries",
         lambda: aov_across_countries(engine, top_n=args.top_n, alpha=args.alpha)),
        ("AOV across RFM segments",
         lambda: aov_across_segments(engine, alpha=args.alpha)),
        ("Churn vs country",
         lambda: churn_by_country(engine, top_n=args.top_n,
                                  churn_days=args.churn_days, alpha=args.alpha)),
        ("Churn vs RFM segment",
         lambda: churn_by_segment(engine, churn_days=args.churn_days, alpha=args.alpha)),
    ]

    completed = 0
    skipped = 0
    for name, runner in tests:
        try:
            _print_result(runner())
            completed += 1
        except (InsufficientDataError, StatisticalTestError) as exc:
            # A test the data cannot support is a legitimate outcome, not a
            # crash - report it and carry on.
            print(f"\n{RULE}")
            print(f"  {name}: SKIPPED")
            print(RULE)
            print(f"  {exc.message}")
            if exc.details:
                for key, value in exc.details.items():
                    print(f"    {key}: {value}")
            skipped += 1

    print(f"\n{RULE}")
    print(f"  {completed} test(s) completed, {skipped} skipped.")
    print(f"{RULE}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
