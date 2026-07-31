"""ETL cleaning tests.

The central claim these tests defend is that the cleaner *audits* rather than
silently discards. A pipeline that drops 30% of rows and says nothing is worse
than one that crashes, because the resulting dashboard looks fine and is wrong.

Each retention decision below is asserted against a business justification, not
just a row count: cancellations must survive so returns net out of revenue, and
guest checkouts must survive so revenue totals stay correct even though those
rows cannot be segmented.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from app.core.exceptions import ETLValidationError
from app.etl.clean import (
    MAX_PLAUSIBLE_QUANTITY,
    clean_transactions,
    normalise_columns,
)
from app.etl.quality import DataQualityReport


@pytest.fixture
def report() -> DataQualityReport:
    return DataQualityReport(source_file="unit-test")


@pytest.fixture
def cleaned(malformed_transactions, report):
    """Run the hostile fixture through the cleaner once, reuse everywhere."""
    return clean_transactions(malformed_transactions, report)


# ---------------------------------------------------------------------------
# The headline requirement: malformed input is handled and logged, not fatal
# ---------------------------------------------------------------------------
def test_malformed_data_does_not_crash_the_pipeline(cleaned):
    """18 rows, most of them broken in a different way, must clean cleanly."""
    assert isinstance(cleaned, pd.DataFrame)
    assert not cleaned.empty, "every row was dropped - the cleaner is too aggressive"


def test_every_cleaning_decision_is_recorded(cleaned, report):
    """Nothing is dropped or altered without an audit entry.

    This is the assertion that makes the report trustworthy: the sum of rows
    dropped, plus the rows that survived, must reconcile to the input count.
    Any unexplained shortfall means a rule removed rows without recording it.
    """
    assert report.rows_in == 18
    assert len(cleaned) == report.rows_in - report.rows_dropped, (
        f"{report.rows_in} in, {len(cleaned)} out, but only "
        f"{report.rows_dropped} rows accounted for as dropped"
    )
    assert report.decisions, "no cleaning decisions were recorded at all"


def test_report_serialises_to_json(cleaned, report, tmp_path):
    """The audit trail must survive a round-trip to data_quality_report.json."""
    path = report.save(tmp_path / "data_quality_report.json")
    assert path.exists()

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["summary"]["rows_in"] == 18
    assert payload["summary"]["rows_out"] == len(cleaned)
    assert payload["summary"]["finished_at"]  # set on save
    assert isinstance(payload["decisions"], list) and payload["decisions"]

    # Every decision must carry enough context to be audited later.
    for decision in payload["decisions"]:
        assert decision["rule"]
        assert decision["action"] in {"dropped", "retained", "modified", "flagged"}
        assert decision["reason"], f"{decision['rule']} recorded no justification"
        assert isinstance(decision["rows_affected"], int)


# ---------------------------------------------------------------------------
# Rows that MUST be dropped
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "rule",
    [
        "quantity_not_numeric",
        "price_not_numeric",
        "invoice_date_unparseable",
        "invoice_no_missing",
        "stock_code_missing",
        "zero_quantity",
        "negative_unit_price",
        "implausible_quantity",
        "exact_duplicate_rows",
    ],
)
def test_unusable_rows_are_dropped_and_logged(cleaned, report, rule):
    """Each unusable-row rule fired exactly once on the fixture."""
    decisions = {d.rule: d for d in report.decisions}
    assert rule in decisions, f"the cleaner never evaluated {rule!r}"

    decision = decisions[rule]
    assert decision.action == "dropped", (
        f"{rule} recorded {decision.action!r}, expected 'dropped'"
    )
    assert decision.rows_affected == 1
    assert decision.reason, f"{rule} was dropped without explaining why"


def test_dropped_rows_include_examples_for_auditing(cleaned, report):
    """A count alone is not auditable - the report keeps sample offenders."""
    dropped = [d for d in report.decisions if d.action == "dropped" and d.rows_affected]
    assert dropped
    assert all(d.examples for d in dropped), (
        "some dropped-row decisions carry no example rows, so a reviewer cannot "
        "verify the rule matched what it claims"
    )


def test_the_absurd_quantity_row_is_gone(cleaned):
    assert (cleaned["quantity"].abs() <= MAX_PLAUSIBLE_QUANTITY).all()


def test_duplicate_control_row_collapses_to_one(cleaned):
    """The fixture contains 536365 twice; exactly one copy may survive."""
    assert (cleaned["invoice_no"] == "536365").sum() == 1


# ---------------------------------------------------------------------------
# Rows that MUST be kept - dropping these would misrepresent the business
# ---------------------------------------------------------------------------
def test_cancellations_are_kept_and_flagged(cleaned):
    """Returns are real events; deleting them would inflate net revenue."""
    cancelled = cleaned[cleaned["is_cancelled"]]
    assert not cancelled.empty, "cancellations were dropped - net revenue is now wrong"
    assert "C536372" in set(cleaned["invoice_no"])
    # Returns must carry negative revenue so they subtract from the total.
    assert (cancelled["revenue"] < 0).all()


def test_negative_quantity_without_c_prefix_is_reclassified(cleaned, report):
    """A negative quantity is a return even when the invoice lacks the prefix.

    Leaving it unflagged would let a return count as a sale in the churn
    features while still subtracting from revenue - inconsistent either way.
    """
    row = cleaned[cleaned["invoice_no"] == "536373"]
    assert not row.empty, "row was dropped rather than reclassified"
    assert bool(row.iloc[0]["is_cancelled"]) is True

    decisions = {d.rule: d for d in report.decisions}
    assert decisions["negative_qty_without_c_prefix"].action == "modified"


def test_guest_checkouts_are_kept_with_null_customer(cleaned, report):
    """~22% of real rows have no customer id; they are still revenue."""
    guests = cleaned[cleaned["customer_id"].isna()]
    assert len(guests) == 1
    assert "536377" in set(guests["invoice_no"])
    assert bool(guests.iloc[0]["has_customer"]) is False

    decisions = {d.rule: d for d in report.decisions}
    assert decisions["missing_customer_id"].action == "flagged"


def test_zero_price_rows_are_kept_and_flagged(cleaned, report):
    """Free samples and promos are genuine activity, not corruption."""
    assert "536376" in set(cleaned["invoice_no"])
    decisions = {d.rule: d for d in report.decisions}
    assert decisions["zero_unit_price"].action == "flagged"


def test_service_codes_are_kept_and_flagged_as_non_product(cleaned, report):
    """Postage is not a product but it is revenue, so it stays - labelled."""
    postage = cleaned[cleaned["stock_code"].str.upper() == "POST"]
    assert not postage.empty
    assert bool(postage.iloc[0]["is_service_code"]) is True

    decisions = {d.rule: d for d in report.decisions}
    assert decisions["service_stock_codes"].action == "flagged"


def test_missing_description_is_filled_not_dropped(cleaned, report):
    """A nameless product is still a sale; the id carries the join."""
    row = cleaned[cleaned["invoice_no"] == "536379"]
    assert not row.empty
    assert row.iloc[0]["description"] == "UNKNOWN PRODUCT"

    decisions = {d.rule: d for d in report.decisions}
    assert decisions["missing_description"].action == "modified"


def test_blank_country_is_normalised(cleaned, report):
    row = cleaned[cleaned["invoice_no"] == "536380"]
    assert not row.empty
    assert row.iloc[0]["country"] == "Unspecified"

    decisions = {d.rule: d for d in report.decisions}
    assert decisions["missing_country"].action == "modified"


# ---------------------------------------------------------------------------
# Derived columns
# ---------------------------------------------------------------------------
def test_revenue_is_derived_not_trusted(cleaned):
    """Revenue must equal quantity x price for every surviving row."""
    expected = (cleaned["quantity"] * cleaned["unit_price"]).round(2)
    pd.testing.assert_series_equal(
        cleaned["revenue"], expected, check_names=False
    )


def test_expected_flag_columns_exist(cleaned):
    for column in ("revenue", "is_cancelled", "is_service_code", "has_customer"):
        assert column in cleaned.columns, f"missing derived column {column!r}"


def test_dtypes_are_stable_after_cleaning(cleaned):
    """Downstream transforms assume real dtypes, not object columns."""
    assert pd.api.types.is_numeric_dtype(cleaned["quantity"])
    assert pd.api.types.is_numeric_dtype(cleaned["unit_price"])
    assert pd.api.types.is_datetime64_any_dtype(cleaned["invoice_date"])
    assert pd.api.types.is_bool_dtype(cleaned["is_cancelled"])


# ---------------------------------------------------------------------------
# Column normalisation and structural failure
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("source_columns", "expected"),
    [
        # UCI has shipped several spellings across dataset revisions.
        (["Invoice", "StockCode", "Quantity", "InvoiceDate", "Price",
          "Customer ID", "Country"], "customer_id"),
        (["InvoiceNo", "StockCode", "Quantity", "InvoiceDate", "UnitPrice",
          "CustomerID", "Country"], "customer_id"),
    ],
)
def test_column_aliases_are_normalised(source_columns, expected):
    df = pd.DataFrame([dict.fromkeys(source_columns, 1)])
    out = normalise_columns(df)
    assert expected in out.columns
    assert "invoice_no" in out.columns
    assert "unit_price" in out.columns


def test_missing_required_column_fails_loudly():
    """A structural problem cannot be cleaned around, so it must raise.

    Continuing would produce a half-empty warehouse that looks populated.
    """
    df = pd.DataFrame([{"Invoice": "1", "Quantity": 1}])  # no stock code, date...

    with pytest.raises(ETLValidationError) as exc_info:
        normalise_columns(df)

    error = exc_info.value
    assert "stock_code" in error.details["missing_columns"]
    assert error.details["found_columns"]  # tells the reviewer what WAS present


def test_optional_columns_are_synthesised_when_absent():
    """Description and customer id may legitimately be missing entirely."""
    df = pd.DataFrame([{
        "Invoice": "536365", "StockCode": "85123A", "Quantity": 6,
        "InvoiceDate": "2011-01-01 10:00:00", "Price": 2.55,
        "Country": "United Kingdom",
    }])
    out = normalise_columns(df)
    assert "description" in out.columns
    assert "customer_id" in out.columns


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
def test_empty_frame_cleans_to_empty_without_crashing(report):
    """An empty extract is a data problem, not a code problem."""
    empty = pd.DataFrame(columns=[
        "Invoice", "StockCode", "Description", "Quantity",
        "InvoiceDate", "Price", "Customer ID", "Country",
    ])
    result = clean_transactions(empty, report)

    assert result.empty
    assert report.rows_in == 0
    assert report.retention_rate == 0.0  # must not divide by zero


def test_all_valid_rows_are_fully_retained(synthetic_transactions, report):
    """Clean input must pass through untouched - no false positives.

    The synthetic fixture is deliberately well-formed, so a drop here would
    mean a cleaning rule is over-matching on legitimate data.
    """
    # The cleaner derives these itself; hand them back in raw shape.
    raw = synthetic_transactions.drop(columns=["revenue", "is_cancelled"])

    result = clean_transactions(raw, report)

    assert len(result) == len(raw)
    assert report.rows_dropped == 0
    assert report.retention_rate == 100.0
