"""Shared pytest fixtures.

The synthetic dataset here is the backbone of the analytics tests: it is small
enough that every expected RFM value can be computed by hand and asserted
exactly, which is the only way to catch a silently inverted recency score or a
frequency that counts line items instead of orders.
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest

# Reference date used by every synthetic-data test. Fixed rather than derived
# from "today" so expected values never drift.
REFERENCE_DATE = date(2011, 12, 10)


def _line(
    invoice: str,
    customer: int | None,
    day: str,
    qty: int,
    price: float,
    *,
    stock: str = "85123A",
    country: str = "United Kingdom",
    description: str = "WHITE HANGING HEART T-LIGHT HOLDER",
) -> dict:
    """One invoice line.

    Revenue is intentionally NOT precomputed here - the cleaner derives it, so
    tests exercise that derivation rather than trusting a value the fixture
    supplied.
    """
    return {
        "invoice_no": invoice,
        "stock_code": stock,
        "description": description,
        "quantity": qty,
        "invoice_date": datetime.fromisoformat(day),
        "unit_price": price,
        "customer_id": customer,
        "country": country,
    }


@pytest.fixture
def synthetic_transactions() -> pd.DataFrame:
    """A hand-built transaction set with known, hand-checkable RFM outcomes.

    Five customers, each constructed to land in a specific place:

    ==========  ======  =========  ========  ==================================
    customer    orders  recency*   monetary  intent
    ==========  ======  =========  ========  ==================================
    1001        3       1 day      600.00    best customer: recent + frequent
    1002        2       10 days    300.00    solid mid-tier
    1003        1       40 days    150.00    single recent-ish purchase
    1004        1       300 days   50.00     long dormant, low value
    1005        2       200 days   400.00    high spend but lapsed
    ==========  ======  =========  ========  ==================================

    (*) recency measured against REFERENCE_DATE (2011-12-10).

    Customer 1001 has a 30-line invoice to prove frequency counts *invoices*:
    if it counted rows, 1001 would show 32 orders instead of 3.
    """
    rows: list[dict] = []

    # --- 1001: 3 invoices, last on 2011-12-09 (recency 1) ------------------
    # First invoice deliberately has 30 lines of 10.00 = 300.00.
    #
    # Each line must carry a DISTINCT stock code. Thirty lines with the same
    # invoice, product, quantity, price and timestamp are exact duplicates -
    # a data-entry artefact the ETL correctly collapses to one row - not a
    # 30-product basket. Using one repeated code made this fixture represent
    # data the cleaner would never emit, so the RFM tests were asserting
    # against an impossible input.
    rows += [
        _line("536365", 1001, "2011-01-15T10:00:00", 1, 10.00, stock=f"1001L{i:02d}")
        for i in range(30)
    ]
    rows.append(_line("536366", 1001, "2011-06-20T11:00:00", 10, 20.00))  # 200.00
    rows.append(_line("536367", 1001, "2011-12-09T09:00:00", 10, 10.00))  # 100.00
    # 1001 total: 300 + 200 + 100 = 600.00 across 3 invoices

    # --- 1002: 2 invoices, last on 2011-11-30 (recency 10) -----------------
    rows.append(_line("536368", 1002, "2011-03-01T10:00:00", 10, 10.00))  # 100.00
    rows.append(_line("536369", 1002, "2011-11-30T10:00:00", 20, 10.00))  # 200.00
    # 1002 total: 300.00 across 2 invoices

    # --- 1003: 1 invoice, on 2011-10-31 (recency 40) -----------------------
    rows.append(_line("536370", 1003, "2011-10-31T10:00:00", 15, 10.00))  # 150.00

    # --- 1004: 1 invoice, on 2011-02-13 (recency 300) ----------------------
    rows.append(_line("536371", 1004, "2011-02-13T10:00:00", 5, 10.00))  # 50.00

    # --- 1005: 2 invoices, last on 2011-05-24 (recency 200) ----------------
    rows.append(_line("536372", 1005, "2011-01-10T10:00:00", 15, 10.00))  # 150.00
    rows.append(_line("536373", 1005, "2011-05-24T10:00:00", 25, 10.00))  # 250.00
    # 1005 total: 400.00 across 2 invoices

    # --- guest checkout: no customer id, must not appear in RFM ------------
    rows.append(_line("536374", None, "2011-12-01T10:00:00", 5, 10.00))

    df = pd.DataFrame(rows)
    # The RFM service consumes post-clean data, where revenue and the
    # cancellation flag already exist.
    df["revenue"] = (df["quantity"] * df["unit_price"]).round(2)
    df["is_cancelled"] = df["invoice_no"].str.upper().str.startswith("C")
    return df


@pytest.fixture
def expected_rfm() -> dict[int, dict[str, float]]:
    """Hand-computed RFM values for ``synthetic_transactions``.

    Recency is days between the customer's last invoice and REFERENCE_DATE.
    Frequency is distinct invoice count. Monetary is summed revenue.
    """
    return {
        1001: {"recency": 1, "frequency": 3, "monetary": 600.00},
        1002: {"recency": 10, "frequency": 2, "monetary": 300.00},
        1003: {"recency": 40, "frequency": 1, "monetary": 150.00},
        1004: {"recency": 300, "frequency": 1, "monetary": 50.00},
        1005: {"recency": 200, "frequency": 2, "monetary": 400.00},
    }


@pytest.fixture
def malformed_transactions() -> pd.DataFrame:
    """A deliberately hostile frame: every row breaks something different.

    Used to prove the ETL cleans and *audits* bad input rather than crashing.
    Each row is annotated with the rule it is designed to trigger.
    """
    return pd.DataFrame([
        # -- valid control rows, must survive -------------------------------
        _line("536365", 1001, "2011-01-15T10:00:00", 10, 5.00),
        _line("536366", 1002, "2011-02-15T10:00:00", 5, 10.00),
        # -- non-numeric quantity -> dropped --------------------------------
        {**_line("536367", 1003, "2011-03-15T10:00:00", 0, 5.00),
         "quantity": "not-a-number"},
        # -- non-numeric price -> dropped -----------------------------------
        {**_line("536368", 1004, "2011-04-15T10:00:00", 5, 0.0),
         "unit_price": "free!"},
        # -- unparseable date -> dropped ------------------------------------
        {**_line("536369", 1005, "2011-05-15T10:00:00", 5, 5.00),
         "invoice_date": "31/31/2011"},
        # -- blank invoice number -> dropped --------------------------------
        _line("", 1006, "2011-06-15T10:00:00", 5, 5.00),
        # -- blank stock code -> dropped ------------------------------------
        {**_line("536371", 1007, "2011-07-15T10:00:00", 5, 5.00),
         "stock_code": "  "},
        # -- cancellation (C prefix, negative qty) -> KEPT and flagged ------
        _line("C536372", 1008, "2011-08-15T10:00:00", -5, 5.00),
        # -- negative qty, no C prefix -> KEPT, reclassified as cancelled ---
        _line("536373", 1009, "2011-09-15T10:00:00", -3, 5.00),
        # -- zero quantity -> dropped (no revenue, no movement) -------------
        _line("536374", 1010, "2011-10-15T10:00:00", 0, 5.00),
        # -- negative price -> dropped --------------------------------------
        _line("536375", 1011, "2011-11-15T10:00:00", 5, -10.00),
        # -- zero price -> KEPT and flagged (free sample) -------------------
        _line("536376", 1012, "2011-11-16T10:00:00", 5, 0.00),
        # -- missing customer id -> KEPT with NULL customer -----------------
        _line("536377", None, "2011-11-17T10:00:00", 5, 5.00),
        # -- absurd quantity -> dropped as data entry error -----------------
        _line("536378", 1013, "2011-11-18T10:00:00", 500_000, 5.00),
        # -- missing description -> KEPT, placeholder filled ----------------
        {**_line("536379", 1014, "2011-11-19T10:00:00", 5, 5.00),
         "description": None},
        # -- blank country -> KEPT, normalised to Unspecified ---------------
        {**_line("536380", 1015, "2011-11-20T10:00:00", 5, 5.00), "country": ""},
        # -- exact duplicate of the first control row -> dropped ------------
        _line("536365", 1001, "2011-01-15T10:00:00", 10, 5.00),
        # -- service stock code -> KEPT and flagged as non-product ----------
        _line("536381", 1016, "2011-11-21T10:00:00", 1, 18.00, stock="POST",
              description="POSTAGE"),
    ])
