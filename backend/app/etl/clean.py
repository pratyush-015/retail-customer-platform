"""Cleaning and normalisation of the raw Online Retail II extract.

Cleaning philosophy
-------------------
Rows are only *dropped* when they cannot represent a real transaction. Anything
that is merely unusual is **kept and flagged**, because dropping it would make
the warehouse misrepresent the business:

===========================  ======================================================
Cancelled invoices (``C``)   Kept, ``is_cancelled=True``. They are real returns;
                             deleting them would inflate net revenue.
Negative quantities          Kept. They are the mechanic by which returns are
                             expressed and must net out of revenue.
Missing CustomerID (~22%)    Kept with ``customer_id=NULL``. Real guest-checkout
                             revenue; excluded from RFM (which needs identity)
                             but included in sales totals.
Zero-price rows              Kept unless the stock code is a known adjustment
                             code. Free samples/promos are genuine activity.
Exact duplicates             Dropped - same invoice, product, qty, price and
                             timestamp is a data-entry artefact, not two sales.
Malformed/unparseable rows   Dropped and audited (bad dates, non-numeric
                             quantities, missing invoice or stock code).
===========================  ======================================================

Every one of these produces an entry in ``data_quality_report.json``.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from app.core.exceptions import ETLValidationError
from app.core.logging_config import get_logger
from app.etl.quality import DataQualityReport

logger = get_logger(__name__)

# Canonical internal column names, mapped from the several spellings UCI has
# shipped across dataset revisions ("Customer ID" vs "CustomerID", etc.).
COLUMN_ALIASES: dict[str, str] = {
    "invoice": "invoice_no",
    "invoiceno": "invoice_no",
    "stockcode": "stock_code",
    "description": "description",
    "quantity": "quantity",
    "invoicedate": "invoice_date",
    "price": "unit_price",
    "unitprice": "unit_price",
    "customerid": "customer_id",
    "customer id": "customer_id",
    "country": "country",
}

REQUIRED_COLUMNS = [
    "invoice_no",
    "stock_code",
    "quantity",
    "invoice_date",
    "unit_price",
    "country",
]

# Stock codes that are not products: postage, fees, manual adjustments, debt.
SERVICE_CODES = {
    "POST", "DOT", "C2", "M", "BANK CHARGES", "PADS", "B", "CRUK",
    "AMAZONFEE", "S", "D", "SP1002", "gift_0001", "TEST001", "TEST002",
    "ADJUST", "ADJUST2", "m",
}
_SERVICE_CODE_UPPER = {c.upper() for c in SERVICE_CODES}

# Extreme values that indicate a data-entry error rather than a real order.
# The largest genuine order in this dataset is ~4 800 units.
MAX_PLAUSIBLE_QUANTITY = 80_000
MAX_PLAUSIBLE_PRICE = 50_000.0


def _examples(df: pd.DataFrame, mask: pd.Series, limit: int = 3) -> list[dict]:
    """Grab a few representative offending rows for the audit report."""
    if not mask.any():
        return []
    cols = [c for c in ("invoice_no", "stock_code", "description", "quantity",
                        "unit_price", "customer_id", "invoice_date") if c in df.columns]
    sample = df.loc[mask, cols].head(limit)
    records: list[dict] = []
    for row in sample.to_dict(orient="records"):
        records.append({k: (None if pd.isna(v) else v) for k, v in row.items()})
    return records


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename source columns to canonical snake_case names.

    Raises
    ------
    ETLValidationError
        If a required column is absent. This is a *structural* failure - unlike
        dirty rows, it cannot be cleaned around, so the pipeline stops rather
        than silently producing a half-empty warehouse.
    """
    renamed = {}
    for col in df.columns:
        key = str(col).strip().lower().replace("_", "").replace(" ", "")
        alt = str(col).strip().lower()
        target = COLUMN_ALIASES.get(key) or COLUMN_ALIASES.get(alt)
        renamed[col] = target or str(col).strip().lower().replace(" ", "_")

    out = df.rename(columns=renamed)

    missing = [c for c in REQUIRED_COLUMNS if c not in out.columns]
    if missing:
        raise ETLValidationError(
            f"Source data is missing required column(s): {', '.join(missing)}.",
            details={
                "missing_columns": missing,
                "found_columns": [str(c) for c in df.columns],
            },
        )

    if "description" not in out.columns:
        out["description"] = None
    if "customer_id" not in out.columns:
        out["customer_id"] = np.nan

    return out


def clean_transactions(
    df: pd.DataFrame,
    report: DataQualityReport,
) -> pd.DataFrame:
    """Clean the raw transaction frame, auditing every decision.

    Returns a frame with canonical dtypes and these added columns:
    ``revenue``, ``is_cancelled``, ``is_service_code``, ``has_customer``.
    """
    df = normalise_columns(df)
    report.rows_in = len(df)
    logger.info("Cleaning %s raw rows...", f"{len(df):,}")

    # ---------------------------------------------------------------- types --
    # Coerce first so every later rule operates on real dtypes rather than a
    # mix of strings and numbers.
    df["invoice_no"] = df["invoice_no"].astype(str).str.strip()
    df["stock_code"] = df["stock_code"].astype(str).str.strip()
    df["description"] = (
        df["description"].astype(str).str.strip().replace({"nan": None, "None": None, "": None})
    )
    df["country"] = df["country"].astype(str).str.strip()

    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
    df["customer_id"] = pd.to_numeric(df["customer_id"], errors="coerce")
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")

    # ---------------------------------------------- rule: unparseable values --
    bad_qty = df["quantity"].isna()
    if bad_qty.any():
        report.record(
            "quantity_not_numeric", "dropped", int(bad_qty.sum()),
            "Quantity could not be parsed as a number; the row cannot be valued.",
            _examples(df, bad_qty),
        )
        df = df[~bad_qty]
    else:
        report.record("quantity_not_numeric", "retained", 0,
                      "All quantity values parsed as numeric.")

    bad_price = df["unit_price"].isna()
    if bad_price.any():
        report.record(
            "price_not_numeric", "dropped", int(bad_price.sum()),
            "Unit price could not be parsed as a number; revenue is undefined.",
            _examples(df, bad_price),
        )
        df = df[~bad_price]
    else:
        report.record("price_not_numeric", "retained", 0,
                      "All unit price values parsed as numeric.")

    bad_date = df["invoice_date"].isna()
    if bad_date.any():
        report.record(
            "invoice_date_unparseable", "dropped", int(bad_date.sum()),
            "Invoice date could not be parsed; the row cannot join to dim_date.",
            _examples(df, bad_date),
        )
        df = df[~bad_date]
    else:
        report.record("invoice_date_unparseable", "retained", 0,
                      "All invoice dates parsed successfully.")

    # ------------------------------------------------ rule: missing keys ----
    blank_invoice = df["invoice_no"].isin(["", "nan", "None", "NaN"])
    if blank_invoice.any():
        report.record(
            "invoice_no_missing", "dropped", int(blank_invoice.sum()),
            "Invoice number is blank; the row cannot be attributed to an order.",
            _examples(df, blank_invoice),
        )
        df = df[~blank_invoice]
    else:
        report.record("invoice_no_missing", "retained", 0,
                      "All rows carry an invoice number.")

    blank_stock = df["stock_code"].isin(["", "nan", "None", "NaN"])
    if blank_stock.any():
        report.record(
            "stock_code_missing", "dropped", int(blank_stock.sum()),
            "Stock code is blank; the row cannot join to dim_product.",
            _examples(df, blank_stock),
        )
        df = df[~blank_stock]
    else:
        report.record("stock_code_missing", "retained", 0,
                      "All rows carry a stock code.")

    # ------------------------------------------------- rule: cancellations --
    # Cancellations are marked by a "C" prefix on the invoice number.
    df["is_cancelled"] = df["invoice_no"].str.upper().str.startswith("C")
    n_cancelled = int(df["is_cancelled"].sum())
    report.record(
        "cancelled_invoices", "flagged", n_cancelled,
        "Invoice numbers prefixed 'C' are cancellations/returns. RETAINED and "
        "flagged is_cancelled=True so returns analysis is possible; queries "
        "exclude them from gross revenue explicitly.",
        _examples(df, df["is_cancelled"]),
    )

    # ------------------------------------------- rule: negative quantities --
    neg_qty = df["quantity"] < 0
    neg_not_cancelled = neg_qty & ~df["is_cancelled"]
    report.record(
        "negative_quantity", "flagged", int(neg_qty.sum()),
        "Negative quantities represent returned goods. RETAINED so they net "
        "correctly against revenue rather than inflating it.",
        _examples(df, neg_qty),
    )
    if neg_not_cancelled.any():
        # Returns booked without a C-prefix: still returns, so treat them as
        # cancelled for consistency instead of letting the flag disagree with
        # the sign of the quantity.
        df.loc[neg_not_cancelled, "is_cancelled"] = True
        report.record(
            "negative_qty_without_c_prefix", "modified",
            int(neg_not_cancelled.sum()),
            "Negative quantity on a non-'C' invoice - reclassified as "
            "is_cancelled=True so the flag matches the economics of the row.",
            _examples(df, neg_not_cancelled),
        )

    # The mirror case: a "C" invoice carrying a *positive* quantity. Rare (a
    # single row in this dataset) but contradictory, so it is surfaced rather
    # than silently averaged into the returns total. The C-prefix is treated as
    # authoritative because it is an explicit operator action, whereas the sign
    # could be a keying error; the alternative - flipping is_cancelled off -
    # would let a credit note masquerade as a sale.
    pos_qty_cancelled = (df["quantity"] > 0) & df["is_cancelled"]
    if pos_qty_cancelled.any():
        report.record(
            "cancelled_invoice_with_positive_qty", "flagged",
            int(pos_qty_cancelled.sum()),
            "Invoice is 'C'-prefixed (a cancellation) but the quantity is "
            "positive, so the row contributes positive revenue to the returns "
            "bucket. RETAINED with is_cancelled=True - the explicit C-prefix is "
            "trusted over the sign - but flagged here because the two signals "
            "disagree and the row is therefore ambiguous.",
            _examples(df, pos_qty_cancelled),
        )
    else:
        report.record("cancelled_invoice_with_positive_qty", "retained", 0,
                      "No cancelled invoices carry a positive quantity.")

    # ---------------------------------------------------- rule: zero qty ----
    zero_qty = df["quantity"] == 0
    if zero_qty.any():
        report.record(
            "zero_quantity", "dropped", int(zero_qty.sum()),
            "Zero quantity generates zero revenue and no inventory movement; "
            "the row carries no information.",
            _examples(df, zero_qty),
        )
        df = df[~zero_qty]
    else:
        report.record("zero_quantity", "retained", 0, "No zero-quantity rows found.")

    # ------------------------------------------------- rule: bad prices ----
    neg_price = df["unit_price"] < 0
    if neg_price.any():
        report.record(
            "negative_unit_price", "dropped", int(neg_price.sum()),
            "Negative unit price is not economically meaningful (returns are "
            "expressed via negative quantity, not negative price); these are "
            "debt-adjustment artefacts.",
            _examples(df, neg_price),
        )
        df = df[~neg_price]
    else:
        report.record("negative_unit_price", "retained", 0,
                      "No negative unit prices found.")

    zero_price = df["unit_price"] == 0
    report.record(
        "zero_unit_price", "flagged", int(zero_price.sum()),
        "Zero-price lines are free samples, promotional items or manual "
        "adjustments. RETAINED (they are genuine activity) but contribute no "
        "revenue.",
        _examples(df, zero_price),
    )

    # ------------------------------------------------ rule: outlier values --
    outlier_qty = df["quantity"].abs() > MAX_PLAUSIBLE_QUANTITY
    if outlier_qty.any():
        report.record(
            "implausible_quantity", "dropped", int(outlier_qty.sum()),
            f"|quantity| > {MAX_PLAUSIBLE_QUANTITY:,} is orders of magnitude "
            "beyond any genuine order in this dataset and skews every "
            "aggregate; treated as data entry error.",
            _examples(df, outlier_qty),
        )
        df = df[~outlier_qty]
    else:
        report.record("implausible_quantity", "retained", 0,
                      "No implausible quantities found.")

    outlier_price = df["unit_price"] > MAX_PLAUSIBLE_PRICE
    if outlier_price.any():
        report.record(
            "implausible_unit_price", "dropped", int(outlier_price.sum()),
            f"Unit price > {MAX_PLAUSIBLE_PRICE:,.0f} indicates an adjustment "
            "or error rather than a retail product price.",
            _examples(df, outlier_price),
        )
        df = df[~outlier_price]
    else:
        report.record("implausible_unit_price", "retained", 0,
                      "No implausible unit prices found.")

    # -------------------------------------------------- rule: duplicates ----
    # "Exact duplicate" means every business-meaningful field matches, incl.
    # the timestamp. Two identical lines seconds apart would be kept.
    dup_cols = ["invoice_no", "stock_code", "quantity", "unit_price", "invoice_date"]
    dup_mask = df.duplicated(subset=dup_cols, keep="first")
    if dup_mask.any():
        report.record(
            "exact_duplicate_rows", "dropped", int(dup_mask.sum()),
            "Identical invoice/product/qty/price/timestamp - a data-entry "
            "artefact, not two separate sales. First occurrence kept.",
            _examples(df, dup_mask),
        )
        df = df[~dup_mask]
    else:
        report.record("exact_duplicate_rows", "retained", 0, "No exact duplicates found.")

    # --------------------------------------------- rule: missing customer --
    df["has_customer"] = df["customer_id"].notna()
    n_missing_cust = int((~df["has_customer"]).sum())
    report.record(
        "missing_customer_id", "flagged", n_missing_cust,
        "Guest checkouts with no customer id. RETAINED with customer_id=NULL: "
        "the revenue is real and belongs in sales totals, but these rows are "
        "excluded from RFM/churn, which require a customer identity.",
        _examples(df, ~df["has_customer"]),
    )

    # ------------------------------------------------------- derived cols ---
    df["customer_id"] = df["customer_id"].astype("Int64")
    df["stock_code"] = df["stock_code"].str.upper()
    df["is_service_code"] = df["stock_code"].isin(_SERVICE_CODE_UPPER) | df[
        "stock_code"
    ].str.match(r"^(BANK|ADJUST|AMAZON|TEST)", na=False)

    n_service = int(df["is_service_code"].sum())
    report.record(
        "service_stock_codes", "flagged", n_service,
        "Non-product lines (postage, bank charges, manual adjustments). "
        "RETAINED because they are real revenue/cost, but flagged so product "
        "analytics can exclude them.",
        _examples(df, df["is_service_code"]),
    )

    # Blank descriptions are cosmetic; a placeholder beats dropping the sale.
    missing_desc = df["description"].isna()
    if missing_desc.any():
        df.loc[missing_desc, "description"] = "UNKNOWN PRODUCT"
        report.record(
            "missing_description", "modified", int(missing_desc.sum()),
            "Description was blank. Filled with 'UNKNOWN PRODUCT' - the "
            "transaction is still valid and the stock code identifies the item.",
        )
    else:
        report.record("missing_description", "retained", 0,
                      "All rows carry a product description.")

    blank_country = df["country"].isin(["", "nan", "None", "Unspecified"])
    if blank_country.any():
        df.loc[blank_country, "country"] = "Unspecified"
        report.record(
            "missing_country", "modified", int(blank_country.sum()),
            "Country was blank or literal 'Unspecified'. Normalised to "
            "'Unspecified' so it forms one explicit bucket instead of several.",
        )
    else:
        report.record("missing_country", "retained", 0, "All rows carry a country.")

    # Revenue is materialised once, here, so every consumer agrees on it.
    df["revenue"] = (df["quantity"] * df["unit_price"]).round(2)

    df = df.reset_index(drop=True)
    report.rows_out = len(df)

    # ------------------------------------------------------------ metrics ---
    report.add_metric("date_range_start", df["invoice_date"].min())
    report.add_metric("date_range_end", df["invoice_date"].max())
    report.add_metric("unique_invoices", int(df["invoice_no"].nunique()))
    report.add_metric("unique_customers", int(df["customer_id"].nunique()))
    report.add_metric("unique_products", int(df["stock_code"].nunique()))
    report.add_metric("unique_countries", int(df["country"].nunique()))
    report.add_metric("gross_revenue", float(df.loc[~df["is_cancelled"], "revenue"].sum()))
    report.add_metric("returns_value", float(df.loc[df["is_cancelled"], "revenue"].sum()))
    report.add_metric("net_revenue", float(df["revenue"].sum()))
    report.add_metric(
        "pct_rows_without_customer",
        round(float((~df["has_customer"]).mean() * 100), 2),
    )

    return df


__all__ = [
    "clean_transactions",
    "normalise_columns",
    "SERVICE_CODES",
    "REQUIRED_COLUMNS",
    "MAX_PLAUSIBLE_QUANTITY",
    "MAX_PLAUSIBLE_PRICE",
]
