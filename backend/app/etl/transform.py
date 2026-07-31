"""Transform cleaned transactions into star-schema dimension and fact frames.

The source is a single flat table, so the dimensions are *derived*:

* ``dim_date``    - a contiguous calendar spanning the invoice date range.
* ``dim_country`` - distinct countries, enriched with a region rollup.
* ``dim_product`` - distinct stock codes, with a keyword-derived category.
* ``dim_customer``- distinct customer ids with first/last purchase dates.
                    (RFM/churn columns are filled later by the analytics run.)
* ``fact_sales``  - one row per invoice line, joined to the surrogate keys.
"""

from __future__ import annotations

import pandas as pd

from app.core.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Country -> region. The dataset is UK-centric with 40-odd countries; grouping
# them makes the geography charts legible instead of a 40-bar sprawl.
# ---------------------------------------------------------------------------
REGION_MAP: dict[str, str] = {
    # United Kingdom & Ireland
    "United Kingdom": "UK & Ireland",
    "EIRE": "UK & Ireland",
    "Channel Islands": "UK & Ireland",
    # Western Europe
    "France": "Western Europe",
    "Germany": "Western Europe",
    "Netherlands": "Western Europe",
    "Belgium": "Western Europe",
    "Switzerland": "Western Europe",
    "Austria": "Western Europe",
    "Luxembourg": "Western Europe",
    "Monaco": "Western Europe",
    "Liechtenstein": "Western Europe",
    # Southern Europe
    "Spain": "Southern Europe",
    "Portugal": "Southern Europe",
    "Italy": "Southern Europe",
    "Greece": "Southern Europe",
    "Malta": "Southern Europe",
    "Cyprus": "Southern Europe",
    "San Marino": "Southern Europe",
    # Northern Europe
    "Sweden": "Northern Europe",
    "Norway": "Northern Europe",
    "Denmark": "Northern Europe",
    "Finland": "Northern Europe",
    "Iceland": "Northern Europe",
    "Lithuania": "Northern Europe",
    "Latvia": "Northern Europe",
    "Estonia": "Northern Europe",
    # Eastern Europe
    "Poland": "Eastern Europe",
    "Czech Republic": "Eastern Europe",
    "Slovakia": "Eastern Europe",
    "Slovenia": "Eastern Europe",
    "Hungary": "Eastern Europe",
    "Romania": "Eastern Europe",
    "Bulgaria": "Eastern Europe",
    "RSA": "Africa",
    # Americas
    "USA": "Americas",
    "Canada": "Americas",
    "Brazil": "Americas",
    # Asia-Pacific
    "Australia": "Asia-Pacific",
    "Japan": "Asia-Pacific",
    "Singapore": "Asia-Pacific",
    "Hong Kong": "Asia-Pacific",
    "Korea": "Asia-Pacific",
    "Thailand": "Asia-Pacific",
    "Indonesia": "Asia-Pacific",
    "India": "Asia-Pacific",
    # Middle East & Africa
    "Israel": "Middle East",
    "United Arab Emirates": "Middle East",
    "Bahrain": "Middle East",
    "Saudi Arabia": "Middle East",
    "Lebanon": "Middle East",
    "Nigeria": "Africa",
    "West Indies": "Americas",
    "European Community": "Other",
    "Unspecified": "Unspecified",
}

ISO_MAP: dict[str, str] = {
    "United Kingdom": "GBR", "EIRE": "IRL", "France": "FRA", "Germany": "DEU",
    "Netherlands": "NLD", "Belgium": "BEL", "Switzerland": "CHE", "Austria": "AUT",
    "Spain": "ESP", "Portugal": "PRT", "Italy": "ITA", "Greece": "GRC",
    "Sweden": "SWE", "Norway": "NOR", "Denmark": "DNK", "Finland": "FIN",
    "Iceland": "ISL", "Poland": "POL", "Czech Republic": "CZE", "USA": "USA",
    "Canada": "CAN", "Brazil": "BRA", "Australia": "AUS", "Japan": "JPN",
    "Singapore": "SGP", "Hong Kong": "HKG", "Israel": "ISR", "Cyprus": "CYP",
    "Malta": "MLT", "Lithuania": "LTU", "Lebanon": "LBN", "RSA": "ZAF",
    "United Arab Emirates": "ARE", "Saudi Arabia": "SAU", "Bahrain": "BHR",
    "Channel Islands": "GBR", "Luxembourg": "LUX", "Nigeria": "NGA",
    "Korea": "KOR", "Thailand": "THA", "Indonesia": "IDN", "India": "IND",
}

# ---------------------------------------------------------------------------
# Product category keywords, checked in order (first match wins). Ordering
# matters: "CHRISTMAS" outranks "DECORATION" so seasonal lines group together.
# ---------------------------------------------------------------------------
CATEGORY_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("Seasonal & Christmas", ("CHRISTMAS", "XMAS", "ADVENT", "SANTA", "SNOWMAN",
                              "REINDEER", "EASTER", "HALLOWEEN", "VALENTINE")),
    ("Kitchen & Dining", ("MUG", "PLATE", "BOWL", "CUTLERY", "TEAPOT", "JUG",
                          "CAKE", "BAKING", "KITCHEN", "APRON", "TRAY", "CUP",
                          "SAUCER", "GLASS", "BOTTLE", "TEASPOON", "NAPKIN",
                          "COASTER", "PLACEMAT", "LUNCH BOX", "CANISTER")),
    ("Home Decor", ("HEART", "FRAME", "CANDLE", "LANTERN", "HOLDER", "ORNAMENT",
                    "DECORATION", "HANGING", "MIRROR", "CLOCK", "VASE", "LIGHT",
                    "CUSHION", "DOORMAT", "WALL ART", "PHOTO", "SIGN")),
    ("Storage & Organisation", ("BOX", "STORAGE", "BASKET", "TIN", "JAR",
                               "ORGANISER", "RACK", "HOOK", "DRAWER", "CADDY",
                               "CRATE", "CABINET")),
    ("Bags & Accessories", ("BAG", "PURSE", "WALLET", "UMBRELLA", "SCARF",
                            "HAT", "GLOVES", "JEWELLERY", "NECKLACE",
                            "BRACELET", "EARRINGS", "RING", "KEYRING")),
    ("Toys & Games", ("TOY", "GAME", "PUZZLE", "DOLL", "TEDDY", "BEAR",
                      "PLAYHOUSE", "SKITTLES", "SPINNING TOP", "SOLDIER",
                      "BUILDING BLOCK", "YOYO", "KIDS", "CHILDREN")),
    ("Stationery & Craft", ("PEN", "PENCIL", "NOTEBOOK", "CARD", "PAPER",
                            "CRAFT", "STICKER", "RUBBER", "CHALK", "ERASER",
                            "JOURNAL", "GIFT WRAP", "RIBBON", "TAPE", "ENVELOPE",
                            "BOOKMARK", "NOTE PAD")),
    ("Garden & Outdoor", ("GARDEN", "PLANT", "FLOWER", "POT", "WATERING",
                          "BIRD", "OUTDOOR", "PICNIC", "PARASOL", "SEED")),
    ("Bath & Beauty", ("SOAP", "BATH", "TOWEL", "TOOTHBRUSH", "PERFUME",
                       "LOTION", "MIRROR COMPACT", "COSMETIC")),
    ("Party & Celebration", ("PARTY", "BUNTING", "BALLOON", "CONFETTI",
                             "CAKE STAND", "WEDDING", "BIRTHDAY", "GARLAND")),
]

SERVICE_CATEGORY = "Postage & Service"


def derive_category(description: str, is_service: bool = False) -> str:
    """Map a product description to a coarse category.

    The dataset has ~4 500 distinct descriptions and no category column;
    keyword rules give a dashboard-friendly grouping. Unmatched items fall into
    "Other" rather than being force-fit into a bucket that would mislead.
    """
    if is_service:
        return SERVICE_CATEGORY
    if not description:
        return "Other"

    text = str(description).upper()
    for category, keywords in CATEGORY_RULES:
        if any(keyword in text for keyword in keywords):
            return category
    return "Other"


def build_dim_date(df: pd.DataFrame) -> pd.DataFrame:
    """Build a gapless calendar dimension spanning the invoice date range."""
    # A contiguous calendar, not merely the dates that happen to appear in the
    # data. The retailer does not trade every day (there is exactly one
    # Saturday in the whole two-year span), so a sparse dimension would make
    # non-trading days vanish from time series rather than render as zero, and
    # would quietly bias any day-of-week aggregate. Building the full range
    # costs a few hundred rows and keeps trend queries honest.
    observed = pd.to_datetime(df["invoice_date"].dt.date)
    dates = pd.date_range(observed.min(), observed.max(), freq="D")

    dim = pd.DataFrame({"full_date": dates})
    dim["date_id"] = dim["full_date"].dt.strftime("%Y%m%d").astype(int)
    dim["year"] = dim["full_date"].dt.year.astype(int)
    dim["quarter"] = dim["full_date"].dt.quarter.astype(int)
    dim["month"] = dim["full_date"].dt.month.astype(int)
    dim["month_name"] = dim["full_date"].dt.strftime("%B")
    dim["week_of_year"] = dim["full_date"].dt.isocalendar().week.astype(int)
    dim["day_of_month"] = dim["full_date"].dt.day.astype(int)
    dim["day_of_week"] = dim["full_date"].dt.dayofweek.astype(int)
    dim["day_name"] = dim["full_date"].dt.strftime("%A")
    dim["is_weekend"] = dim["day_of_week"].isin([5, 6])
    dim["year_month"] = dim["full_date"].dt.strftime("%Y-%m")
    dim["full_date"] = dim["full_date"].dt.date

    trading_days = observed.nunique()
    logger.info(
        "  dim_date      : %s rows (%s .. %s), %s with trading activity",
        f"{len(dim):,}", dim["full_date"].min(), dim["full_date"].max(),
        f"{trading_days:,}",
    )
    return dim


def build_dim_country(df: pd.DataFrame) -> pd.DataFrame:
    """Build the country dimension with region and ISO enrichment."""
    countries = sorted(df["country"].dropna().unique())
    dim = pd.DataFrame({"country_name": countries})
    dim["region"] = dim["country_name"].map(REGION_MAP).fillna("Other")
    dim["iso_code"] = dim["country_name"].map(ISO_MAP)
    # Surrogate keys are assigned deterministically (sorted order) so repeated
    # runs produce identical ids - important for idempotency.
    dim.insert(0, "country_id", range(1, len(dim) + 1))

    unmapped = dim.loc[dim["region"] == "Other", "country_name"].tolist()
    if unmapped:
        logger.debug("  Countries without an explicit region: %s", unmapped)

    logger.info("  dim_country   : %s rows across %s regions",
                f"{len(dim):,}", dim["region"].nunique())
    return dim


def build_dim_product(df: pd.DataFrame) -> pd.DataFrame:
    """Build the product dimension, one row per stock code.

    A stock code can carry several descriptions across its history (typos,
    renames). The most frequent description is chosen as canonical.
    """
    grouped = (
        df.groupby("stock_code", as_index=False)
        .agg(
            description=("description", lambda s: s.mode().iat[0] if not s.mode().empty else "UNKNOWN"),
            is_service_code=("is_service_code", "max"),
        )
        .sort_values("stock_code")
        .reset_index(drop=True)
    )

    grouped["category"] = [
        derive_category(desc, bool(svc))
        for desc, svc in zip(grouped["description"], grouped["is_service_code"])
    ]
    grouped["description"] = grouped["description"].astype(str).str.slice(0, 256)
    grouped.insert(0, "product_id", range(1, len(grouped) + 1))

    cat_counts = grouped["category"].value_counts()
    logger.info("  dim_product   : %s rows, %s categories (largest: %s=%s)",
                f"{len(grouped):,}", grouped["category"].nunique(),
                cat_counts.index[0], cat_counts.iloc[0])
    return grouped


def build_dim_customer(df: pd.DataFrame) -> pd.DataFrame:
    """Build the customer dimension from identified transactions only.

    Guest-checkout rows (no customer id) are excluded here by design - they are
    not a customer - but their revenue still lands in ``fact_sales``.
    """
    identified = df[df["has_customer"]].copy()
    if identified.empty:
        return pd.DataFrame(
            columns=["customer_id", "country", "first_purchase_date", "last_purchase_date"]
        )

    grouped = identified.groupby("customer_id", as_index=False).agg(
        first_purchase_date=("invoice_date", "min"),
        last_purchase_date=("invoice_date", "max"),
        # A handful of customers order from more than one country; the modal
        # country is the sensible single-valued attribute.
        country=("country", lambda s: s.mode().iat[0] if not s.mode().empty else "Unspecified"),
    )
    grouped["first_purchase_date"] = grouped["first_purchase_date"].dt.date
    grouped["last_purchase_date"] = grouped["last_purchase_date"].dt.date
    grouped["customer_id"] = grouped["customer_id"].astype(int)

    logger.info("  dim_customer  : %s rows", f"{len(grouped):,}")
    return grouped


def build_fact_sales(
    df: pd.DataFrame,
    dim_product: pd.DataFrame,
    dim_country: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble the fact table with surrogate keys resolved."""
    fact = df.copy()

    product_key = dim_product.set_index("stock_code")["product_id"]
    country_key = dim_country.set_index("country_name")["country_id"]

    fact["product_id"] = fact["stock_code"].map(product_key)
    fact["country_id"] = fact["country"].map(country_key)
    fact["date_id"] = fact["invoice_date"].dt.strftime("%Y%m%d").astype(int)
    fact["invoice_timestamp"] = fact["invoice_date"]

    # line_number disambiguates the legitimate case of one product appearing
    # twice on the same invoice, and is what makes the uniqueness constraint
    # (and therefore the idempotent load) correct.
    fact["line_number"] = fact.groupby(["invoice_no", "product_id"]).cumcount()

    # A null surrogate key here would mean a dimension build missed a value -
    # that is a bug, not dirty data, so fail loudly rather than load orphans.
    missing_product = int(fact["product_id"].isna().sum())
    missing_country = int(fact["country_id"].isna().sum())
    if missing_product or missing_country:
        raise ValueError(
            f"Dimension lookup failed: {missing_product} rows without a "
            f"product_id, {missing_country} without a country_id."
        )

    columns = [
        "invoice_no", "line_number", "product_id", "customer_id", "date_id",
        "country_id", "invoice_timestamp", "quantity", "unit_price", "revenue",
        "is_cancelled",
    ]
    fact = fact[columns].copy()
    fact["product_id"] = fact["product_id"].astype(int)
    fact["country_id"] = fact["country_id"].astype(int)
    fact["quantity"] = fact["quantity"].astype(int)

    logger.info("  fact_sales    : %s rows", f"{len(fact):,}")
    return fact


def transform_all(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build every dimension and the fact table from cleaned transactions."""
    logger.info("Building star schema...")
    dim_date = build_dim_date(df)
    dim_country = build_dim_country(df)
    dim_product = build_dim_product(df)
    dim_customer = build_dim_customer(df)
    fact = build_fact_sales(df, dim_product, dim_country)

    return {
        "dim_date": dim_date,
        "dim_country": dim_country,
        "dim_product": dim_product,
        "dim_customer": dim_customer,
        "fact_sales": fact,
    }


__all__ = [
    "transform_all",
    "build_dim_date",
    "build_dim_country",
    "build_dim_product",
    "build_dim_customer",
    "build_fact_sales",
    "derive_category",
    "REGION_MAP",
    "CATEGORY_RULES",
]
