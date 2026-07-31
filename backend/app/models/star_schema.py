"""Star schema for the retail sales warehouse (SQLAlchemy 2.0 declarative).

    dim_date ─┐
   dim_product┼──> fact_sales <──┬─ dim_customer
    dim_country┘                 │
                                 └─ (grain: one row per invoice line item)

Modelling decisions worth calling out
-------------------------------------
* **Grain**: one row per *invoice line* (an invoice + a stock code). This is the
  finest grain the source provides and lets every downstream metric - revenue,
  AOV, basket size - be derived by aggregation rather than by storing
  pre-computed totals that can drift out of sync.

* **Surrogate keys on dimensions, natural keys preserved.** ``dim_product`` has
  an integer ``product_id`` PK plus the natural ``stock_code``. The source's
  stock codes are messy (mixed case, whitespace, non-product codes like
  ``POST``/``BANK CHARGES``), so pinning facts to a surrogate key keeps joins
  fast and insulates the fact table from source key cleanup.

* **``dim_customer.customer_id`` is the *natural* key.** Customer IDs in this
  dataset are already stable 5-digit integers, and analysts recognise them, so
  a surrogate would add indirection with no benefit.

* **RFM and churn columns live on ``dim_customer``.** They are slowly-changing
  customer attributes recomputed by the analytics pipeline. Denormalising them
  onto the dimension keeps the segment/churn dashboards to single-table scans
  instead of re-aggregating a million fact rows per request.

* **``is_cancelled`` is a column, not a filter applied at load time.** Cancelled
  invoices (``C``-prefixed, negative quantities) are *retained* so returns
  analysis is possible; every query decides for itself whether to exclude them.
  Silently dropping them at ETL time would make the warehouse lie about returns.
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base shared by every model and by Alembic autogenerate."""


# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------
class DimDate(Base):
    """Calendar dimension, pre-exploded so time-series queries never parse dates.

    ``date_id`` uses the conventional ``YYYYMMDD`` integer form, which is both
    human-readable in raw query output and naturally sortable.
    """

    __tablename__ = "dim_date"

    date_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    full_date: Mapped[date] = mapped_column(Date, nullable=False, unique=True, index=True)
    year: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    quarter: Mapped[int] = mapped_column(Integer, nullable=False)
    month: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    month_name: Mapped[str] = mapped_column(String(16), nullable=False)
    week_of_year: Mapped[int] = mapped_column(Integer, nullable=False)
    day_of_month: Mapped[int] = mapped_column(Integer, nullable=False)
    day_of_week: Mapped[int] = mapped_column(Integer, nullable=False)  # 0=Mon
    day_name: Mapped[str] = mapped_column(String(16), nullable=False)
    is_weekend: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    year_month: Mapped[str] = mapped_column(String(7), nullable=False, index=True)

    sales: Mapped[list["FactSales"]] = relationship(back_populates="date")

    __table_args__ = (
        Index("ix_dim_date_year_month_num", "year", "month"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<DimDate {self.date_id}>"


class DimCountry(Base):
    """Country dimension with a coarse region rollup for map/bar charts."""

    __tablename__ = "dim_country"

    country_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    country_name: Mapped[str] = mapped_column(String(96), nullable=False, unique=True, index=True)
    region: Mapped[str] = mapped_column(String(48), nullable=False, default="Unknown", index=True)
    iso_code: Mapped[str | None] = mapped_column(String(3), nullable=True)

    sales: Mapped[list["FactSales"]] = relationship(back_populates="country")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<DimCountry {self.country_name}>"


class DimProduct(Base):
    """Product dimension keyed by a surrogate id, natural stock code retained.

    ``category`` is derived heuristically from the description during ETL - the
    source has no category field, and a coarse grouping is far more useful on a
    dashboard than 4 000 raw product descriptions.
    """

    __tablename__ = "dim_product"

    product_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stock_code: Mapped[str] = mapped_column(String(32), nullable=False, unique=True, index=True)
    description: Mapped[str] = mapped_column(String(256), nullable=False, default="UNKNOWN")
    category: Mapped[str] = mapped_column(String(64), nullable=False, default="Other", index=True)
    # Non-product lines: postage, bank charges, manual adjustments, samples.
    is_service_code: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    sales: Mapped[list["FactSales"]] = relationship(back_populates="product")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<DimProduct {self.stock_code} {self.description[:24]!r}>"


class DimCustomer(Base):
    """Customer dimension enriched by the analytics pipeline.

    The ``rfm_*`` and ``churn_*`` columns are nullable because they only exist
    after ``scripts/run_analytics.py`` has run. A freshly-loaded warehouse is
    valid with them empty; the API reports "analytics not yet computed" rather
    than pretending every customer scores zero.
    """

    __tablename__ = "dim_customer"

    customer_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    country: Mapped[str] = mapped_column(String(96), nullable=False, default="Unspecified", index=True)
    first_purchase_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    last_purchase_date: Mapped[date | None] = mapped_column(Date, nullable=True, index=True)

    # ---- RFM (recomputed by the analytics pipeline) ----
    rfm_recency: Mapped[int | None] = mapped_column(Integer, nullable=True)      # days
    rfm_frequency: Mapped[int | None] = mapped_column(Integer, nullable=True)    # orders
    rfm_monetary: Mapped[float | None] = mapped_column(Numeric(14, 2), nullable=True)
    r_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    f_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    m_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    rfm_score: Mapped[str | None] = mapped_column(String(3), nullable=True)
    rfm_segment: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)

    # ---- Behavioural aggregates (features for the churn model) ----
    # server_default matters here: the ETL bulk-loads through pandas'
    # ``to_sql``, which emits raw INSERTs and therefore bypasses SQLAlchemy's
    # Python-side ``default=``. Without a database-level default these NOT NULL
    # columns reject every bulk-loaded row. The analytics step fills them in.
    total_orders: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    total_revenue: Mapped[float] = mapped_column(
        Numeric(14, 2), nullable=False, default=0, server_default="0"
    )
    total_items: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    avg_order_value: Mapped[float | None] = mapped_column(Numeric(12, 2), nullable=True)
    distinct_products: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    return_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    tenure_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    avg_days_between_orders: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ---- Churn ----
    churn_risk_score: Mapped[float | None] = mapped_column(Float, nullable=True, index=True)
    is_churned: Mapped[bool | None] = mapped_column(Boolean, nullable=True, index=True)

    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True, default=func.now(), onupdate=func.now()
    )

    sales: Mapped[list["FactSales"]] = relationship(back_populates="customer")

    __table_args__ = (
        Index("ix_dim_customer_segment_risk", "rfm_segment", "churn_risk_score"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<DimCustomer {self.customer_id} {self.rfm_segment}>"


# ---------------------------------------------------------------------------
# Fact
# ---------------------------------------------------------------------------
class FactSales(Base):
    """One row per invoice line item.

    ``revenue`` is stored rather than always computed as ``quantity *
    unit_price``. It is the single most aggregated column in the application,
    and materialising it lets Postgres answer revenue queries straight from an
    index without evaluating an expression per row.

    The ``(invoice_no, stock_code, line_number)`` unique constraint is what
    makes the load **idempotent**: re-running the ETL collides on this key
    instead of duplicating the warehouse. ``line_number`` disambiguates the
    genuine case of the same product appearing twice on one invoice.
    """

    __tablename__ = "fact_sales"

    sale_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    invoice_no: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    line_number: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    product_id: Mapped[int] = mapped_column(
        ForeignKey("dim_product.product_id", ondelete="RESTRICT"), nullable=False, index=True
    )
    # Nullable by design: ~22% of source rows are guest checkouts with no
    # customer id. They are real revenue and must count toward sales totals,
    # so they are kept with a NULL customer rather than discarded.
    customer_id: Mapped[int | None] = mapped_column(
        ForeignKey("dim_customer.customer_id", ondelete="SET NULL"), nullable=True, index=True
    )
    date_id: Mapped[int] = mapped_column(
        ForeignKey("dim_date.date_id", ondelete="RESTRICT"), nullable=False, index=True
    )
    country_id: Mapped[int] = mapped_column(
        ForeignKey("dim_country.country_id", ondelete="RESTRICT"), nullable=False, index=True
    )

    invoice_timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    unit_price: Mapped[float] = mapped_column(Numeric(12, 3), nullable=False)
    revenue: Mapped[float] = mapped_column(Numeric(14, 2), nullable=False, index=True)
    is_cancelled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)

    product: Mapped["DimProduct"] = relationship(back_populates="sales")
    customer: Mapped["DimCustomer | None"] = relationship(back_populates="sales")
    date: Mapped["DimDate"] = relationship(back_populates="sales")
    country: Mapped["DimCountry"] = relationship(back_populates="sales")

    __table_args__ = (
        UniqueConstraint(
            "invoice_no", "product_id", "line_number", name="uq_fact_sales_line"
        ),
        # Covers the dashboard's hottest query: revenue over time, optionally
        # sliced by country, excluding cancellations.
        Index("ix_fact_sales_date_country", "date_id", "country_id"),
        Index("ix_fact_sales_customer_date", "customer_id", "date_id"),
        Index("ix_fact_sales_cancelled_date", "is_cancelled", "date_id"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<FactSales {self.invoice_no} p={self.product_id} rev={self.revenue}>"


__all__ = [
    "Base",
    "DimDate",
    "DimCountry",
    "DimProduct",
    "DimCustomer",
    "FactSales",
]
