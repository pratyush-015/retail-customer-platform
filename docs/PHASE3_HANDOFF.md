# Phase 3 Handoff — FastAPI Backend

Phases 1–2 are complete, committed, and green. Phase 3 is **partially started**:
`backend/app/core/error_handlers.py` is written and ready. This note is the
spec for finishing the API layer in a fresh session.

## Current state

| Layer | Status |
|---|---|
| Star schema + Alembic migration | ✅ done |
| ETL (clean/transform/load, `data_quality_report.json`) | ✅ done, idempotent |
| RFM segmentation | ✅ done, persisted to `dim_customer` |
| Statistical tests (t-test/ANOVA/chi²) | ✅ done |
| Churn model (logistic regression) | ✅ trained, ROC-AUC **0.8036**, CV 0.7922 ± 0.011 |
| Global error handlers | ✅ **just added**, not yet wired to an app |
| Everything else in Phase 3 | ❌ not started |

Test suite: **122 passing** (`cd backend && python -m pytest -q`).

Warehouse is loaded (SQLite at `data/retail_insights.db`): 5,942 customers,
~1M fact rows. Model artifacts live in `backend/app/ml/artifacts/`
(`churn_model.joblib`, `model_metadata.json`).

## Verified contracts (don't re-derive these)

**Settings** (`app.core.config.get_settings()`): `api_v1_prefix="/api"`,
`cors_origin_list`, `artifact_path`, `churn_threshold_days=90`, `is_sqlite`,
`environment`, `app_name`.

**DB** (`app.core.database`): `get_db()` FastAPI dependency, `get_engine()`,
`wait_for_database(...)` (exponential backoff, raises `DatabaseConnectionError`).

**Exceptions** (`app.core.exceptions`): `RetailInsightsError` base with
`.code/.status/.message/.details/.to_dict()`, plus `DatabaseConnectionError`
(503), `CustomerNotFoundError(customer_id)` (404), `InvalidDateRangeError`
(422, has `.start_after_end(start, end)`), `ModelNotLoadedError` (503),
`InsufficientDataError`, `StatisticalTestError`, `SegmentNotFoundError`,
`InvalidParameterError`, `DatabaseQueryError`.

**Key columns** — `fact_sales`: `invoice_no, product_id, customer_id, date_id,
country_id, invoice_timestamp, quantity, unit_price, revenue, is_cancelled`.
`dim_customer`: `customer_id, country, first/last_purchase_date, rfm_recency,
rfm_frequency, rfm_monetary, r_score, f_score, m_score, rfm_score, rfm_segment,
total_orders, total_revenue, avg_order_value, return_rate, tenure_days,
avg_days_between_orders, churn_risk_score, is_churned`. `dim_date`: `date_id,
full_date, year, quarter, month, month_name, week_of_year, day_of_week,
day_name, is_weekend, year_month`. `dim_country`: `country_id, country_name,
region, iso_code`. `dim_product`: `product_id, stock_code, description,
category, is_service_code`.

> Note: revenue/monetary columns are `Numeric` → SQLAlchemy returns `Decimal`.
> Cast to `float` in the service layer or Pydantic serialisation will be ugly.

**Services**: `app.services.rfm` (`compute_rfm`, `summarise_segments`,
`SEGMENT_ORDER`, `SEGMENT_DESCRIPTIONS`, `load_transactions`, `dataset_max_date`),
`app.services.statistics` (`aov_by_country`, `aov_across_countries`,
`aov_across_segments`, `churn_by_country`, `churn_by_segment` — all return a
`TestResult` dataclass that is already API-shaped), `app.ml.features`
(`FEATURE_NAMES`, `build_scoring_features`, `latest_warehouse_date`).

## Remaining work, in build order

1. **`app/ml/registry.py`** — lazy singleton loading `churn_model.joblib` +
   `model_metadata.json`. Cache the loaded pipeline; on missing/corrupt file
   raise `ModelNotLoadedError` so those endpoints 503 while the rest of the API
   stays up. Expose `get_model()`, `get_metadata()`, `is_available()`,
   `reset()` (for tests). Must not raise at import time.
2. **`app/schemas/`** — Pydantic response + query models. Include a
   `DateRangeParams` validator raising `InvalidDateRangeError.start_after_end`,
   a `threshold: float = Field(ge=0, le=1)` for churn, and a
   `ChurnPredictRequest` with non-negative RFM fields.
3. **`app/services/kpi.py`** — summary aggregation (revenue, orders, customers,
   AOV, MoM growth) and the trend/by-country queries. Exclude
   `is_cancelled` rows from revenue; join `dim_date` for granularity.
4. **`app/api/`** — one router per resource (`health`, `kpis`, `sales`,
   `customers`, `churn`, `stats`, `predict`), aggregated in `app/api/router.py`.
5. **`app/main.py`** — `create_app()` factory: CORS from settings,
   `register_exception_handlers(app)`, lifespan calling `wait_for_database`
   (log-and-continue so `/api/health` can report `degraded` rather than the
   container crash-looping), OpenAPI metadata.
6. **Tests** (`backend/tests/test_api_*.py`) — `TestClient` against a synthetic
   SQLite warehouse via dependency override of `get_db`. Cover happy path +
   error path per endpoint, and assert the envelope shape
   (`error.code`/`message`/`status`) for: unknown customer → 404
   `CUSTOMER_NOT_FOUND`; `threshold=-1` → 422 with `details.fields[0].field ==
   "threshold"`; model absent → 503 `MODEL_NOT_LOADED` (monkeypatch
   `registry.reset()` + a bogus artifact dir).

The minimal slice the user approved for this session was: health, KPIs, error
handlers, model loader + tests. Error handlers are done; start at step 1.
