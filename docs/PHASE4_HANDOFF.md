# Phase 4 Handoff — API layer (in progress)

Context ran out mid-phase. This captures the state and the contracts needed to
finish without re-deriving them.

## Done

- **Phases 1–3** complete (see `PHASE3_HANDOFF.md`): star schema + Alembic, ETL
  with `data_quality_report.json`, RFM, statistics, churn model + registry.
- **Phase 4 partial**:
  - `app/core/error_handlers.py` — global handler, `{"error": {code, message, status}}`
  - `app/schemas/common.py` — `APIModel`, `ErrorResponse`, `DateRangeParams`,
    `PaginationParams`, `Page[T]`, health models
  - `app/schemas/kpi.py`, `app/api/deps.py`, `health.py`, `kpis.py`, `sales.py`,
    `router.py`, `main.py`
  - `app/services/kpi.py` — summary, trend, by-country
  - `app/services/customer.py` — **just written**, not yet wired to a router
  - Smoke test: 8/8 endpoints passing before this file was added

## Remaining in Phase 4

1. `app/schemas/customer.py` — response models for segments, profile, at-risk list
2. `app/schemas/churn.py` — model-performance, predict request/response
3. `app/schemas/stats.py` — hypothesis test envelope
4. `app/api/customers.py` — `/customers/segments`, `/customers/{customer_id}`
5. `app/api/churn.py` — `/churn/at-risk`, `/churn/model-performance`, `POST /predict/churn`
6. `app/api/stats.py` — `/stats/hypothesis-test?type=ttest|anova|chi2`
7. Register all three on `app/api/router.py`
8. `tests/test_api_*.py` — happy path + error path per endpoint

## Contracts (verified — do not re-derive)

**Session vs Engine.** Routers get a `Session` via `Depends(get_db)`. But
`services/rfm.py`, `services/statistics.py` and `ml/features.py` all take an
`Engine`. Bridge with `db.get_bind()` inside the router — do *not* add a second
FastAPI dependency for the engine, and do not change the service signatures
(the CLI scripts in `scripts/` call them with an engine directly).

`services/kpi.py` and `services/customer.py` take a `Session`, not an engine.

**Numeric coercion.** Postgres returns `Decimal`; `_to_float` / `_to_int` in
`services/kpi.py` normalise it. `customer.py` imports these two helpers from
`kpi.py`. If a third consumer appears, promote them to a shared module rather
than importing `_`-prefixed names more widely.

**Exceptions** (`app/core/exceptions.py`, all extend `RetailInsightsError`):
- `CustomerNotFoundError(customer_id)` — positional arg, → 404 `CUSTOMER_NOT_FOUND`
- `InvalidDateRangeError.start_after_end(start, end)` — classmethod, → 422
- `ModelNotLoadedError` — → 503, raised by the registry
- also `SegmentNotFoundError`, `InvalidParameterError`, `InsufficientDataError`,
  `StatisticalTestError`, `DatabaseConnectionError`, `DatabaseQueryError`

**Model registry** (`app/ml/registry.py`): `load_model()`, `get_metadata()`,
`is_available()`, `predict_proba(features)`, `reset()`. Raises
`ModelNotLoadedError` when the artifact is missing/corrupt — let it propagate to
the global handler for the 503; do not catch it in the router.

**Statistics service** returns `TestResult` with `.to_dict()` — the stats router
should call that rather than rebuilding the envelope field by field.

**`dim_customer` enrichment columns are NULL until analytics runs.**
`get_segment_distribution` returns `computed: False` with an empty list in that
case; the profile returns `rfm_computed: False` with null scores. Endpoints must
preserve this distinction — "not yet computed" is not "zero".

## Design decisions worth keeping

- `SEGMENT_ORDER` and `SEGMENT_HEALTH` live in `services/customer.py` and ship in
  the API response (`health` field per segment). The frontend should colour from
  that field, not hardcode a parallel mapping.
- `risk_band()` fixes the high/medium/low thresholds at 0.7 / 0.4 server-side for
  the same reason.
- `/churn/at-risk` sorts by `churn_risk_score × total_revenue`, not probability.
  A 95%-likely churner worth £40 ranks below a 60%-likely one worth £12,000.
  Response carries both `revenue_at_risk` (worst case) and
  `expected_revenue_loss` (probability-weighted).
- Segment revenue sums to *customer-attributed* revenue, not total revenue —
  guest invoices have a NULL customer. Exposed as `attributed_revenue` so it
  reconciles against `/kpis/summary` instead of reading as a bug.

## Then

- **Phase 5**: React frontend, Executive Overview first
- **Phase 6**: Docker + compose + GitHub Actions
- **Phase 7**: README + polish
