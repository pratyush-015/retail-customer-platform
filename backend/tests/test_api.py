"""API-layer tests: happy paths, error paths, and response contracts.

These run against whatever database the environment points at. Endpoints that
need loaded data are skipped rather than failed when the warehouse is empty, so
the suite stays meaningful in CI (where no data exists) without being a false
green locally. The error-path tests need no data at all and always run, which is
deliberate: the error contract is the part most likely to regress silently.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.ml import registry


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(scope="module")
def has_data(client: TestClient) -> bool:
    """Whether the warehouse has rows, for skipping data-dependent assertions."""
    response = client.get("/api/kpis/summary")
    return response.status_code == 200 and response.json().get("total_orders", 0) > 0


@pytest.fixture(scope="module")
def has_model() -> bool:
    try:
        registry.load_model()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Health and contract basics
# ---------------------------------------------------------------------------


def test_health_reports_each_dependency(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    # The value of a health check is the per-dependency detail, not the 200: a
    # 200 masking a dead database is exactly the failure worth catching.
    components = {c["name"] for c in body["components"]}
    assert {"database", "churn_model"} <= components


def test_openapi_schema_builds(client: TestClient) -> None:
    """A broken response_model surfaces here rather than at request time."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    for required in (
        "/api/kpis/summary",
        "/api/sales/trend",
        "/api/sales/by-country",
        "/api/customers/segments",
        "/api/customers/{customer_id}",
        "/api/churn/at-risk",
        "/api/churn/model-performance",
        "/api/stats/hypothesis-test",
        "/api/predict/churn",
        "/api/health",
    ):
        assert required in paths, f"{required} missing from the OpenAPI schema"


# ---------------------------------------------------------------------------
# Error contract
# ---------------------------------------------------------------------------


def test_unknown_route_uses_the_standard_error_shape(client: TestClient) -> None:
    """404s from the framework must match the documented envelope too.

    Without an explicit handler FastAPI emits ``{"detail": "Not Found"}``, so a
    frontend written against ``error.code`` would break on exactly the responses
    it is most likely to hit.
    """
    response = client.get("/api/does-not-exist")
    assert response.status_code == 404
    error = response.json()["error"]
    assert set(error) >= {"code", "message", "status"}
    assert error["status"] == 404


def test_customer_not_found_returns_typed_404(client: TestClient) -> None:
    response = client.get("/api/customers/999999999")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "CUSTOMER_NOT_FOUND"


def test_invalid_customer_id_is_422_naming_the_field(client: TestClient) -> None:
    response = client.get("/api/customers/-1")
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["status"] == 422
    # The response has to say *which* field failed, or the client cannot fix it.
    assert "customer_id" in str(error).lower()


def test_negative_threshold_is_rejected(client: TestClient) -> None:
    assert client.get("/api/churn/at-risk?threshold=-0.5").status_code == 422


def test_threshold_above_one_is_rejected(client: TestClient) -> None:
    assert client.get("/api/churn/at-risk?threshold=1.5").status_code == 422


def test_inverted_date_range_is_rejected(client: TestClient) -> None:
    response = client.get(
        "/api/sales/trend?granularity=month&start_date=2011-12-01&end_date=2011-01-01"
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] in {"INVALID_DATE_RANGE", "VALIDATION_ERROR"}


def test_invalid_granularity_is_rejected(client: TestClient) -> None:
    assert client.get("/api/sales/trend?granularity=fortnight").status_code == 422


def test_unknown_test_type_is_rejected(client: TestClient) -> None:
    assert client.get("/api/stats/hypothesis-test?type=magic").status_code == 422


def test_ttest_without_both_countries_is_rejected(client: TestClient) -> None:
    response = client.get("/api/stats/hypothesis-test?type=ttest&country_a=Germany")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_PARAMETER"


def test_ttest_against_itself_is_rejected(client: TestClient) -> None:
    """Comparing a group with itself always yields p=1; that is a bug, not a result."""
    response = client.get(
        "/api/stats/hypothesis-test?type=ttest&country_a=Germany&country_b=germany"
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def test_predict_returns_probability_and_band(client: TestClient, has_model: bool) -> None:
    if not has_model:
        pytest.skip("No churn model artifact available.")
    response = client.post(
        "/api/predict/churn", json={"recency": 200, "frequency": 3, "monetary": 500}
    )
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_band"] in {"low", "medium", "high"}
    # Ten features must reach the model even though three were supplied.
    assert len(body["features_used"]) == 10
    assert "avg_order_value" in body["derived_features"]
    assert body["interpretation"]


def test_predict_is_monotonic_in_recency(client: TestClient, has_model: bool) -> None:
    """A longer-absent customer must not score as safer, all else equal.

    This is the assertion that caught the skew bug in the first model: a
    high-frequency customer who had not purchased for a year scored 0.001.
    """
    if not has_model:
        pytest.skip("No churn model artifact available.")
    base = {"frequency": 5, "monetary": 1000, "avg_days_between_orders": 30}
    recent = client.post("/api/predict/churn", json={**base, "recency": 15}).json()
    stale = client.post("/api/predict/churn", json={**base, "recency": 400}).json()
    assert stale["churn_probability"] > recent["churn_probability"]


def test_predict_rejects_negative_recency(client: TestClient) -> None:
    response = client.post(
        "/api/predict/churn", json={"recency": -5, "frequency": 3, "monetary": 500}
    )
    assert response.status_code == 422


def test_predict_rejects_missing_required_field(client: TestClient) -> None:
    response = client.post("/api/predict/churn", json={"recency": 30})
    assert response.status_code == 422
    assert "frequency" in str(response.json()).lower()


def test_predict_rejects_contradictory_aov(client: TestClient) -> None:
    """A 999 AOV cannot coexist with 400 total across 4 orders."""
    response = client.post(
        "/api/predict/churn",
        json={"recency": 50, "frequency": 4, "monetary": 400, "avg_order_value": 999},
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Data endpoints
# ---------------------------------------------------------------------------


def test_kpi_summary_is_internally_consistent(client: TestClient, has_data: bool) -> None:
    if not has_data:
        pytest.skip("Warehouse is empty; run the ETL first.")
    body = client.get("/api/kpis/summary").json()
    assert body["total_revenue"] > 0
    assert body["total_orders"] > 0
    # AOV is revenue/orders by definition; a mismatch means one of them is wrong.
    assert body["avg_order_value"] == pytest.approx(
        body["total_revenue"] / body["total_orders"], rel=0.01
    )


def test_sales_trend_is_chronological(client: TestClient, has_data: bool) -> None:
    if not has_data:
        pytest.skip("Warehouse is empty; run the ETL first.")
    points = client.get("/api/sales/trend?granularity=month").json()["points"]
    assert points
    periods = [p["period"] for p in points]
    assert periods == sorted(periods), "Chart would render with time running backwards."


def test_country_revenue_is_ordered_descending(client: TestClient, has_data: bool) -> None:
    if not has_data:
        pytest.skip("Warehouse is empty; run the ETL first.")
    rows = client.get("/api/sales/by-country").json()["items"]
    revenues = [r["revenue"] for r in rows]
    assert revenues == sorted(revenues, reverse=True)


def test_segments_flag_whether_analytics_ran(client: TestClient) -> None:
    """Empty must be reported as ``computed: false``, never as zeros.

    Zeros would render as a real chart showing no customers, which is a wrong
    answer; the flag lets the UI say "run the pipeline" instead.
    """
    body = client.get("/api/customers/segments").json()
    assert isinstance(body["computed"], bool)
    if not body["computed"]:
        assert body["segments"] == []


def test_at_risk_respects_threshold_and_sorts_by_money(
    client: TestClient, has_data: bool
) -> None:
    if not has_data:
        pytest.skip("Warehouse is empty; run the ETL first.")
    body = client.get("/api/churn/at-risk?threshold=0.6&limit=25").json()
    assert body["threshold"] == 0.6
    for item in body["items"]:
        assert item["churn_risk_score"] >= 0.6
    losses = [i["expected_revenue_loss"] for i in body["items"]]
    assert losses == sorted(losses, reverse=True), "Worklist is not ranked by value at stake."


def test_at_risk_high_threshold_yields_empty_not_error(client: TestClient) -> None:
    """Nobody above 0.999 is a valid answer, and must not be a 404 or a crash."""
    response = client.get("/api/churn/at-risk?threshold=0.999")
    assert response.status_code == 200
    assert isinstance(response.json()["items"], list)


def test_model_performance_reports_honest_metrics(
    client: TestClient, has_model: bool
) -> None:
    if not has_model:
        pytest.skip("No churn model artifact available.")
    body = client.get("/api/churn/model-performance").json()
    assert 0.5 < body["roc_auc"] < 1.0, "AUC of 1.0 would mean label leakage, not skill."
    matrix = body["confusion_matrix"]
    assert sum(matrix.values()) > 0
    assert len(body["feature_importance"]) == len(body["features"])
    assert body["roc_curve"]


def test_model_performance_503s_when_artifact_missing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing model must degrade to 503, not take the server down.

    This is the state a reviewer is in before the training script has run, so it
    has to be a clean typed response rather than an unhandled exception.
    """
    from app.core.exceptions import ModelNotLoadedError

    def raise_missing(*args: object, **kwargs: object) -> None:
        raise ModelNotLoadedError("Simulated missing artifact.")

    monkeypatch.setattr(registry, "get_metadata", raise_missing)
    response = client.get("/api/churn/model-performance")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "MODEL_NOT_LOADED"


def test_hypothesis_test_returns_effect_size_with_p_value(
    client: TestClient, has_data: bool
) -> None:
    """A p-value without an effect size invites calling trivia significant."""
    if not has_data:
        pytest.skip("Warehouse is empty; run the ETL first.")
    body = client.get("/api/stats/hypothesis-test?type=anova_country&top_n=5").json()
    assert body["p_value"] is None or 0.0 <= body["p_value"] <= 1.0
    assert body["effect_size_name"]
    assert body["interpretation"]
    assert len(body["groups"]) >= 2
