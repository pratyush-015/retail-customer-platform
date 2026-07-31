"""Smoke-test the customer endpoints against the loaded warehouse.

Exercises happy paths *and* error paths in-process via TestClient, so it needs no
running server. Kept as a script rather than a one-liner because the multi-line
quoting required to do this inline is not portable across cmd and PowerShell.

Run:  python scripts/smoke_customers.py
"""

from __future__ import annotations

import sys

import _bootstrap  # noqa: F401  (puts backend/ on sys.path)
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {label}" + (f" — {detail}" if detail else ""))
    if not condition:
        failures.append(label)


def main() -> int:
    print("=" * 78)
    print("CUSTOMER ENDPOINT SMOKE TEST")
    print("=" * 78)

    # ---------------------------------------------------------------- segments
    print("\nGET /api/customers/segments")
    r = client.get("/api/customers/segments")
    check("200 OK", r.status_code == 200, f"status={r.status_code}")
    if r.status_code != 200:
        print(f"    body: {r.text[:400]}")
        return 1

    seg = r.json()
    check("computed flag is true", seg["computed"] is True)
    check("segments returned", len(seg["segments"]) > 0, f"{len(seg['segments'])} segments")
    print(
        f"    total_customers={seg['total_customers']:,}  "
        f"attributed_revenue=£{seg['attributed_revenue']:,.2f}"
    )
    print(f"    {'segment':<20}{'n':>7}{'cust%':>8}{'rev%':>9}{'rev/cust':>12}  health")
    for s in seg["segments"]:
        print(
            f"    {s['segment']:<20}{s['customer_count']:>7,}{s['customer_pct']:>8.1f}"
            f"{s['revenue_pct']:>9.1f}{s['avg_revenue_per_customer']:>12,.0f}  {s['health']}"
        )

    pct_sum = sum(s["revenue_pct"] for s in seg["segments"])
    check("revenue_pct sums to ~100", abs(pct_sum - 100) < 0.5, f"{pct_sum:.2f}")
    cust_sum = sum(s["customer_count"] for s in seg["segments"])
    check(
        "customer counts sum to total",
        cust_sum == seg["total_customers"],
        f"{cust_sum} vs {seg['total_customers']}",
    )

    # Champions should out-earn Lost per customer; if not, the segment mapping
    # is inverted somewhere and every downstream chart is misleading.
    by_name = {s["segment"]: s for s in seg["segments"]}
    if "Champions" in by_name and "Lost" in by_name:
        check(
            "Champions out-earn Lost per customer",
            by_name["Champions"]["avg_revenue_per_customer"]
            > by_name["Lost"]["avg_revenue_per_customer"],
            f"{by_name['Champions']['avg_revenue_per_customer']:,.0f} vs "
            f"{by_name['Lost']['avg_revenue_per_customer']:,.0f}",
        )
        check(
            "Champions more recent than Lost",
            by_name["Champions"]["avg_recency_days"] < by_name["Lost"]["avg_recency_days"],
            f"{by_name['Champions']['avg_recency_days']:.0f}d vs "
            f"{by_name['Lost']['avg_recency_days']:.0f}d",
        )

    # ------------------------------------------------- profile (happy path)
    # Pick a real customer from the at-risk-agnostic path: reuse a segment row's
    # existence by querying a known-good id from the DB via the API itself is
    # circular, so take the first id that resolves from a small probe range.
    print("\nGET /api/customers/{customer_id}")
    probe_id = None
    for candidate in (17850, 14606, 12748, 13089, 14911):
        if client.get(f"/api/customers/{candidate}").status_code == 200:
            probe_id = candidate
            break

    if probe_id is None:
        check("found a live customer id to profile", False, "none of the probe ids resolved")
    else:
        r = client.get(f"/api/customers/{probe_id}")
        check(f"200 OK for customer {probe_id}", r.status_code == 200)
        p = r.json()
        print(
            f"    id={p['customer_id']}  country={p['country']}  "
            f"segment={p['rfm']['segment']}  rfm_score={p['rfm']['rfm_score']}"
        )
        print(
            f"    orders={p['total_orders']}  revenue=£{p['total_revenue']:,.2f}  "
            f"AOV=£{(p['avg_order_value'] or 0):,.2f}  percentile={p['revenue_percentile']}"
        )
        print(
            f"    churn={p['churn_risk_score']} ({p['churn_risk_band']})  "
            f"is_churned={p['is_churned']}  recency={p['rfm']['recency_days']}d"
        )
        check("rfm_computed is true", p["rfm_computed"] is True)
        check("order history present", len(p["order_history"]) > 0, f"{len(p['order_history'])} invoices")
        check("top products present", len(p["top_products"]) > 0, f"{len(p['top_products'])} products")

        # History must read oldest-first for the timeline chart.
        dates = [e["order_date"] for e in p["order_history"]]
        check("history is chronological", dates == sorted(dates))

        # Frequency should agree with the distinct invoices we can see.
        if p["rfm"]["frequency"] is not None and len(p["order_history"]) < 100:
            check(
                "frequency matches invoice count",
                p["rfm"]["frequency"] == len(p["order_history"]),
                f"rfm.frequency={p['rfm']['frequency']} vs history={len(p['order_history'])}",
            )

        print(f"    history_limit honoured:")
        r_lim = client.get(f"/api/customers/{probe_id}?history_limit=3")
        check(
            "history_limit=3 returns <=3",
            len(r_lim.json()["order_history"]) <= 3,
            f"{len(r_lim.json()['order_history'])} rows",
        )

    # ------------------------------------------------------- profile (errors)
    print("\nError paths")
    r = client.get("/api/customers/99999999")
    check("unknown id -> 404", r.status_code == 404, f"status={r.status_code}")
    body = r.json()
    check(
        "404 uses the standard error envelope",
        body.get("error", {}).get("code") == "CUSTOMER_NOT_FOUND",
        str(body)[:160],
    )
    check("404 carries status in body", body.get("error", {}).get("status") == 404)

    r = client.get("/api/customers/0")
    check("id=0 rejected as 422", r.status_code == 422, f"status={r.status_code}")

    r = client.get("/api/customers/-5")
    check("negative id rejected as 422", r.status_code == 422, f"status={r.status_code}")

    r = client.get("/api/customers/abc")
    check("non-numeric id rejected as 422", r.status_code == 422, f"status={r.status_code}")

    if probe_id:
        r = client.get(f"/api/customers/{probe_id}?history_limit=0")
        check("history_limit=0 rejected as 422", r.status_code == 422, f"status={r.status_code}")
        r = client.get(f"/api/customers/{probe_id}?history_limit=9999")
        check("history_limit over cap rejected as 422", r.status_code == 422, f"status={r.status_code}")

    # Route-ordering regression: /segments must not be captured by /{customer_id}.
    check(
        "/segments not shadowed by /{customer_id}",
        client.get("/api/customers/segments").status_code == 200,
    )

    print("\n" + "=" * 78)
    if failures:
        print(f"FAILED ({len(failures)}): " + "; ".join(failures))
        return 1
    print("All customer endpoint checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
