"""Smoke-test the live API against the loaded warehouse.

Exercises happy paths plus two error paths (bad date range -> 422, unknown
route -> 404 envelope) so the error contract is verified, not assumed.
Run with: python scripts/smoke_api.py
"""

from __future__ import annotations

import json

import _bootstrap  # noqa: F401  (puts backend/ on sys.path, loads .env)
from fastapi.testclient import TestClient

from app.main import app


def show(client: TestClient, method: str, url: str, expect: int, **kwargs) -> None:
    response = client.request(method, url, **kwargs)
    ok = "PASS" if response.status_code == expect else "FAIL"
    print(f"\n[{ok}] {method} {url} -> {response.status_code} (expected {expect})")
    body = response.json()
    text = json.dumps(body, indent=2, default=str)
    print(text if len(text) <= 900 else text[:900] + "\n  ... truncated ...")


def main() -> int:
    with TestClient(app) as client:
        print("=" * 70)
        print("HAPPY PATHS")
        print("=" * 70)
        show(client, "GET", "/api/health", 200)
        show(client, "GET", "/api/kpis/summary", 200)
        show(client, "GET", "/api/sales/trend?granularity=month", 200)
        show(client, "GET", "/api/sales/by-country?limit=5", 200)

        print()
        print("=" * 70)
        print("ERROR PATHS")
        print("=" * 70)
        show(
            client,
            "GET",
            "/api/kpis/summary?start_date=2011-12-01&end_date=2010-01-01",
            422,
        )
        show(client, "GET", "/api/sales/trend?granularity=fortnight", 422)
        show(client, "GET", "/api/sales/by-country?limit=-3", 422)
        show(client, "GET", "/api/does-not-exist", 404)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
