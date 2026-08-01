"""Aggregate router.

Each resource keeps its own module; this is the single place they are composed,
so ``main.py`` mounts one router and the prefix lives in settings rather than
being repeated per module.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api import churn, customers, health, kpis, sales, stats

api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(kpis.router)
api_router.include_router(sales.router)
api_router.include_router(customers.router)
api_router.include_router(churn.router)
api_router.include_router(stats.router)



__all__ = ["api_router"]
