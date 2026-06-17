"""
VoltEdge API — Application factory.

Usage:
    uvicorn voltedge.api.app:create_app --factory --reload --port 8000
    # Or via CLI: voltedge api start
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from voltedge.api.auth import auth_router
from voltedge.api.routes import (
    analytics_router,
    health_router,
    pipeline_router,
    sites_router,
)


def create_app() -> FastAPI:
    from config.settings import settings

    app = FastAPI(
        title="VoltEdge API",
        description="Enterprise energy intelligence — real-time ingestion, ML analytics, ESG reporting.",
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — restrict in production via environment variable
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.is_development else [],
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        allow_credentials=True,
    )

    from voltedge.api.ai_routes import ai_router
    from voltedge.api.ws_routes import ws_router
    from voltedge.api.causal_routes import causal_router

    app.include_router(auth_router)
    app.include_router(health_router)
    app.include_router(pipeline_router)
    app.include_router(sites_router)
    app.include_router(analytics_router)
    app.include_router(ai_router)
    app.include_router(ws_router)
    app.include_router(causal_router)

    return app


# WSGI/ASGI entry point
app = create_app()
