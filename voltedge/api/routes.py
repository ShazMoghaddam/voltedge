"""
VoltEdge API — Route handlers.

Organised into four routers:
  /health         — liveness / readiness
  /pipeline       — data ingestion (simulate, status)
  /sites          — data access (list, read)
  /analytics      — ML + ESG endpoints (forecast, anomaly, esg, optimize)

All routes are stateless: store and pipeline objects are injected
via FastAPI dependency injection, making them trivially swappable in tests.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from voltedge.api.models import (
    AnomalyPoint, AnomalyResponse,
    ESGResponse,
    ForecastPoint, ForecastResponse,
    HealthResponse,
    OptimizationResponse, RecommendationItem,
    SimulateRequest, SimulateResponse,
    SiteDataResponse, SiteListResponse,
    TariffType,
)
from voltedge.core.pipeline import EnergyPipeline
from voltedge.esg.metrics import ESGCalculator
from voltedge.ingestion.simulators import SimulatedSiteConnector
from voltedge.models.anomaly import AnomalyDetector
from voltedge.models.forecasting import DemandForecaster
from voltedge.models.optimization import CostOptimizer
from voltedge.processing.transformer import EnergyTransformer
from voltedge.storage.base import BaseStore, LocalStore
from voltedge.utils.logger import get_logger

log = get_logger(__name__)

# ── Dependency ────────────────────────────────────────────────────────────────

_default_store: BaseStore = LocalStore()


def get_store() -> BaseStore:
    """FastAPI dependency — override in tests via app.dependency_overrides."""
    return _default_store


StoreDep = Annotated[BaseStore, Depends(get_store)]

# ── Routers ───────────────────────────────────────────────────────────────────

health_router    = APIRouter(tags=["Health"])
pipeline_router  = APIRouter(prefix="/pipeline",  tags=["Pipeline"])
sites_router     = APIRouter(prefix="/sites",     tags=["Sites"])
analytics_router = APIRouter(prefix="/analytics", tags=["Analytics"])


# ── /health ───────────────────────────────────────────────────────────────────

@health_router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness probe. Returns 200 when the API is running."""
    from config.settings import settings
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        timestamp=datetime.now(timezone.utc),
    )


@health_router.get("/ready", response_model=HealthResponse)
async def ready(store: StoreDep) -> HealthResponse:
    """Readiness probe. Checks storage connectivity."""
    from config.settings import settings
    try:
        store.list_sites()
        return HealthResponse(
            status="ready",
            version=settings.app_version,
            timestamp=datetime.now(timezone.utc),
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Storage unavailable: {exc}")


# ── /pipeline ─────────────────────────────────────────────────────────────────

@pipeline_router.post("/simulate", response_model=SimulateResponse, status_code=201)
async def simulate(body: SimulateRequest, store: StoreDep) -> SimulateResponse:
    """
    Generate and ingest simulated energy data for a site.
    Creates both raw and processed Parquet files in storage.
    """
    import time
    start = time.perf_counter()

    connector = SimulatedSiteConnector(
        site_id=body.site_id,
        site_type=body.site_type.value,
        hours=body.hours,
        seed=body.seed or abs(hash(body.site_id)) % 9999,
    )
    pipeline = EnergyPipeline(store=store)
    pipeline.register_connector(body.site_id, connector)
    summary = await pipeline.run_all()

    if not summary.success and summary.sites_processed == 0:
        raise HTTPException(status_code=500, detail="Pipeline run produced no data.")

    site_result = summary.site_results.get(body.site_id, {})
    elapsed = time.perf_counter() - start

    return SimulateResponse(
        site_id=body.site_id,
        records_ingested=site_result.get("records_valid", 0),
        raw_path=site_result.get("raw_path", ""),
        processed_path=site_result.get("processed_path", ""),
        duration_seconds=round(elapsed, 3),
    )


# ── /sites ────────────────────────────────────────────────────────────────────

@sites_router.get("/", response_model=SiteListResponse)
def list_sites(store: StoreDep) -> SiteListResponse:
    """Return all site IDs that have data in storage."""
    sites = store.list_sites()
    return SiteListResponse(sites=sites, count=len(sites))


@sites_router.get("/{site_id}/data", response_model=SiteDataResponse)
def get_site_data(
    site_id: str,
    store: StoreDep,
    layer: str = Query(default="processed", pattern="^(raw|processed)$"),
    days:  int = Query(default=7, ge=1, le=365),
) -> SiteDataResponse:
    """Read recent data for a site. Returns metadata + first 5 rows as preview."""
    df = store.read(site_id, layer=layer, days=days)
    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for site '{site_id}' in layer '{layer}' over last {days} days."
        )

    preview_df = df.head(5).copy()
    for col in preview_df.select_dtypes(include=["datetime64[ns, UTC]", "datetime64[ns]"]).columns:
        preview_df[col] = preview_df[col].astype(str)

    return SiteDataResponse(
        site_id=site_id,
        layer=layer,
        days=days,
        rows=len(df),
        columns=list(df.columns),
        preview=preview_df.to_dict(orient="records"),
    )


# ── /analytics ────────────────────────────────────────────────────────────────

@analytics_router.post("/{site_id}/forecast", response_model=ForecastResponse)
def get_forecast(
    site_id: str,
    store: StoreDep,
    horizon_hours: int = Query(default=24, ge=1, le=168),
    days: int = Query(default=90, ge=30, le=365),
) -> ForecastResponse:
    """Train DemandForecaster on recent history and return 24–168h forecast."""
    df = store.read(site_id, layer="processed", days=days)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No processed data for '{site_id}'.")
    if len(df) < 48:
        raise HTTPException(
            status_code=422,
            detail=f"Insufficient data: need ≥48 rows, got {len(df)}."
        )

    forecaster = DemandForecaster(site_id=site_id, horizon_hours=horizon_hours)
    metrics = forecaster.train(df)
    forecast_df = forecaster.predict(df)

    points = [
        ForecastPoint(
            timestamp=row["timestamp"],
            kwh_forecast=row["kwh_forecast"],
            lower_bound=row["lower_bound"],
            upper_bound=row["upper_bound"],
            horizon_h=int(row["horizon_h"]),
        )
        for _, row in forecast_df.iterrows()
    ]

    return ForecastResponse(
        site_id=site_id,
        horizon_hours=horizon_hours,
        model_version=forecaster.model_version,
        training_metrics=metrics,
        forecast=points,
    )


@analytics_router.post("/{site_id}/anomaly", response_model=AnomalyResponse)
def get_anomaly(
    site_id: str,
    store: StoreDep,
    days: int = Query(default=30, ge=7, le=365),
    contamination: float = Query(default=0.05, ge=0.01, le=0.2),
) -> AnomalyResponse:
    """Run anomaly detection over recent data and return flagged readings."""
    df = store.read(site_id, layer="processed", days=days)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No processed data for '{site_id}'.")

    detector = AnomalyDetector(site_id=site_id, contamination=contamination)
    detector.train(df)
    result_df = detector.predict(df)

    flagged = result_df[result_df["is_anomaly"]].copy()
    anomaly_points = [
        AnomalyPoint(
            timestamp=row["timestamp"],
            kwh=row["kwh"],
            anomaly_score=row["anomaly_score"],
            is_anomaly=True,
            anomaly_label=row.get("anomaly_label", "anomaly"),
        )
        for _, row in flagged.iterrows()
    ]

    return AnomalyResponse(
        site_id=site_id,
        total_readings=len(result_df),
        anomaly_count=len(flagged),
        anomaly_rate=round(len(flagged) / max(len(result_df), 1), 4),
        anomalies=anomaly_points,
    )


@analytics_router.post("/{site_id}/esg", response_model=ESGResponse)
def get_esg(
    site_id: str,
    store: StoreDep,
    days: int = Query(default=30, ge=1, le=365),
    country: str = Query(default="GB", min_length=2, max_length=2),
) -> ESGResponse:
    """Compute Scope 2 emissions and GRI 302-1 metrics for a site."""
    df = store.read(site_id, layer="processed", days=days)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No processed data for '{site_id}'.")

    calc = ESGCalculator(country_code=country)
    m = calc.compute(df, site_id)

    return ESGResponse(
        site_id=m.site_id,
        period_start=m.period_start,
        period_end=m.period_end,
        total_kwh=m.total_kwh,
        total_co2_kg=m.total_co2_kg,
        scope_2_location_tco2e=m.scope_2_location_based_tco2e,
        scope_2_market_tco2e=m.scope_2_market_based_tco2e,
        gri_302_1_gj=m.gri_302_1,
        renewable_fraction=m.renewable_fraction,
        peak_demand_kw=m.peak_demand_kw,
        avg_power_factor=m.avg_power_factor,
        carbon_intensity_kgco2_per_kwh=m.carbon_intensity_kgco2_per_kwh,
    )


@analytics_router.post("/{site_id}/optimize", response_model=OptimizationResponse)
def get_optimization(
    site_id: str,
    store: StoreDep,
    tariff: TariffType = Query(default=TariffType.UK_HALF_HOURLY),
    days: int = Query(default=30, ge=7, le=365),
) -> OptimizationResponse:
    """Run cost optimization analysis and return prioritised recommendations."""
    df = store.read(site_id, layer="processed", days=days)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No processed data for '{site_id}'.")

    optimizer = CostOptimizer(site_id=site_id, tariff=tariff.value)
    report = optimizer.analyse(df)

    recs = [
        RecommendationItem(
            category=r.category,
            description=r.description,
            estimated_annual_saving=r.estimated_annual_saving,
            currency=r.currency,
            effort=r.effort,
            payback_months=r.payback_months,
            kwh_shift_potential=r.kwh_shift_potential,
        )
        for r in report.recommendations
    ]

    return OptimizationResponse(
        site_id=report.site_id,
        tariff_name=report.tariff_name,
        current_annual_cost=report.current_annual_cost,
        optimised_annual_cost=report.optimised_annual_cost,
        currency=report.currency,
        saving_pct=round(report.saving_pct, 2),
        recommendations=recs,
    )
