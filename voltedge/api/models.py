"""
VoltEdge API — Pydantic request/response schemas.
Kept separate from DB models so the API contract is explicit and versioned.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────

class SiteType(str, Enum):
    factory    = "factory"
    office     = "office"
    warehouse  = "warehouse"
    data_center = "data_center"


class LayerType(str, Enum):
    raw       = "raw"
    processed = "processed"


class TariffType(str, Enum):
    UK_HALF_HOURLY = "UK_HALF_HOURLY"
    EU_INDUSTRIAL  = "EU_INDUSTRIAL"
    US_COMMERCIAL  = "US_COMMERCIAL"


# ── Common ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime


class ErrorResponse(BaseModel):
    detail: str
    code: str = "INTERNAL_ERROR"


# ── Pipeline ──────────────────────────────────────────────────────────────────

class SimulateRequest(BaseModel):
    site_id:   str        = Field(..., json_schema_extra={"example": "LONDON-FACTORY-01"})
    site_type: SiteType   = SiteType.factory
    hours:     int        = Field(default=168, ge=1, le=8760)
    seed:      int | None = None


class SimulateResponse(BaseModel):
    site_id:         str
    records_ingested: int
    raw_path:        str
    processed_path:  str
    duration_seconds: float


class PipelineStatusResponse(BaseModel):
    registered_sites: list[str]
    last_run:         datetime | None
    total_runs:       int


# ── Sites ─────────────────────────────────────────────────────────────────────

class SiteListResponse(BaseModel):
    sites: list[str]
    count: int


class SiteDataResponse(BaseModel):
    site_id: str
    layer:   str
    days:    int
    rows:    int
    columns: list[str]
    preview: list[dict[str, Any]]   # First 5 rows as dicts


# ── Forecast ──────────────────────────────────────────────────────────────────

class ForecastPoint(BaseModel):
    timestamp:    datetime
    kwh_forecast: float
    lower_bound:  float
    upper_bound:  float
    horizon_h:    int


class ForecastResponse(BaseModel):
    site_id:        str
    horizon_hours:  int
    model_version:  str
    training_metrics: dict[str, float]
    forecast:       list[ForecastPoint]


# ── Anomaly ───────────────────────────────────────────────────────────────────

class AnomalyPoint(BaseModel):
    timestamp:     datetime
    kwh:           float
    anomaly_score: float
    is_anomaly:    bool
    anomaly_label: str


class AnomalyResponse(BaseModel):
    site_id:       str
    total_readings: int
    anomaly_count: int
    anomaly_rate:  float
    anomalies:     list[AnomalyPoint]


# ── ESG ───────────────────────────────────────────────────────────────────────

class ESGResponse(BaseModel):
    site_id:                      str
    period_start:                 datetime
    period_end:                   datetime
    total_kwh:                    float
    total_co2_kg:                 float
    scope_2_location_tco2e:       float
    scope_2_market_tco2e:         float
    gri_302_1_gj:                 float
    renewable_fraction:           float
    peak_demand_kw:               float
    avg_power_factor:             float | None
    carbon_intensity_kgco2_per_kwh: float


# ── Optimization ──────────────────────────────────────────────────────────────

class RecommendationItem(BaseModel):
    category:                str
    description:             str
    estimated_annual_saving: float
    currency:                str
    effort:                  str
    payback_months:          float | None
    kwh_shift_potential:     float


class OptimizationResponse(BaseModel):
    site_id:               str
    tariff_name:           str
    current_annual_cost:   float
    optimised_annual_cost: float
    currency:              str
    saving_pct:            float
    recommendations:       list[RecommendationItem]
