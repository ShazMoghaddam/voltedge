"""ESG router — /esg endpoints."""
from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from voltedge.api.auth import TokenData, require_role
from voltedge.esg.metrics import ESGCalculator
from voltedge.storage.base import LocalStore

router = APIRouter(prefix="/esg", tags=["esg"])
_store = LocalStore()

class ESGResponse(BaseModel):
    site_id: str
    total_kwh: float
    total_co2_kg: float
    scope_2_location_tco2e: float
    scope_2_market_tco2e: float
    gri_302_1_gj: float
    peak_demand_kw: float
    renewable_fraction: float
    avg_power_factor: float | None

@router.get("/{site_id}", response_model=ESGResponse)
def esg_report(
    site_id: str, days: int = 30, country: str = "GB",
    user: TokenData = Depends(require_role("viewer", "operator", "admin")),
):
    df = _store.read(site_id, layer="processed", days=days)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data for site '{site_id}'")
    m = ESGCalculator(country_code=country).compute(df, site_id)
    return ESGResponse(
        site_id=m.site_id, total_kwh=m.total_kwh, total_co2_kg=m.total_co2_kg,
        scope_2_location_tco2e=m.scope_2_location_based_tco2e,
        scope_2_market_tco2e=m.scope_2_market_based_tco2e,
        gri_302_1_gj=m.gri_302_1, peak_demand_kw=m.peak_demand_kw,
        renewable_fraction=m.renewable_fraction, avg_power_factor=m.avg_power_factor,
    )
