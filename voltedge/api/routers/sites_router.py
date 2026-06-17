"""Sites router — /sites endpoints."""
from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from voltedge.api.auth import TokenData, require_role
from voltedge.storage.base import LocalStore

router = APIRouter(prefix="/sites", tags=["sites"])
_store = LocalStore()

class SiteListResponse(BaseModel):
    sites: list[str]
    count: int

class SiteSummary(BaseModel):
    site_id: str
    total_kwh: float
    record_count: int
    latest_timestamp: str | None

@router.get("", response_model=SiteListResponse)
def list_sites(user: TokenData = Depends(require_role("viewer", "operator", "admin"))):
    sites = _store.list_sites()
    return SiteListResponse(sites=sites, count=len(sites))

@router.get("/{site_id}/summary", response_model=SiteSummary)
def site_summary(
    site_id: str, days: int = 7,
    user: TokenData = Depends(require_role("viewer", "operator", "admin")),
):
    df = _store.read(site_id, layer="processed", days=days)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data for site '{site_id}'")
    latest = str(df["timestamp"].max()) if "timestamp" in df.columns else None
    return SiteSummary(
        site_id=site_id,
        total_kwh=round(float(df["kwh"].sum()), 2),
        record_count=len(df),
        latest_timestamp=latest,
    )
