"""Pipeline router — /pipeline endpoints."""
from __future__ import annotations
import asyncio
from typing import Literal
from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel
from voltedge.api.auth import TokenData, require_role
from voltedge.core.pipeline import EnergyPipeline
from voltedge.ingestion.simulators import SimulatedSiteConnector
from voltedge.storage.base import LocalStore

router = APIRouter(prefix="/pipeline", tags=["pipeline"])

class SimulateRequest(BaseModel):
    site_id: str = "SITE-DEMO-01"
    site_type: Literal["factory", "office", "warehouse", "data_center"] = "factory"
    hours: int = 168
    seed: int = 42

class PipelineRunResponse(BaseModel):
    status: str
    sites_processed: int
    total_records: int
    total_errors: int
    duration_seconds: float

def _run_simulation(req: SimulateRequest) -> PipelineRunResponse:
    store = LocalStore()
    pipeline = EnergyPipeline(store=store)
    pipeline.register_connector(
        req.site_id,
        SimulatedSiteConnector(req.site_id, req.site_type, req.hours, req.seed),
    )
    summary = asyncio.run(pipeline.run_all())
    return PipelineRunResponse(
        status="ok" if summary.success else "partial",
        sites_processed=summary.sites_processed,
        total_records=summary.total_records,
        total_errors=summary.total_errors,
        duration_seconds=round(summary.duration_seconds, 3),
    )

@router.post("/simulate", response_model=PipelineRunResponse)
def simulate(req: SimulateRequest, user: TokenData = Depends(require_role("operator", "admin"))):
    return _run_simulation(req)

@router.post("/simulate/async")
def simulate_async(req: SimulateRequest, bg: BackgroundTasks,
                   user: TokenData = Depends(require_role("operator", "admin"))):
    bg.add_task(_run_simulation, req)
    return {"status": "queued", "site_id": req.site_id}
