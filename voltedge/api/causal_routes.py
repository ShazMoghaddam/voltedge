"""
VoltEdge API — Causal AI Routes (v8.0)

Exposes causal graph, root-cause, counterfactual, and digital twin
endpoints for the enterprise dashboard and API consumers.

Endpoints:
  POST /causal/graph/{site_id}           Build causal graph from stored data
  POST /causal/root-cause/{site_id}      Root-cause analysis for an anomaly
  POST /causal/counterfactual/{site_id}  What-if intervention analysis
  POST /causal/simulate/{site_id}        Digital twin scenario simulation
  POST /causal/compare/{site_id}         Compare multiple scenarios
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from voltedge.storage.base import LocalStore
from voltedge.utils.logger import get_logger

log = get_logger(__name__)
causal_router = APIRouter(prefix="/causal", tags=["Causal AI"])

_store = LocalStore()


# ── Request models ────────────────────────────────────────────────────────────

class RootCauseRequest(BaseModel):
    anomaly_timestamp: str = Field(...,
        json_schema_extra={"example": "2026-05-15T14:00:00+00:00"})
    anomaly_variable:  str = Field(default="kwh")
    window_hours:      int = Field(default=168, ge=24, le=720)


class CounterfactualRequest(BaseModel):
    intervention_variable: str   = Field(...,
        json_schema_extra={"example": "is_business_hour"})
    intervention_value:    float = Field(...,
        json_schema_extra={"example": 0.0})
    description:           str   = Field(default="")
    target_variable:       str   = Field(default="kwh")


class SimulateRequest(BaseModel):
    scenario_type:    str  = Field(...,
        json_schema_extra={"example": "solar_pv"})
    scenario_params:  dict = Field(default_factory=dict,
        json_schema_extra={"example": {"capacity_kwp": 100}})
    simulation_hours: int  = Field(default=8760, ge=24, le=8760)


class CompareRequest(BaseModel):
    scenarios: list[dict] = Field(...,
        json_schema_extra={"example": [
            {"type": "solar_pv", "params": {"capacity_kwp": 100}},
            {"type": "bess",     "params": {"capacity_kwh": 500, "charge_rate_kw": 200}},
        ]})
    simulation_hours: int = Field(default=8760, ge=24, le=8760)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_df(site_id: str, days: int = 90):
    import pandas as pd
    df = _store.read(site_id, layer="processed", days=days)
    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No processed data found for site '{site_id}'. "
                   "Run the ingestion pipeline first."
        )
    return df


def _build_graph(df, site_id: str):
    from voltedge.causal.graph import CausalGraphBuilder
    return CausalGraphBuilder(alpha=0.05, min_samples=48).build(df, site_id)


# ── Causal graph ──────────────────────────────────────────────────────────────

@causal_router.post("/graph/{site_id}")
def build_causal_graph(site_id: str, days: int = 90) -> dict:
    """Learn a causal DAG from the site's processed energy data."""
    df    = _load_df(site_id, days)
    graph = _build_graph(df, site_id)
    return graph.summary()


# ── Root-cause analysis ───────────────────────────────────────────────────────

@causal_router.post("/root-cause/{site_id}")
def root_cause_analysis(site_id: str, body: RootCauseRequest) -> dict:
    """
    Identify the causal drivers of an anomaly at a given timestamp.

    Returns ranked causal factors, narrative explanation, and recommendations.
    """
    from voltedge.causal.root_cause import RootCauseAnalyser

    df    = _load_df(site_id, days=max(30, body.window_hours // 24 + 7))
    graph = _build_graph(df, site_id)
    analyser = RootCauseAnalyser(graph)
    report = analyser.analyse(
        df,
        anomaly_timestamp=body.anomaly_timestamp,
        anomaly_variable=body.anomaly_variable,
        window_hours=body.window_hours,
    )
    return report.to_dict()


# ── Counterfactual ────────────────────────────────────────────────────────────

@causal_router.post("/counterfactual/{site_id}")
def counterfactual(site_id: str, body: CounterfactualRequest) -> dict:
    """
    Compute do(X=x) → target: what would the outcome be if we
    intervened to set intervention_variable to intervention_value?
    """
    from voltedge.causal.counterfactual import CounterfactualEngine, Intervention

    df     = _load_df(site_id, days=90)
    graph  = _build_graph(df, site_id)
    engine = CounterfactualEngine(graph)
    iv     = Intervention(
        variable=body.intervention_variable,
        new_value=body.intervention_value,
        description=body.description or f"Set {body.intervention_variable}={body.intervention_value}",
    )
    result = engine.compute(df, iv, target=body.target_variable)
    return result.to_dict()


# ── Digital twin simulation ───────────────────────────────────────────────────

@causal_router.post("/simulate/{site_id}")
def simulate_scenario(site_id: str, body: SimulateRequest) -> dict:
    """
    Run a digital twin scenario simulation.
    Returns total kWh, CO₂, cost, and peak demand vs baseline.
    """
    from voltedge.causal.digital_twin import EnergyDigitalTwin

    df    = _load_df(site_id, days=90)
    graph = _build_graph(df, site_id)
    twin  = EnergyDigitalTwin(graph=graph, df=df, n_simulations=50)

    SCENARIO_MAP = {
        "baseline":    lambda p: twin.run_baseline(hours=body.simulation_hours),
        "solar_pv":    lambda p: twin.run_scenario(
            EnergyDigitalTwin.solar_scenario(p.get("capacity_kwp", 100)),
            hours=body.simulation_hours),
        "bess":        lambda p: twin.run_scenario(
            EnergyDigitalTwin.bess_scenario(
                p.get("capacity_kwh", 500), p.get("charge_rate_kw", 200)),
            hours=body.simulation_hours),
        "ev_fleet":    lambda p: twin.run_scenario(
            EnergyDigitalTwin.ev_fleet_scenario(p.get("vehicles", 10)),
            hours=body.simulation_hours),
        "night_shift": lambda p: twin.run_scenario(
            EnergyDigitalTwin.night_shift_scenario(),
            hours=body.simulation_hours),
    }

    fn = SCENARIO_MAP.get(body.scenario_type)
    if not fn:
        raise HTTPException(status_code=400,
                            detail=f"Unknown scenario: '{body.scenario_type}'")

    result   = fn(body.scenario_params)
    baseline = twin.run_baseline(hours=body.simulation_hours)
    d        = result.to_dict()
    d["vs_baseline"] = {
        "delta_kwh":    round(result.total_kwh   - baseline.total_kwh,   1),
        "delta_co2_kg": round(result.total_co2_kg - baseline.total_co2_kg, 1),
        "delta_cost":   round(result.total_cost   - baseline.total_cost,  2),
        "delta_pct":    round(
            (result.total_kwh - baseline.total_kwh) /
            (abs(baseline.total_kwh) + 1e-8) * 100, 1),
    }
    return d


# ── Multi-scenario comparison ─────────────────────────────────────────────────

@causal_router.post("/compare/{site_id}")
def compare_scenarios(site_id: str, body: CompareRequest) -> dict:
    """
    Compare multiple what-if scenarios against the current baseline.
    Returns ranked results showing which scenario saves most energy/cost.
    """
    from voltedge.causal.digital_twin import EnergyDigitalTwin, Scenario

    df    = _load_df(site_id, days=90)
    graph = _build_graph(df, site_id)
    twin  = EnergyDigitalTwin(graph=graph, df=df, n_simulations=30)

    def _build(s: dict) -> "Scenario":  # type: ignore
        stype  = s.get("type", "baseline")
        params = s.get("params", {})
        template_map = {
            "solar_pv":    EnergyDigitalTwin.solar_scenario,
            "bess":        EnergyDigitalTwin.bess_scenario,
            "ev_fleet":    EnergyDigitalTwin.ev_fleet_scenario,
            "night_shift": lambda **_: EnergyDigitalTwin.night_shift_scenario(),
        }
        fn = template_map.get(stype)
        if fn:
            try:
                return fn(**params)
            except TypeError:
                return fn()
        return Scenario(name=stype, description=stype)

    scenarios  = [_build(s) for s in body.scenarios]
    comparison = twin.compare_scenarios(scenarios, hours=body.simulation_hours)
    return comparison.to_dict()
