"""Tests for causal AI API routes."""
from __future__ import annotations
import asyncio
import pytest
from fastapi.testclient import TestClient

from voltedge.api.app import create_app
from voltedge.ingestion.simulators import SimulatedSiteConnector
from voltedge.processing.transformer import EnergyTransformer
from voltedge.storage.base import LocalStore

SITE = "CAUSAL-API-TEST"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp   = tmp_path_factory.mktemp("causal_api")
    store = LocalStore(base_path=tmp)

    # Seed processed data for the test site
    conn   = SimulatedSiteConnector(SITE, "factory", hours=400, seed=42)
    result = asyncio.run(conn.fetch())
    df     = EnergyTransformer().transform(result.data)
    store.write(df, site_id=SITE, layer="processed")

    from voltedge.api import causal_routes
    causal_routes._store = store   # inject test store

    app = create_app()
    return TestClient(app)


# ── /causal/graph ─────────────────────────────────────────────────────────────

def test_causal_graph_200(client):
    r = client.post(f"/causal/graph/{SITE}")
    assert r.status_code == 200


def test_causal_graph_has_keys(client):
    r = client.post(f"/causal/graph/{SITE}")
    body = r.json()
    for k in ("site_id", "variables", "n_edges", "n_samples", "edges"):
        assert k in body


def test_causal_graph_site_id(client):
    r = client.post(f"/causal/graph/{SITE}")
    assert r.json()["site_id"] == SITE


def test_causal_graph_404_unknown_site(client):
    r = client.post("/causal/graph/NO-SUCH-SITE")
    assert r.status_code == 404


# ── /causal/root-cause ────────────────────────────────────────────────────────

def test_root_cause_200(client):
    r = client.post(f"/causal/root-cause/{SITE}", json={
        "anomaly_timestamp": "2026-05-10T14:00:00+00:00",
    })
    assert r.status_code == 200


def test_root_cause_has_narrative(client):
    r = client.post(f"/causal/root-cause/{SITE}", json={
        "anomaly_timestamp": "2026-05-10T14:00:00+00:00",
    })
    assert "narrative" in r.json()
    assert len(r.json()["narrative"]) > 0


def test_root_cause_has_recommendations(client):
    r = client.post(f"/causal/root-cause/{SITE}", json={
        "anomaly_timestamp": "2026-05-10T14:00:00+00:00",
    })
    assert "recommendations" in r.json()


def test_root_cause_404_unknown_site(client):
    r = client.post("/causal/root-cause/UNKNOWN", json={
        "anomaly_timestamp": "2026-05-10T14:00:00+00:00",
    })
    assert r.status_code == 404


# ── /causal/counterfactual ────────────────────────────────────────────────────

def test_counterfactual_200(client):
    r = client.post(f"/causal/counterfactual/{SITE}", json={
        "intervention_variable": "is_business_hour",
        "intervention_value":    0.0,
        "description":           "Move to off-peak",
    })
    assert r.status_code == 200


def test_counterfactual_has_delta(client):
    r = client.post(f"/causal/counterfactual/{SITE}", json={
        "intervention_variable": "is_business_hour",
        "intervention_value":    0.0,
    })
    body = r.json()
    for k in ("baseline_value", "counterfactual_value", "delta", "delta_pct"):
        assert k in body


def test_counterfactual_has_annual_impacts(client):
    r = client.post(f"/causal/counterfactual/{SITE}", json={
        "intervention_variable": "is_business_hour",
        "intervention_value":    0.0,
    })
    body = r.json()
    for k in ("annual_kwh_impact", "annual_co2_impact", "annual_cost_impact"):
        assert k in body


def test_counterfactual_has_narrative(client):
    r = client.post(f"/causal/counterfactual/{SITE}", json={
        "intervention_variable": "is_business_hour",
        "intervention_value":    0.0,
    })
    assert len(r.json().get("narrative", "")) > 0


# ── /causal/simulate ──────────────────────────────────────────────────────────

def test_simulate_baseline_200(client):
    r = client.post(f"/causal/simulate/{SITE}", json={
        "scenario_type":    "baseline",
        "simulation_hours": 168,
    })
    assert r.status_code == 200


def test_simulate_solar_pv(client):
    r = client.post(f"/causal/simulate/{SITE}", json={
        "scenario_type":    "solar_pv",
        "scenario_params":  {"capacity_kwp": 50},
        "simulation_hours": 168,
    })
    assert r.status_code == 200
    body = r.json()
    assert "total_kwh" in body
    assert "vs_baseline" in body


def test_simulate_bess(client):
    r = client.post(f"/causal/simulate/{SITE}", json={
        "scenario_type":    "bess",
        "scenario_params":  {"capacity_kwh": 200, "charge_rate_kw": 100},
        "simulation_hours": 168,
    })
    assert r.status_code == 200


def test_simulate_ev_fleet(client):
    r = client.post(f"/causal/simulate/{SITE}", json={
        "scenario_type":    "ev_fleet",
        "scenario_params":  {"vehicles": 5},
        "simulation_hours": 168,
    })
    assert r.status_code == 200


def test_simulate_night_shift(client):
    r = client.post(f"/causal/simulate/{SITE}", json={
        "scenario_type":    "night_shift",
        "simulation_hours": 168,
    })
    assert r.status_code == 200


def test_simulate_unknown_scenario_400(client):
    r = client.post(f"/causal/simulate/{SITE}", json={
        "scenario_type": "magic_beans",
    })
    assert r.status_code == 400


def test_simulate_has_vs_baseline(client):
    r = client.post(f"/causal/simulate/{SITE}", json={
        "scenario_type":    "solar_pv",
        "simulation_hours": 168,
    })
    vs = r.json()["vs_baseline"]
    for k in ("delta_kwh", "delta_co2_kg", "delta_cost", "delta_pct"):
        assert k in vs


# ── /causal/compare ───────────────────────────────────────────────────────────

def test_compare_200(client):
    r = client.post(f"/causal/compare/{SITE}", json={
        "scenarios": [
            {"type": "solar_pv", "params": {"capacity_kwp": 50}},
            {"type": "bess",     "params": {"capacity_kwh": 200, "charge_rate_kw": 100}},
        ],
        "simulation_hours": 168,
    })
    assert r.status_code == 200


def test_compare_has_ranked_by_saving(client):
    r = client.post(f"/causal/compare/{SITE}", json={
        "scenarios": [
            {"type": "solar_pv", "params": {"capacity_kwp": 50}},
        ],
        "simulation_hours": 168,
    })
    body = r.json()
    assert "ranked_by_saving" in body
    assert len(body["ranked_by_saving"]) == 1


def test_compare_has_baseline(client):
    r = client.post(f"/causal/compare/{SITE}", json={
        "scenarios": [{"type": "night_shift", "params": {}}],
        "simulation_hours": 168,
    })
    assert "baseline" in r.json()
    assert r.json()["baseline"]["total_kwh"] > 0
