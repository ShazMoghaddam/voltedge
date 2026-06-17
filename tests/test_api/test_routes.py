"""
VoltEdge API — route tests.
Uses FastAPI TestClient so no server process is needed.
Store is overridden with a tmp LocalStore per test.
"""

from __future__ import annotations

import asyncio
import pytest
from fastapi.testclient import TestClient

from voltedge.api.app import create_app
from voltedge.api.routes import get_store
from voltedge.ingestion.simulators import SimulatedSiteConnector
from voltedge.processing.transformer import EnergyTransformer
from voltedge.storage.base import LocalStore


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_store(tmp_path):
    return LocalStore(base_path=tmp_path)


@pytest.fixture
def client(tmp_store):
    app = create_app()
    app.dependency_overrides[get_store] = lambda: tmp_store
    return TestClient(app)


@pytest.fixture
def seeded_client(tmp_store):
    """Client backed by a store pre-loaded with 200h of factory data."""
    connector = SimulatedSiteConnector("TEST-SITE", "factory", hours=200, seed=42)
    result = asyncio.run(connector.fetch())
    df = EnergyTransformer().transform(result.data)
    tmp_store.write(df, site_id="TEST-SITE", layer="raw")
    tmp_store.write(df, site_id="TEST-SITE", layer="processed")

    app = create_app()
    app.dependency_overrides[get_store] = lambda: tmp_store
    return TestClient(app)


# ── Health ────────────────────────────────────────────────────────────────────

def test_health_returns_200(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "version" in r.json()
    assert "timestamp" in r.json()


def test_ready_returns_200(client):
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


def test_health_has_correct_content_type(client):
    r = client.get("/health")
    assert "application/json" in r.headers["content-type"]


# ── Pipeline ──────────────────────────────────────────────────────────────────

def test_simulate_creates_data(client):
    r = client.post("/pipeline/simulate", json={
        "site_id": "API-SITE-01",
        "site_type": "factory",
        "hours": 48,
        "seed": 42,
    })
    assert r.status_code == 201
    body = r.json()
    assert body["site_id"] == "API-SITE-01"
    assert body["records_ingested"] == 48
    assert body["duration_seconds"] > 0


def test_simulate_office_site(client):
    r = client.post("/pipeline/simulate", json={
        "site_id": "OFFICE-01", "site_type": "office", "hours": 24
    })
    assert r.status_code == 201
    assert r.json()["records_ingested"] == 24


def test_simulate_all_site_types(client):
    for stype in ["factory", "office", "warehouse", "data_center"]:
        r = client.post("/pipeline/simulate", json={
            "site_id": f"SITE-{stype}", "site_type": stype, "hours": 12
        })
        assert r.status_code == 201, f"Failed for {stype}: {r.text}"


def test_simulate_rejects_zero_hours(client):
    r = client.post("/pipeline/simulate", json={
        "site_id": "BAD-SITE", "site_type": "factory", "hours": 0
    })
    assert r.status_code == 422


def test_simulate_rejects_over_8760_hours(client):
    r = client.post("/pipeline/simulate", json={
        "site_id": "BAD-SITE", "site_type": "factory", "hours": 9000
    })
    assert r.status_code == 422


# ── Sites ─────────────────────────────────────────────────────────────────────

def test_list_sites_empty(client):
    r = client.get("/sites/")
    assert r.status_code == 200
    assert r.json()["sites"] == []
    assert r.json()["count"] == 0


def test_list_sites_after_simulate(client):
    client.post("/pipeline/simulate", json={"site_id": "A", "hours": 12})
    client.post("/pipeline/simulate", json={"site_id": "B", "hours": 12})
    r = client.get("/sites/")
    assert r.status_code == 200
    assert set(r.json()["sites"]) == {"A", "B"}
    assert r.json()["count"] == 2


def test_get_site_data_404_when_missing(client):
    r = client.get("/sites/GHOST-SITE/data")
    assert r.status_code == 404


def test_get_site_data_returns_preview(seeded_client):
    r = seeded_client.get("/sites/TEST-SITE/data?layer=processed&days=30")
    assert r.status_code == 200
    body = r.json()
    assert body["site_id"] == "TEST-SITE"
    assert body["rows"] > 0
    assert len(body["preview"]) <= 5
    assert "kwh" in body["columns"]


def test_get_site_data_raw_layer(seeded_client):
    r = seeded_client.get("/sites/TEST-SITE/data?layer=raw&days=30")
    assert r.status_code == 200
    assert r.json()["rows"] > 0


def test_get_site_data_rejects_invalid_layer(seeded_client):
    r = seeded_client.get("/sites/TEST-SITE/data?layer=invalid")
    assert r.status_code == 422


# ── Analytics — Forecast ──────────────────────────────────────────────────────

def test_forecast_returns_correct_horizon(seeded_client):
    r = seeded_client.post("/analytics/TEST-SITE/forecast?horizon_hours=12")
    assert r.status_code == 200
    body = r.json()
    assert body["site_id"] == "TEST-SITE"
    assert len(body["forecast"]) == 12
    assert "training_metrics" in body
    assert "mae" in body["training_metrics"]


def test_forecast_24h_default(seeded_client):
    r = seeded_client.post("/analytics/TEST-SITE/forecast")
    assert r.status_code == 200
    assert len(r.json()["forecast"]) == 24


def test_forecast_bounds_ordering(seeded_client):
    r = seeded_client.post("/analytics/TEST-SITE/forecast")
    for point in r.json()["forecast"]:
        assert point["lower_bound"] <= point["kwh_forecast"] <= point["upper_bound"]


def test_forecast_404_missing_site(client):
    r = client.post("/analytics/MISSING/forecast")
    assert r.status_code == 404


def test_forecast_rejects_horizon_over_168(seeded_client):
    r = seeded_client.post("/analytics/TEST-SITE/forecast?horizon_hours=200")
    assert r.status_code == 422


# ── Analytics — Anomaly ───────────────────────────────────────────────────────

def test_anomaly_returns_response(seeded_client):
    r = seeded_client.post("/analytics/TEST-SITE/anomaly")
    assert r.status_code == 200
    body = r.json()
    assert body["site_id"] == "TEST-SITE"
    assert "anomaly_count" in body
    assert "anomaly_rate" in body
    assert 0.0 <= body["anomaly_rate"] <= 1.0


def test_anomaly_404_missing_site(client):
    r = client.post("/analytics/MISSING/anomaly")
    assert r.status_code == 404


def test_anomaly_flagged_have_scores(seeded_client):
    r = seeded_client.post("/analytics/TEST-SITE/anomaly")
    for point in r.json()["anomalies"]:
        assert 0.0 <= point["anomaly_score"] <= 1.0
        assert point["is_anomaly"] is True


# ── Analytics — ESG ───────────────────────────────────────────────────────────

def test_esg_returns_metrics(seeded_client):
    r = seeded_client.post("/analytics/TEST-SITE/esg?country=GB")
    assert r.status_code == 200
    body = r.json()
    assert body["total_kwh"] > 0
    assert body["scope_2_location_tco2e"] > 0
    assert "gri_302_1_gj" in body


def test_esg_404_missing_site(client):
    r = client.post("/analytics/MISSING/esg")
    assert r.status_code == 404


def test_esg_different_countries(seeded_client):
    for country in ["GB", "DE", "FR", "US"]:
        r = seeded_client.post(f"/analytics/TEST-SITE/esg?country={country}")
        assert r.status_code == 200, f"Failed for {country}"


# ── Analytics — Optimization ──────────────────────────────────────────────────

def test_optimize_returns_recommendations(seeded_client):
    r = seeded_client.post("/analytics/TEST-SITE/optimize")
    assert r.status_code == 200
    body = r.json()
    assert body["current_annual_cost"] > 0
    assert len(body["recommendations"]) > 0


def test_optimize_savings_are_positive(seeded_client):
    r = seeded_client.post("/analytics/TEST-SITE/optimize")
    for rec in r.json()["recommendations"]:
        assert rec["estimated_annual_saving"] >= 0


def test_optimize_eu_tariff(seeded_client):
    r = seeded_client.post("/analytics/TEST-SITE/optimize?tariff=EU_INDUSTRIAL")
    assert r.status_code == 200
    assert r.json()["currency"] == "EUR"


def test_optimize_404_missing_site(client):
    r = client.post("/analytics/MISSING/optimize")
    assert r.status_code == 404


# ── OpenAPI schema ────────────────────────────────────────────────────────────

def test_openapi_schema_is_accessible(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    schema = r.json()
    assert "paths" in schema
    assert "/health" in schema["paths"]
    assert "/pipeline/simulate" in schema["paths"]


def test_docs_endpoint_accessible(client):
    r = client.get("/docs")
    assert r.status_code == 200
