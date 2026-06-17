"""Tests for the SimulatedSiteConnector."""

from __future__ import annotations

import asyncio

import pytest

from voltedge.ingestion.simulators import SimulatedSiteConnector


@pytest.mark.asyncio
async def test_simulator_returns_expected_record_count():
    connector = SimulatedSiteConnector("SITE-TEST", site_type="factory", hours=24, seed=42)
    result = await connector.fetch()
    assert result.records_fetched == 24
    assert result.records_valid == 24
    assert result.records_invalid == 0


@pytest.mark.asyncio
async def test_simulator_dataframe_schema():
    connector = SimulatedSiteConnector("SITE-TEST", site_type="office", hours=12, seed=0)
    result = await connector.fetch()
    df = result.data
    for col in ("kwh", "timestamp", "site_id", "sensor_id", "voltage_v"):
        assert col in df.columns, f"Missing column: {col}"


@pytest.mark.asyncio
async def test_simulator_no_negative_kwh():
    connector = SimulatedSiteConnector("SITE-TEST", site_type="warehouse", hours=168, seed=7)
    result = await connector.fetch()
    assert (result.data["kwh"] >= 0).all()


@pytest.mark.asyncio
async def test_simulator_site_profiles():
    factory = SimulatedSiteConnector("F", "factory", hours=24, seed=1)
    office = SimulatedSiteConnector("O", "office", hours=24, seed=1)
    r_factory = await factory.fetch()
    r_office = await office.fetch()
    # Factory should consume more than office on average
    assert r_factory.data["kwh"].mean() > r_office.data["kwh"].mean()


@pytest.mark.asyncio
async def test_anomaly_injection():
    """With 100% anomaly probability, all readings should be elevated."""
    connector = SimulatedSiteConnector(
        "SITE-ANOMALY", site_type="factory", hours=50, seed=99, anomaly_prob=1.0
    )
    result = await connector.fetch()
    normal = SimulatedSiteConnector(
        "SITE-NORMAL", site_type="factory", hours=50, seed=99, anomaly_prob=0.0
    )
    r_normal = await normal.fetch()
    assert result.data["kwh"].mean() > r_normal.data["kwh"].mean() * 1.5
