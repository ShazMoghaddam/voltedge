"""Tests for DemandForecaster."""

from __future__ import annotations

import asyncio
import pytest
from voltedge.ingestion.simulators import SimulatedSiteConnector
from voltedge.models.forecasting import DemandForecaster
from voltedge.processing.transformer import EnergyTransformer


def _get_processed_df(hours: int = 500):
    connector = SimulatedSiteConnector("TEST-SITE", "factory", hours=hours, seed=42)
    result = asyncio.run(connector.fetch())
    transformer = EnergyTransformer()
    return transformer.transform(result.data)


def test_forecaster_trains_and_returns_metrics():
    df = _get_processed_df()
    model = DemandForecaster(site_id="TEST-SITE", horizon_hours=6)
    metrics = model.train(df)
    assert "mae" in metrics and "rmse" in metrics and "r2" in metrics
    assert metrics["mae"] >= 0
    assert metrics["r2"] <= 1.0


def test_forecaster_predict_shape():
    df = _get_processed_df()
    model = DemandForecaster(site_id="TEST-SITE", horizon_hours=12)
    model.train(df)
    forecast = model.predict(df.tail(100))
    assert len(forecast) == 12
    assert "kwh_forecast" in forecast.columns
    assert "lower_bound" in forecast.columns
    assert "upper_bound" in forecast.columns


def test_forecaster_no_negative_forecasts():
    df = _get_processed_df()
    model = DemandForecaster(site_id="TEST-SITE", horizon_hours=24)
    model.train(df)
    forecast = model.predict(df.tail(200))
    assert (forecast["kwh_forecast"] >= 0).all()
    assert (forecast["lower_bound"] >= 0).all()


def test_forecaster_bounds_ordering():
    df = _get_processed_df()
    model = DemandForecaster(site_id="TEST-SITE", horizon_hours=6)
    model.train(df)
    forecast = model.predict(df.tail(100))
    assert (forecast["lower_bound"] <= forecast["kwh_forecast"]).all()
    assert (forecast["kwh_forecast"] <= forecast["upper_bound"]).all()


def test_forecaster_save_load(tmp_path):
    df = _get_processed_df()
    model = DemandForecaster(site_id="SAVE-SITE", horizon_hours=6,
                              artifact_dir=str(tmp_path))
    metrics = model.train(df)
    model.save(metrics)

    loaded = DemandForecaster(site_id="SAVE-SITE", horizon_hours=6,
                               artifact_dir=str(tmp_path))
    loaded.load()
    forecast = loaded.predict(df.tail(50))
    assert len(forecast) == 6
