"""
Tests for LSTMForecaster.
Uses a short lookback (24h) and few epochs to keep test runtime under 60s.
"""
from __future__ import annotations

import asyncio
import pytest
import numpy as np

from voltedge.ingestion.simulators import SimulatedSiteConnector
try:
    import tensorflow  # noqa: F401
except ImportError:
    import pytest
    pytest.skip(
        'TensorFlow not installed — skipping LSTM tests. '
        'Install tensorflow-macos (Apple Silicon) or tensorflow to run.',
        allow_module_level=True,
    )

from voltedge.models.lstm_forecaster import LSTMForecaster
from voltedge.processing.transformer import EnergyTransformer

SITE = "LSTM-TEST"
LOOKBACK = 24   # Short window for fast tests


def _get_df(hours: int = 300, seed: int = 42) -> object:
    conn = SimulatedSiteConnector(SITE, "factory", hours=hours, seed=seed)
    result = asyncio.run(conn.fetch())
    return EnergyTransformer().transform(result.data)


def _make_model(**kwargs) -> LSTMForecaster:
    return LSTMForecaster(
        site_id=SITE,
        horizon_hours=6,
        lookback_hours=LOOKBACK,
        epochs=5,          # Minimal epochs for speed
        batch_size=32,
        **kwargs,
    )


# ── Training ──────────────────────────────────────────────────────────────────

def test_lstm_train_returns_metrics():
    df = _get_df()
    model = _make_model()
    metrics = model.train(df)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "epochs_run" in metrics
    assert metrics["mae"] >= 0


def test_lstm_train_insufficient_data_raises():
    model = _make_model()
    import pandas as pd
    tiny = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=10, freq="h", tz="UTC"),
                         "kwh": [1.0] * 10})
    with pytest.raises(ValueError, match="Need"):
        model.train(tiny)


def test_lstm_epochs_run_leq_max():
    df = _get_df()
    model = _make_model()
    metrics = model.train(df)
    assert metrics["epochs_run"] <= 5


def test_lstm_train_samples_positive():
    df = _get_df()
    model = _make_model()
    metrics = model.train(df)
    assert metrics["train_samples"] > 0
    assert metrics["val_samples"] > 0


# ── Prediction ────────────────────────────────────────────────────────────────

def test_lstm_predict_correct_horizon():
    df = _get_df()
    model = _make_model()
    model.train(df)
    forecast = model.predict(df)
    assert len(forecast) == 6


def test_lstm_predict_columns_present():
    df = _get_df()
    model = _make_model()
    model.train(df)
    forecast = model.predict(df)
    for col in ("timestamp", "kwh_forecast", "lower_bound", "upper_bound",
                "horizon_h", "uncertainty"):
        assert col in forecast.columns, f"Missing: {col}"


def test_lstm_predict_no_negative_forecasts():
    df = _get_df()
    model = _make_model()
    model.train(df)
    forecast = model.predict(df)
    assert (forecast["kwh_forecast"] >= 0).all()
    assert (forecast["lower_bound"] >= 0).all()


def test_lstm_predict_bounds_ordered():
    df = _get_df()
    model = _make_model()
    model.train(df)
    forecast = model.predict(df)
    assert (forecast["lower_bound"] <= forecast["kwh_forecast"]).all()
    assert (forecast["kwh_forecast"] <= forecast["upper_bound"]).all()


def test_lstm_predict_uncertainty_nonnegative():
    df = _get_df()
    model = _make_model()
    model.train(df)
    forecast = model.predict(df)
    assert (forecast["uncertainty"] >= 0).all()


def test_lstm_horizon_in_forecast():
    df = _get_df()
    model = _make_model()
    model.train(df)
    forecast = model.predict(df)
    assert list(forecast["horizon_h"]) == list(range(1, 7))


def test_lstm_predict_without_train_raises():
    model = _make_model()
    df = _get_df()
    with pytest.raises(RuntimeError, match="train"):
        model.predict(df)


def test_lstm_predict_insufficient_context_raises():
    df = _get_df()
    model = _make_model()
    model.train(df)
    import pandas as pd
    short = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC"),
        "kwh": [10.0] * 5,
    })
    with pytest.raises(ValueError, match="Need"):
        model.predict(short)


# ── Persistence ───────────────────────────────────────────────────────────────

def test_lstm_save_load_roundtrip(tmp_path):
    df = _get_df()
    model = _make_model(artifact_dir=str(tmp_path))
    metrics = model.train(df)
    model.save(metrics)

    loaded = _make_model(artifact_dir=str(tmp_path))
    loaded.load()

    forecast = loaded.predict(df)
    assert len(forecast) == 6
    assert (forecast["kwh_forecast"] >= 0).all()


def test_lstm_save_without_train_raises(tmp_path):
    model = _make_model(artifact_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match="No model"):
        model.save()


def test_lstm_metadata_written(tmp_path):
    import json
    df = _get_df()
    model = _make_model(artifact_dir=str(tmp_path))
    metrics = model.train(df)
    model.save(metrics)
    meta = json.loads((tmp_path / "lstm_forecaster" / SITE / "metadata.json").read_text())
    assert meta["site_id"] == SITE
    assert meta["horizon_hours"] == 6
    assert "mae" in meta["metrics"]


# ── Model properties ──────────────────────────────────────────────────────────

def test_lstm_model_name():
    assert LSTMForecaster.model_name == "lstm_forecaster"


def test_lstm_model_version_semver():
    parts = LSTMForecaster.model_version.split(".")
    assert len(parts) == 3


def test_lstm_mc_dropout_produces_variance():
    """With MC Dropout, repeated predictions should have non-zero std."""
    df = _get_df()
    model = _make_model()
    model.train(df)
    f1 = model.predict(df, mc_samples=10)["kwh_forecast"].values
    f2 = model.predict(df, mc_samples=10)["kwh_forecast"].values
    # Two separate MC runs won't be identical (stochastic dropout)
    assert not np.allclose(f1, f2, atol=0)
