"""Tests for AnomalyDetector."""

from __future__ import annotations

import asyncio
import pytest
from voltedge.ingestion.simulators import SimulatedSiteConnector
from voltedge.models.anomaly import AnomalyDetector
from voltedge.processing.transformer import EnergyTransformer


def _get_processed_df(anomaly_prob: float = 0.05, hours: int = 400):
    connector = SimulatedSiteConnector(
        "TEST-ANOMALY", "factory", hours=hours, seed=7, anomaly_prob=anomaly_prob
    )
    result = asyncio.run(connector.fetch())
    transformer = EnergyTransformer()
    return transformer.transform(result.data)


def test_anomaly_detector_trains():
    df = _get_processed_df()
    model = AnomalyDetector("TEST-ANOMALY", contamination=0.05)
    metrics = model.train(df)
    assert "training_rows" in metrics
    assert metrics["training_rows"] > 0


def test_anomaly_predict_adds_columns():
    df = _get_processed_df()
    model = AnomalyDetector("TEST-ANOMALY")
    model.train(df)
    result = model.predict(df)
    for col in ("anomaly_score", "is_anomaly", "anomaly_label"):
        assert col in result.columns, f"Missing: {col}"


def test_anomaly_score_in_range():
    df = _get_processed_df()
    model = AnomalyDetector("TEST-ANOMALY")
    model.train(df)
    result = model.predict(df)
    assert (result["anomaly_score"] >= 0).all()
    assert (result["anomaly_score"] <= 1).all()


def test_high_anomaly_prob_detected():
    """With 50% injected anomalies, detector should flag a meaningful fraction."""
    df = _get_processed_df(anomaly_prob=0.5)
    model = AnomalyDetector("TEST-ANOMALY", contamination=0.1)
    model.train(df)
    result = model.predict(df)
    flagged_pct = result["is_anomaly"].mean()
    # Should flag at least 5% (contamination set to 10%)
    assert flagged_pct >= 0.04, f"Only {flagged_pct:.1%} flagged"
