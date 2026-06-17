"""Tests for the anomaly model cache."""
from __future__ import annotations

import time
import pytest
import pandas as pd

from voltedge.models.anomaly.cache import (
    get_anomaly_results, invalidate, cache_stats, MIN_ROWS, CACHE_TTL
)


@pytest.fixture(autouse=True)
def clear_cache():
    """Start each test with an empty cache."""
    invalidate()
    yield
    invalidate()


@pytest.fixture()
def good_df(tmp_path, monkeypatch):
    """Provide a real 7-day DataFrame from the DB loader (via seeder)."""
    # Use a tiny in-memory DB for speed
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from voltedge.db.database import Base
    import voltedge.db.database as dbmod

    engine  = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Session = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    monkeypatch.setattr(dbmod, "engine",  engine)
    monkeypatch.setattr(dbmod, "Session", Session)
    Base.metadata.create_all(engine)

    from voltedge.db.seeder import seed
    seed()

    from voltedge.db.loader import load_site_readings
    from voltedge.processing.transformer import EnergyTransformer
    raw = load_site_readings("LONDON-FACTORY-01", hours=168)
    return EnergyTransformer().transform(raw)


def test_returns_results_with_sufficient_data(good_df):
    detector, anom_df = get_anomaly_results("LONDON-FACTORY-01", good_df, hours=168)
    assert detector is not None
    assert anom_df is not None
    assert "anomaly_score" in anom_df.columns
    assert "is_anomaly" in anom_df.columns


def test_returns_none_with_insufficient_data():
    tiny_df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=10, freq="h", tz="UTC"),
        "kwh": [100.0] * 10,
        "hour": list(range(10)),
        "is_weekend": [0] * 10,
        "is_business_hour": [1] * 10,
    })
    detector, anom_df = get_anomaly_results("TEST-SITE", tiny_df, hours=10)
    assert detector is None
    assert anom_df is None


def test_second_call_hits_cache(good_df):
    # First call — trains model
    get_anomaly_results("LONDON-FACTORY-01", good_df, hours=168)
    stats_before = cache_stats()
    assert len(stats_before) == 1

    # Second call — should hit cache (age_s > 0 but TTL not exceeded)
    get_anomaly_results("LONDON-FACTORY-01", good_df, hours=168)
    stats_after = cache_stats()
    assert len(stats_after) == 1  # same entry, not duplicated


def test_different_sites_cached_separately(good_df):
    get_anomaly_results("LONDON-FACTORY-01", good_df, hours=168)
    get_anomaly_results("FRANKFURT-DC-01",   good_df, hours=168)
    assert len(cache_stats()) == 2


def test_different_windows_cached_separately(good_df):
    get_anomaly_results("LONDON-FACTORY-01", good_df, hours=24)
    get_anomaly_results("LONDON-FACTORY-01", good_df, hours=168)
    assert len(cache_stats()) == 2


def test_invalidate_specific_site(good_df):
    get_anomaly_results("LONDON-FACTORY-01", good_df, hours=168)
    get_anomaly_results("FRANKFURT-DC-01",   good_df, hours=168)
    invalidate("LONDON-FACTORY-01")
    stats = cache_stats()
    assert not any("LONDON" in str(k) for k in stats)
    assert any("FRANKFURT" in str(k) for k in stats)


def test_invalidate_all(good_df):
    get_anomaly_results("LONDON-FACTORY-01", good_df, hours=168)
    invalidate()
    assert cache_stats() == {}


def test_anomaly_scores_in_valid_range(good_df):
    _, anom_df = get_anomaly_results("LONDON-FACTORY-01", good_df, hours=168)
    assert anom_df["anomaly_score"].between(0, 1).all()


def test_is_anomaly_is_boolean_like(good_df):
    _, anom_df = get_anomaly_results("LONDON-FACTORY-01", good_df, hours=168)
    assert set(anom_df["is_anomaly"].unique()).issubset({True, False})


def test_anomaly_label_column_present(good_df):
    _, anom_df = get_anomaly_results("LONDON-FACTORY-01", good_df, hours=168)
    assert "anomaly_label" in anom_df.columns


def test_cache_stats_reports_ttl(good_df):
    get_anomaly_results("LONDON-FACTORY-01", good_df, hours=168)
    stats = cache_stats()
    key   = list(stats.keys())[0]
    assert stats[key]["ttl_remaining_s"] > 0
    assert stats[key]["rows"] > 0
