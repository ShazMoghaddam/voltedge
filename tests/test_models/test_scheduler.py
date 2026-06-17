"""Tests for ModelScheduler."""

from __future__ import annotations

import asyncio
import time
import pytest

from voltedge.core.scheduler import ModelScheduler
from voltedge.storage.base import LocalStore


def test_scheduler_starts_and_stops(tmp_path):
    store = LocalStore(base_path=tmp_path)
    scheduler = ModelScheduler(store=store, sites=["S1", "S2"], retrain_hours=168)
    scheduler.start()
    assert scheduler.is_running
    scheduler.stop()
    assert not scheduler.is_running


def test_scheduler_status_structure(tmp_path):
    store = LocalStore(base_path=tmp_path)
    scheduler = ModelScheduler(store=store, sites=["S1"], retrain_hours=168)
    scheduler.start()
    status = scheduler.status()
    assert "running" in status
    assert "sites" in status
    assert "last_run" in status
    assert status["running"] is True
    scheduler.stop()


def test_scheduler_trigger_now_with_no_data(tmp_path):
    """trigger_now with no stored data should not raise — just log a warning."""
    store = LocalStore(base_path=tmp_path)
    scheduler = ModelScheduler(store=store, sites=["EMPTY-SITE"], retrain_hours=168)
    scheduler.trigger_now()   # Should complete without exception


def test_scheduler_callback_called(tmp_path):
    """on_retrain callback should be invoked after a successful retrain."""
    from voltedge.ingestion.simulators import SimulatedSiteConnector
    from voltedge.processing.transformer import EnergyTransformer

    # Seed some data
    store = LocalStore(base_path=tmp_path)
    connector = SimulatedSiteConnector("CB-SITE", "factory", hours=500, seed=0)
    result = asyncio.run(connector.fetch())
    df = EnergyTransformer().transform(result.data)
    store.write(df, site_id="CB-SITE", layer="processed")

    called: list[tuple] = []
    def on_retrain(site_id, model_name, metrics):
        called.append((site_id, model_name, metrics))

    scheduler = ModelScheduler(
        store=store, sites=["CB-SITE"],
        retrain_hours=168, artifact_dir=str(tmp_path),
        on_retrain=on_retrain,
    )
    scheduler.trigger_now()
    assert len(called) >= 1
    assert any(c[0] == "CB-SITE" for c in called)
