"""Tests for the EnergyEvent model."""
from __future__ import annotations
import json
from datetime import datetime, timezone
import pytest
from voltedge.streaming.events import (
    EnergyEvent, EventType, heartbeat_event, pipeline_status_event,
)

TS  = "2026-05-15T12:00:00+00:00"
SITE = "EU-LONDON-01"


def _raw(kwh=100.0, seq=0):
    return EnergyEvent(
        event_type=EventType.RAW, site_id=SITE, sensor_id="M1",
        timestamp=TS, kwh=kwh, sequence=seq,
    )


# ── Construction ──────────────────────────────────────────────────────────────

def test_event_auto_id():
    e = _raw()
    assert len(e.event_id) == 36   # UUID4

def test_event_auto_produced_at():
    e = _raw()
    dt = datetime.fromisoformat(e.produced_at)
    assert dt.tzinfo is not None

def test_event_is_immutable():
    e = _raw()
    with pytest.raises(Exception):
        e.kwh = 999  # frozen dataclass

def test_two_events_have_different_ids():
    assert _raw().event_id != _raw().event_id

# ── Serialisation ─────────────────────────────────────────────────────────────

def test_to_dict_has_all_keys():
    d = _raw().to_dict()
    for k in ("event_id","event_type","site_id","sensor_id","timestamp","kwh","payload","produced_at","sequence"):
        assert k in d

def test_round_trip_dict():
    e = _raw(kwh=42.5, seq=7)
    restored = EnergyEvent.from_dict(e.to_dict())
    assert restored.kwh == 42.5
    assert restored.sequence == 7
    assert restored.event_type == EventType.RAW

def test_round_trip_json():
    e = _raw(kwh=77.3)
    restored = EnergyEvent.from_json(e.to_json())
    assert restored.kwh == pytest.approx(77.3)
    assert restored.site_id == SITE

def test_to_json_is_valid_json():
    raw = _raw().to_json()
    d = json.loads(raw)
    assert d["event_type"] == "raw"

def test_event_type_serialised_as_string():
    d = _raw().to_dict()
    assert isinstance(d["event_type"], str)

# ── Derived events ────────────────────────────────────────────────────────────

def test_as_anomaly_type():
    a = _raw().as_anomaly(score=0.92, label="sudden_spike", is_anomaly=True)
    assert a.event_type == EventType.ANOMALY

def test_as_anomaly_carries_score():
    a = _raw().as_anomaly(0.87, "flat_line", True)
    assert a.payload["anomaly_score"] == 0.87
    assert a.payload["anomaly_label"] == "flat_line"
    assert a.payload["is_anomaly"] is True

def test_as_anomaly_links_source():
    raw = _raw()
    a = raw.as_anomaly(0.5, "normal", False)
    assert a.payload["source_event"] == raw.event_id

def test_as_anomaly_preserves_kwh():
    a = _raw(kwh=200.0).as_anomaly(0.5, "x", False)
    assert a.kwh == 200.0

def test_as_processed_type():
    p = _raw().as_processed({"hour_sin": 0.5})
    assert p.event_type == EventType.PROCESSED

def test_as_processed_carries_features():
    p = _raw().as_processed({"hour_sin": 0.5, "kwh_lag_1h": 99.0})
    assert p.payload["features"]["hour_sin"] == 0.5

def test_as_alert_type():
    a = _raw().as_alert("Spike", "high", "Sudden 40% spike")
    assert a.event_type == EventType.ALERT

def test_as_alert_payload():
    a = _raw().as_alert("T", "critical", "M")
    assert a.payload["severity"] == "critical"
    assert a.payload["title"] == "T"

# ── Properties ────────────────────────────────────────────────────────────────

def test_reading_datetime_is_utc():
    dt = _raw().reading_datetime
    assert dt.tzinfo is not None

def test_is_anomaly_event_true():
    a = _raw().as_anomaly(0.95, "spike", True)
    assert a.is_anomaly_event is True

def test_is_anomaly_event_false_when_not_anomaly():
    a = _raw().as_anomaly(0.3, "normal", False)
    assert a.is_anomaly_event is False

def test_is_anomaly_event_false_for_raw():
    assert _raw().is_anomaly_event is False

def test_repr_contains_site():
    assert SITE in repr(_raw())

# ── System events ─────────────────────────────────────────────────────────────

def test_heartbeat_event():
    h = heartbeat_event("SITE-01")
    assert h.event_type == EventType.SYSTEM
    assert h.payload["type"] == "heartbeat"

def test_pipeline_status_event():
    s = pipeline_status_event("main-pipeline", "running", {"lag_ms": 45})
    assert s.event_type == EventType.SYSTEM
    assert s.payload["status"] == "running"
    assert s.payload["lag_ms"] == 45
