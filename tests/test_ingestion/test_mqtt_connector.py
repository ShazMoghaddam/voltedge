"""
Tests for MQTTConnector.
No real broker needed — we patch the network layer and use inject_message().
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from voltedge.ingestion.mqtt_connector import MQTTConnector


SITE = "MQTT-TEST-SITE"


def _make_payload(kwh: float = 45.23, sensor: str = "MAIN") -> dict:
    return {
        "ts":      datetime.now(timezone.utc).isoformat(),
        "kwh":     kwh,
        "voltage": 230.1,
        "current": 197.5,
        "pf":      0.92,
        "temp_c":  21.3,
        "sensor":  sensor,
    }


# ── Unit: _parse ──────────────────────────────────────────────────────────────

def test_parse_maps_all_fields():
    connector = MQTTConnector(SITE)
    parsed = connector._parse(_make_payload(kwh=100.0))
    assert parsed["kwh"] == 100.0
    assert parsed["voltage_v"] == 230.1
    assert parsed["current_a"] == 197.5
    assert parsed["power_factor"] == 0.92
    assert parsed["temperature_c"] == 21.3


def test_parse_defaults_missing_kwh_to_zero():
    connector = MQTTConnector(SITE)
    parsed = connector._parse({"ts": datetime.now(timezone.utc).isoformat()})
    assert parsed["kwh"] == 0.0


def test_parse_sensor_id_uses_site_id_fallback():
    connector = MQTTConnector(SITE)
    parsed = connector._parse({"kwh": 10.0})
    assert SITE in parsed["sensor_id"]


def test_parse_metadata_has_source():
    connector = MQTTConnector(SITE)
    parsed = connector._parse(_make_payload())
    assert parsed["metadata"]["source"] == "mqtt"
    assert parsed["metadata"]["topic"] == f"voltedge/{SITE}/energy"


# ── Unit: inject_message ──────────────────────────────────────────────────────

def test_inject_message_appends_to_messages():
    connector = MQTTConnector(SITE)
    connector.inject_message(_make_payload(kwh=55.0))
    assert len(connector._messages) == 1
    assert connector._messages[0]["kwh"] == 55.0


def test_inject_multiple_messages():
    connector = MQTTConnector(SITE)
    for i in range(10):
        connector.inject_message(_make_payload(kwh=float(i)))
    assert len(connector._messages) == 10


# ── Integration: fetch via inject_message (no broker) ─────────────────────────

@pytest.fixture
def patched_connector():
    """
    Connector whose _connect_and_collect is a no-op, so fetch()
    only processes what was injected via inject_message().
    """
    connector = MQTTConnector(SITE, collect_window_seconds=0)
    # Patch out all network I/O
    with patch.object(connector, "_connect_and_collect", return_value=None):
        yield connector


async def _fetch_with_injected(connector, payloads: list[dict]) -> object:
    """Inject payloads then call fetch(), bypassing the broker."""
    for p in payloads:
        connector.inject_message(p)
    return await connector.fetch()


def test_fetch_returns_ingestion_result(patched_connector):
    payloads = [_make_payload(kwh=float(i)) for i in range(5)]
    result = asyncio.run(_fetch_with_injected(patched_connector, payloads))
    assert result.records_fetched == 5
    assert result.records_valid == 5
    assert result.records_invalid == 0


def test_fetch_produces_dataframe_with_kwh(patched_connector):
    payloads = [_make_payload(kwh=100.0), _make_payload(kwh=200.0)]
    result = asyncio.run(_fetch_with_injected(patched_connector, payloads))
    assert not result.data.empty
    assert "kwh" in result.data.columns
    assert list(result.data["kwh"]) == [100.0, 200.0]


def test_fetch_no_messages_gives_empty_df(patched_connector):
    result = asyncio.run(_fetch_with_injected(patched_connector, []))
    assert result.records_fetched == 0
    assert result.data.empty


def test_fetch_invalid_kwh_handled(patched_connector):
    """Negative kwh should be caught by EnergyReading validator."""
    bad = {"ts": datetime.now(timezone.utc).isoformat(), "kwh": -10.0}
    result = asyncio.run(_fetch_with_injected(patched_connector, [bad]))
    # Pydantic validates kwh >= 0; negative value becomes invalid record
    assert result.records_invalid == 1
    assert result.records_valid == 0


def test_fetch_site_id_propagated(patched_connector):
    result = asyncio.run(_fetch_with_injected(patched_connector, [_make_payload()]))
    assert result.site_id == SITE


def test_source_name_is_mqtt():
    connector = MQTTConnector(SITE)
    assert connector.source_name == "mqtt"


def test_default_topic_includes_site_id():
    connector = MQTTConnector("FACTORY-GB-01")
    assert "FACTORY-GB-01" in connector.topic


def test_custom_topic_respected():
    connector = MQTTConnector(SITE, topic_pattern="custom/topic/path")
    assert connector.topic == "custom/topic/path"


def test_tls_flag_stored():
    connector = MQTTConnector(SITE, tls=True, broker_port=8883)
    assert connector.tls is True
    assert connector.broker_port == 8883


def test_fetch_mixed_valid_invalid(patched_connector):
    """Mix of valid and invalid payloads — check correct split."""
    payloads = [
        _make_payload(kwh=50.0),          # valid
        {"kwh": -5.0},                     # invalid: negative kwh
        _make_payload(kwh=75.0),          # valid
    ]
    result = asyncio.run(_fetch_with_injected(patched_connector, payloads))
    assert result.records_valid == 2
    assert result.records_invalid == 1
    assert result.success_rate == pytest.approx(2 / 3)
