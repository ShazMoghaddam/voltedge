"""Tests for WebSocketHub — uses starlette.testclient WebSocket support."""
from __future__ import annotations
import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from voltedge.streaming.hub import Connection, WebSocketHub
from voltedge.streaming.events import EnergyEvent, EventType

SITE = "HUB-TEST-01"
TS   = "2026-05-15T14:00:00+00:00"


def _event(event_type=EventType.PROCESSED, site_id=SITE, kwh=100.0):
    return EnergyEvent(
        event_type=event_type, site_id=site_id,
        sensor_id="M1", timestamp=TS, kwh=kwh,
    )


def _mock_ws():
    """Mock WebSocket that records sent messages."""
    ws = AsyncMock()
    ws.sent = []
    async def send_json(data):
        ws.sent.append(data)
    ws.send_json.side_effect = send_json
    return ws


def _make_conn(hub=None, tenant="test-tenant"):
    ws  = _mock_ws()
    hub = hub or WebSocketHub()
    conn = Connection(ws, tenant_id=tenant)
    hub._connections[conn.id] = conn
    hub._by_tenant[tenant].add(conn.id)
    return conn, ws, hub


# ── Connection ────────────────────────────────────────────────────────────────

def test_connection_has_id():
    ws   = _mock_ws()
    conn = Connection(ws)
    assert len(conn.id) == 8


def test_connection_matches_all_sites_when_none_subscribed():
    ws   = _mock_ws()
    conn = Connection(ws)
    assert conn.matches(_event())


def test_connection_matches_subscribed_site():
    ws   = _mock_ws()
    conn = Connection(ws)
    conn.subscribed_sites = {SITE}
    assert conn.matches(_event(site_id=SITE))


def test_connection_does_not_match_other_site():
    ws   = _mock_ws()
    conn = Connection(ws)
    conn.subscribed_sites = {"OTHER-SITE"}
    assert not conn.matches(_event(site_id=SITE))


def test_connection_filters_by_event_type():
    ws   = _mock_ws()
    conn = Connection(ws)
    conn.subscribed_types = {EventType.ANOMALY}
    assert not conn.matches(_event(event_type=EventType.PROCESSED))
    assert conn.matches(_event(event_type=EventType.ANOMALY))


def test_connection_info_keys():
    ws   = _mock_ws()
    conn = Connection(ws)
    info = conn.info
    for k in ("connection_id", "tenant_id", "subscribed_sites", "messages_sent", "alive"):
        assert k in info


# ── Hub connection management ─────────────────────────────────────────────────

def test_hub_connect_registers_connection():
    async def run():
        hub = WebSocketHub()
        ws  = _mock_ws()
        conn = Connection(ws, tenant_id="t1")
        hub._connections[conn.id] = conn
        hub._by_tenant["t1"].add(conn.id)
        return hub.connection_count

    count = asyncio.run(run())
    assert count == 1


def test_hub_disconnect_removes_connection():
    async def run():
        conn, ws, hub = _make_conn()
        await hub.disconnect(conn.id)
        return hub.connection_count

    count = asyncio.run(run())
    assert count == 0


def test_hub_disconnect_unknown_id_is_noop():
    async def run():
        hub = WebSocketHub()
        await hub.disconnect("nonexistent")   # no error

    asyncio.run(run())


def test_hub_tenant_connection_count():
    conn, ws, hub = _make_conn(tenant="acme")
    assert hub.tenant_connection_count("acme") == 1
    assert hub.tenant_connection_count("other") == 0


def test_hub_connection_limit_per_tenant():
    async def run():
        hub = WebSocketHub(max_connections_per_tenant=2)
        # Add 2 connections for tenant
        for _ in range(2):
            conn, ws, _ = _make_conn(hub=hub, tenant="limited")
        # Third should reject
        ws3 = _mock_ws()
        ws3.close = AsyncMock()
        result = await hub.connect(ws3, tenant_id="limited")
        return result

    result = asyncio.run(run())
    assert result is None


# ── Broadcast ─────────────────────────────────────────────────────────────────

def test_broadcast_to_matching_connection():
    async def run():
        conn, ws, hub = _make_conn()
        n = await hub.broadcast(_event())
        return n, ws.sent

    n, sent = asyncio.run(run())
    assert n == 1
    assert len(sent) == 1
    assert sent[0]["event"] == "data"


def test_broadcast_with_no_connections():
    async def run():
        hub = WebSocketHub()
        return await hub.broadcast(_event())

    n = asyncio.run(run())
    assert n == 0


def test_broadcast_does_not_send_to_non_matching_site():
    async def run():
        conn, ws, hub = _make_conn()
        conn.subscribed_sites = {"DIFFERENT-SITE"}
        n = await hub.broadcast(_event(site_id=SITE))
        return n

    n = asyncio.run(run())
    assert n == 0


def test_broadcast_payload_contains_event_dict():
    async def run():
        conn, ws, hub = _make_conn()
        evt = _event(kwh=250.0)
        await hub.broadcast(evt)
        return ws.sent

    sent = asyncio.run(run())
    assert sent[0]["payload"]["kwh"] == 250.0   # directly inserted, no preamble


def test_broadcast_removes_dead_connection():
    async def run():
        conn, ws, hub = _make_conn()
        conn.alive = False
        await hub.broadcast(_event())
        return hub.connection_count

    count = asyncio.run(run())
    assert count == 0


def test_broadcast_many():
    async def run():
        conn, ws, hub = _make_conn()
        events = [_event() for _ in range(3)]
        total = await hub.broadcast_many(events)
        return total

    total = asyncio.run(run())
    assert total == 3


# ── Control messages ──────────────────────────────────────────────────────────

def test_handle_subscribe_message():
    async def run():
        conn, ws, hub = _make_conn()
        msg = json.dumps({"action": "subscribe", "sites": ["SITE-A", "SITE-B"]})
        await hub.handle_client_message(conn, msg)
        return conn.subscribed_sites

    sites = asyncio.run(run())
    assert sites == {"SITE-A", "SITE-B"}


def test_handle_ping_returns_pong():
    async def run():
        conn, ws, hub = _make_conn()
        await hub.handle_client_message(conn, json.dumps({"action": "ping"}))
        return ws.sent

    sent = asyncio.run(run())
    pong = next((m for m in sent if m.get("event") == "pong"), None)
    assert pong is not None


def test_handle_invalid_json():
    async def run():
        conn, ws, hub = _make_conn()
        await hub.handle_client_message(conn, "not-json")
        return ws.sent

    sent = asyncio.run(run())
    error = next((m for m in sent if m.get("event") == "error"), None)
    assert error is not None


def test_handle_unknown_action():
    async def run():
        conn, ws, hub = _make_conn()
        await hub.handle_client_message(conn, json.dumps({"action": "fly"}))
        return ws.sent

    sent = asyncio.run(run())
    error = next((m for m in sent if m.get("event") == "error"), None)
    assert error is not None


# ── Metrics ───────────────────────────────────────────────────────────────────

def test_metrics_structure():
    hub = WebSocketHub()
    m = hub.metrics()
    for k in ("active_connections", "tenants", "total_received", "total_sent"):
        assert k in m


def test_metrics_total_sent_increments():
    async def run():
        conn, ws, hub = _make_conn()
        await hub.broadcast(_event())
        return hub.metrics()["total_sent"]

    total = asyncio.run(run())
    assert total == 1
