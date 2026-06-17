"""
VoltEdge Streaming — WebSocket Hub

Manages all active WebSocket connections and broadcasts
EnergyEvents to subscribed clients in real time.

Responsibilities:
  - Accept and track WebSocket connections
  - Route events to the right subscribers (by site_id or wildcard)
  - Handle client disconnects gracefully (no loud errors)
  - Enforce per-tenant connection limits
  - Expose connection metrics for dashboard health

The hub subscribes to the broker's processed/anomaly topics
and pushes events to clients — decoupling the broker from WebSockets.

Client subscription message (sent after connect):
    {"action": "subscribe", "sites": ["SITE-01", "SITE-02"], "types": ["anomaly"]}

Server messages:
    {"event": "connected",   "connection_id": "...", "subscribed_sites": [...]}
    {"event": "data",        "payload": EnergyEvent.to_dict()}
    {"event": "error",       "message": "..."}
    {"event": "heartbeat",   "ts": "ISO-8601"}
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from voltedge.streaming.events import EnergyEvent, EventType
from voltedge.utils.logger import get_logger

log = get_logger(__name__)


# ── Connection record ─────────────────────────────────────────────────────────

class Connection:
    """Represents one active WebSocket client."""

    def __init__(
        self,
        websocket: WebSocket,
        connection_id: str | None = None,
        tenant_id: str = "anonymous",
    ) -> None:
        self.ws            = websocket
        self.id            = connection_id or str(uuid.uuid4())[:8]
        self.tenant_id     = tenant_id
        self.subscribed_sites: set[str]       = set()
        self.subscribed_types: set[EventType] = {
            EventType.PROCESSED, EventType.ANOMALY, EventType.ALERT
        }
        self.connected_at  = datetime.now(timezone.utc)
        self.messages_sent = 0
        self.alive         = True

    async def send_json(self, data: dict) -> bool:
        """Send a JSON message. Returns False if the connection is gone."""
        try:
            await self.ws.send_json(data)
            self.messages_sent += 1
            return True
        except Exception:
            self.alive = False
            return False

    def matches(self, event: EnergyEvent) -> bool:
        """True if this connection should receive the given event."""
        if event.event_type not in self.subscribed_types:
            return False
        if not self.subscribed_sites:
            return True     # wildcard — all sites
        return event.site_id in self.subscribed_sites

    @property
    def info(self) -> dict[str, Any]:
        return {
            "connection_id":     self.id,
            "tenant_id":         self.tenant_id,
            "subscribed_sites":  list(self.subscribed_sites),
            "subscribed_types":  [t.value for t in self.subscribed_types],
            "connected_at":      self.connected_at.isoformat(),
            "messages_sent":     self.messages_sent,
            "alive":             self.alive,
        }


# ── WebSocket hub ─────────────────────────────────────────────────────────────

class WebSocketHub:
    """
    Central registry of all WebSocket connections.
    Receives events from the broker and broadcasts to matching clients.

    Args:
        max_connections_per_tenant: Hard limit per tenant (default 20).
        heartbeat_interval:         Seconds between keepalive pings (default 30).
    """

    def __init__(
        self,
        max_connections_per_tenant: int = 20,
        heartbeat_interval: float = 30.0,
    ) -> None:
        self._connections:  dict[str, Connection]       = {}   # {conn_id: Connection}
        self._by_tenant:    dict[str, set[str]]         = defaultdict(set)
        self._max_per_tenant = max_connections_per_tenant
        self._heartbeat_sec  = heartbeat_interval
        self._total_received = 0
        self._total_sent     = 0
        self._hb_task: asyncio.Task | None = None

    # ── Connection lifecycle ──────────────────────────────────────────────────

    async def connect(
        self,
        websocket: WebSocket,
        tenant_id: str = "anonymous",
    ) -> Connection | None:
        """
        Accept a WebSocket and register it.
        Returns None if the tenant has hit the connection limit.
        """
        if len(self._by_tenant[tenant_id]) >= self._max_per_tenant:
            await websocket.close(code=4029, reason="Connection limit reached")
            log.warning("hub.connection_limit", tenant=tenant_id,
                        limit=self._max_per_tenant)
            return None

        await websocket.accept()
        conn = Connection(websocket, tenant_id=tenant_id)
        self._connections[conn.id] = conn
        self._by_tenant[tenant_id].add(conn.id)

        await conn.send_json({
            "event":            "connected",
            "connection_id":    conn.id,
            "subscribed_sites": [],
            "message":          "Connected to VoltEdge live stream. Send subscribe message.",
        })
        log.info("hub.connected", conn=conn.id, tenant=tenant_id,
                 total=len(self._connections))
        return conn

    async def disconnect(self, conn_id: str) -> None:
        conn = self._connections.pop(conn_id, None)
        if conn:
            self._by_tenant[conn.tenant_id].discard(conn_id)
            log.info("hub.disconnected", conn=conn_id,
                     tenant=conn.tenant_id, msgs_sent=conn.messages_sent)

    async def handle_client_message(self, conn: Connection, raw: str) -> None:
        """Process an incoming control message from a WebSocket client."""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await conn.send_json({"event": "error", "message": "Invalid JSON"})
            return

        action = msg.get("action", "")

        if action == "subscribe":
            sites = msg.get("sites", [])
            types = msg.get("types", [])
            conn.subscribed_sites = set(sites)
            if types:
                conn.subscribed_types = {
                    EventType(t) for t in types if t in EventType._value2member_map_
                }
            await conn.send_json({
                "event":            "subscribed",
                "subscribed_sites": list(conn.subscribed_sites),
                "subscribed_types": [t.value for t in conn.subscribed_types],
            })
            log.debug("hub.subscribed", conn=conn.id, sites=sites)

        elif action == "unsubscribe":
            conn.subscribed_sites.clear()
            conn.subscribed_types = set(EventType)
            await conn.send_json({"event": "unsubscribed"})

        elif action == "ping":
            await conn.send_json({"event": "pong",
                                   "ts": datetime.now(timezone.utc).isoformat()})

        else:
            await conn.send_json({"event": "error",
                                   "message": f"Unknown action: '{action}'"})

    # ── Broadcast ─────────────────────────────────────────────────────────────

    async def broadcast(self, event: EnergyEvent) -> int:
        """
        Send an event to all matching connections.
        Dead connections are removed during the sweep.
        Returns the number of clients that received the event.
        """
        self._total_received += 1
        if not self._connections:
            return 0

        payload = {"event": "data", "payload": event.to_dict()}
        dead: list[str] = []
        sent = 0

        for conn_id, conn in self._connections.items():
            if not conn.alive:
                dead.append(conn_id)
                continue
            if conn.matches(event):
                ok = await conn.send_json(payload)
                if ok:
                    sent += 1
                    self._total_sent += 1
                else:
                    dead.append(conn_id)

        for conn_id in dead:
            await self.disconnect(conn_id)

        return sent

    async def broadcast_many(self, events: list[EnergyEvent]) -> int:
        """Broadcast a batch of events. Returns total sent across all events."""
        total = 0
        for event in events:
            total += await self.broadcast(event)
        return total

    # ── Heartbeat ─────────────────────────────────────────────────────────────

    async def start_heartbeat(self) -> None:
        """Send periodic pings to all connections to detect stale clients."""
        async def _loop():
            while True:
                await asyncio.sleep(self._heartbeat_sec)
                await self._send_heartbeat()

        self._hb_task = asyncio.create_task(_loop(), name="ws_hub_heartbeat")

    async def _send_heartbeat(self) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        dead: list[str] = []
        for conn_id, conn in self._connections.items():
            ok = await conn.send_json({"event": "heartbeat", "ts": ts})
            if not ok:
                dead.append(conn_id)
        for conn_id in dead:
            await self.disconnect(conn_id)

    async def stop_heartbeat(self) -> None:
        if self._hb_task:
            self._hb_task.cancel()
            self._hb_task = None

    # ── Metrics ───────────────────────────────────────────────────────────────

    def metrics(self) -> dict[str, Any]:
        return {
            "active_connections": len(self._connections),
            "tenants":            len(self._by_tenant),
            "total_received":     self._total_received,
            "total_sent":         self._total_sent,
            "connections":        [c.info for c in self._connections.values()],
        }

    @property
    def connection_count(self) -> int:
        return len(self._connections)

    def tenant_connection_count(self, tenant_id: str) -> int:
        return len(self._by_tenant.get(tenant_id, set()))
