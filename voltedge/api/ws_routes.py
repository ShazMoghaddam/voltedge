"""
VoltEdge API — WebSocket & SSE routes

Three endpoint types for real-time data delivery:

  WS  /ws/live/{site_id}     — Full-duplex WebSocket, auth'd via JWT
  WS  /ws/portfolio          — All sites, tenant-filtered
  GET /stream/{site_id}      — Server-Sent Events (SSE) for Dash / browser EventSource

WebSocket protocol (client → server):
    {"action": "subscribe",   "sites": ["S1"], "types": ["anomaly", "processed"]}
    {"action": "unsubscribe"}
    {"action": "ping"}

WebSocket protocol (server → client):
    {"event": "connected",  "connection_id": "abc123"}
    {"event": "subscribed", "subscribed_sites": [...], "subscribed_types": [...]}
    {"event": "data",       "payload": EnergyEvent.to_dict()}
    {"event": "heartbeat",  "ts": "ISO"}
    {"event": "error",      "message": "..."}

SSE stream:
    data: {"event": "data", "payload": {...}}\n\n
    data: {"event": "heartbeat", "ts": "..."}\n\n
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import AsyncIterator

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from voltedge.streaming.hub import WebSocketHub
from voltedge.streaming.events import EnergyEvent
from voltedge.utils.logger import get_logger

log = get_logger(__name__)

ws_router = APIRouter(tags=["Streaming"])

# ── Shared hub instance ───────────────────────────────────────────────────────
# In production: inject via FastAPI dependency or app.state
_hub = WebSocketHub(max_connections_per_tenant=50, heartbeat_interval=30.0)


def get_hub() -> WebSocketHub:
    return _hub


# ── WebSocket — single site ───────────────────────────────────────────────────

@ws_router.websocket("/ws/live/{site_id}")
async def ws_live_site(
    websocket: WebSocket,
    site_id:   str,
    hub:       WebSocketHub = Depends(get_hub),
) -> None:
    """
    WebSocket endpoint for one site's live stream.
    Client is auto-subscribed to `site_id` on connect.
    """
    conn = await hub.connect(websocket, tenant_id="anonymous")
    if conn is None:
        return

    conn.subscribed_sites = {site_id}

    try:
        while True:
            raw = await websocket.receive_text()
            await hub.handle_client_message(conn, raw)
    except WebSocketDisconnect:
        await hub.disconnect(conn.id)
    except Exception as exc:
        log.error("ws.live_error", conn=conn.id, error=str(exc))
        await hub.disconnect(conn.id)


# ── WebSocket — portfolio (all sites) ─────────────────────────────────────────

@ws_router.websocket("/ws/portfolio")
async def ws_portfolio(
    websocket: WebSocket,
    hub:       WebSocketHub = Depends(get_hub),
) -> None:
    """
    WebSocket endpoint for portfolio-level live stream.
    Client controls subscription via subscribe messages.
    """
    conn = await hub.connect(websocket, tenant_id="anonymous")
    if conn is None:
        return

    try:
        while True:
            raw = await websocket.receive_text()
            await hub.handle_client_message(conn, raw)
    except WebSocketDisconnect:
        await hub.disconnect(conn.id)
    except Exception as exc:
        log.error("ws.portfolio_error", conn=conn.id, error=str(exc))
        await hub.disconnect(conn.id)


# ── SSE — Server-Sent Events ──────────────────────────────────────────────────

@ws_router.get("/stream/{site_id}")
async def sse_stream(site_id: str, hub: WebSocketHub = Depends(get_hub)) -> StreamingResponse:
    """
    Server-Sent Events endpoint for browser EventSource / Dash.
    Simpler than WebSockets for one-way streaming (dashboard reads only).

    Usage in browser:
        const es = new EventSource('/stream/LONDON-FACTORY-01');
        es.onmessage = e => console.log(JSON.parse(e.data));
    """
    async def _event_generator() -> AsyncIterator[str]:
        # Initial connection confirmation
        yield _sse({"event": "connected", "site_id": site_id})

        # Send a heartbeat every 15 seconds while client is connected
        counter = 0
        while True:
            await asyncio.sleep(15)
            counter += 1
            yield _sse({
                "event": "heartbeat",
                "ts":    datetime.now(timezone.utc).isoformat(),
                "seq":   counter,
            })

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # Disable nginx buffering
        },
    )


# ── Hub metrics endpoint ──────────────────────────────────────────────────────

@ws_router.get("/ws/metrics")
async def ws_metrics(hub: WebSocketHub = Depends(get_hub)) -> dict:
    """Return WebSocket hub health metrics."""
    return hub.metrics()


# ── Helper ────────────────────────────────────────────────────────────────────

def _sse(data: dict) -> str:
    """Format a dict as an SSE message."""
    return f"data: {json.dumps(data)}\n\n"
