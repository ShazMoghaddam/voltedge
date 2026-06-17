"""
VoltEdge v5.0 — Live Dashboard WebSocket Client

Replaces the 60-second dcc.Interval polling with a WebSocket
connection that receives events in real time and updates the
dashboard instantly.

Uses a background thread to maintain the WebSocket connection
and push events into a thread-safe queue consumed by Dash callbacks.

Pattern:
    ws_client = LiveDataClient(ws_url="ws://localhost:8000/ws/stream")
    ws_client.start()                    # Background thread

    # In Dash callback triggered by dcc.Interval (1s — just to check queue)
    events = ws_client.drain()           # Get all queued events
    for event in events:
        update_dashboard(event)

    ws_client.stop()
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
from datetime import datetime, timezone
from typing import Any

from voltedge.utils.logger import get_logger

log = get_logger(__name__)


class LiveDataClient:
    """
    Thread-safe WebSocket client for the Dash dashboard.

    Maintains a persistent connection to the VoltEdge streaming API
    and buffers received events in a thread-safe queue.

    Args:
        ws_url:      WebSocket endpoint URL.
        site_id:     Optional site filter (uses /ws/stream/{site_id}).
        max_queue:   Maximum buffered events before dropping old ones.
        reconnect_s: Seconds to wait before reconnecting on error.
    """

    def __init__(
        self,
        ws_url:      str   = "ws://localhost:8000/ws/stream",
        site_id:     str | None = None,
        max_queue:   int   = 500,
        reconnect_s: float = 3.0,
    ) -> None:
        self._url        = f"{ws_url}/{site_id}" if site_id else ws_url
        self._max_q      = max_queue
        self._reconnect  = reconnect_s
        self._queue:     queue.Queue[dict[str, Any]] = queue.Queue(maxsize=max_queue)
        self._thread:    threading.Thread | None = None
        self._stop_evt:  threading.Event = threading.Event()
        self._connected: bool = False
        self._stats: dict[str, int] = {
            "received": 0, "dropped": 0, "reconnects": 0
        }
        # In-memory KPI state (updated as events arrive)
        self._kpi: dict[str, Any] = {}
        self._latest_per_site: dict[str, dict[str, Any]] = {}
        self._anomalous_sites: set[str] = set()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background WebSocket thread."""
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="voltedge-ws-client",
            daemon=True,
        )
        self._thread.start()
        log.info("live_client.started", url=self._url)

    def stop(self) -> None:
        """Signal the background thread to stop."""
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=5)
        log.info("live_client.stopped", stats=self._stats)

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── Data access (called from Dash callbacks) ──────────────────────────────

    def drain(self, max_events: int = 50) -> list[dict[str, Any]]:
        """
        Return up to max_events queued events, emptying the queue.
        Non-blocking — returns immediately with whatever is available.
        """
        events = []
        for _ in range(max_events):
            try:
                events.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return events

    def latest_kpi(self) -> dict[str, Any]:
        """Return the most recent KPI snapshot."""
        return dict(self._kpi)

    def latest_reading(self, site_id: str) -> dict[str, Any] | None:
        """Return the latest reading for a specific site."""
        return self._latest_per_site.get(site_id)

    def is_anomalous(self, site_id: str) -> bool:
        """True if the site is currently in an anomalous state."""
        return site_id in self._anomalous_sites

    def anomalous_sites(self) -> list[str]:
        """Return list of currently anomalous site IDs."""
        return list(self._anomalous_sites)

    def stats(self) -> dict[str, Any]:
        return {
            **self._stats,
            "connected":      self._connected,
            "queue_depth":    self._queue.qsize(),
            "tracked_sites":  len(self._latest_per_site),
            "anomalous":      len(self._anomalous_sites),
        }

    # ── Background thread ─────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        """Asyncio event loop running in the background thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._ws_loop())
        finally:
            loop.close()

    async def _ws_loop(self) -> None:
        """Maintain WebSocket connection with automatic reconnection."""
        import websockets

        while not self._stop_evt.is_set():
            try:
                log.info("live_client.connecting", url=self._url)
                async with websockets.connect(
                    self._url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    self._connected = True
                    log.info("live_client.connected")
                    async for raw in ws:
                        if self._stop_evt.is_set():
                            break
                        try:
                            msg = json.loads(raw)
                            self._process_message(msg)
                        except Exception as exc:
                            log.warning("live_client.parse_error", error=str(exc))

            except Exception as exc:
                self._connected = False
                self._stats["reconnects"] += 1
                log.warning("live_client.disconnected", error=str(exc),
                            reconnects=self._stats["reconnects"])

            if not self._stop_evt.is_set():
                await asyncio.sleep(self._reconnect)

        self._connected = False

    def _process_message(self, msg: dict[str, Any]) -> None:
        """Update in-memory state and queue the message for Dash."""
        event_type = msg.get("type", "")
        site_id    = msg.get("site_id", "")

        # Update in-memory state
        if event_type == "reading.ingested" and site_id:
            self._latest_per_site[site_id] = msg.get("data", {})

        elif event_type == "anomaly.detected" and site_id:
            self._anomalous_sites.add(site_id)

        elif event_type == "anomaly.cleared" and site_id:
            self._anomalous_sites.discard(site_id)

        elif event_type == "kpi.updated":
            self._kpi.update(msg.get("data", {}))

        elif event_type in ("connected", "ping"):
            return   # Control messages — don't queue

        # Queue for Dash callbacks
        self._stats["received"] += 1
        try:
            self._queue.put_nowait(msg)
        except queue.Full:
            # Drop oldest message to make room
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(msg)
                self._stats["dropped"] += 1
            except queue.Empty:
                pass
