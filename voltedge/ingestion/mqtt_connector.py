"""
VoltEdge — MQTT IoT Connector

Subscribes to an MQTT broker topic and collects energy readings
for a configurable window before returning them to the pipeline.

Designed for:
  - Industrial IoT sensors publishing to HiveMQ / Mosquitto / AWS IoT Core
  - Smart meters with DLMS/COSEM over MQTT
  - Custom ESP32/Raspberry Pi sensor nodes

Topic convention (configurable):
    voltedge/{site_id}/energy       — main energy readings
    voltedge/{site_id}/power        — real-time power (W)
    voltedge/{site_id}/status       — sensor heartbeat / metadata

Message format (JSON):
    {
        "ts":        "2026-05-13T08:00:00Z",   # ISO timestamp
        "kwh":       45.23,
        "voltage":   230.1,
        "current":   197.5,
        "pf":        0.92,
        "temp_c":    21.3,
        "sensor":    "MAIN-METER-01"
    }
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from datetime import datetime, timezone
from typing import Any

try:
    import paho.mqtt.client as mqtt
    PAHO_AVAILABLE = True
except ImportError:
    PAHO_AVAILABLE = False

from voltedge.ingestion.base import BaseConnector
from voltedge.utils.logger import get_logger

log = get_logger(__name__)


class MQTTConnector(BaseConnector):
    """
    MQTT-backed connector: subscribes to a broker, collects messages
    for `collect_window_seconds`, then yields them to the pipeline.

    Args:
        site_id:               Site identifier (used in topic pattern).
        broker_host:           MQTT broker hostname.
        broker_port:           Default 1883 (8883 for TLS).
        topic_pattern:         MQTT topic to subscribe to.
                               Defaults to "voltedge/{site_id}/energy".
        collect_window_seconds: How long to listen before returning.
        username / password:   Optional broker credentials.
        tls:                   Enable TLS (requires broker_port=8883).
        client_id:             MQTT client ID (auto-generated if None).
        qos:                   MQTT QoS level (0, 1, or 2).
    """

    source_name = "mqtt"

    def __init__(
        self,
        site_id: str,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        topic_pattern: str | None = None,
        collect_window_seconds: float = 55.0,
        username: str | None = None,
        password: str | None = None,
        tls: bool = False,
        client_id: str | None = None,
        qos: int = 1,
    ) -> None:
        super().__init__(site_id=site_id)
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.topic = topic_pattern or f"voltedge/{site_id}/energy"
        self.collect_window = collect_window_seconds
        self.username = username
        self.password = password
        self.tls = tls
        self.client_id = client_id or f"voltedge-{site_id}-{int(time.time())}"
        self.qos = qos

        self._messages: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._connected = threading.Event()
        self._client: Any = None

    # ── BaseConnector interface ───────────────────────────────────────────────

    async def _fetch_raw(self) -> list[dict[str, Any]]:
        """
        Connect to the broker, subscribe, collect for `collect_window_seconds`,
        disconnect, and return raw message payloads.
        """
        if not PAHO_AVAILABLE:
            raise RuntimeError(
                "paho-mqtt is required for MQTTConnector. "
                "Install it with: pip install paho-mqtt"
            )

        self._connected.clear()

        client = mqtt.Client(client_id=self.client_id, protocol=mqtt.MQTTv5)
        self._client = client

        if self.username:
            client.username_pw_set(self.username, self.password)
        if self.tls:
            client.tls_set()

        client.on_connect    = self._on_connect
        client.on_message    = self._on_message
        client.on_disconnect = self._on_disconnect

        log.info(
            "mqtt.connecting",
            site=self.site_id,
            broker=f"{self.broker_host}:{self.broker_port}",
            topic=self.topic,
        )

        # Connect in a thread so we don't block the event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._connect_and_collect)

        log.info(
            "mqtt.collection_complete",
            site=self.site_id,
            messages=len(self._messages),
        )
        return list(self._messages)

    def _parse(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Map MQTT JSON payload to EnergyReading fields."""
        return {
            "sensor_id": raw.get("sensor", f"{self.site_id}-MQTT"),
            "timestamp":  raw.get("ts", datetime.now(timezone.utc).isoformat()),
            "kwh":        float(raw.get("kwh", 0.0)),
            "voltage_v":  raw.get("voltage"),
            "current_a":  raw.get("current"),
            "power_factor": raw.get("pf"),
            "temperature_c": raw.get("temp_c"),
            "metadata":   {
                "source": "mqtt",
                "topic":  self.topic,
                "raw_keys": list(raw.keys()),
            },
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _connect_and_collect(self) -> None:
        """Blocking: connect → wait for connection → collect → disconnect."""
        # Reset here so inject_message() in tests survives the no-op patch
        with self._lock:
            self._messages = []
        try:
            self._client.connect(self.broker_host, self.broker_port, keepalive=60)
            self._client.loop_start()

            # Wait for connection confirmation (max 10s)
            connected = self._connected.wait(timeout=10.0)
            if not connected:
                log.error("mqtt.connection_timeout", broker=self.broker_host)
                return

            # Collect messages for the window
            time.sleep(self.collect_window)

        except Exception as exc:
            log.error("mqtt.connection_error", error=str(exc), broker=self.broker_host)
        finally:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:
                pass

    def _on_connect(self, client, userdata, flags, rc, properties=None) -> None:
        if rc == 0:
            client.subscribe(self.topic, qos=self.qos)
            self._connected.set()
            log.info("mqtt.subscribed", topic=self.topic, site=self.site_id)
        else:
            log.error("mqtt.connect_failed", rc=rc, site=self.site_id)

    def _on_message(self, client, userdata, msg) -> None:
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            with self._lock:
                self._messages.append(payload)
            log.debug("mqtt.message_received", topic=msg.topic, site=self.site_id)
        except json.JSONDecodeError as exc:
            log.warning("mqtt.invalid_json", error=str(exc), topic=msg.topic)

    def _on_disconnect(self, client, userdata, rc, properties=None) -> None:
        if rc != 0:
            log.warning("mqtt.unexpected_disconnect", rc=rc, site=self.site_id)

    def inject_message(self, payload: dict[str, Any]) -> None:
        """
        Inject a message directly — used in tests to bypass the broker.
        Simulates a received MQTT message.
        """
        with self._lock:
            self._messages.append(payload)
