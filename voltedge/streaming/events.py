"""
VoltEdge Streaming — Canonical Event Model

Every message that flows through the streaming pipeline is an EnergyEvent.
It is the single contract between:

  producers  →  broker  →  pipeline  →  hub  →  WebSocket clients

Design principles:
  - Immutable once created (frozen dataclass)
  - Always UTC-timestamped at creation
  - Self-describing: carries source, site, sensor identity
  - Serialisable to/from JSON with no information loss
  - Typed: EventType enum prevents string typos across the codebase

Event lifecycle:
  RAW       → arrives from IoT/MQTT/simulator
  PROCESSED → after EnergyTransformer (features added)
  ANOMALY   → flagged by AnomalyDetector (score + label)
  FORECAST  → model output, pushed to clients
  SYSTEM    → heartbeat, reconnect, pipeline-status
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class EventType(str, Enum):
    RAW       = "raw"        # Fresh sensor reading, unprocessed
    PROCESSED = "processed"  # Feature-engineered reading
    ANOMALY   = "anomaly"    # Flagged anomaly
    FORECAST  = "forecast"   # Model prediction horizon
    ALERT     = "alert"      # Threshold breach / SLA event
    SYSTEM    = "system"     # Heartbeat / pipeline status


@dataclass(frozen=True)
class EnergyEvent:
    """
    Canonical streaming message. Immutable after creation.

    Every field that crosses a network boundary must be JSON-serialisable.
    No pandas DataFrames, numpy arrays, or datetime objects in payload —
    use ISO strings and plain Python dicts instead.

    Args:
        event_type: Lifecycle stage of this event.
        site_id:    VoltEdge site identifier.
        sensor_id:  Individual sensor/meter identifier.
        timestamp:  UTC ISO-8601 string of the reading time.
        kwh:        Energy reading in kWh.
        payload:    Event-type-specific additional data.
        event_id:   Auto-generated UUID4 for deduplication.
        produced_at: UTC ISO string when this event was created.
        sequence:   Monotonically increasing counter per site (best-effort).
    """
    event_type:  EventType
    site_id:     str
    sensor_id:   str
    timestamp:   str          # ISO-8601 UTC string
    kwh:         float

    payload:     dict[str, Any] = field(default_factory=dict)
    event_id:    str            = field(default_factory=lambda: str(uuid.uuid4()))
    produced_at: str            = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    sequence:    int            = 0

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id":   self.event_id,
            "event_type": self.event_type.value,
            "site_id":    self.site_id,
            "sensor_id":  self.sensor_id,
            "timestamp":  self.timestamp,
            "kwh":        self.kwh,
            "payload":    self.payload,
            "produced_at": self.produced_at,
            "sequence":   self.sequence,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EnergyEvent":
        return cls(
            event_type=EventType(d["event_type"]),
            site_id=d["site_id"],
            sensor_id=d["sensor_id"],
            timestamp=d["timestamp"],
            kwh=float(d["kwh"]),
            payload=d.get("payload", {}),
            event_id=d.get("event_id", str(uuid.uuid4())),
            produced_at=d.get("produced_at", datetime.now(timezone.utc).isoformat()),
            sequence=int(d.get("sequence", 0)),
        )

    @classmethod
    def from_json(cls, s: str) -> "EnergyEvent":
        return cls.from_dict(json.loads(s))

    # ── Derived events ────────────────────────────────────────────────────────

    def as_anomaly(
        self,
        score: float,
        label: str,
        is_anomaly: bool,
    ) -> "EnergyEvent":
        """Promote this event to ANOMALY type with scoring metadata."""
        return EnergyEvent(
            event_type=EventType.ANOMALY,
            site_id=self.site_id,
            sensor_id=self.sensor_id,
            timestamp=self.timestamp,
            kwh=self.kwh,
            sequence=self.sequence,
            payload={
                **self.payload,
                "anomaly_score": round(score, 4),
                "anomaly_label": label,
                "is_anomaly":    is_anomaly,
                "source_event":  self.event_id,
            },
        )

    def as_processed(self, features: dict[str, Any]) -> "EnergyEvent":
        """Promote to PROCESSED with feature-engineering metadata."""
        return EnergyEvent(
            event_type=EventType.PROCESSED,
            site_id=self.site_id,
            sensor_id=self.sensor_id,
            timestamp=self.timestamp,
            kwh=self.kwh,
            sequence=self.sequence,
            payload={**self.payload, "features": features, "source_event": self.event_id},
        )

    def as_alert(self, title: str, severity: str, message: str) -> "EnergyEvent":
        """Promote to ALERT for threshold breach notification."""
        return EnergyEvent(
            event_type=EventType.ALERT,
            site_id=self.site_id,
            sensor_id=self.sensor_id,
            timestamp=self.timestamp,
            kwh=self.kwh,
            sequence=self.sequence,
            payload={
                "title":    title,
                "severity": severity,
                "message":  message,
                "source_event": self.event_id,
            },
        )

    # ── Convenience ───────────────────────────────────────────────────────────

    @property
    def reading_datetime(self) -> datetime:
        return datetime.fromisoformat(self.timestamp.rstrip("Z")).replace(tzinfo=timezone.utc)

    @property
    def is_anomaly_event(self) -> bool:
        return self.event_type == EventType.ANOMALY and self.payload.get("is_anomaly", False)

    def __repr__(self) -> str:
        return (
            f"EnergyEvent({self.event_type.value} | {self.site_id} | "
            f"{self.timestamp[:16]} | {self.kwh:.1f} kWh)"
        )


# ── System events ─────────────────────────────────────────────────────────────

def heartbeat_event(site_id: str) -> EnergyEvent:
    return EnergyEvent(
        event_type=EventType.SYSTEM,
        site_id=site_id,
        sensor_id="system",
        timestamp=datetime.now(timezone.utc).isoformat(),
        kwh=0.0,
        payload={"type": "heartbeat"},
    )


def pipeline_status_event(
    pipeline_name: str,
    status: str,
    details: dict[str, Any] | None = None,
) -> EnergyEvent:
    return EnergyEvent(
        event_type=EventType.SYSTEM,
        site_id="platform",
        sensor_id=pipeline_name,
        timestamp=datetime.now(timezone.utc).isoformat(),
        kwh=0.0,
        payload={"type": "pipeline_status", "status": status, **(details or {})},
    )
