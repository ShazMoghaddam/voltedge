"""
VoltEdge — Base data connector interface.

All ingestion sources (IoT sensors, REST APIs, MQTT brokers) implement
this interface. The pipeline layer calls `fetch()` without knowing the
underlying transport.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import pandas as pd
from pydantic import BaseModel, Field, field_validator


# ── Domain model for a single energy reading ──────────────────────────────────

class EnergyReading(BaseModel):
    """Validated, typed representation of one sensor reading."""

    reading_id: str = Field(default_factory=lambda: str(uuid4()))
    site_id: str
    sensor_id: str
    timestamp: datetime
    kwh: float = Field(ge=0, description="Energy consumed in kWh")
    voltage_v: float | None = Field(default=None, ge=0)
    current_a: float | None = Field(default=None, ge=0)
    power_factor: float | None = Field(default=None, ge=0, le=1)
    temperature_c: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("timestamp", mode="before")
    @classmethod
    def ensure_utc(cls, v: Any) -> datetime:
        if isinstance(v, str):
            v = datetime.fromisoformat(v)
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


# ── Connector result envelope ──────────────────────────────────────────────────

@dataclass
class IngestionResult:
    source: str
    site_id: str
    records_fetched: int
    records_valid: int
    records_invalid: int
    data: pd.DataFrame
    errors: list[str] = field(default_factory=list)
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def success_rate(self) -> float:
        if self.records_fetched == 0:
            return 0.0
        return self.records_valid / self.records_fetched


# ── Abstract base connector ────────────────────────────────────────────────────

class BaseConnector(abc.ABC):
    """
    Abstract base class for all VoltEdge data connectors.

    Subclasses must implement `_fetch_raw()` and `_parse()`.
    The public `fetch()` method handles validation, error collection,
    and returns a standardised IngestionResult.
    """

    source_name: str = "unknown"

    def __init__(self, site_id: str, **kwargs: Any) -> None:
        self.site_id = site_id
        self._kwargs = kwargs

    @abc.abstractmethod
    async def _fetch_raw(self) -> list[dict[str, Any]]:
        """Retrieve raw records from the data source. Must be async."""
        ...

    @abc.abstractmethod
    def _parse(self, raw: dict[str, Any]) -> dict[str, Any]:
        """
        Map a raw source record to EnergyReading field names.
        Return a dict that can be passed to EnergyReading(**result).
        """
        ...

    async def fetch(self) -> IngestionResult:
        """
        Orchestrate: fetch → parse → validate → return IngestionResult.
        Invalid records are collected as errors, not raised.
        """
        from voltedge.utils.logger import get_logger
        log = get_logger(self.__class__.__name__)

        raw_records = await self._fetch_raw()
        log.debug("connector.raw_fetched", site=self.site_id, count=len(raw_records))

        valid_readings: list[EnergyReading] = []
        errors: list[str] = []

        for raw in raw_records:
            try:
                parsed = self._parse(raw)
                parsed.setdefault("site_id", self.site_id)
                reading = EnergyReading(**parsed)
                valid_readings.append(reading)
            except Exception as exc:
                errors.append(f"Parse error: {exc} | raw={raw}")

        df = (
            pd.DataFrame([r.to_dict() for r in valid_readings])
            if valid_readings
            else pd.DataFrame()
        )

        return IngestionResult(
            source=self.source_name,
            site_id=self.site_id,
            records_fetched=len(raw_records),
            records_valid=len(valid_readings),
            records_invalid=len(errors),
            data=df,
            errors=errors,
        )
