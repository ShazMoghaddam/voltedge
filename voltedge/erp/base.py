"""
VoltEdge ERP Adapters — Base Interface

Defines the contract all ERP connectors implement.
Enterprise systems (SAP, Oracle, custom REST) all expose energy
meter readings and cost centre mappings via this interface.

The adapter pattern means VoltEdge's pipeline never knows whether
it's talking to SAP PM, Oracle Utilities, or a custom REST API —
it just calls fetch_readings() and normalise().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd


@dataclass
class ERPReading:
    """Canonical energy reading from any ERP system."""
    site_id:        str
    meter_id:       str
    timestamp:      datetime
    kwh:            float
    cost_centre:    str              = ""
    currency:       str              = "USD"
    cost_amount:    float | None     = None
    unit_of_measure: str             = "KWH"
    erp_source:     str              = "generic"
    raw_payload:    dict[str, Any]   = field(default_factory=dict)


@dataclass
class ERPFetchResult:
    """Result of a single ERP fetch operation."""
    source:           str
    site_id:          str
    records_fetched:  int
    records_valid:    int
    records_invalid:  int
    data:             pd.DataFrame
    errors:           list[str]   = field(default_factory=list)
    fetched_at:       datetime    = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def success_rate(self) -> float:
        if self.records_fetched == 0:
            return 0.0
        return self.records_valid / self.records_fetched


class BaseERPAdapter(ABC):
    """
    Abstract base for all ERP connectors.

    Subclasses implement:
        _fetch_raw()  — pull raw records from the ERP system
        _parse()      — map one raw record → ERPReading

    The base class handles validation, error collection, and DataFrame assembly.
    """

    source_name: str = "generic_erp"

    def __init__(self, site_id: str, **kwargs) -> None:
        self.site_id = site_id
        self._config = kwargs

    async def fetch(self) -> ERPFetchResult:
        raw_records = await self._fetch_raw()
        valid: list[dict] = []
        invalid = 0
        errors: list[str] = []

        for rec in raw_records:
            try:
                reading = self._parse(rec)
                if reading.kwh < 0:
                    raise ValueError(f"Negative kWh: {reading.kwh}")
                valid.append({
                    "site_id":        reading.site_id,
                    "meter_id":       reading.meter_id,
                    "timestamp":      reading.timestamp,
                    "kwh":            reading.kwh,
                    "cost_centre":    reading.cost_centre,
                    "currency":       reading.currency,
                    "cost_amount":    reading.cost_amount,
                    "erp_source":     reading.erp_source,
                })
            except Exception as exc:
                invalid += 1
                errors.append(str(exc))

        df = pd.DataFrame(valid) if valid else pd.DataFrame()

        return ERPFetchResult(
            source=self.source_name,
            site_id=self.site_id,
            records_fetched=len(raw_records),
            records_valid=len(valid),
            records_invalid=invalid,
            data=df,
            errors=errors,
        )

    @abstractmethod
    async def _fetch_raw(self) -> list[dict[str, Any]]:
        """Pull raw records from the ERP system."""

    @abstractmethod
    def _parse(self, raw: dict[str, Any]) -> ERPReading:
        """Map one raw ERP record to ERPReading."""

    def inject_records(self, records: list[dict[str, Any]]) -> None:
        """Test helper — inject records bypassing the network."""
        self._injected = records

    async def _get_injected(self) -> list[dict[str, Any]]:
        return getattr(self, "_injected", [])
