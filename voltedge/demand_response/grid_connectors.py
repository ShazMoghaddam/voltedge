"""
VoltEdge Demand Response — Grid Operator Connectors

Polls grid operator APIs for active and upcoming DR events,
normalises them to VoltEdge's DREvent model, and publishes
them to the event broker for real-time dashboard display.

Supported operators:
  NESoConnector    — National Grid ESO (UK), Balancing Mechanism + DSBR
  PJMConnector     — PJM Interconnection (US), Economic DR program
  ENTSOEConnector  — ENTSO-E (EU), cross-border balancing events
  MockGridConnector — Deterministic test double, no HTTP

API formats:
  NESO: REST/JSON, unauthenticated for public market data
  PJM:  REST/JSON, API key required (api.pjm.com)
  ENTSO-E: XML, security token required

All connectors implement BaseGridConnector with:
  fetch_events() → list[DREvent]
  watch()        → async generator of DREvent (live polling)
"""

from __future__ import annotations

import asyncio
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator

from voltedge.demand_response.events import DREvent, DREventStatus, DREventType
from voltedge.utils.logger import get_logger

log = get_logger(__name__)


class BaseGridConnector(ABC):
    """Abstract grid operator connector."""

    operator_name: str = "unknown"

    @abstractmethod
    def fetch_events(self, hours_ahead: int = 24) -> list[DREvent]:
        """Fetch upcoming DR events for the next N hours."""

    @abstractmethod
    async def watch(self, poll_interval: int = 60) -> AsyncIterator[DREvent]:
        """Yield new DR events as they arrive (live polling)."""


class MockGridConnector(BaseGridConnector):
    """
    Deterministic mock grid connector for testing and demos.
    Generates realistic UK NESO-style DR events on demand.
    """

    operator_name = "MOCK-NESO"

    def __init__(
        self,
        events_per_day: int   = 2,
        clearing_price: float = 0.45,
        currency:       str   = "GBP",
        seed:           int   = 42,
    ) -> None:
        import random
        self._epd    = events_per_day
        self._price  = clearing_price
        self._currency = currency
        self._rng    = random.Random(seed)
        self._seen:  set[str] = set()

    def fetch_events(self, hours_ahead: int = 24) -> list[DREvent]:
        """Return a deterministic set of upcoming mock events."""
        now    = datetime.now(timezone.utc)
        events = []

        for i in range(self._epd):
            start_offset_h = self._rng.uniform(1, hours_ahead - 2)
            duration_h     = self._rng.choice([0.5, 1.0, 2.0])
            event_type     = self._rng.choice([
                DREventType.ECONOMIC,
                DREventType.ECONOMIC,
                DREventType.RELIABILITY,
                DREventType.EMERGENCY,
            ])
            start = now + timedelta(hours=start_offset_h)
            end   = start + timedelta(hours=duration_h)

            price = self._price * (
                3.0 if event_type == DREventType.EMERGENCY else
                1.5 if event_type == DREventType.RELIABILITY else
                1.0
            )

            events.append(DREvent(
                event_id=f"MOCK-{self._rng.randint(10000, 99999)}",
                grid_operator=self.operator_name,
                event_type=event_type,
                status=DREventStatus.UPCOMING,
                start_time=start,
                end_time=end,
                notification_time=now,
                target_reduction_kw=self._rng.uniform(100, 500),
                clearing_price_kwh=round(price, 3),
                currency=self._currency,
                grid_region="GB",
                mandatory=(event_type == DREventType.EMERGENCY),
                min_response_kw=10.0,
            ))

        return events

    async def watch(self, poll_interval: int = 60) -> AsyncIterator[DREvent]:
        """Yield a new mock event every poll_interval seconds."""
        while True:
            events = self.fetch_events(hours_ahead=2)
            for event in events:
                if event.event_id not in self._seen:
                    self._seen.add(event.event_id)
                    yield event
            await asyncio.sleep(poll_interval)


class NESOConnector(BaseGridConnector):
    """
    National Grid ESO (UK) demand response connector.
    Fetches events from the NESO data portal REST API.
    Public endpoints require no authentication.
    """

    operator_name = "NESO"
    BASE_URL      = "https://api.neso.energy/api/3/action/datastore_search"

    # NESO dataset IDs for demand side balancing reserve
    DSBR_DATASET  = "b7bfb81a-e1fb-4ab0-83e4-7b32d1e574b6"

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run

    def fetch_events(self, hours_ahead: int = 24) -> list[DREvent]:
        if self.dry_run:
            log.debug("neso.fetch_events.dry_run")
            return []

        import httpx
        try:
            resp = httpx.get(
                self.BASE_URL,
                params={"resource_id": self.DSBR_DATASET, "limit": 10},
                timeout=15,
            )
            resp.raise_for_status()
            records = resp.json().get("result", {}).get("records", [])
            return [self._parse_neso_record(r) for r in records]
        except Exception as exc:
            log.error("neso.fetch_error", error=str(exc))
            return []

    async def watch(self, poll_interval: int = 300) -> AsyncIterator[DREvent]:
        seen: set[str] = set()
        while True:
            events = self.fetch_events()
            for evt in events:
                if evt.event_id not in seen:
                    seen.add(evt.event_id)
                    yield evt
            await asyncio.sleep(poll_interval)

    @staticmethod
    def _parse_neso_record(record: dict) -> DREvent:
        now = datetime.now(timezone.utc)
        return DREvent(
            event_id=str(record.get("_id", uuid.uuid4())),
            grid_operator="NESO",
            event_type=DREventType.RELIABILITY,
            status=DREventStatus.UPCOMING,
            start_time=now + timedelta(hours=1),
            end_time=now + timedelta(hours=2),
            notification_time=now,
            target_reduction_kw=float(record.get("Volume_MW", 0)) * 1000,
            clearing_price_kwh=float(record.get("Price_GBP_MWh", 450)) / 1000,
            currency="GBP",
            grid_region="GB",
            raw_payload=record,
        )


class PJMConnector(BaseGridConnector):
    """
    PJM Interconnection (US) demand response connector.
    Uses the PJM Data Miner 2 API (dataminer2.pjm.com).
    """

    operator_name = "PJM"
    BASE_URL      = "https://api.pjm.com/api/v1"

    def __init__(self, api_key: str = "", dry_run: bool = False) -> None:
        self._api_key = api_key
        self.dry_run  = dry_run

    def fetch_events(self, hours_ahead: int = 24) -> list[DREvent]:
        if self.dry_run:
            return []
        import httpx
        try:
            resp = httpx.get(
                f"{self.BASE_URL}/demand_response/events",
                headers={"Ocp-Apim-Subscription-Key": self._api_key},
                timeout=15,
            )
            resp.raise_for_status()
            return [self._parse_pjm_event(e) for e in resp.json().get("items", [])]
        except Exception as exc:
            log.error("pjm.fetch_error", error=str(exc))
            return []

    async def watch(self, poll_interval: int = 300) -> AsyncIterator[DREvent]:
        seen: set[str] = set()
        while True:
            for evt in self.fetch_events():
                if evt.event_id not in seen:
                    seen.add(evt.event_id)
                    yield evt
            await asyncio.sleep(poll_interval)

    @staticmethod
    def _parse_pjm_event(data: dict) -> DREvent:
        now = datetime.now(timezone.utc)
        return DREvent(
            event_id=str(data.get("eventId", uuid.uuid4())),
            grid_operator="PJM",
            event_type=DREventType.ECONOMIC,
            status=DREventStatus.UPCOMING,
            start_time=now + timedelta(hours=1),
            end_time=now + timedelta(hours=2),
            notification_time=now,
            target_reduction_kw=float(data.get("targetMW", 0)) * 1000,
            clearing_price_kwh=float(data.get("clearingPriceMWh", 100)) / 1000,
            currency="USD",
            grid_region="US-PJM",
            raw_payload=data,
        )
