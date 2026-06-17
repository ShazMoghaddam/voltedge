"""
VoltEdge — SAP Plant Maintenance (PM) / IS-U Adapter

Connects to SAP via:
  1. SAP OData services (preferred — modern S/4HANA)
  2. RFC-over-HTTP via SAP Web Services
  3. IDoc flat-file extraction (legacy ECC)

SAP energy data lives in:
  - IS-U (Industry Specific — Utilities): device readings, meters, billing
  - PM (Plant Maintenance): equipment measurements, counters
  - CO (Controlling): cost centre allocations

This adapter targets the SAP IS-U meter reading OData endpoint.

Endpoint pattern (SAP Netweaver Gateway):
    https://{host}:{port}/sap/opu/odata/sap/ZUTIL_METER_READINGS/MeterReadings
    ?$filter=Site eq '{site_id}' and ReadingDate ge datetime'{from}' and ReadingDate le datetime'{to}'
    &$format=json

Requires:
    - SAP user with IS-U read access
    - OData service ZUTIL_METER_READINGS enabled (or equivalent)
    - Network connectivity to SAP backend

For demo/test: uses inject_records() instead of real HTTP.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Any

import httpx

from voltedge.erp.base import BaseERPAdapter, ERPReading
from voltedge.utils.logger import get_logger

log = get_logger(__name__)


class SAPAdapter(BaseERPAdapter):
    """
    SAP IS-U / PM energy data adapter.

    Args:
        site_id:       VoltEdge site identifier (mapped to SAP functional location).
        sap_host:      SAP Netweaver Gateway hostname.
        sap_port:      Default 443 (HTTPS) or 8443.
        sap_client:    SAP client/mandant number (e.g. "100").
        username:      SAP service user.
        password:      SAP service user password.
        fl_mapping:    Optional dict mapping site_id → SAP functional location code.
        days_back:     How many days of readings to fetch per call.
        timeout:       HTTP timeout in seconds.
        use_mock:      If True, generate synthetic SAP-shaped records (for testing).
    """

    source_name = "sap_is_u"

    ODATA_PATH = "/sap/opu/odata/sap/ZUTIL_METER_READINGS/MeterReadings"

    def __init__(
        self,
        site_id: str,
        sap_host: str = "sap-gateway.example.com",
        sap_port: int = 443,
        sap_client: str = "100",
        username: str | None = None,
        password: str | None = None,
        fl_mapping: dict[str, str] | None = None,
        days_back: int = 1,
        timeout: float = 30.0,
        use_mock: bool = False,
    ) -> None:
        super().__init__(site_id=site_id)
        self.sap_host   = sap_host
        self.sap_port   = sap_port
        self.sap_client = sap_client
        self.username   = username
        self.password   = password
        self.fl_mapping = fl_mapping or {}
        self.days_back  = days_back
        self.timeout    = timeout
        self.use_mock   = use_mock

    async def _fetch_raw(self) -> list[dict[str, Any]]:
        if self.use_mock:
            return self._generate_mock_readings()

        injected = await self._get_injected()
        if injected:
            return injected

        fl_id = self.fl_mapping.get(self.site_id, self.site_id)
        now  = datetime.now(timezone.utc)
        from_dt = (now - timedelta(days=self.days_back)).strftime("%Y-%m-%dT%H:%M:%S")
        to_dt   = now.strftime("%Y-%m-%dT%H:%M:%S")

        url = (
            f"https://{self.sap_host}:{self.sap_port}{self.ODATA_PATH}"
            f"?$filter=FunctionalLocation eq '{fl_id}'"
            f" and ReadingDate ge datetime'{from_dt}'"
            f" and ReadingDate le datetime'{to_dt}'"
            f"&sap-client={self.sap_client}&$format=json"
        )

        log.info("sap.fetch_start", site=self.site_id, fl=fl_id, url=url)

        async with httpx.AsyncClient(
            auth=(self.username, self.password) if self.username else None,
            timeout=self.timeout,
            verify=True,
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            records = data.get("d", {}).get("results", [])
            log.info("sap.fetch_complete", site=self.site_id, records=len(records))
            return records

    def _parse(self, raw: dict[str, Any]) -> ERPReading:
        """Map SAP IS-U OData record → ERPReading."""
        # SAP OData dates come as "/Date(1234567890000)/" — strip to epoch ms
        reading_date = raw.get("ReadingDate", "")
        if reading_date.startswith("/Date("):
            epoch_ms = int(reading_date[6:-2])
            ts = datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc)
        else:
            ts = datetime.fromisoformat(reading_date.rstrip("Z")).replace(tzinfo=timezone.utc)

        kwh_str = raw.get("ReadingValue", raw.get("EnergyValue", "0"))
        kwh = float(kwh_str) if kwh_str else 0.0

        # SAP may report in units other than kWh — normalise
        uom = raw.get("UnitOfMeasure", "KWH").upper()
        if uom == "MWH":
            kwh *= 1000.0
        elif uom == "GWH":
            kwh *= 1_000_000.0

        return ERPReading(
            site_id=self.site_id,
            meter_id=raw.get("MeterID", raw.get("DeviceID", f"{self.site_id}-MAIN")),
            timestamp=ts,
            kwh=kwh,
            cost_centre=raw.get("CostCentre", raw.get("CostCenter", "")),
            currency=raw.get("Currency", "EUR"),
            cost_amount=float(raw["Amount"]) if raw.get("Amount") else None,
            unit_of_measure=uom,
            erp_source=self.source_name,
            raw_payload=raw,
        )

    def _generate_mock_readings(self) -> list[dict[str, Any]]:
        """
        Generate synthetic SAP-shaped OData records for demo/test.
        Produces hourly readings for the last `days_back` days.
        """
        import random
        random.seed(abs(hash(self.site_id)) % 9999)
        now = datetime.now(timezone.utc)
        records = []
        for h in range(self.days_back * 24):
            ts = now - timedelta(hours=h)
            epoch_ms = int(ts.timestamp() * 1000)
            kwh = round(random.gauss(350, 40), 3)
            records.append({
                "MeterID":           f"SAP-{self.site_id}-METER-01",
                "FunctionalLocation": self.fl_mapping.get(self.site_id, self.site_id),
                "ReadingDate":        f"/Date({epoch_ms})/",
                "ReadingValue":       str(max(0, kwh)),
                "UnitOfMeasure":      "KWH",
                "CostCentre":         "CC-OPERATIONS-001",
                "Currency":           "EUR",
                "Amount":             str(round(kwh * 0.22, 4)),
            })
        return records
