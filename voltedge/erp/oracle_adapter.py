"""
VoltEdge — Oracle Utilities (CC&B / MWM) Adapter

Connects to Oracle Utilities via:
  1. Oracle REST Data Services (ORDS) — preferred for modern deployments
  2. Oracle Integration Cloud (OIC) — enterprise middleware
  3. Direct JDBC-over-HTTP via Oracle Application Express (APEX)

Oracle Utilities data lives in:
  - CC&B (Customer Care & Billing): meter reads, usage intervals
  - MWM (Mobile Workforce Management): field meter data
  - Analytics Cloud: aggregated energy intelligence

This adapter targets the Oracle ORDS meter reading endpoint.

Endpoint pattern:
    https://{host}:{port}/ords/{schema}/meter_reads/
    ?site_id={site_id}&from_dt={iso}&to_dt={iso}

For demo/test: uses inject_records() or use_mock=True.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any

import httpx

from voltedge.erp.base import BaseERPAdapter, ERPReading
from voltedge.utils.logger import get_logger

log = get_logger(__name__)


class OracleAdapter(BaseERPAdapter):
    """
    Oracle Utilities (CC&B / MWM) energy data adapter.

    Args:
        site_id:        VoltEdge site identifier.
        ords_host:      Oracle ORDS hostname.
        ords_port:      Default 8080 (HTTP) or 8443 (HTTPS).
        ords_schema:    Oracle ORDS schema name.
        username:       Oracle REST user.
        password:       Oracle REST user password.
        days_back:      How many days of readings per fetch.
        timeout:        HTTP timeout seconds.
        use_mock:       Generate synthetic records (for testing).
        tls_verify:     Whether to verify TLS certificate (False for dev).
    """

    source_name = "oracle_utilities"

    def __init__(
        self,
        site_id: str,
        ords_host: str = "oracle-ords.example.com",
        ords_port: int = 443,
        ords_schema: str = "voltedge",
        username: str | None = None,
        password: str | None = None,
        days_back: int = 1,
        timeout: float = 30.0,
        use_mock: bool = False,
        tls_verify: bool = True,
    ) -> None:
        super().__init__(site_id=site_id)
        self.ords_host   = ords_host
        self.ords_port   = ords_port
        self.ords_schema = ords_schema
        self.username    = username
        self.password    = password
        self.days_back   = days_back
        self.timeout     = timeout
        self.use_mock    = use_mock
        self.tls_verify  = tls_verify

    async def _fetch_raw(self) -> list[dict[str, Any]]:
        if self.use_mock:
            return self._generate_mock_readings()

        injected = await self._get_injected()
        if injected:
            return injected

        now     = datetime.now(timezone.utc)
        from_dt = (now - timedelta(days=self.days_back)).isoformat()
        to_dt   = now.isoformat()

        url = (
            f"https://{self.ords_host}:{self.ords_port}"
            f"/ords/{self.ords_schema}/meter_reads/"
            f"?site_id={self.site_id}"
            f"&from_dt={from_dt}"
            f"&to_dt={to_dt}"
        )

        log.info("oracle.fetch_start", site=self.site_id, url=url)

        async with httpx.AsyncClient(
            auth=(self.username, self.password) if self.username else None,
            timeout=self.timeout,
            verify=self.tls_verify,
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            records = data.get("items", data.get("rows", []))
            log.info("oracle.fetch_complete", site=self.site_id, records=len(records))
            return records

    def _parse(self, raw: dict[str, Any]) -> ERPReading:
        """Map Oracle ORDS meter_reads record → ERPReading."""
        ts_str = raw.get("reading_datetime", raw.get("READ_DATE", ""))
        try:
            ts = datetime.fromisoformat(ts_str.rstrip("Z")).replace(tzinfo=timezone.utc)
        except (ValueError, AttributeError):
            ts = datetime.now(timezone.utc)

        kwh_raw = raw.get("energy_kwh", raw.get("READ_VALUE", raw.get("QUANTITY", 0)))
        kwh = float(kwh_raw) if kwh_raw is not None else 0.0

        uom = str(raw.get("unit_of_measure", raw.get("UOM", "KWH"))).upper()
        if uom == "MWH":
            kwh *= 1000.0

        cost_raw = raw.get("cost_amount", raw.get("CHARGE_AMOUNT"))

        return ERPReading(
            site_id=self.site_id,
            meter_id=raw.get("meter_id", raw.get("METER_ID", f"{self.site_id}-ORACLE")),
            timestamp=ts,
            kwh=kwh,
            cost_centre=raw.get("cost_centre", raw.get("COST_CENTRE", "")),
            currency=raw.get("currency", raw.get("CURRENCY_CODE", "USD")),
            cost_amount=float(cost_raw) if cost_raw is not None else None,
            unit_of_measure=uom,
            erp_source=self.source_name,
            raw_payload=raw,
        )

    def _generate_mock_readings(self) -> list[dict[str, Any]]:
        """Synthetic Oracle ORDS-shaped records for demo/test."""
        import random
        random.seed(abs(hash(self.site_id + "oracle")) % 9999)
        now = datetime.now(timezone.utc)
        records = []
        for h in range(self.days_back * 24):
            ts = now - timedelta(hours=h)
            kwh = round(max(0, random.gauss(280, 35)), 3)
            records.append({
                "meter_id":        f"ORC-{self.site_id}-M01",
                "site_id":         self.site_id,
                "reading_datetime": ts.isoformat(),
                "energy_kwh":       kwh,
                "unit_of_measure":  "KWH",
                "cost_centre":      "CC-PLANT-002",
                "currency":         "USD",
                "cost_amount":      round(kwh * 0.18, 4),
            })
        return records
