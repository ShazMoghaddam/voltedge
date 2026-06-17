"""
VoltEdge — Deterministic data simulator.

Generates realistic energy consumption data for development, testing,
and stakeholder demos without requiring live sensor connections.

Features:
  - Weekday / weekend demand curves
  - Site-specific base loads (factory vs office vs warehouse)
  - Gaussian noise + occasional anomaly spikes
  - Seasonal temperature variation
"""

from __future__ import annotations

import asyncio
import math
import random
from datetime import datetime, timedelta, timezone
from typing import Any

from voltedge.ingestion.base import BaseConnector

# ── Site profiles ──────────────────────────────────────────────────────────────

SITE_PROFILES: dict[str, dict[str, Any]] = {
    "factory": {
        "base_kwh": 450.0,
        "peak_multiplier": 2.8,
        "peak_hours": (6, 20),   # 06:00 – 20:00
        "weekend_factor": 0.35,
    },
    "office": {
        "base_kwh": 80.0,
        "peak_multiplier": 3.5,
        "peak_hours": (8, 18),
        "weekend_factor": 0.05,
    },
    "warehouse": {
        "base_kwh": 120.0,
        "peak_multiplier": 1.8,
        "peak_hours": (7, 22),
        "weekend_factor": 0.6,
    },
    "data_center": {
        "base_kwh": 900.0,
        "peak_multiplier": 1.1,  # Relatively flat load
        "peak_hours": (0, 24),
        "weekend_factor": 0.95,
    },
}


def _demand_curve(hour: int, profile: dict[str, Any]) -> float:
    """Smooth sinusoidal demand curve scaled by site profile."""
    start, end = profile["peak_hours"]
    if start <= hour < end:
        # Bell curve peaking at mid-point of working hours
        mid = (start + end) / 2
        sigma = (end - start) / 4
        bell = math.exp(-0.5 * ((hour - mid) / sigma) ** 2)
        return profile["base_kwh"] * (1 + (profile["peak_multiplier"] - 1) * bell)
    return profile["base_kwh"]


class SimulatedSiteConnector(BaseConnector):
    """
    Generates a batch of synthetic hourly energy readings.

    Args:
        site_id:       Unique site identifier (e.g. "SITE-LONDON-01")
        site_type:     One of factory | office | warehouse | data_center
        hours:         Number of hours of history to generate (default: 24)
        seed:          Random seed for reproducibility
        anomaly_prob:  Probability of injecting an anomalous reading (0–1)
    """

    source_name = "simulator"

    def __init__(
        self,
        site_id: str,
        site_type: str = "factory",
        hours: int = 24,
        seed: int | None = None,
        anomaly_prob: float = 0.02,
    ) -> None:
        super().__init__(site_id=site_id)
        self.profile = SITE_PROFILES.get(site_type, SITE_PROFILES["factory"])
        self.hours = hours
        self.anomaly_prob = anomaly_prob
        self._rng = random.Random(seed)

    async def _fetch_raw(self) -> list[dict[str, Any]]:
        await asyncio.sleep(0)  # Yield to event loop — real connectors would I/O here
        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        records: list[dict[str, Any]] = []

        for h in range(self.hours, 0, -1):
            ts = now - timedelta(hours=h)
            is_weekend = ts.weekday() >= 5
            hour = ts.hour

            base = _demand_curve(hour, self.profile)
            if is_weekend:
                base *= self.profile["weekend_factor"]

            # Seasonal variation: ~±15% based on day-of-year
            day_of_year = ts.timetuple().tm_yday
            seasonal = 1 + 0.15 * math.sin(2 * math.pi * (day_of_year - 80) / 365)

            kwh = base * seasonal
            kwh *= 1 + self._rng.gauss(0, 0.05)  # ±5% noise

            # Inject anomaly
            is_anomaly = self._rng.random() < self.anomaly_prob
            if is_anomaly:
                kwh *= self._rng.uniform(2.5, 4.0)

            records.append({
                "ts": ts.isoformat(),
                "sensor": f"{self.site_id}-MAIN",
                "energy_kwh": max(0.0, round(kwh, 2)),
                "voltage": round(self._rng.gauss(230, 2), 1),
                "current": round(kwh / 0.23 / 1000 * 1000, 1),  # Approx
                "pf": round(self._rng.gauss(0.92, 0.02), 3),
                "temp_c": round(20 + 5 * math.sin(2 * math.pi * hour / 24), 1),
                "anomaly": is_anomaly,
            })

        return records

    def _parse(self, raw: dict[str, Any]) -> dict[str, Any]:
        return {
            "sensor_id": raw["sensor"],
            "timestamp": raw["ts"],
            "kwh": raw["energy_kwh"],
            "voltage_v": raw["voltage"],
            "current_a": raw["current"],
            "power_factor": raw["pf"],
            "temperature_c": raw["temp_c"],
            "metadata": {"injected_anomaly": raw.get("anomaly", False)},
        }
