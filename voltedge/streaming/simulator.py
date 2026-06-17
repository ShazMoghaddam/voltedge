"""
VoltEdge Streaming — Live Event Simulator

Generates a continuous stream of realistic EnergyEvents at a configurable
rate, publishing them to the broker's raw topics. Replaces the 60-second
batch simulator with sub-second event generation for v5.0 real-time mode.

Designed for:
  - Development without real IoT hardware
  - Load testing the WebSocket hub
  - Demo environments (Shell/BP pilots)

Features:
  - Configurable events-per-second per site
  - Realistic demand profiles (factory / office / warehouse / data_center)
  - Synthetic anomaly injection at configurable probability
  - Graceful shutdown via asyncio.Event

Usage:
    broker = InMemoryBroker()
    sim = LiveSimulator(broker, sites=["LONDON-01", "DUBAI-01"], events_per_second=1.0)
    await sim.start()
    await asyncio.sleep(60)
    await sim.stop()
"""

from __future__ import annotations

import asyncio
import math
import random
from datetime import datetime, timezone
from typing import Any

from voltedge.streaming.broker import BaseBroker
from voltedge.streaming.events import EnergyEvent, EventType
from voltedge.utils.logger import get_logger

log = get_logger(__name__)

SITE_PROFILES = {
    "factory":     {"base_kwh": 400, "amplitude": 120, "noise": 25},
    "office":      {"base_kwh": 120, "amplitude":  60, "noise": 10},
    "warehouse":   {"base_kwh": 200, "amplitude":  80, "noise": 15},
    "data_center": {"base_kwh": 350, "amplitude":  30, "noise":  8},
}


class LiveSimulator:
    """
    Pushes synthetic EnergyEvents to the broker at a steady tick rate.

    Args:
        broker:            The event broker to publish to.
        sites:             List of site_ids to simulate.
        site_types:        Optional dict mapping site_id → site_type.
        events_per_second: Tick rate per site (default 1 = one reading/sec).
        anomaly_prob:      Probability of injecting an anomaly spike (default 2%).
        seed:              Random seed for reproducibility (None = random).
    """

    def __init__(
        self,
        broker:            BaseBroker,
        sites:             list[str],
        site_types:        dict[str, str] | None = None,
        events_per_second: float = 1.0,
        anomaly_prob:      float = 0.02,
        seed:              int | None = None,
    ) -> None:
        self._broker   = broker
        self._sites    = sites
        self._types    = site_types or {}
        self._eps      = max(0.01, events_per_second)
        self._anom_p   = anomaly_prob
        self._rng      = random.Random(seed)
        self._running  = False
        self._tasks:   list[asyncio.Task] = []
        self._counts:  dict[str, int]     = {s: 0 for s in sites}
        self._stop_ev  = asyncio.Event()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._running = True
        self._stop_ev.clear()
        for site_id in self._sites:
            task = asyncio.create_task(
                self._site_loop(site_id),
                name=f"sim.{site_id}",
            )
            self._tasks.append(task)
        log.info("simulator.started", sites=self._sites, eps=self._eps)

    async def stop(self) -> None:
        self._running = False
        self._stop_ev.set()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        log.info("simulator.stopped", counts=self._counts)

    # ── Site loop ─────────────────────────────────────────────────────────────

    async def _site_loop(self, site_id: str) -> None:
        interval = 1.0 / self._eps
        stype    = self._types.get(site_id, "factory")
        profile  = SITE_PROFILES.get(stype, SITE_PROFILES["factory"])

        log.debug("simulator.site_loop_start", site=site_id, type=stype)
        try:
            while self._running:
                event = self._generate_event(site_id, profile, self._counts[site_id])
                topic = self._broker.topic_for(site_id, "raw")
                await self._broker.publish(topic, event)
                self._counts[site_id] += 1
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            log.debug("simulator.site_loop_stopped", site=site_id)

    def _generate_event(
        self,
        site_id:  str,
        profile:  dict[str, float],
        sequence: int,
    ) -> EnergyEvent:
        """Generate one synthetic energy reading."""
        now = datetime.now(timezone.utc)
        h   = now.hour + now.minute / 60.0

        # Demand profile: sinusoidal daily curve
        demand_factor = 0.7 + 0.3 * math.sin(math.pi * (h - 6) / 12)
        base_kwh = profile["base_kwh"] * demand_factor
        noise    = self._rng.gauss(0, profile["noise"])
        kwh      = max(0.0, base_kwh + noise)

        anomaly_injected = False
        if self._rng.random() < self._anom_p:
            kwh *= self._rng.uniform(3.0, 8.0)    # Spike: 3–8× normal
            anomaly_injected = True

        return EnergyEvent(
            event_type=EventType.RAW,
            site_id=site_id,
            sensor_id=f"{site_id}-MAIN-METER",
            timestamp=now.isoformat(),
            kwh=round(kwh, 3),
            sequence=sequence,
            payload={"injected_anomaly": anomaly_injected, "profile": profile},
        )

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        return {
            "running":           self._running,
            "sites":             self._sites,
            "events_per_second": self._eps,
            "counts":            dict(self._counts),
            "total":             sum(self._counts.values()),
        }
