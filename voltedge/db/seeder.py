"""
VoltEdge — 90-day energy reading seeder.

Generates realistic hourly readings for all four demo sites:
  LONDON-FACTORY-01      — factory: high weekday peaks, low weekends
  DUBAI-OFFICE-01        — office: business hours, Sun-Thu work week
  ROTTERDAM-WAREHOUSE-01 — warehouse: extended hours, steady load
  FRANKFURT-DC-01        — data centre: flat 24/7 baseload

Design decisions
----------------
- 90 days × 24 h × 4 sites = 8,640 readings per site, 34,560 total
- Timestamps are hourly UTC, ending at the current truncated hour
- Seeder is idempotent: checks row count before inserting
- Anomalies injected deliberately: 1-2 per site per month (~4 per 90 days)
  placed at realistic times (peak hours, Mondays, weather events)
- Seasonal variation: ±15% following a sine curve (winter higher for factory/WH,
  summer higher for office due to cooling load)
- Random noise: ±5% Gaussian per reading
"""
from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from voltedge.db import database as _dbmod


def _get_session():
    return _dbmod.Session()

from voltedge.db.models import EnergyReading
from voltedge.utils.logger import get_logger

log = get_logger(__name__)

DAYS = 90


# ── Site profiles ──────────────────────────────────────────────────────────────

PROFILES = {
    "LONDON-FACTORY-01": {
        "type":            "factory",
        "base_kwh":        450.0,
        "peak_multiplier": 2.8,
        "peak_start":      6,
        "peak_end":        20,
        "weekend_factor":  0.32,
        "voltage":         400.0,   # 3-phase
        "pf_mean":         0.91,
        "pf_std":          0.02,
        "temp_base":       10.0,    # cooler — UK
        "temp_amp":        8.0,
        "anomaly_hours":   [10, 14, 16],  # production peaks — likely anomaly times
        "work_week":       range(5),      # Mon-Fri
    },
    "DUBAI-OFFICE-01": {
        "type":            "office",
        "base_kwh":        85.0,
        "peak_multiplier": 3.8,
        "peak_start":      7,
        "peak_end":        19,
        "weekend_factor":  0.04,
        "voltage":         230.0,
        "pf_mean":         0.94,
        "pf_std":          0.015,
        "temp_base":       30.0,    # hot baseline — Dubai
        "temp_amp":        6.0,
        "anomaly_hours":   [13, 14, 15],  # post-lunch AC surge
        "work_week":       [6, 0, 1, 2, 3],  # Sun-Thu (0=Mon, 6=Sun)
    },
    "ROTTERDAM-WAREHOUSE-01": {
        "type":            "warehouse",
        "base_kwh":        140.0,
        "peak_multiplier": 1.75,
        "peak_start":      6,
        "peak_end":        22,
        "weekend_factor":  0.58,
        "voltage":         400.0,
        "pf_mean":         0.88,
        "pf_std":          0.025,
        "temp_base":       9.0,
        "temp_amp":        7.0,
        "anomaly_hours":   [7, 8, 18, 19],  # shift change
        "work_week":       range(5),
    },
    "FRANKFURT-DC-01": {
        "type":            "data_center",
        "base_kwh":        920.0,
        "peak_multiplier": 1.08,
        "peak_start":      0,
        "peak_end":        24,
        "weekend_factor":  0.97,
        "voltage":         400.0,
        "pf_mean":         0.97,
        "pf_std":          0.008,
        "temp_base":       21.0,    # controlled server room temp
        "temp_amp":        1.5,
        "anomaly_hours":   [2, 3, 4],   # overnight backup jobs
        "work_week":       range(7),    # 24/7
    },
}


# ── Demand curve ───────────────────────────────────────────────────────────────

def _kwh_for_hour(hour: int, weekday: int, profile: dict, rng: random.Random,
                  day_of_year: int, is_anomaly: bool) -> float:
    """Compute realistic kWh for one hour given site profile."""
    start, end = profile["peak_start"], profile["peak_end"]
    is_work_day = weekday in profile["work_week"]

    # Bell-curve demand during operational hours
    if start <= hour < end:
        mid   = (start + end) / 2.0
        sigma = (end - start) / 4.0
        bell  = math.exp(-0.5 * ((hour - mid) / sigma) ** 2)
        base  = profile["base_kwh"] * (1 + (profile["peak_multiplier"] - 1) * bell)
    else:
        base = profile["base_kwh"]

    if not is_work_day:
        base *= profile["weekend_factor"]

    # Seasonal variation ±15% (winter peak for factory/warehouse, summer for office)
    seasonal_phase = 80 if profile["type"] != "office" else 170
    seasonal = 1.0 + 0.15 * math.sin(2 * math.pi * (day_of_year - seasonal_phase) / 365)
    base *= seasonal

    # Gaussian noise ±5%
    base *= 1 + rng.gauss(0, 0.05)

    # Anomaly: 2.5–4× normal (realistic equipment fault / unplanned startup)
    if is_anomaly:
        base *= rng.uniform(2.5, 4.0)

    return max(0.0, round(base, 2))


def _temperature(hour: int, day_of_year: int, profile: dict,
                 rng: random.Random) -> float:
    """Ambient temperature with seasonal + diurnal variation."""
    seasonal = profile["temp_amp"] * math.sin(2 * math.pi * (day_of_year - 80) / 365)
    diurnal  = 3.0 * math.sin(2 * math.pi * (hour - 6) / 24)
    noise    = rng.gauss(0, 0.5)
    return round(profile["temp_base"] + seasonal + diurnal + noise, 1)


# ── Anomaly schedule ──────────────────────────────────────────────────────────

def _build_anomaly_set(site_id: str, timestamps: list[datetime],
                       profile: dict) -> set[str]:
    """
    Deterministically place ~4 anomalies per 90 days:
    one per ~3-week block, at realistic hours for that site.
    """
    rng = random.Random(hash(site_id) % 999983)
    anomaly_ts: set[str] = set()

    # Divide 90 days into 4 blocks; place one anomaly in each
    block_size = len(timestamps) // 4
    for block in range(4):
        start_idx = block * block_size
        end_idx   = min(start_idx + block_size, len(timestamps))
        # Pick a random day in the block
        day_idx = rng.randint(start_idx, end_idx - 1)
        ts      = timestamps[day_idx]
        # Snap to a realistic anomaly hour for this site
        anom_hour = rng.choice(profile["anomaly_hours"])
        anom_ts   = ts.replace(hour=anom_hour, minute=0, second=0, microsecond=0)
        anomaly_ts.add(anom_ts.isoformat())

    return anomaly_ts


# ── Main seeder ────────────────────────────────────────────────────────────────

def seed(force: bool = False) -> int:
    """
    Seed the database with 90 days of hourly readings for all sites.

    Args:
        force: Re-seed even if data already exists (wipes existing readings).

    Returns:
        Number of rows inserted.
    """
    _dbmod.init_db()

    with _dbmod.Session() as session:
        existing = session.query(EnergyReading).count()
        if existing > 0 and not force:
            log.info("seeder.skipped", existing_rows=existing)
            return 0

        if force and existing > 0:
            session.query(EnergyReading).delete()
            session.commit()
            log.info("seeder.wiped", rows=existing)

        log.info("seeder.start", sites=len(PROFILES), days=DAYS)

        # Build timestamp spine: 90 days of hourly UTC, ending at current hour
        now   = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        spine = [now - timedelta(hours=h) for h in range(DAYS * 24 - 1, -1, -1)]

        total_inserted = 0
        batch: list[EnergyReading] = []
        BATCH_SIZE = 500

        for site_id, profile in PROFILES.items():
            rng          = random.Random(abs(hash(site_id)) % 999983)
            anomaly_set  = _build_anomaly_set(site_id, spine, profile)

            for ts in spine:
                is_anomaly  = ts.isoformat() in anomaly_set
                day_of_year = ts.timetuple().tm_yday
                weekday     = ts.weekday()   # 0=Mon, 6=Sun

                kwh  = _kwh_for_hour(ts.hour, weekday, profile, rng,
                                     day_of_year, is_anomaly)
                temp = _temperature(ts.hour, day_of_year, profile, rng)
                volt = round(rng.gauss(profile["voltage"], profile["voltage"] * 0.01), 1)
                pf   = round(max(0.7, min(1.0,
                                rng.gauss(profile["pf_mean"], profile["pf_std"]))), 3)
                # Approximate current from kWh and voltage
                phases    = 3 if profile["voltage"] > 300 else 1
                current_a = round((kwh * 1000) / (volt * phases * pf + 1e-9), 1)

                batch.append(EnergyReading(
                    site_id       = site_id,
                    timestamp     = ts.isoformat(),
                    kwh           = kwh,
                    voltage_v     = volt,
                    current_a     = current_a,
                    power_factor  = pf,
                    temperature_c = temp,
                    is_anomaly    = int(is_anomaly),
                ))

                if len(batch) >= BATCH_SIZE:
                    session.bulk_save_objects(batch)
                    session.commit()
                    total_inserted += len(batch)
                    batch.clear()

            log.info("seeder.site_done", site=site_id, anomalies=len(anomaly_set))

        if batch:
            session.bulk_save_objects(batch)
            session.commit()
            total_inserted += len(batch)

        log.info("seeder.complete", total_rows=total_inserted)
        return total_inserted


if __name__ == "__main__":
    n = seed(force=True)
    print(f"Seeded {n} rows")
