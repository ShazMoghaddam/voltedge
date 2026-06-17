"""
VoltEdge — Database → DataFrame loader.

Replaces the SimulatedSiteConnector as the primary data source.
Falls back to the simulator only if the database has no data for the
requested site (e.g. on a fresh install before seeding completes).
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pandas as pd

from voltedge.db import database as _dbmod
from voltedge.db.models import EnergyReading
from voltedge.utils.logger import get_logger

log = get_logger(__name__)

SITE_TYPES = {
    "LONDON-FACTORY-01":      "factory",
    "DUBAI-OFFICE-01":        "office",
    "ROTTERDAM-WAREHOUSE-01": "warehouse",
    "FRANKFURT-DC-01":        "data_center",
}


def load_site_readings(site_id: str, hours: int = 168) -> pd.DataFrame:
    """
    Load the most recent `hours` of readings for `site_id` from SQLite.

    Returns a DataFrame with the exact columns the EnergyTransformer expects:
        timestamp, site_id, kwh, voltage_v, current_a, power_factor, temperature_c

    Falls back to the simulator if the database is empty for this site.
    """
    try:
        _dbmod.init_db()
        cutoff = (
            datetime.now(timezone.utc) - timedelta(hours=hours)
        ).isoformat()

        with _dbmod.Session() as session:
            rows = (
                session.query(EnergyReading)
                .filter(
                    EnergyReading.site_id == site_id,
                    EnergyReading.timestamp >= cutoff,
                )
                .order_by(EnergyReading.timestamp.asc())
                .all()
            )

        if not rows:
            log.warning("loader.no_db_data", site=site_id, hours=hours)
            return _fallback(site_id, hours)

        df = pd.DataFrame([{
            "timestamp":     r.timestamp,
            "site_id":       r.site_id,
            "kwh":           r.kwh,
            "voltage_v":     r.voltage_v,
            "current_a":     r.current_a,
            "power_factor":  r.power_factor,
            "temperature_c": r.temperature_c,
        } for r in rows])

        log.debug("loader.db_hit", site=site_id, rows=len(df))
        return df

    except Exception as exc:
        log.warning("loader.db_error", error=str(exc), site=site_id)
        return _fallback(site_id, hours)


def _fallback(site_id: str, hours: int) -> pd.DataFrame:
    """Simulator fallback — used only if DB is empty or unavailable."""
    from voltedge.ingestion.simulators import SimulatedSiteConnector
    from voltedge.processing.transformer import EnergyTransformer

    log.info("loader.fallback_to_simulator", site=site_id, hours=hours)
    stype  = SITE_TYPES.get(site_id, "factory")
    conn   = SimulatedSiteConnector(
        site_id, stype, hours=hours, seed=abs(hash(site_id)) % 9999
    )
    result = asyncio.run(conn.fetch())
    if result.data.empty:
        return pd.DataFrame()
    # Simulator data needs transformer too — return raw here, caller transforms
    return result.data
