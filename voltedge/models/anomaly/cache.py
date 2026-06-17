"""
VoltEdge — Anomaly model cache.

Keeps trained AnomalyDetector instances in memory so the dashboard
never retrains on every callback. Design decisions:

  - Keyed by (site_id, hours_window) — different time windows produce
    different feature distributions, so they need separate models.
  - TTL of 30 minutes — models auto-expire so they stay current as new
    DB readings accumulate (seeder writes are infrequent; 30 min is safe).
  - Thread-safe — the pre-warmer and the callback thread share the cache.
  - MIN_ROWS lowered to 48 (2 days of hourly data) — with the SQLite
    database we always have 90 days, but this guards against edge cases.
  - Graceful degradation — on cache miss AND training failure, returns
    (None, None) so the caller can show a helpful placeholder rather
    than a blank chart.
"""
from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from voltedge.models.anomaly import AnomalyDetector
from voltedge.utils.logger import get_logger

log = get_logger(__name__)

MIN_ROWS     = 48           # 2 days of hourly data — minimum to train reliably
CACHE_TTL    = timedelta(minutes=30)
PREWARM_HOURS = 168         # 7-day window used for pre-warming

# ── Cache store ───────────────────────────────────────────────────────────────
# { (site_id, hours): (detector, predictions_df, cached_at) }
_cache: dict[tuple, tuple[AnomalyDetector, pd.DataFrame, datetime]] = {}
_lock  = threading.Lock()


# ── Public API ─────────────────────────────────────────────────────────────────

def get_anomaly_results(
    site_id: str,
    df: pd.DataFrame,
    hours: int = 168,
    contamination: float = 0.05,
) -> tuple[Optional[AnomalyDetector], Optional[pd.DataFrame]]:
    """
    Return (detector, predictions_df) for the given site and data.

    Hits the in-memory cache when possible; trains and caches when not.
    Returns (None, None) if training is impossible (too few rows).
    """
    key = (site_id, hours)

    # Cache hit — return if within TTL
    with _lock:
        if key in _cache:
            detector, anom_df, cached_at = _cache[key]
            age = datetime.now(timezone.utc) - cached_at
            if age < CACHE_TTL:
                log.debug("anomaly.cache_hit", site=site_id, age_s=int(age.total_seconds()))
                # Re-predict on the current df (data may have changed even if model hasn't)
                try:
                    fresh_predictions = detector.predict(df)
                    return detector, fresh_predictions
                except Exception:
                    pass  # Fall through to retrain

    # Cache miss or stale — train a fresh model
    return _train_and_cache(site_id, df, hours, contamination, key)


def _train_and_cache(
    site_id: str,
    df: pd.DataFrame,
    hours: int,
    contamination: float,
    key: tuple,
) -> tuple[Optional[AnomalyDetector], Optional[pd.DataFrame]]:
    """Train a new model, cache it, and return results."""
    if len(df) < MIN_ROWS:
        log.warning(
            "anomaly.insufficient_data",
            site=site_id,
            rows=len(df),
            min_rows=MIN_ROWS,
        )
        return None, None

    try:
        detector = AnomalyDetector(site_id=site_id, contamination=contamination)
        detector.train(df)
        anom_df  = detector.predict(df)

        with _lock:
            _cache[key] = (detector, anom_df, datetime.now(timezone.utc))

        log.info("anomaly.cache_stored", site=site_id, hours=hours, rows=len(df))
        return detector, anom_df

    except Exception as exc:
        log.warning("anomaly.train_failed", site=site_id, error=str(exc))
        return None, None


# ── Pre-warmer ─────────────────────────────────────────────────────────────────

SITES = [
    "LONDON-FACTORY-01",
    "DUBAI-OFFICE-01",
    "ROTTERDAM-WAREHOUSE-01",
    "FRANKFURT-DC-01",
]


def prewarm_all() -> None:
    """
    Train anomaly models for all four sites on the 7-day window.
    Called in a background thread on app startup so the first dashboard
    load hits a warm cache instead of waiting for training.
    """
    log.info("anomaly.prewarm_start", sites=len(SITES))
    from voltedge.db.loader import load_site_readings
    from voltedge.processing.transformer import EnergyTransformer

    for site_id in SITES:
        try:
            raw = load_site_readings(site_id, hours=PREWARM_HOURS)
            if raw.empty:
                log.warning("anomaly.prewarm_no_data", site=site_id)
                continue
            df = EnergyTransformer().transform(raw)
            get_anomaly_results(site_id, df, hours=PREWARM_HOURS)
            log.info("anomaly.prewarm_done", site=site_id)
        except Exception as exc:
            log.warning("anomaly.prewarm_failed", site=site_id, error=str(exc))

    log.info("anomaly.prewarm_complete")


def invalidate(site_id: str | None = None) -> None:
    """Clear cache for a specific site, or all sites if site_id is None."""
    with _lock:
        if site_id:
            keys_to_drop = [k for k in _cache if k[0] == site_id]
        else:
            keys_to_drop = list(_cache.keys())
        for k in keys_to_drop:
            del _cache[k]
    log.info("anomaly.cache_invalidated", site=site_id or "all")


def cache_stats() -> dict:
    """Return a snapshot of cache contents for debugging."""
    with _lock:
        return {
            str(k): {
                "rows": len(v[1]) if v[1] is not None else 0,
                "age_s": int((datetime.now(timezone.utc) - v[2]).total_seconds()),
                "ttl_remaining_s": max(
                    0, int((CACHE_TTL - (datetime.now(timezone.utc) - v[2])).total_seconds())
                ),
            }
            for k, v in _cache.items()
        }
