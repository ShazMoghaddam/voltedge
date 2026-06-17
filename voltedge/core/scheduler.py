"""
VoltEdge — Auto-Retraining Scheduler

Uses APScheduler to trigger model retraining on a configurable cadence.
Runs alongside the ingestion pipeline in the same process.

Usage:
    from voltedge.core.scheduler import ModelScheduler
    from voltedge.storage.base import LocalStore

    store = LocalStore()
    scheduler = ModelScheduler(store=store, sites=["SITE-01", "SITE-02"])
    scheduler.start()          # Non-blocking: runs in background thread
    # ... application runs ...
    scheduler.stop()
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from voltedge.models.anomaly import AnomalyDetector
from voltedge.models.forecasting import DemandForecaster
from voltedge.storage.base import BaseStore
from voltedge.utils.logger import get_logger

log = get_logger(__name__)


class ModelScheduler:
    """
    Schedules periodic model retraining for all registered sites.

    Args:
        store:          Storage backend to read processed data from.
        sites:          List of site IDs to manage.
        retrain_hours:  How often to retrain (default: 168h = weekly).
        on_retrain:     Optional callback called after each retrain with
                        (site_id, model_name, metrics) for monitoring hooks.
    """

    def __init__(
        self,
        store: BaseStore,
        sites: list[str],
        retrain_hours: int = 168,
        artifact_dir: str = "data/models",
        on_retrain: Callable[[str, str, dict], None] | None = None,
    ) -> None:
        self._store = store
        self._sites = sites
        self._retrain_hours = retrain_hours
        self._artifact_dir = artifact_dir
        self._on_retrain = on_retrain
        self._scheduler = BackgroundScheduler(timezone="UTC")
        self._last_run: dict[str, datetime] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background scheduler. Non-blocking."""
        trigger = IntervalTrigger(hours=self._retrain_hours)
        self._scheduler.add_job(
            func=self._retrain_all,
            trigger=trigger,
            id="voltedge_model_retrain",
            replace_existing=True,
            name="VoltEdge Model Retraining",
        )
        self._scheduler.start()
        log.info(
            "scheduler.started",
            sites=self._sites,
            retrain_every_hours=self._retrain_hours,
        )

    def stop(self) -> None:
        """Gracefully stop the background scheduler."""
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            log.info("scheduler.stopped")

    def trigger_now(self) -> None:
        """Manually trigger an immediate retraining cycle (useful for testing)."""
        self._retrain_all()

    @property
    def is_running(self) -> bool:
        return self._scheduler.running

    # ── Internal ──────────────────────────────────────────────────────────────

    def _retrain_all(self) -> None:
        """Retrain all models for all sites. Called by the scheduler."""
        log.info("scheduler.retrain_cycle_start", sites=len(self._sites))
        for site_id in self._sites:
            self._retrain_site(site_id)

    def _retrain_site(self, site_id: str) -> None:
        """Retrain forecaster + anomaly detector for one site."""
        log.info("scheduler.site_retrain_start", site=site_id)

        df = self._store.read(site_id, layer="processed", days=90)
        if df.empty or len(df) < 48:
            log.warning("scheduler.insufficient_data", site=site_id, rows=len(df))
            return

        # Retrain forecaster
        try:
            forecaster = DemandForecaster(
                site_id=site_id, horizon_hours=24, artifact_dir=self._artifact_dir
            )
            metrics = forecaster.train(df)
            forecaster.save(metrics)
            log.info("scheduler.forecaster_retrained", site=site_id, mae=metrics.get("mae"))
            if self._on_retrain:
                self._on_retrain(site_id, "forecaster", metrics)
        except Exception as exc:
            log.error("scheduler.forecaster_retrain_failed", site=site_id, error=str(exc))

        # Retrain anomaly detector
        try:
            detector = AnomalyDetector(
                site_id=site_id, contamination=0.05, artifact_dir=self._artifact_dir
            )
            metrics = detector.train(df)
            detector.save(metrics)
            log.info("scheduler.anomaly_retrained", site=site_id,
                     rows=metrics.get("training_rows"))
            if self._on_retrain:
                self._on_retrain(site_id, "anomaly", metrics)
        except Exception as exc:
            log.error("scheduler.anomaly_retrain_failed", site=site_id, error=str(exc))

        self._last_run[site_id] = datetime.now(timezone.utc)
        log.info("scheduler.site_retrain_done", site=site_id)

    def status(self) -> dict:
        """Return a status dict for monitoring endpoints."""
        return {
            "running": self.is_running,
            "sites": self._sites,
            "retrain_interval_hours": self._retrain_hours,
            "last_run": {
                sid: ts.isoformat() for sid, ts in self._last_run.items()
            },
            "jobs": [
                {
                    "id": job.id,
                    "name": job.name,
                    "next_run": job.next_run_time.isoformat()
                    if job.next_run_time else None,
                }
                for job in self._scheduler.get_jobs()
            ],
        }
