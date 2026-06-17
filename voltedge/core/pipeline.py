"""
VoltEdge — Core pipeline orchestrator.

Wires together: Ingestion → Processing → Storage → (ML trigger).
Runs on a schedule via APScheduler or on-demand via the CLI / API.

Usage:
    import asyncio
    from voltedge.core.pipeline import EnergyPipeline
    from voltedge.ingestion.simulators import SimulatedSiteConnector
    from voltedge.storage.base import LocalStore

    store = LocalStore()
    pipeline = EnergyPipeline(store=store)
    pipeline.register_connector("SITE-01", SimulatedSiteConnector("SITE-01", "factory"))
    asyncio.run(pipeline.run_all())
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from voltedge.ingestion.base import BaseConnector, IngestionResult
from voltedge.processing.transformer import EnergyTransformer
from voltedge.storage.base import BaseStore
from voltedge.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class PipelineRunSummary:
    """Returned after every pipeline run for observability / alerting."""

    started_at: datetime
    finished_at: datetime | None = None
    sites_processed: int = 0
    total_records: int = 0
    total_errors: int = 0
    site_results: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        if self.finished_at is None:
            return 0.0
        return (self.finished_at - self.started_at).total_seconds()

    @property
    def success(self) -> bool:
        return self.total_errors == 0


class EnergyPipeline:
    """
    Orchestrates the full data pipeline for all registered sites.

    Stages:
      1. Fetch   — call each connector's async `fetch()` method concurrently
      2. Process — clean, feature-engineer the raw DataFrame
      3. Store   — write raw + processed layers to the store
      4. Notify  — emit structured log events (hook for alerting later)
    """

    def __init__(
        self,
        store: BaseStore,
        transformer: EnergyTransformer | None = None,
        concurrency: int = 5,
    ) -> None:
        self._store = store
        self._transformer = transformer or EnergyTransformer()
        self._connectors: dict[str, BaseConnector] = {}
        self._semaphore = asyncio.Semaphore(concurrency)

    # ── Registration ──────────────────────────────────────────────────────────

    def register_connector(self, site_id: str, connector: BaseConnector) -> None:
        self._connectors[site_id] = connector
        log.info("pipeline.connector_registered", site=site_id, source=connector.source_name)

    def register_many(self, connectors: dict[str, BaseConnector]) -> None:
        for site_id, connector in connectors.items():
            self.register_connector(site_id, connector)

    # ── Execution ─────────────────────────────────────────────────────────────

    async def run_all(self) -> PipelineRunSummary:
        """Fetch and process all registered sites concurrently."""
        summary = PipelineRunSummary(started_at=datetime.now(timezone.utc))
        log.info("pipeline.run_start", sites=list(self._connectors.keys()))

        tasks = [
            self._run_site(site_id, connector)
            for site_id, connector in self._connectors.items()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for site_id, result in zip(self._connectors.keys(), results):
            if isinstance(result, Exception):
                log.error("pipeline.site_failed", site=site_id, error=str(result))
                summary.total_errors += 1
                summary.site_results[site_id] = {"status": "error", "error": str(result)}
            else:
                summary.sites_processed += 1
                summary.total_records += result.get("records_valid", 0)
                summary.total_errors += result.get("errors", 0)
                summary.site_results[site_id] = result

        summary.finished_at = datetime.now(timezone.utc)
        log.info(
            "pipeline.run_complete",
            duration_s=round(summary.duration_seconds, 2),
            sites=summary.sites_processed,
            records=summary.total_records,
            errors=summary.total_errors,
        )
        return summary

    async def run_site(self, site_id: str) -> dict[str, Any]:
        """Run the pipeline for a single site on demand."""
        connector = self._connectors.get(site_id)
        if connector is None:
            raise ValueError(f"No connector registered for site: {site_id}")
        return await self._run_site(site_id, connector)

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _run_site(self, site_id: str, connector: BaseConnector) -> dict[str, Any]:
        async with self._semaphore:
            log.info("pipeline.site_start", site=site_id)

            # Stage 1: Ingest
            result: IngestionResult = await connector.fetch()
            log.info(
                "pipeline.ingested",
                site=site_id,
                records=result.records_valid,
                invalid=result.records_invalid,
                success_rate=round(result.success_rate, 3),
            )

            if result.data.empty:
                return {"status": "no_data", "records_valid": 0, "errors": len(result.errors)}

            # Stage 2: Store raw
            raw_path = self._store.write(result.data, site_id=site_id, layer="raw")

            # Stage 3: Transform
            processed_df = self._transformer.transform(result.data)

            # Stage 4: Store processed
            proc_path = self._store.write(processed_df, site_id=site_id, layer="processed")

            return {
                "status": "ok",
                "records_valid": result.records_valid,
                "records_invalid": result.records_invalid,
                "errors": len(result.errors),
                "raw_path": raw_path,
                "processed_path": proc_path,
            }
