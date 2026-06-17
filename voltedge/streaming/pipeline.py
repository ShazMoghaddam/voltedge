"""
VoltEdge Streaming — Real-time Event Pipeline

Consumes raw EnergyEvents from the broker, applies:
  1. Lightweight feature extraction (cyclic time features, lag approximation)
  2. Online anomaly scoring (cached IsolationForest per site)
  3. Alert threshold checking (score > threshold → alert event)
  4. Fan-out: publishes processed + anomaly events back to broker
             for WebSocket hub and storage consumers

Designed for sub-100ms end-to-end latency from IoT sensor to dashboard.

Architecture:
    Broker(raw topic)
        ↓
    StreamingPipeline._process_one()
        ├── feature_extract() → PROCESSED event → broker
        ├── anomaly_score()   → ANOMALY event   → broker (+ alert if threshold)
        └── metrics.inc()     → pipeline health counters

Unlike the batch pipeline, the streaming pipeline:
  - Never blocks: all I/O is async
  - Maintains a rolling window of recent readings per site (for lag features)
  - Uses pre-trained IsolationForest models loaded at startup
  - Gracefully degrades: if a model is not loaded, scoring is skipped
"""

from __future__ import annotations

import asyncio
import math
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any

from voltedge.streaming.broker import BaseBroker
from voltedge.streaming.events import EnergyEvent, EventType, pipeline_status_event
from voltedge.utils.logger import get_logger

log = get_logger(__name__)

# Rolling window depth per site (used to approximate lag features in real time)
WINDOW_DEPTH = 50


class StreamingPipeline:
    """
    Async streaming pipeline: raw → processed → anomaly → alert.

    Args:
        broker:              Event broker (InMemory or SQS).
        anomaly_threshold:   Score above which an alert event is emitted.
        alert_callback:      Async callable(EnergyEvent) invoked on each alert.
        sites:               Site IDs to subscribe to (empty = all via wildcard).
        model_artifact_dir:  Directory containing trained IsolationForest models.
    """

    def __init__(
        self,
        broker:             BaseBroker,
        anomaly_threshold:  float = 0.75,
        alert_callback:     Any   = None,
        sites:              list[str] | None = None,
        model_artifact_dir: str = "data/models",
    ) -> None:
        self._broker     = broker
        self._threshold  = anomaly_threshold
        self._alert_cb   = alert_callback
        self._sites      = sites or []
        self._model_dir  = model_artifact_dir

        # Rolling windows: {site_id: deque of recent kwh values}
        self._windows:  dict[str, deque] = defaultdict(lambda: deque(maxlen=WINDOW_DEPTH))

        # Loaded anomaly models: {site_id: AnomalyDetector}
        self._models:   dict[str, Any]   = {}

        # Health counters
        self._stats: dict[str, int] = defaultdict(int)
        self._running = False
        self._tasks:  list[asyncio.Task] = []

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self, site_ids: list[str] | None = None) -> None:
        """Start consuming events. Non-blocking — returns immediately."""
        sites = site_ids or self._sites
        if not sites:
            log.warning("pipeline.no_sites", msg="No sites specified; nothing to consume.")
            return

        self._running = True
        for site_id in sites:
            self._try_load_model(site_id)
            raw_topic = self._broker.topic_for(site_id, "raw")
            task = asyncio.create_task(
                self._consume_site(site_id, raw_topic),
                name=f"pipeline.{site_id}",
            )
            self._tasks.append(task)

        log.info("pipeline.started", sites=sites, tasks=len(self._tasks))

        # Publish startup status
        await self._broker.publish(
            "voltedge.platform.system",
            pipeline_status_event("streaming_pipeline", "running",
                                  {"sites": sites, "threshold": self._threshold}),
        )

    async def stop(self) -> None:
        """Gracefully stop all site consumers."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        log.info("pipeline.stopped")

    # ── Core processing ───────────────────────────────────────────────────────

    async def _consume_site(self, site_id: str, topic: str) -> None:
        """Endless consumer loop for one site's raw topic."""
        log.info("pipeline.site_consumer_start", site=site_id, topic=topic)
        try:
            async for raw_event in self._broker.subscribe(topic):
                if not self._running:
                    break
                try:
                    await self._process_one(raw_event)
                    self._stats["processed"] += 1
                except Exception as exc:
                    self._stats["errors"] += 1
                    log.error("pipeline.process_error",
                              site=site_id, error=str(exc), event_id=raw_event.event_id)
        except asyncio.CancelledError:
            log.info("pipeline.site_consumer_stopped", site=site_id)

    async def _process_one(self, raw: EnergyEvent) -> None:
        """
        Full processing chain for one raw event:
          1. Update rolling window
          2. Extract features
          3. Publish PROCESSED event
          4. Run anomaly scoring
          5. Publish ANOMALY event
          6. If above threshold → fire alert
        """
        site_id = raw.site_id
        self._windows[site_id].append(raw.kwh)

        # ── 1. Feature extraction ─────────────────────────────────────────────
        features = self._extract_features(raw)

        # ── 2. Publish PROCESSED ─────────────────────────────────────────────
        processed = raw.as_processed(features)
        proc_topic = self._broker.topic_for(site_id, "processed")
        await self._broker.publish(proc_topic, processed)

        # ── 3. Anomaly scoring ────────────────────────────────────────────────
        score, label = self._score(site_id, features)
        is_anomaly   = score >= self._threshold

        anom_event = raw.as_anomaly(score=score, label=label, is_anomaly=is_anomaly)
        anom_topic = self._broker.topic_for(site_id, "anomaly")
        await self._broker.publish(anom_topic, anom_event)

        # ── 4. Alert if threshold exceeded ────────────────────────────────────
        if is_anomaly:
            self._stats["anomalies"] += 1
            alert = raw.as_alert(
                title=f"Anomaly — {site_id}",
                severity="high" if score >= 0.90 else "medium",
                message=f"Score {score:.3f}: {label} at {raw.timestamp[:16]}",
            )
            alert_topic = self._broker.topic_for(site_id, "alert")
            await self._broker.publish(alert_topic, alert)

            if self._alert_cb:
                try:
                    await self._alert_cb(alert)
                except Exception as exc:
                    log.error("pipeline.alert_callback_error", error=str(exc))

    # ── Feature extraction (online, no pandas) ────────────────────────────────

    def _extract_features(self, event: EnergyEvent) -> dict[str, float]:
        """
        Lightweight real-time feature extraction.
        Mirrors EnergyTransformer but operates on single events + rolling window.
        No pandas — pure Python for sub-millisecond latency.
        """
        try:
            dt = datetime.fromisoformat(event.timestamp.rstrip("Z")).replace(tzinfo=timezone.utc)
        except (ValueError, AttributeError):
            dt = datetime.now(timezone.utc)

        h   = dt.hour
        dow = dt.weekday()
        mon = dt.month

        features: dict[str, float] = {
            # Cyclic time encoding
            "hour_sin":         math.sin(2 * math.pi * h   / 24),
            "hour_cos":         math.cos(2 * math.pi * h   / 24),
            "dow_sin":          math.sin(2 * math.pi * dow / 7),
            "dow_cos":          math.cos(2 * math.pi * dow / 7),
            "month_sin":        math.sin(2 * math.pi * mon / 12),
            "month_cos":        math.cos(2 * math.pi * mon / 12),
            "is_weekend":       float(dow >= 5),
            "is_business_hour": float(7 <= h < 19 and dow < 5),
            "kwh":              event.kwh,
        }

        # Rolling window statistics
        window = list(self._windows[event.site_id])
        if len(window) >= 2:
            features["kwh_lag_1"]     = window[-2]
            features["kwh_delta"]     = event.kwh - window[-2]
            features["kwh_pct_change"] = (
                (event.kwh - window[-2]) / (abs(window[-2]) + 1e-8) * 100
            )
        if len(window) >= 6:
            features["kwh_roll_mean_6"] = sum(window[-6:]) / 6
            features["kwh_roll_std_6"]  = _std(window[-6:])
        if len(window) >= 24:
            features["kwh_roll_mean_24"] = sum(window[-24:]) / 24
            features["kwh_roll_std_24"]  = _std(window[-24:])

        return features

    # ── Anomaly scoring ───────────────────────────────────────────────────────

    def _score(self, site_id: str, features: dict[str, float]) -> tuple[float, str]:
        """
        Score one feature dict.
        Returns (anomaly_score 0–1, label).
        Falls back to rule-based scoring if no model loaded.
        """
        model = self._models.get(site_id)
        if model:
            try:
                import numpy as np
                feat_names = list(features.keys())
                X = np.array([[features[k] for k in feat_names]])
                # Normalise using the model's scaler if available
                if hasattr(model, "_scaler") and model._scaler:
                    X = model._scaler.transform(X)
                raw_score = float(model._model.decision_function(X)[0])
                # Map decision function to 0–1
                score = 1 / (1 + math.exp(raw_score * 3))
                label = model._classify_anomaly(type("Row", (), features)())
                return round(score, 4), label
            except Exception:
                pass   # Fall through to rule-based

        # Rule-based fallback (no model loaded)
        return self._rule_based_score(features)

    @staticmethod
    def _rule_based_score(features: dict[str, float]) -> tuple[float, str]:
        """
        Simple rule-based anomaly detection as a fallback.
        Good enough for alerting without a trained model.
        """
        score = 0.0
        labels: list[str] = []

        pct = features.get("kwh_pct_change", 0.0)
        if abs(pct) > 200.0:
            score = max(score, 0.95)
            labels.append("extreme_spike" if pct > 0 else "extreme_drop")
        elif abs(pct) > 80.0:
            score = max(score, 0.80)
            labels.append("sudden_spike" if pct > 0 else "sudden_drop")

        std24 = features.get("kwh_roll_std_24", 1.0)
        if std24 < 0.05 and features.get("kwh", 1.0) > 0:
            score = max(score, 0.85)
            labels.append("flat_line_sensor")

        kwh  = features.get("kwh", 0.0)
        if "kwh_roll_mean_24" in features:
            mean  = features["kwh_roll_mean_24"]
            ratio = kwh / (mean + 1e-8)
            if ratio > 4.0:
                score = max(score, 0.90)
                labels.append("extreme_consumption")
            elif ratio < 0.05 and features.get("is_business_hour", 0) == 1.0:
                score = max(score, 0.78)
                labels.append("unexpected_low_load")

        return round(score, 4), (", ".join(labels) if labels else "normal")

    # ── Model loading ─────────────────────────────────────────────────────────

    def _try_load_model(self, site_id: str) -> None:
        """Attempt to load a pre-trained AnomalyDetector for this site."""
        try:
            from voltedge.models.anomaly import AnomalyDetector
            detector = AnomalyDetector(site_id=site_id,
                                       artifact_dir=self._model_dir)
            detector.load()
            self._models[site_id] = detector
            log.info("pipeline.model_loaded", site=site_id)
        except Exception:
            log.debug("pipeline.model_not_found_using_rules", site=site_id)

    # ── Health ────────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        return {
            "running":   self._running,
            "sites":     self._sites,
            "processed": self._stats["processed"],
            "anomalies": self._stats["anomalies"],
            "errors":    self._stats["errors"],
        }

    def window_size(self, site_id: str) -> int:
        return len(self._windows.get(site_id, []))


# ── Utility ───────────────────────────────────────────────────────────────────

def _std(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return math.sqrt(variance)
