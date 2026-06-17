"""
VoltEdge — Alert Webhook Dispatcher

Sends structured alerts to Slack and PagerDuty when anomalies are
detected, model retraining completes, or ESG thresholds are breached.

Designed for:
  - Slack Incoming Webhooks (free)
  - PagerDuty Events API v2 (free tier: 5 users)
  - Generic HTTP webhooks (custom integrations)

Usage:
    from voltedge.alerts.webhooks import AlertDispatcher, Alert, AlertSeverity

    dispatcher = AlertDispatcher(
        slack_webhook_url="https://hooks.slack.com/services/...",
        pagerduty_routing_key="abc123...",
    )

    alert = Alert(
        title="Anomaly Detected",
        message="LONDON-FACTORY-01: sudden spike at 14:00 UTC. Score: 0.94",
        severity=AlertSeverity.HIGH,
        site_id="LONDON-FACTORY-01",
        source="anomaly_detector",
        details={"anomaly_score": 0.94, "label": "sudden_spike"},
    )

    results = await dispatcher.send(alert)
    # results: {"slack": True, "pagerduty": True}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import httpx

from voltedge.utils.logger import get_logger

log = get_logger(__name__)


# ── Types ──────────────────────────────────────────────────────────────────────

class AlertSeverity(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


# Severity → Slack colour sidebar
_SLACK_COLORS = {
    AlertSeverity.LOW:      "#7fff6b",
    AlertSeverity.MEDIUM:   "#ffcc00",
    AlertSeverity.HIGH:     "#ff6b35",
    AlertSeverity.CRITICAL: "#ff0000",
}

# Severity → PagerDuty event severity
_PD_SEVERITY = {
    AlertSeverity.LOW:      "info",
    AlertSeverity.MEDIUM:   "warning",
    AlertSeverity.HIGH:     "error",
    AlertSeverity.CRITICAL: "critical",
}


@dataclass
class Alert:
    """
    Platform-agnostic alert structure. Serialised differently per channel.

    Args:
        title:      Short subject line (displayed in notification banner).
        message:    Full human-readable description.
        severity:   LOW / MEDIUM / HIGH / CRITICAL.
        site_id:    Source site identifier.
        source:     Which VoltEdge module raised the alert.
        details:    Arbitrary key-value dict (attached as fields in Slack, context in PD).
        dedup_key:  Optional deduplication key (used by PagerDuty to correlate alerts).
    """
    title:     str
    message:   str
    severity:  AlertSeverity = AlertSeverity.MEDIUM
    site_id:   str           = ""
    source:    str           = "voltedge"
    details:   dict[str, Any] = field(default_factory=dict)
    dedup_key: str | None    = None
    timestamp: datetime      = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass
class DispatchResult:
    """Result of a single webhook dispatch attempt."""
    channel:     str
    success:     bool
    status_code: int | None = None
    error:       str | None = None


# ── Dispatcher ────────────────────────────────────────────────────────────────

class AlertDispatcher:
    """
    Sends Alert objects to one or more webhook destinations.

    Args:
        slack_webhook_url:       Slack Incoming Webhook URL. None = disabled.
        pagerduty_routing_key:   PagerDuty Events API v2 routing key. None = disabled.
        generic_webhook_url:     Any HTTP endpoint receiving JSON POST. None = disabled.
        timeout_seconds:         HTTP request timeout per channel.
        dry_run:                 Log alerts but do not make real HTTP calls (for testing).
    """

    PAGERDUTY_EVENTS_URL = "https://events.pagerduty.com/v2/enqueue"

    def __init__(
        self,
        slack_webhook_url:     str | None = None,
        pagerduty_routing_key: str | None = None,
        generic_webhook_url:   str | None = None,
        timeout_seconds:       float = 10.0,
        dry_run:               bool = False,
    ) -> None:
        self.slack_url      = slack_webhook_url
        self.pd_routing_key = pagerduty_routing_key
        self.generic_url    = generic_webhook_url
        self.timeout        = timeout_seconds
        self.dry_run        = dry_run

    # ── Public ────────────────────────────────────────────────────────────────

    async def send(self, alert: Alert) -> list[DispatchResult]:
        """
        Dispatch the alert to all configured channels.
        Returns a list of DispatchResult — one per channel attempted.
        Failures in one channel do not block others.
        """
        results: list[DispatchResult] = []

        if self.slack_url:
            results.append(await self._send_slack(alert))
        if self.pd_routing_key:
            results.append(await self._send_pagerduty(alert))
        if self.generic_url:
            results.append(await self._send_generic(alert))

        if not results:
            log.warning("alerts.no_channels_configured")

        return results

    async def send_many(self, alerts: list[Alert]) -> dict[str, list[DispatchResult]]:
        """Send multiple alerts. Returns {alert.title: [results]}."""
        import asyncio
        tasks = [self.send(a) for a in alerts]
        all_results = await asyncio.gather(*tasks)
        return {a.title: r for a, r in zip(alerts, all_results)}

    def resolve(self, dedup_key: str, message: str = "Alert resolved") -> None:
        """
        Synchronous resolve — sends a PagerDuty RESOLVE event to close an incident.
        """
        if not self.pd_routing_key:
            return
        import asyncio
        payload = {
            "routing_key":  self.pd_routing_key,
            "event_action": "resolve",
            "dedup_key":    dedup_key,
            "payload": {"summary": message, "severity": "info",
                        "source": "voltedge"},
        }
        if not self.dry_run:
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    r = client.post(self.PAGERDUTY_EVENTS_URL, json=payload)
                    log.info("alerts.pd_resolved", dedup_key=dedup_key, status=r.status_code)
            except Exception as exc:
                log.error("alerts.pd_resolve_error", error=str(exc))

    # ── Channel implementations ───────────────────────────────────────────────

    async def _send_slack(self, alert: Alert) -> DispatchResult:
        payload = self._build_slack_payload(alert)
        return await self._http_post("slack", self.slack_url, payload)

    async def _send_pagerduty(self, alert: Alert) -> DispatchResult:
        payload = self._build_pagerduty_payload(alert)
        return await self._http_post("pagerduty", self.PAGERDUTY_EVENTS_URL, payload)

    async def _send_generic(self, alert: Alert) -> DispatchResult:
        payload = {
            "title":     alert.title,
            "message":   alert.message,
            "severity":  alert.severity.value,
            "site_id":   alert.site_id,
            "source":    alert.source,
            "details":   alert.details,
            "timestamp": alert.timestamp.isoformat(),
        }
        return await self._http_post("generic", self.generic_url, payload)

    async def _http_post(
        self, channel: str, url: str, payload: dict
    ) -> DispatchResult:
        log.info(
            "alerts.dispatch",
            channel=channel, title=payload.get("title", ""),
            dry_run=self.dry_run,
        )
        if self.dry_run:
            log.info("alerts.dry_run", channel=channel, payload=json.dumps(payload)[:200])
            return DispatchResult(channel=channel, success=True, status_code=200)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(url, json=payload)
                success = r.status_code < 300
                if not success:
                    log.warning("alerts.dispatch_failed",
                                channel=channel, status=r.status_code, body=r.text[:200])
                else:
                    log.info("alerts.dispatch_ok", channel=channel, status=r.status_code)
                return DispatchResult(channel=channel, success=success, status_code=r.status_code)
        except Exception as exc:
            log.error("alerts.dispatch_error", channel=channel, error=str(exc))
            return DispatchResult(channel=channel, success=False, error=str(exc))

    # ── Payload builders ──────────────────────────────────────────────────────

    def _build_slack_payload(self, alert: Alert) -> dict:
        fields = [
            {"title": "Site", "value": alert.site_id or "—", "short": True},
            {"title": "Source", "value": alert.source,      "short": True},
            {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
            {"title": "Time", "value": alert.timestamp.strftime("%Y-%m-%d %H:%M UTC"), "short": True},
        ]
        for k, v in (alert.details or {}).items():
            fields.append({"title": k.replace("_", " ").title(), "value": str(v), "short": True})

        return {
            "attachments": [{
                "color":    _SLACK_COLORS.get(alert.severity, "#888"),
                "title":    f"⚡ VoltEdge | {alert.title}",
                "text":     alert.message,
                "fields":   fields,
                "footer":   "VoltEdge Energy Intelligence",
                "ts":       int(alert.timestamp.timestamp()),
            }]
        }

    def _build_pagerduty_payload(self, alert: Alert) -> dict:
        payload: dict[str, Any] = {
            "routing_key":  self.pd_routing_key,
            "event_action": "trigger",
            "payload": {
                "summary":   alert.title,
                "severity":  _PD_SEVERITY.get(alert.severity, "warning"),
                "source":    alert.site_id or "voltedge",
                "component": alert.source,
                "custom_details": {
                    "message": alert.message,
                    **alert.details,
                },
                "timestamp": alert.timestamp.isoformat(),
            },
            "client":     "VoltEdge",
            "client_url": "http://localhost:8050",
        }
        if alert.dedup_key:
            payload["dedup_key"] = alert.dedup_key
        return payload


# ── Convenience factory functions ─────────────────────────────────────────────

def anomaly_alert(
    site_id: str,
    anomaly_score: float,
    label: str,
    timestamp: str,
) -> Alert:
    """Pre-built alert for anomaly detector events."""
    severity = (
        AlertSeverity.CRITICAL if anomaly_score >= 0.95 else
        AlertSeverity.HIGH     if anomaly_score >= 0.80 else
        AlertSeverity.MEDIUM
    )
    return Alert(
        title=f"Anomaly Detected — {site_id}",
        message=f"Anomaly scored {anomaly_score:.3f} ({label}) at {timestamp}.",
        severity=severity,
        site_id=site_id,
        source="anomaly_detector",
        details={
            "anomaly_score": anomaly_score,
            "label":         label,
            "timestamp":     timestamp,
        },
        dedup_key=f"anomaly-{site_id}-{timestamp}",
    )


def esg_threshold_alert(
    site_id: str,
    metric: str,
    actual: float,
    threshold: float,
    unit: str = "tCO₂e",
) -> Alert:
    """Pre-built alert when an ESG metric exceeds a configured threshold."""
    pct_over = (actual / threshold - 1) * 100
    severity = AlertSeverity.CRITICAL if pct_over > 20 else AlertSeverity.HIGH
    return Alert(
        title=f"ESG Threshold Exceeded — {site_id}",
        message=(
            f"{metric} is {actual:.3f} {unit}, which is {pct_over:.1f}% "
            f"above the threshold of {threshold:.3f} {unit}."
        ),
        severity=severity,
        site_id=site_id,
        source="esg_calculator",
        details={"metric": metric, "actual": actual, "threshold": threshold, "unit": unit},
        dedup_key=f"esg-{site_id}-{metric}",
    )


def retrain_complete_alert(site_id: str, model: str, metrics: dict) -> Alert:
    """Notification when a model finishes retraining."""
    return Alert(
        title=f"Model Retrained — {site_id} / {model}",
        message=f"{model} retrained for {site_id}. MAE: {metrics.get('mae', 'N/A')}",
        severity=AlertSeverity.LOW,
        site_id=site_id,
        source="scheduler",
        details=metrics,
    )
