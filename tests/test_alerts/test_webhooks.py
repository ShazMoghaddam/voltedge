"""
Tests for AlertDispatcher and Alert payload builders.
Uses dry_run=True and httpx.MockTransport to avoid real HTTP calls.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from voltedge.alerts.webhooks import (
    Alert, AlertDispatcher, AlertSeverity, DispatchResult,
    anomaly_alert, esg_threshold_alert, retrain_complete_alert,
)


TS = datetime(2026, 5, 14, 12, 0, 0, tzinfo=timezone.utc)


def _alert(**kwargs) -> Alert:
    defaults = dict(
        title="Test Alert", message="Test message",
        severity=AlertSeverity.HIGH, site_id="SITE-01",
        source="test", timestamp=TS,
    )
    return Alert(**{**defaults, **kwargs})


# ── Alert dataclass ───────────────────────────────────────────────────────────

def test_alert_defaults_severity_to_medium():
    a = Alert(title="T", message="M")
    assert a.severity == AlertSeverity.MEDIUM


def test_alert_has_timestamp():
    a = Alert(title="T", message="M")
    assert a.timestamp is not None
    assert a.timestamp.tzinfo is not None


def test_alert_details_defaults_to_empty():
    a = Alert(title="T", message="M")
    assert a.details == {}


# ── Dry run ───────────────────────────────────────────────────────────────────

def test_dry_run_slack_returns_success():
    dispatcher = AlertDispatcher(
        slack_webhook_url="https://hooks.slack.com/test", dry_run=True
    )
    results = asyncio.run(dispatcher.send(_alert()))
    assert len(results) == 1
    assert results[0].channel == "slack"
    assert results[0].success is True
    assert results[0].status_code == 200


def test_dry_run_pagerduty_returns_success():
    dispatcher = AlertDispatcher(pagerduty_routing_key="fake-key", dry_run=True)
    results = asyncio.run(dispatcher.send(_alert()))
    assert results[0].channel == "pagerduty"
    assert results[0].success is True


def test_dry_run_generic_returns_success():
    dispatcher = AlertDispatcher(generic_webhook_url="https://example.com/hook", dry_run=True)
    results = asyncio.run(dispatcher.send(_alert()))
    assert results[0].channel == "generic"
    assert results[0].success is True


def test_no_channels_returns_empty_list():
    dispatcher = AlertDispatcher(dry_run=True)
    results = asyncio.run(dispatcher.send(_alert()))
    assert results == []


def test_all_channels_dry_run():
    dispatcher = AlertDispatcher(
        slack_webhook_url="https://slack.test",
        pagerduty_routing_key="pd-key",
        generic_webhook_url="https://generic.test",
        dry_run=True,
    )
    results = asyncio.run(dispatcher.send(_alert()))
    assert len(results) == 3
    channels = {r.channel for r in results}
    assert channels == {"slack", "pagerduty", "generic"}
    assert all(r.success for r in results)


def test_send_many_dry_run():
    dispatcher = AlertDispatcher(slack_webhook_url="https://slack.test", dry_run=True)
    alerts = [_alert(title=f"Alert {i}") for i in range(3)]
    result_map = asyncio.run(dispatcher.send_many(alerts))
    assert len(result_map) == 3


# ── Payload builders ──────────────────────────────────────────────────────────

def test_slack_payload_has_attachments():
    dispatcher = AlertDispatcher()
    payload = dispatcher._build_slack_payload(_alert())
    assert "attachments" in payload
    assert len(payload["attachments"]) == 1


def test_slack_payload_includes_title():
    dispatcher = AlertDispatcher()
    payload = dispatcher._build_slack_payload(_alert(title="My Alert"))
    title = payload["attachments"][0]["title"]
    assert "My Alert" in title


def test_slack_payload_includes_message():
    dispatcher = AlertDispatcher()
    payload = dispatcher._build_slack_payload(_alert(message="My message"))
    assert payload["attachments"][0]["text"] == "My message"


def test_slack_payload_colour_for_critical():
    dispatcher = AlertDispatcher()
    payload = dispatcher._build_slack_payload(_alert(severity=AlertSeverity.CRITICAL))
    assert payload["attachments"][0]["color"] == "#ff0000"


def test_slack_payload_colour_for_low():
    dispatcher = AlertDispatcher()
    payload = dispatcher._build_slack_payload(_alert(severity=AlertSeverity.LOW))
    assert payload["attachments"][0]["color"] == "#7fff6b"


def test_slack_payload_includes_details_as_fields():
    dispatcher = AlertDispatcher()
    a = _alert(details={"anomaly_score": 0.95})
    payload = dispatcher._build_slack_payload(a)
    fields = payload["attachments"][0]["fields"]
    detail_titles = [f["title"] for f in fields]
    assert "Anomaly Score" in detail_titles


def test_pagerduty_payload_has_correct_structure():
    dispatcher = AlertDispatcher(pagerduty_routing_key="test-key")
    payload = dispatcher._build_pagerduty_payload(_alert())
    assert payload["routing_key"] == "test-key"
    assert payload["event_action"] == "trigger"
    assert "payload" in payload
    assert "summary" in payload["payload"]


def test_pagerduty_payload_severity_critical():
    dispatcher = AlertDispatcher(pagerduty_routing_key="key")
    payload = dispatcher._build_pagerduty_payload(_alert(severity=AlertSeverity.CRITICAL))
    assert payload["payload"]["severity"] == "critical"


def test_pagerduty_payload_includes_dedup_key():
    dispatcher = AlertDispatcher(pagerduty_routing_key="key")
    a = _alert(dedup_key="my-dedup-key")
    payload = dispatcher._build_pagerduty_payload(a)
    assert payload["dedup_key"] == "my-dedup-key"


# ── Mock HTTP transport ───────────────────────────────────────────────────────

def test_real_dispatch_with_mock_transport():
    """Simulate a real HTTP call using httpx mock transport."""
    send_calls: list = []

    async def mock_send(request, **kwargs):
        send_calls.append(request)
        return httpx.Response(200, text="ok")

    transport = httpx.MockTransport(handler=mock_send)

    dispatcher = AlertDispatcher(slack_webhook_url="https://hooks.slack.com/test")

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        results = asyncio.run(dispatcher.send(_alert()))
        assert results[0].success is True
        assert results[0].status_code == 200


def test_dispatch_handles_connection_error():
    """AlertDispatcher should catch network errors and return success=False."""
    dispatcher = AlertDispatcher(slack_webhook_url="https://hooks.slack.com/test")

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_client_cls.return_value = mock_client

        results = asyncio.run(dispatcher.send(_alert()))
        assert results[0].success is False
        assert results[0].error is not None


# ── Factory functions ─────────────────────────────────────────────────────────

def test_anomaly_alert_critical_when_high_score():
    a = anomaly_alert("SITE-01", 0.97, "sudden_spike", "2026-05-14T12:00:00Z")
    assert a.severity == AlertSeverity.CRITICAL
    assert "SITE-01" in a.title
    assert a.dedup_key is not None


def test_anomaly_alert_high_when_medium_score():
    a = anomaly_alert("SITE-01", 0.85, "flat_line", "2026-05-14T12:00:00Z")
    assert a.severity == AlertSeverity.HIGH


def test_anomaly_alert_medium_when_low_score():
    a = anomaly_alert("SITE-01", 0.72, "anomaly", "2026-05-14T12:00:00Z")
    assert a.severity == AlertSeverity.MEDIUM


def test_esg_threshold_alert_over_20pct_is_critical():
    a = esg_threshold_alert("SITE-01", "scope_2_location", actual=1.25, threshold=1.0)
    assert a.severity == AlertSeverity.CRITICAL


def test_esg_threshold_alert_has_correct_fields():
    a = esg_threshold_alert("SITE-01", "scope_2", 1.1, 1.0, "tCO2e")
    assert a.details["actual"] == 1.1
    assert a.details["threshold"] == 1.0


def test_retrain_complete_alert_is_low_severity():
    a = retrain_complete_alert("SITE-01", "forecaster", {"mae": 12.5})
    assert a.severity == AlertSeverity.LOW
    assert a.details["mae"] == 12.5


def test_all_severity_levels_valid():
    for sev in AlertSeverity:
        a = _alert(severity=sev)
        dispatcher = AlertDispatcher(slack_webhook_url="https://test", dry_run=True)
        payload = dispatcher._build_slack_payload(a)
        assert "attachments" in payload
