"""Tests for demand response events, dispatcher, and grid connectors."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import pytest

from voltedge.demand_response.events import (
    DREvent, DREventStatus, DREventType, SiteResponse, SiteResponseStatus,
)
from voltedge.demand_response.dispatcher import DemandResponseDispatcher, SHEDDABLE_ASSETS
from voltedge.demand_response.grid_connectors import MockGridConnector

NOW = datetime.now(timezone.utc)


def _event(**kwargs) -> DREvent:
    defaults = dict(
        event_id="EVT-001",
        grid_operator="MOCK-NESO",
        event_type=DREventType.ECONOMIC,
        status=DREventStatus.UPCOMING,
        start_time=NOW + timedelta(hours=1),
        end_time=NOW + timedelta(hours=2),
        notification_time=NOW,
        target_reduction_kw=100.0,
        clearing_price_kwh=0.45,
        currency="GBP",
        min_response_kw=10.0,
        mandatory=False,
    )
    return DREvent(**{**defaults, **kwargs})


# ── DREvent ───────────────────────────────────────────────────────────────────

def test_event_duration_minutes():
    evt = _event()
    assert evt.duration_minutes == pytest.approx(60.0)

def test_event_potential_revenue():
    # 100 kW × 1h × £0.45/kWh = £45
    evt = _event(target_reduction_kw=100.0, clearing_price_kwh=0.45)
    assert evt.potential_revenue == pytest.approx(45.0)

def test_event_is_active_false_for_future():
    evt = _event(start_time=NOW + timedelta(hours=5))
    assert evt.is_active is False

def test_event_is_active_true_for_current():
    evt = _event(
        start_time=NOW - timedelta(minutes=30),
        end_time=NOW + timedelta(minutes=30),
    )
    assert evt.is_active is True

def test_event_mandatory_emergency():
    evt = _event(event_type=DREventType.EMERGENCY, mandatory=True)
    assert evt.mandatory is True

def test_event_to_dict_keys():
    d = _event().to_dict()
    for k in ("event_id","event_type","duration_minutes","target_reduction_kw","potential_revenue"):
        assert k in d


# ── SiteResponse ──────────────────────────────────────────────────────────────

def test_site_response_reduction_pct():
    r = SiteResponse(
        event_id="E1", site_id="S1",
        baseline_kw=100.0, response_kw=70.0,
        actual_reduction_kw=30.0,
    )
    assert r.reduction_pct == pytest.approx(30.0)

def test_site_response_net_revenue():
    r = SiteResponse(revenue_earned=50.0, penalty_charged=5.0)
    assert r.net_revenue == pytest.approx(45.0)

def test_site_response_to_dict_keys():
    d = SiteResponse().to_dict()
    for k in ("response_id","status","revenue_earned","net_revenue","reduction_pct"):
        assert k in d


# ── DemandResponseDispatcher ──────────────────────────────────────────────────

@pytest.fixture
def dispatcher():
    return DemandResponseDispatcher(auto_respond=True, min_revenue_per_event=10.0)


def test_evaluate_event_recommends_opt_in(dispatcher):
    evt = _event(target_reduction_kw=50.0, clearing_price_kwh=0.45)
    result = dispatcher.evaluate_event(evt, "SITE-01")
    assert result["recommendation"] == "opt_in"
    assert result["expected_revenue"] > 0


def test_evaluate_event_has_shed_plan(dispatcher):
    evt = _event(target_reduction_kw=30.0)
    result = dispatcher.evaluate_event(evt, "SITE-01")
    assert "shed_plan" in result
    assert len(result["shed_plan"]) > 0


def test_evaluate_mandatory_event_always_opt_in(dispatcher):
    evt = _event(target_reduction_kw=50.0, mandatory=True, clearing_price_kwh=0.01)
    result = dispatcher.evaluate_event(evt, "SITE-01")
    assert result["recommendation"] == "opt_in"


def test_shed_plan_priorities_ascending(dispatcher):
    evt = _event(target_reduction_kw=200.0)
    result = dispatcher.evaluate_event(evt, "SITE-01")
    priorities = [a["priority"] for a in result["shed_plan"]]
    assert priorities == sorted(priorities)


def test_shed_plan_does_not_exceed_target(dispatcher):
    target = 30.0
    evt = _event(target_reduction_kw=target)
    result = dispatcher.evaluate_event(evt, "SITE-01")
    total_kw = sum(a["target_kw"] for a in result["shed_plan"])
    assert total_kw <= target + 0.5   # allow rounding


def test_opt_in_creates_response(dispatcher):
    evt = _event()
    response = dispatcher.opt_in(evt, "SITE-01", committed_kw=80.0)
    assert response.status == SiteResponseStatus.OPTED_IN
    assert response.committed_kw == 80.0


def test_opt_out_creates_response(dispatcher):
    evt = _event()
    response = dispatcher.opt_out(evt, "SITE-01", reason="maintenance")
    assert response.status == SiteResponseStatus.OPTED_OUT
    assert "maintenance" in response.notes


def test_get_shed_actions_after_opt_in(dispatcher):
    evt = _event()
    dispatcher.opt_in(evt, "SITE-01", committed_kw=50.0)
    actions = dispatcher.get_shed_actions("EVT-001", "SITE-01")
    assert len(actions) > 0
    assert all(a.site_id == "SITE-01" for a in actions)


def test_settle_opted_out_returns_unchanged(dispatcher):
    evt = _event()
    dispatcher.opt_out(evt, "SITE-01")
    settled = dispatcher.settle_event(evt, "SITE-01")
    assert settled.status == SiteResponseStatus.OPTED_OUT


def test_settle_without_response_raises(dispatcher):
    evt = _event()
    with pytest.raises(ValueError):
        dispatcher.settle_event(evt, "NEVER-OPTED-IN")


def test_settle_with_dataframe(dispatcher):
    import pandas as pd
    evt = _event(
        start_time=NOW - timedelta(hours=2),
        end_time=NOW - timedelta(hours=1),
    )
    dispatcher.opt_in(evt, "SITE-01", committed_kw=50.0)

    # Simulate reduced consumption during event
    rows = []
    for m in range(60):
        ts = evt.start_time + timedelta(minutes=m)
        rows.append({"timestamp": ts, "kwh": 60.0})  # reduced load
    # Prior same-weekday hours for baseline
    for m in range(60):
        ts = evt.start_time - timedelta(days=7, minutes=-m)
        rows.append({"timestamp": ts, "kwh": 100.0})  # baseline
    df = pd.DataFrame(rows)

    settled = dispatcher.settle_event(evt, "SITE-01", actual_df=df)
    assert settled.status == SiteResponseStatus.COMPLETED
    assert settled.revenue_earned >= 0


def test_portfolio_revenue_structure(dispatcher):
    summary = dispatcher.portfolio_revenue()
    for k in ("events_responded", "total_revenue", "net_revenue", "avg_reduction_pct"):
        assert k in summary


def test_portfolio_revenue_after_settlement(dispatcher):
    import pandas as pd
    evt = _event(
        start_time=NOW - timedelta(hours=2),
        end_time=NOW - timedelta(hours=1),
    )
    dispatcher.opt_in(evt, "SITE-01", committed_kw=50.0)
    df = pd.DataFrame([
        {"timestamp": evt.start_time + timedelta(minutes=i), "kwh": 50.0}
        for i in range(60)
    ])
    dispatcher.settle_event(evt, "SITE-01", actual_df=df)
    summary = dispatcher.portfolio_revenue()
    assert summary["events_responded"] == 1


# ── MockGridConnector ─────────────────────────────────────────────────────────

def test_mock_connector_fetch_events():
    conn = MockGridConnector(events_per_day=3, seed=1)
    events = conn.fetch_events(hours_ahead=24)
    assert len(events) == 3


def test_mock_events_have_future_start_times():
    conn = MockGridConnector(seed=2)
    events = conn.fetch_events()
    for evt in events:
        assert evt.start_time > datetime.now(timezone.utc) - timedelta(seconds=5)


def test_mock_events_all_upcoming_status():
    conn = MockGridConnector(seed=3)
    events = conn.fetch_events()
    assert all(e.status == DREventStatus.UPCOMING for e in events)


def test_mock_events_have_operator_name():
    conn = MockGridConnector(seed=4)
    events = conn.fetch_events()
    assert all(e.grid_operator == "MOCK-NESO" for e in events)


def test_mock_events_deterministic():
    c1 = MockGridConnector(seed=42).fetch_events()
    c2 = MockGridConnector(seed=42).fetch_events()
    assert [e.clearing_price_kwh for e in c1] == [e.clearing_price_kwh for e in c2]


def test_mock_emergency_events_are_mandatory():
    conn = MockGridConnector(seed=99, events_per_day=10)
    events = conn.fetch_events()
    emergency = [e for e in events if e.event_type == DREventType.EMERGENCY]
    assert all(e.mandatory for e in emergency)


def test_mock_watch_yields_events():
    async def run():
        conn = MockGridConnector(events_per_day=2, seed=5)
        received = []
        async for evt in conn.watch(poll_interval=0):
            received.append(evt)
            if len(received) >= 2:
                break
        return received
    evts = asyncio.run(run())
    assert len(evts) == 2
    assert all(isinstance(e, DREvent) for e in evts)


# ── SHEDDABLE_ASSETS catalogue ────────────────────────────────────────────────

def test_sheddable_assets_have_required_fields():
    for asset in SHEDDABLE_ASSETS:
        for k in ("type", "priority", "typical_kw", "command"):
            assert k in asset, f"Missing {k} in asset {asset}"

def test_sheddable_assets_priorities_unique():
    priorities = [a["priority"] for a in SHEDDABLE_ASSETS]
    assert len(priorities) == len(set(priorities))

def test_sheddable_assets_ordered_by_priority():
    priorities = [a["priority"] for a in SHEDDABLE_ASSETS]
    assert priorities == sorted(priorities)
