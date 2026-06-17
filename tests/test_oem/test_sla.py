"""Tests for SLA monitoring and reporting."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from voltedge.oem.sla import (
    CheckStatus, HealthCheck, SLABase, SLAIncident,
    SLAMonitoringService, calculate_credit_pct,
)

NOW = datetime(2026, 5, 15, 12, 0, 0, tzinfo=timezone.utc)
PERIOD_START = datetime(2026, 5, 1, 0, 0, 0, tzinfo=timezone.utc)
PERIOD_END   = datetime(2026, 5, 15, 23, 59, 59, tzinfo=timezone.utc)


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    SLABase.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.close()


@pytest.fixture
def svc(db):
    return SLAMonitoringService(db)


def _seed_checks(svc, tenant_id, n_up, n_down, base_time=None):
    base = base_time or PERIOD_START
    for i in range(n_up):
        svc.record_check(
            tenant_id, CheckStatus.UP,
            response_ms=float(50 + i), status_code=200,
            timestamp=base + timedelta(minutes=i),
        )
    for i in range(n_down):
        svc.record_check(
            tenant_id, CheckStatus.DOWN,
            response_ms=0.0, status_code=503,
            error_msg="Service unavailable",
            timestamp=base + timedelta(minutes=n_up + i),
        )


# ── Credit policy ─────────────────────────────────────────────────────────────

def test_no_breach_no_credit():
    assert calculate_credit_pct(99.9, 99.5) == 0.0


def test_credit_10pct_near_miss():
    assert calculate_credit_pct(99.3, 99.5) == 10.0


def test_credit_25pct_significant_breach():
    assert calculate_credit_pct(97.0, 99.5) == 25.0


def test_credit_50pct_severe_breach():
    assert calculate_credit_pct(90.0, 99.5) == 50.0


def test_credit_exactly_at_threshold():
    assert calculate_credit_pct(99.0, 99.5) == 10.0


# ── Record checks ─────────────────────────────────────────────────────────────

def test_record_check_stores_entry(svc, db):
    svc.record_check("T1", CheckStatus.UP, response_ms=42.5, timestamp=NOW)
    checks = db.query(HealthCheck).all()
    assert len(checks) == 1
    assert checks[0].is_up


def test_record_check_down_stores_error(svc, db):
    svc.record_check("T1", CheckStatus.DOWN, error_msg="timeout", timestamp=NOW)
    c = db.query(HealthCheck).first()
    assert not c.is_up
    assert c.error_msg == "timeout"


def test_record_bulk_checks(svc, db):
    checks = [
        {"tenant_id": "T1", "status": "up", "response_ms": 30.0,
         "status_code": 200, "checked_at": NOW},
        {"tenant_id": "T1", "status": "down", "response_ms": 0.0,
         "status_code": 503, "checked_at": NOW + timedelta(minutes=1)},
    ]
    count = svc.record_checks_bulk(checks)
    assert count == 2
    assert db.query(HealthCheck).count() == 2


# ── Incident management ───────────────────────────────────────────────────────

def test_open_incident(svc, db):
    inc = svc.open_incident("T1", started_at=NOW)
    assert inc.is_open
    assert inc.impact == CheckStatus.DOWN.value


def test_resolve_incident(svc, db):
    inc = svc.open_incident("T1", started_at=NOW)
    resolved = svc.resolve_incident(inc.id, resolved_at=NOW + timedelta(hours=2))
    assert not resolved.is_open
    assert abs(resolved.duration_minutes - 120) < 1


def test_incident_duration_open(svc):
    inc = svc.open_incident("T1", started_at=NOW - timedelta(hours=1))
    assert inc.duration_minutes >= 55  # at least ~1h


def test_resolve_unknown_incident_raises(svc):
    with pytest.raises(ValueError):
        svc.resolve_incident("nonexistent-id")


def test_exclude_incident(svc, db):
    inc = svc.open_incident("T1", started_at=NOW)
    svc.exclude_incident(inc.id, reason="customer_caused")
    db.refresh(inc)
    assert inc.is_excluded


def test_get_open_incidents(svc):
    i1 = svc.open_incident("T1", started_at=NOW)
    i2 = svc.open_incident("T1", started_at=NOW)
    svc.resolve_incident(i1.id)
    open_incs = svc.get_open_incidents("T1")
    assert len(open_incs) == 1
    assert open_incs[0].id == i2.id


# ── SLA Report ────────────────────────────────────────────────────────────────

def test_report_all_up_100pct(svc):
    _seed_checks(svc, "T1", n_up=100, n_down=0)
    r = svc.generate_report("T1", sla_target_pct=99.9,
                             period_start=PERIOD_START, period_end=PERIOD_END)
    assert r.uptime_pct == 100.0
    assert r.sla_breached is False
    assert r.credit_pct == 0.0


def test_report_all_down_0pct(svc):
    _seed_checks(svc, "T1", n_up=0, n_down=100)
    r = svc.generate_report("T1", sla_target_pct=99.9,
                             period_start=PERIOD_START, period_end=PERIOD_END)
    assert r.uptime_pct == 0.0
    assert r.sla_breached is True
    assert r.credit_pct == 50.0


def test_report_50pct_up(svc):
    _seed_checks(svc, "T1", n_up=50, n_down=50)
    r = svc.generate_report("T1", sla_target_pct=99.9,
                             period_start=PERIOD_START, period_end=PERIOD_END)
    assert abs(r.uptime_pct - 50.0) < 0.1
    assert r.sla_breached is True


def test_report_no_checks_defaults_100pct(svc):
    r = svc.generate_report("T1", sla_target_pct=99.5,
                             period_start=PERIOD_START, period_end=PERIOD_END)
    assert r.uptime_pct == 100.0
    assert r.sla_breached is False


def test_report_counts_correct(svc):
    _seed_checks(svc, "T1", n_up=80, n_down=20)
    r = svc.generate_report("T1", sla_target_pct=99.9,
                             period_start=PERIOD_START, period_end=PERIOD_END)
    assert r.total_checks == 100
    assert r.checks_up == 80
    assert r.checks_down == 20


def test_report_includes_incidents(svc):
    _seed_checks(svc, "T1", n_up=100, n_down=0)
    i = svc.open_incident("T1", started_at=PERIOD_START + timedelta(hours=1))
    svc.resolve_incident(i.id, resolved_at=PERIOD_START + timedelta(hours=3))
    r = svc.generate_report("T1", sla_target_pct=99.9,
                             period_start=PERIOD_START, period_end=PERIOD_END)
    assert r.incident_count == 1
    assert r.downtime_minutes == pytest.approx(120, abs=2)


def test_excluded_incidents_not_counted(svc):
    _seed_checks(svc, "T1", n_up=100, n_down=0)
    i = svc.open_incident("T1", started_at=PERIOD_START + timedelta(hours=1))
    svc.resolve_incident(i.id, resolved_at=PERIOD_START + timedelta(hours=3))
    svc.exclude_incident(i.id)
    r = svc.generate_report("T1", sla_target_pct=99.9,
                             period_start=PERIOD_START, period_end=PERIOD_END)
    assert r.incident_count == 0
    assert r.downtime_minutes == 0.0


def test_report_meets_sla_property(svc):
    _seed_checks(svc, "T1", n_up=100, n_down=0)
    r = svc.generate_report("T1", sla_target_pct=99.5,
                             period_start=PERIOD_START, period_end=PERIOD_END)
    assert r.meets_sla is True


def test_report_summary_line_met(svc):
    _seed_checks(svc, "T1", n_up=100, n_down=0)
    r = svc.generate_report("T1", sla_target_pct=99.5,
                             period_start=PERIOD_START, period_end=PERIOD_END)
    assert "MET" in r.summary_line()
    assert "T1" in r.summary_line()


def test_report_summary_line_breached(svc):
    _seed_checks(svc, "T1", n_up=0, n_down=100)
    r = svc.generate_report("T1", sla_target_pct=99.5,
                             period_start=PERIOD_START, period_end=PERIOD_END)
    assert "BREACHED" in r.summary_line()


def test_portfolio_report(svc):
    _seed_checks(svc, "T1", n_up=100, n_down=0)
    _seed_checks(svc, "T2", n_up=50, n_down=50)
    reports = svc.portfolio_report(
        {"T1": 99.5, "T2": 99.9},
        period_start=PERIOD_START, period_end=PERIOD_END,
    )
    assert len(reports) == 2
    t1_report = next(r for r in reports if r.tenant_id == "T1")
    t2_report = next(r for r in reports if r.tenant_id == "T2")
    assert t1_report.meets_sla is True
    assert t2_report.sla_breached is True


def test_recent_checks_query(svc):
    _seed_checks(svc, "T1", n_up=10, n_down=0)
    checks = svc.recent_checks("T1", n=5)
    assert len(checks) == 5
