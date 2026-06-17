"""Tests for SOC 2 Audit Trail — chain integrity, query, export."""

from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from voltedge.rbac.audit import AuditEventType, AuditLog, AuditService
from voltedge.rbac.models import Base


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def audit(db):
    return AuditService(db)


# ── Write events ──────────────────────────────────────────────────────────────

def test_log_event_returns_entry(audit):
    entry = audit.log_event(AuditEventType.AUTH_LOGIN_SUCCESS, actor="alice")
    assert entry.id is not None
    assert entry.actor == "alice"
    assert entry.event_type == "auth.login.success"


def test_log_event_stores_details(audit):
    audit.log_event(
        AuditEventType.DATA_READ, actor="bob",
        resource="sites", action="read",
        site_id="SITE-01",
        details={"rows_returned": 500},
    )
    entry = audit.recent(1)[0]
    assert entry.details["rows_returned"] == 500


def test_log_multiple_events(audit):
    for i in range(5):
        audit.log_event(AuditEventType.DATA_READ, actor=f"user{i}")
    entries = audit.recent(10)
    assert len(entries) == 5


def test_log_event_records_timestamp(audit):
    before = datetime.now(timezone.utc).replace(tzinfo=None)
    audit.log_event(AuditEventType.AUTH_LOGIN_SUCCESS, actor="ts_test")
    after = datetime.now(timezone.utc).replace(tzinfo=None)
    entry = audit.recent(1)[0]
    # SQLite returns naive timestamps; normalise for comparison
    ts = entry.timestamp.replace(tzinfo=None) if entry.timestamp.tzinfo else entry.timestamp
    assert before <= ts <= after


def test_log_event_with_ip_and_user_agent(audit):
    audit.log_event(
        AuditEventType.AUTH_LOGIN_SUCCESS, actor="alice",
        ip_address="192.168.1.10", user_agent="VoltEdge-CLI/1.0",
    )
    entry = audit.recent(1)[0]
    assert entry.ip_address == "192.168.1.10"
    assert entry.user_agent == "VoltEdge-CLI/1.0"


def test_all_event_types_can_be_logged(audit):
    for et in AuditEventType:
        audit.log_event(et, actor="system")
    assert audit.db.query(AuditLog).count() == len(AuditEventType)


# ── Chain integrity ───────────────────────────────────────────────────────────

def test_chain_starts_from_genesis(audit):
    result = audit.verify_chain_integrity()
    assert result["valid"] is True
    assert result["entries"] == 0


def test_chain_is_valid_after_writes(audit):
    for i in range(10):
        audit.log_event(AuditEventType.DATA_READ, actor=f"u{i}")
    result = audit.verify_chain_integrity()
    assert result["valid"] is True
    assert result["entries"] == 10


def test_tampered_hash_detected(audit, db):
    audit.log_event(AuditEventType.AUTH_LOGIN_SUCCESS, actor="alice")
    audit.log_event(AuditEventType.DATA_READ, actor="bob")

    # Tamper with the first entry's hash
    entry = db.query(AuditLog).order_by(AuditLog.timestamp.asc()).first()
    entry.chain_hash = "a" * 64
    db.commit()

    result = audit.verify_chain_integrity()
    assert result["valid"] is False
    assert result["first_breach"] is not None


def test_chain_hash_is_unique_per_event(audit):
    e1 = audit.log_event(AuditEventType.AUTH_LOGIN_SUCCESS, actor="a")
    e2 = audit.log_event(AuditEventType.AUTH_LOGIN_SUCCESS, actor="a")
    assert e1.chain_hash != e2.chain_hash


def test_chain_hash_is_64_hex_chars(audit):
    entry = audit.log_event(AuditEventType.AUTH_LOGIN_SUCCESS, actor="a")
    assert len(entry.chain_hash) == 64
    assert all(c in "0123456789abcdef" for c in entry.chain_hash)


# ── Query ─────────────────────────────────────────────────────────────────────

def test_query_by_event_type(audit):
    audit.log_event(AuditEventType.AUTH_LOGIN_SUCCESS, actor="alice")
    audit.log_event(AuditEventType.DATA_READ, actor="alice")
    audit.log_event(AuditEventType.DATA_READ, actor="bob")
    results = audit.query(event_type="data.read")
    assert len(results) == 2
    assert all(r.event_type == "data.read" for r in results)


def test_query_by_actor(audit):
    audit.log_event(AuditEventType.DATA_READ, actor="alice")
    audit.log_event(AuditEventType.DATA_READ, actor="bob")
    audit.log_event(AuditEventType.DATA_EXPORT, actor="alice")
    results = audit.query(actor="alice")
    assert len(results) == 2
    assert all(r.actor == "alice" for r in results)


def test_query_by_site_id(audit):
    audit.log_event(AuditEventType.DATA_READ, actor="a", site_id="SITE-A")
    audit.log_event(AuditEventType.DATA_READ, actor="a", site_id="SITE-B")
    results = audit.query(site_id="SITE-A")
    assert len(results) == 1
    assert results[0].site_id == "SITE-A"


def test_query_by_time_window(audit):
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    audit.log_event(AuditEventType.DATA_READ, actor="x")
    future = datetime.now(timezone.utc) + timedelta(hours=1)
    results = audit.query(since=past, until=future)
    assert len(results) >= 1


def test_query_returns_newest_first(audit):
    for i in range(3):
        audit.log_event(AuditEventType.DATA_READ, actor=f"u{i}")
    results = audit.recent(3)
    ts = [r.timestamp for r in results]
    assert ts == sorted(ts, reverse=True)


# ── Actor summary ─────────────────────────────────────────────────────────────

def test_actor_summary_counts_correctly(audit):
    audit.log_event(AuditEventType.AUTH_LOGIN_SUCCESS, actor="alice")
    audit.log_event(AuditEventType.DATA_READ, actor="alice")
    audit.log_event(AuditEventType.DATA_READ, actor="alice")
    summary = audit.actor_summary("alice")
    assert summary["auth.login.success"] == 1
    assert summary["data.read"] == 2


def test_actor_summary_empty_for_unknown(audit):
    assert audit.actor_summary("nobody") == {}


# ── CSV export ────────────────────────────────────────────────────────────────

def test_csv_export_has_header(audit):
    csv_str = audit.export_csv()
    assert csv_str.startswith("id,timestamp,event_type")


def test_csv_export_contains_events(audit):
    audit.log_event(AuditEventType.DATA_EXPORT, actor="alice",
                    resource="esg", action="export", site_id="SITE-01")
    csv_str = audit.export_csv()
    lines = [l for l in csv_str.strip().split("\n") if l]
    assert len(lines) == 2   # header + 1 data row


def test_csv_export_contains_event_type(audit):
    audit.log_event(AuditEventType.USER_CREATED, actor="admin")
    csv_str = audit.export_csv()
    assert "admin.user.created" in csv_str


def test_csv_is_chronological(audit):
    audit.log_event(AuditEventType.AUTH_LOGIN_SUCCESS, actor="first")
    audit.log_event(AuditEventType.DATA_READ, actor="second")
    csv_str = audit.export_csv()
    lines = csv_str.strip().split("\n")[1:]   # skip header
    assert "first" in lines[0]
    assert "second" in lines[1]
