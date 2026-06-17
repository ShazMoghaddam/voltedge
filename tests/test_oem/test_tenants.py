"""Tests for enterprise tenant lifecycle management."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from voltedge.oem.tenants import (
    ONBOARDING_STEPS, TIER_ENTITLEMENTS, Tenant, TenantBase,
    TenantService, TenantStatus, TenantTier,
)


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    TenantBase.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.close()


@pytest.fixture
def svc(db):
    return TenantService(db)


def _make_tenant(svc, tenant_id="test-corp", **kwargs):
    return svc.create_lead(
        tenant_id=tenant_id,
        company_name="Test Corp",
        contact_email="admin@test.com",
        **kwargs,
    )


# ── Tier entitlements ─────────────────────────────────────────────────────────

def test_enterprise_has_unlimited_sites():
    assert TIER_ENTITLEMENTS[TenantTier.ENTERPRISE]["max_sites"] == -1


def test_starter_max_5_sites():
    assert TIER_ENTITLEMENTS[TenantTier.STARTER]["max_sites"] == 5


def test_professional_has_lstm():
    assert TIER_ENTITLEMENTS[TenantTier.PROFESSIONAL]["lstm"] is True


def test_starter_no_pdf_export():
    assert TIER_ENTITLEMENTS[TenantTier.STARTER]["pdf_export"] is False


def test_enterprise_has_erp():
    assert TIER_ENTITLEMENTS[TenantTier.ENTERPRISE]["erp_integration"] is True


def test_enterprise_sla_is_99_9():
    assert TIER_ENTITLEMENTS[TenantTier.ENTERPRISE]["sla_uptime_pct"] == 99.9


# ── Lead creation ─────────────────────────────────────────────────────────────

def test_create_lead_returns_tenant(svc):
    t = _make_tenant(svc)
    assert t.tenant_id == "test-corp"
    assert t.status == TenantStatus.LEAD.value


def test_create_lead_sets_default_tier(svc):
    t = _make_tenant(svc)
    assert t.tier == TenantTier.STARTER.value


def test_create_duplicate_raises(svc):
    _make_tenant(svc)
    with pytest.raises(ValueError, match="already exists"):
        _make_tenant(svc)


def test_onboarding_starts_with_account_created(svc):
    t = _make_tenant(svc)
    assert "account_created" in t.onboarding_steps


# ── Trial management ──────────────────────────────────────────────────────────

def test_start_trial_changes_status(svc):
    _make_tenant(svc)
    t = svc.start_trial("test-corp")
    assert t.status == TenantStatus.TRIAL.value


def test_start_trial_sets_end_date(svc):
    _make_tenant(svc)
    t = svc.start_trial("test-corp", duration_days=30)
    assert t.trial_ends_at is not None
    expected = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=30)
    ends = t.trial_ends_at.replace(tzinfo=None) if t.trial_ends_at.tzinfo else t.trial_ends_at
    delta = abs((ends - expected).total_seconds())
    assert delta < 5


def test_trial_is_active(svc):
    _make_tenant(svc)
    t = svc.start_trial("test-corp", duration_days=30)
    assert t.is_trial_active is True


def test_trial_days_remaining(svc):
    _make_tenant(svc)
    t = svc.start_trial("test-corp", duration_days=30)
    assert 28 <= t.trial_days_remaining <= 30


def test_start_trial_upgrades_to_professional(svc):
    _make_tenant(svc)
    t = svc.start_trial("test-corp", tier=TenantTier.PROFESSIONAL)
    assert t.tier == TenantTier.PROFESSIONAL.value


def test_expire_overdue_trials(svc):
    _make_tenant(svc)
    t = svc.start_trial("test-corp", duration_days=30)
    # Manually backdate the trial end
    t.trial_ends_at = datetime.now(timezone.utc) - timedelta(hours=1)
    svc.db.commit()
    expired = svc.expire_overdue_trials()
    assert "test-corp" in expired
    t = svc.get("test-corp")
    assert t.status == TenantStatus.SUSPENDED.value


def test_list_expiring_trials(svc):
    _make_tenant(svc, tenant_id="expiring-corp")
    t = svc.start_trial("expiring-corp", duration_days=5)
    expiring = svc.list_expiring_trials(within_days=7)
    assert any(t.tenant_id == "expiring-corp" for t in expiring)


# ── Activation ────────────────────────────────────────────────────────────────

def test_activate_tenant(svc):
    _make_tenant(svc)
    t = svc.activate("test-corp", tier=TenantTier.ENTERPRISE)
    assert t.status == TenantStatus.ACTIVE.value
    assert t.tier == TenantTier.ENTERPRISE.value


def test_activate_sets_billing_anchor(svc):
    _make_tenant(svc)
    t = svc.activate("test-corp")
    assert t.billing_anchor_date is not None


def test_activate_stores_marketplace_id(svc):
    _make_tenant(svc)
    t = svc.activate("test-corp", marketplace_customer_id="aws-cust-abc123")
    assert t.marketplace_customer_id == "aws-cust-abc123"


# ── Suspend / Churn / Reactivate ──────────────────────────────────────────────

def test_suspend_tenant(svc):
    _make_tenant(svc)
    t = svc.suspend("test-corp", reason="payment_failed")
    assert t.status == TenantStatus.SUSPENDED.value


def test_churn_tenant(svc):
    _make_tenant(svc)
    t = svc.churn("test-corp", reason="competitor")
    assert t.status == TenantStatus.CHURNED.value


def test_reactivate_tenant(svc):
    _make_tenant(svc)
    svc.suspend("test-corp")
    t = svc.reactivate("test-corp")
    assert t.status == TenantStatus.ACTIVE.value


# ── Onboarding checklist ──────────────────────────────────────────────────────

def test_complete_onboarding_step(svc):
    _make_tenant(svc)
    t = svc.complete_step("test-corp", "first_site_added")
    assert "first_site_added" in t.onboarding_steps


def test_complete_step_idempotent(svc):
    _make_tenant(svc)
    svc.complete_step("test-corp", "first_site_added")
    svc.complete_step("test-corp", "first_site_added")
    t = svc.get("test-corp")
    assert t.onboarding_steps.count("first_site_added") == 1


def test_complete_unknown_step_raises(svc):
    _make_tenant(svc)
    with pytest.raises(ValueError, match="Unknown"):
        svc.complete_step("test-corp", "invalid_step")


def test_onboarding_progress_structure(svc):
    _make_tenant(svc)
    svc.complete_step("test-corp", "first_site_added")
    progress = svc.onboarding_progress("test-corp")
    assert "completed" in progress
    assert "pending" in progress
    assert "progress_pct" in progress
    assert progress["total_steps"] == len(ONBOARDING_STEPS)
    assert "first_site_added" in progress["completed"]


def test_onboarding_100_pct_when_all_done(svc):
    _make_tenant(svc)
    for step in ONBOARDING_STEPS:
        svc.complete_step("test-corp", step)
    progress = svc.onboarding_progress("test-corp")
    assert progress["progress_pct"] == 100.0


# ── Usage metering ────────────────────────────────────────────────────────────

def test_record_api_call_increments(svc):
    _make_tenant(svc)
    svc.start_trial("test-corp", tier=TenantTier.PROFESSIONAL)
    result = svc.record_api_call("test-corp")
    assert result is True
    t = svc.get("test-corp")
    assert t.api_calls_today == 1


def test_api_call_quota_exceeded(svc):
    _make_tenant(svc, tier=TenantTier.STARTER)
    t = svc.get("test-corp")
    t.api_calls_today = 1000  # exactly at the starter limit (1000 >= 1000)
    svc.db.commit()
    result = svc.record_api_call("test-corp")
    assert result is False


def test_enterprise_never_exceeds_api_quota(svc):
    _make_tenant(svc)
    svc.activate("test-corp", tier=TenantTier.ENTERPRISE)
    t = svc.get("test-corp")
    t.api_calls_today = 10_000_000
    svc.db.commit()
    result = svc.record_api_call("test-corp")
    assert result is True


def test_reset_daily_counters(svc):
    _make_tenant(svc)
    t = svc.get("test-corp")
    t.api_calls_today = 500
    svc.db.commit()
    count = svc.reset_daily_counters()
    assert count >= 1
    assert svc.get("test-corp").api_calls_today == 0


# ── Pipeline summary ──────────────────────────────────────────────────────────

def test_pipeline_summary_structure(svc):
    _make_tenant(svc, tenant_id="a")
    svc.start_trial("a")
    _make_tenant(svc, tenant_id="b")
    summary = svc.pipeline_summary()
    for status in TenantStatus:
        assert status.value in summary
    assert summary[TenantStatus.TRIAL.value] >= 1
    assert summary[TenantStatus.LEAD.value] >= 1


def test_list_by_status(svc):
    _make_tenant(svc, tenant_id="t1")
    _make_tenant(svc, tenant_id="t2")
    svc.start_trial("t1")
    trials = svc.list_by_status(TenantStatus.TRIAL)
    assert any(t.tenant_id == "t1" for t in trials)
    leads = svc.list_by_status(TenantStatus.LEAD)
    assert any(t.tenant_id == "t2" for t in leads)
