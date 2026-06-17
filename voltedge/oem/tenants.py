"""
VoltEdge OEM — Enterprise Tenant Lifecycle Management

Manages the full commercial lifecycle for enterprise accounts:

  LEAD → TRIAL → ACTIVE → SUSPENDED → CHURNED

Features:
  - Self-serve trial provisioning (30-day default)
  - Usage metering (API calls, sites, data volume)
  - Tier-based entitlements (Starter / Professional / Enterprise)
  - Automated trial-expiry detection
  - Billing anchor date tracking (for marketplace metering)
  - Onboarding checklist progress

Designed to feed directly into:
  - AWS Marketplace Metering API (Step 3)
  - SLA monitoring (Step 4)
  - Billing / Stripe integration (future)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer, String, Text, UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session
import json

from voltedge.utils.logger import get_logger

log = get_logger(__name__)


# ── Enums ─────────────────────────────────────────────────────────────────────

class TenantTier(str, Enum):
    STARTER      = "starter"       # ≤5 sites, no LSTM, no ERP
    PROFESSIONAL = "professional"  # ≤20 sites, all ML, no ERP
    ENTERPRISE   = "enterprise"    # Unlimited, all features, ERP, SLA


class TenantStatus(str, Enum):
    LEAD      = "lead"       # Prospect — not yet onboarded
    TRIAL     = "trial"      # Active 30-day trial
    ACTIVE    = "active"     # Paying customer
    SUSPENDED = "suspended"  # Payment failed / compliance hold
    CHURNED   = "churned"    # Cancelled


# ── Tier entitlements ─────────────────────────────────────────────────────────

TIER_ENTITLEMENTS: dict[TenantTier, dict[str, Any]] = {
    TenantTier.STARTER: {
        "max_sites":             5,
        "max_api_calls_per_day": 1_000,
        "max_data_gb":           10,
        "forecast":              True,
        "anomaly":               True,
        "esg":                   True,
        "pdf_export":            False,
        "lstm":                  False,
        "erp_integration":       False,
        "sla_uptime_pct":        99.0,
        "support_tier":          "community",
    },
    TenantTier.PROFESSIONAL: {
        "max_sites":             20,
        "max_api_calls_per_day": 10_000,
        "max_data_gb":           100,
        "forecast":              True,
        "anomaly":               True,
        "esg":                   True,
        "pdf_export":            True,
        "lstm":                  True,
        "erp_integration":       False,
        "sla_uptime_pct":        99.5,
        "support_tier":          "email",
    },
    TenantTier.ENTERPRISE: {
        "max_sites":             -1,   # unlimited
        "max_api_calls_per_day": -1,   # unlimited
        "max_data_gb":           -1,   # unlimited
        "forecast":              True,
        "anomaly":               True,
        "esg":                   True,
        "pdf_export":            True,
        "lstm":                  True,
        "erp_integration":       True,
        "sla_uptime_pct":        99.9,
        "support_tier":          "dedicated",
    },
}


# ── DB Model ──────────────────────────────────────────────────────────────────

class TenantBase(DeclarativeBase):
    pass


class Tenant(TenantBase):
    """Enterprise tenant record."""
    __tablename__ = "tenants"

    id:             Mapped[str]  = mapped_column(String, primary_key=True,
                                                  default=lambda: str(uuid.uuid4()))
    tenant_id:      Mapped[str]  = mapped_column(String, unique=True, nullable=False)
    company_name:   Mapped[str]  = mapped_column(String, nullable=False)
    contact_email:  Mapped[str]  = mapped_column(String, nullable=False)
    contact_name:   Mapped[str]  = mapped_column(String, default="")
    tier:           Mapped[str]  = mapped_column(String, default=TenantTier.STARTER.value)
    status:         Mapped[str]  = mapped_column(String, default=TenantStatus.LEAD.value)

    # Trial
    trial_starts_at:  Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    trial_ends_at:    Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    trial_duration_days: Mapped[int]          = mapped_column(Integer, default=30)

    # Billing
    billing_anchor_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    marketplace_customer_id: Mapped[str | None]  = mapped_column(String, nullable=True)
    aws_account_id:          Mapped[str | None]  = mapped_column(String, nullable=True)

    # Usage (rolling 24h counters — reset by scheduler)
    api_calls_today: Mapped[int]   = mapped_column(Integer, default=0)
    sites_count:     Mapped[int]   = mapped_column(Integer, default=0)
    data_gb_used:    Mapped[float] = mapped_column(Float,   default=0.0)

    # Onboarding checklist (JSON list of completed step keys)
    onboarding_json: Mapped[str] = mapped_column(Text, default="[]")
    notes_json:      Mapped[str] = mapped_column(Text, default="{}")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    @property
    def onboarding_steps(self) -> list[str]:
        return json.loads(self.onboarding_json)

    @property
    def entitlements(self) -> dict[str, Any]:
        return TIER_ENTITLEMENTS[TenantTier(self.tier)]

    @property
    def is_trial_active(self) -> bool:
        if self.status != TenantStatus.TRIAL.value:
            return False
        if not self.trial_ends_at:
            return False
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        ends = self.trial_ends_at.replace(tzinfo=None) if self.trial_ends_at.tzinfo else self.trial_ends_at
        return now < ends

    @property
    def trial_days_remaining(self) -> int:
        if not self.trial_ends_at:
            return 0
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        ends = self.trial_ends_at.replace(tzinfo=None) if self.trial_ends_at.tzinfo else self.trial_ends_at
        delta = ends - now
        return max(0, delta.days)

    def exceeds_quota(self, resource: str) -> bool:
        """True if tenant has exceeded a metered quota."""
        ents = self.entitlements
        if resource == "api_calls" and ents["max_api_calls_per_day"] != -1:
            return self.api_calls_today >= ents["max_api_calls_per_day"]
        if resource == "sites" and ents["max_sites"] != -1:
            return self.sites_count >= ents["max_sites"]
        if resource == "data_gb" and ents["max_data_gb"] != -1:
            return self.data_gb_used >= ents["max_data_gb"]
        return False

    def has_feature(self, feature: str) -> bool:
        return bool(self.entitlements.get(feature, False))


# ── Onboarding steps ──────────────────────────────────────────────────────────

ONBOARDING_STEPS = [
    "account_created",
    "domain_configured",
    "first_site_added",
    "first_ingestion_run",
    "first_forecast",
    "first_anomaly_detected",
    "esg_report_generated",
    "team_members_invited",
    "erp_integration_configured",
    "sla_agreement_signed",
]


# ── Tenant service ────────────────────────────────────────────────────────────

class TenantService:
    """Full tenant lifecycle operations."""

    def __init__(self, db: Session) -> None:
        self.db = db

    # ── Onboarding ────────────────────────────────────────────────────────────

    def create_lead(
        self,
        tenant_id: str,
        company_name: str,
        contact_email: str,
        contact_name: str = "",
        tier: TenantTier = TenantTier.STARTER,
    ) -> Tenant:
        if self.db.query(Tenant).filter_by(tenant_id=tenant_id).first():
            raise ValueError(f"Tenant '{tenant_id}' already exists.")
        tenant = Tenant(
            tenant_id=tenant_id,
            company_name=company_name,
            contact_email=contact_email,
            contact_name=contact_name,
            tier=tier.value,
            status=TenantStatus.LEAD.value,
            onboarding_json=json.dumps(["account_created"]),
        )
        self.db.add(tenant)
        self.db.commit()
        self.db.refresh(tenant)
        log.info("tenant.lead_created", tenant=tenant_id, tier=tier.value)
        return tenant

    def start_trial(
        self,
        tenant_id: str,
        duration_days: int = 30,
        tier: TenantTier = TenantTier.PROFESSIONAL,
    ) -> Tenant:
        tenant = self._get_or_raise(tenant_id)
        now = datetime.now(timezone.utc)
        tenant.status = TenantStatus.TRIAL.value
        tenant.tier = tier.value
        tenant.trial_starts_at = now
        tenant.trial_ends_at = now + timedelta(days=duration_days)
        tenant.trial_duration_days = duration_days
        self.db.commit()
        log.info("tenant.trial_started", tenant=tenant_id,
                 ends_at=tenant.trial_ends_at.isoformat())
        return tenant

    def activate(
        self,
        tenant_id: str,
        tier: TenantTier | None = None,
        marketplace_customer_id: str | None = None,
        aws_account_id: str | None = None,
    ) -> Tenant:
        tenant = self._get_or_raise(tenant_id)
        now = datetime.now(timezone.utc)
        tenant.status = TenantStatus.ACTIVE.value
        tenant.activated_at = now
        tenant.billing_anchor_date = now
        if tier:
            tenant.tier = tier.value
        if marketplace_customer_id:
            tenant.marketplace_customer_id = marketplace_customer_id
        if aws_account_id:
            tenant.aws_account_id = aws_account_id
        self.db.commit()
        log.info("tenant.activated", tenant=tenant_id, tier=tenant.tier)
        return tenant

    def suspend(self, tenant_id: str, reason: str = "") -> Tenant:
        tenant = self._get_or_raise(tenant_id)
        tenant.status = TenantStatus.SUSPENDED.value
        notes = json.loads(tenant.notes_json)
        notes["suspension_reason"] = reason
        notes["suspended_at"] = datetime.now(timezone.utc).isoformat()
        tenant.notes_json = json.dumps(notes)
        self.db.commit()
        log.warning("tenant.suspended", tenant=tenant_id, reason=reason)
        return tenant

    def churn(self, tenant_id: str, reason: str = "") -> Tenant:
        tenant = self._get_or_raise(tenant_id)
        tenant.status = TenantStatus.CHURNED.value
        notes = json.loads(tenant.notes_json)
        notes["churn_reason"] = reason
        notes["churned_at"] = datetime.now(timezone.utc).isoformat()
        tenant.notes_json = json.dumps(notes)
        self.db.commit()
        log.info("tenant.churned", tenant=tenant_id)
        return tenant

    def reactivate(self, tenant_id: str) -> Tenant:
        tenant = self._get_or_raise(tenant_id)
        tenant.status = TenantStatus.ACTIVE.value
        self.db.commit()
        log.info("tenant.reactivated", tenant=tenant_id)
        return tenant

    # ── Onboarding checklist ──────────────────────────────────────────────────

    def complete_step(self, tenant_id: str, step: str) -> Tenant:
        if step not in ONBOARDING_STEPS:
            raise ValueError(f"Unknown onboarding step: '{step}'. Valid: {ONBOARDING_STEPS}")
        tenant = self._get_or_raise(tenant_id)
        steps = tenant.onboarding_steps
        if step not in steps:
            steps.append(step)
            tenant.onboarding_json = json.dumps(steps)
            self.db.commit()
            log.info("tenant.onboarding_step", tenant=tenant_id, step=step,
                     progress=f"{len(steps)}/{len(ONBOARDING_STEPS)}")
        return tenant

    def onboarding_progress(self, tenant_id: str) -> dict[str, Any]:
        tenant = self._get_or_raise(tenant_id)
        completed = set(tenant.onboarding_steps)
        total = len(ONBOARDING_STEPS)
        done = sum(1 for s in ONBOARDING_STEPS if s in completed)
        return {
            "tenant_id":      tenant_id,
            "completed":      [s for s in ONBOARDING_STEPS if s in completed],
            "pending":        [s for s in ONBOARDING_STEPS if s not in completed],
            "progress_pct":   round(done / total * 100, 1),
            "total_steps":    total,
            "steps_complete": done,
        }

    # ── Usage metering ────────────────────────────────────────────────────────

    def record_api_call(self, tenant_id: str) -> bool:
        """Increment API call counter. Returns False if quota exceeded."""
        tenant = self._get_or_raise(tenant_id)
        if tenant.exceeds_quota("api_calls"):
            log.warning("tenant.quota_exceeded", tenant=tenant_id, resource="api_calls")
            return False
        tenant.api_calls_today += 1
        self.db.commit()
        return True

    def update_site_count(self, tenant_id: str, count: int) -> Tenant:
        tenant = self._get_or_raise(tenant_id)
        tenant.sites_count = count
        self.db.commit()
        return tenant

    def reset_daily_counters(self) -> int:
        """Called by scheduler at midnight UTC. Returns number reset."""
        count = self.db.query(Tenant).filter(Tenant.api_calls_today > 0).count()
        self.db.query(Tenant).update({"api_calls_today": 0})
        self.db.commit()
        log.info("tenant.counters_reset", tenants=count)
        return count

    # ── Queries ───────────────────────────────────────────────────────────────

    def get(self, tenant_id: str) -> Tenant | None:
        return self.db.query(Tenant).filter_by(tenant_id=tenant_id).first()

    def list_by_status(self, status: TenantStatus) -> list[Tenant]:
        return self.db.query(Tenant).filter_by(status=status.value).all()

    def list_expiring_trials(self, within_days: int = 7) -> list[Tenant]:
        cutoff = datetime.now(timezone.utc) + timedelta(days=within_days)
        return (
            self.db.query(Tenant)
            .filter(
                Tenant.status == TenantStatus.TRIAL.value,
                Tenant.trial_ends_at <= cutoff,
            )
            .all()
        )

    def expire_overdue_trials(self) -> list[str]:
        """Mark expired trials as suspended. Returns list of affected tenant_ids."""
        now = datetime.now(timezone.utc)
        expired = (
            self.db.query(Tenant)
            .filter(
                Tenant.status == TenantStatus.TRIAL.value,
                Tenant.trial_ends_at < now,
            )
            .all()
        )
        affected = []
        for t in expired:
            t.status = TenantStatus.SUSPENDED.value
            notes = json.loads(t.notes_json)
            notes["trial_expired_at"] = now.isoformat()
            t.notes_json = json.dumps(notes)
            affected.append(t.tenant_id)
        if affected:
            self.db.commit()
            log.info("tenant.trials_expired", count=len(affected), tenants=affected)
        return affected

    def pipeline_summary(self) -> dict[str, int]:
        """CRM-style pipeline summary."""
        return {
            s.value: self.db.query(Tenant).filter_by(status=s.value).count()
            for s in TenantStatus
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_or_raise(self, tenant_id: str) -> Tenant:
        t = self.db.query(Tenant).filter_by(tenant_id=tenant_id).first()
        if not t:
            raise ValueError(f"Tenant '{tenant_id}' not found.")
        return t
