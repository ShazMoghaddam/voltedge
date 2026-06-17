"""
VoltEdge OEM — SLA Monitoring & Compliance Reporting

Tracks platform uptime and performance against contracted SLA targets
for each enterprise tenant tier:

  Starter:      99.0% uptime  (~3.65 days/year downtime allowed)
  Professional: 99.5% uptime  (~1.83 days/year)
  Enterprise:   99.9% uptime  (~8.77 hours/year)

Monitors:
  - API availability (HTTP health checks)
  - Response time P50 / P95 / P99
  - Error rate (5xx responses)
  - Data ingestion lag (time since last successful pipeline run)

Generates:
  - Per-tenant monthly SLA reports
  - Breach notifications (for alert webhook)
  - Credit calculations (percentage of monthly bill)

SLA credit policy (standard industry practice):
  Uptime 99.0–99.5%  → 10% monthly credit
  Uptime 95.0–99.0%  → 25% monthly credit
  Uptime < 95.0%     → 50% monthly credit
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session
import json

from voltedge.utils.logger import get_logger

log = get_logger(__name__)


# ── Enums ─────────────────────────────────────────────────────────────────────

class CheckStatus(str, Enum):
    UP      = "up"
    DOWN    = "down"
    DEGRADED = "degraded"


# ── DB Models ─────────────────────────────────────────────────────────────────

class SLABase(DeclarativeBase):
    pass


class HealthCheck(SLABase):
    """Single availability check result."""
    __tablename__ = "health_checks"

    id:           Mapped[str]    = mapped_column(String, primary_key=True,
                                                  default=lambda: str(uuid.uuid4()))
    tenant_id:    Mapped[str]    = mapped_column(String, nullable=False, index=True)
    checked_at:   Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )
    status:       Mapped[str]    = mapped_column(String, default=CheckStatus.UP.value)
    response_ms:  Mapped[float]  = mapped_column(Float, default=0.0)
    status_code:  Mapped[int]    = mapped_column(Integer, default=200)
    error_msg:    Mapped[str]    = mapped_column(String, default="")
    region:       Mapped[str]    = mapped_column(String, default="eu-west-1")

    @property
    def is_up(self) -> bool:
        return self.status == CheckStatus.UP.value


class SLAIncident(SLABase):
    """A recorded outage or degradation event."""
    __tablename__ = "sla_incidents"

    id:          Mapped[str]      = mapped_column(String, primary_key=True,
                                                   default=lambda: str(uuid.uuid4()))
    tenant_id:   Mapped[str]      = mapped_column(String, nullable=False, index=True)
    started_at:  Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    impact:      Mapped[str]      = mapped_column(String, default=CheckStatus.DOWN.value)
    description: Mapped[str]      = mapped_column(String, default="")
    regions_affected: Mapped[str] = mapped_column(String, default="")
    is_excluded:  Mapped[bool]    = mapped_column(Boolean, default=False)

    @property
    def duration_minutes(self) -> float:
        if not self.resolved_at:
            end = datetime.now(timezone.utc)
        else:
            end = self.resolved_at if self.resolved_at.tzinfo else \
                self.resolved_at.replace(tzinfo=timezone.utc)
        start = self.started_at if self.started_at.tzinfo else \
            self.started_at.replace(tzinfo=timezone.utc)
        return (end - start).total_seconds() / 60

    @property
    def is_open(self) -> bool:
        return self.resolved_at is None


# ── SLA Report ────────────────────────────────────────────────────────────────

@dataclass
class SLAReport:
    """Monthly SLA compliance report for one tenant."""
    tenant_id:          str
    period_start:       datetime
    period_end:         datetime
    sla_target_pct:     float
    uptime_pct:         float
    total_checks:       int
    checks_up:          int
    checks_down:        int
    downtime_minutes:   float
    incident_count:     int
    avg_response_ms:    float
    p95_response_ms:    float
    p99_response_ms:    float
    error_rate_pct:     float
    sla_breached:       bool
    credit_pct:         float           # % of monthly bill as credit
    open_incidents:     int

    @property
    def availability_pct(self) -> float:
        return self.uptime_pct

    @property
    def meets_sla(self) -> bool:
        return not self.sla_breached

    def summary_line(self) -> str:
        status = "✅ MET" if self.meets_sla else f"❌ BREACHED (credit: {self.credit_pct:.0f}%)"
        return (
            f"{self.tenant_id} | {self.period_start.strftime('%Y-%m')} | "
            f"Uptime: {self.uptime_pct:.3f}% | Target: {self.sla_target_pct:.1f}% | {status}"
        )


# ── Credit policy ─────────────────────────────────────────────────────────────

def calculate_credit_pct(actual_uptime_pct: float, target_uptime_pct: float) -> float:
    """
    Calculate SLA credit percentage of monthly bill.
    Industry standard tiered credit policy.
    """
    if actual_uptime_pct >= target_uptime_pct:
        return 0.0
    if actual_uptime_pct >= 99.0:
        return 10.0
    if actual_uptime_pct >= 95.0:
        return 25.0
    return 50.0


# ── SLA Service ───────────────────────────────────────────────────────────────

class SLAMonitoringService:
    """
    Records health checks, detects incidents, and generates SLA reports.
    """

    def __init__(self, db: Session) -> None:
        self.db = db

    # ── Health checks ─────────────────────────────────────────────────────────

    def record_check(
        self,
        tenant_id:   str,
        status:      CheckStatus,
        response_ms: float = 0.0,
        status_code: int   = 200,
        error_msg:   str   = "",
        region:      str   = "eu-west-1",
        timestamp:   datetime | None = None,
    ) -> HealthCheck:
        check = HealthCheck(
            tenant_id=tenant_id,
            status=status.value,
            response_ms=response_ms,
            status_code=status_code,
            error_msg=error_msg,
            region=region,
            checked_at=timestamp or datetime.now(timezone.utc),
        )
        self.db.add(check)
        self.db.commit()
        if status != CheckStatus.UP:
            log.warning("sla.check_failed", tenant=tenant_id, status=status.value,
                        region=region, code=status_code)
        return check

    def record_checks_bulk(self, checks: list[dict]) -> int:
        """Efficiently insert multiple checks. Returns count inserted."""
        rows = [HealthCheck(**c) for c in checks]
        self.db.add_all(rows)
        self.db.commit()
        return len(rows)

    # ── Incident management ───────────────────────────────────────────────────

    def open_incident(
        self,
        tenant_id:        str,
        started_at:       datetime | None = None,
        impact:           CheckStatus     = CheckStatus.DOWN,
        description:      str             = "",
        regions_affected: str             = "",
    ) -> SLAIncident:
        incident = SLAIncident(
            tenant_id=tenant_id,
            started_at=started_at or datetime.now(timezone.utc),
            impact=impact.value,
            description=description,
            regions_affected=regions_affected,
        )
        self.db.add(incident)
        self.db.commit()
        log.warning("sla.incident_opened", tenant=tenant_id, impact=impact.value)
        return incident

    def resolve_incident(self, incident_id: str, resolved_at: datetime | None = None) -> SLAIncident:
        incident = self.db.query(SLAIncident).filter_by(id=incident_id).first()
        if not incident:
            raise ValueError(f"Incident '{incident_id}' not found.")
        incident.resolved_at = resolved_at or datetime.now(timezone.utc)
        self.db.commit()
        log.info("sla.incident_resolved", incident=incident_id,
                 duration_min=round(incident.duration_minutes, 1))
        return incident

    def exclude_incident(self, incident_id: str, reason: str = "") -> None:
        """Mark an incident as excluded from SLA calc (e.g. customer-caused)."""
        incident = self.db.query(SLAIncident).filter_by(id=incident_id).first()
        if incident:
            incident.is_excluded = True
            self.db.commit()
            log.info("sla.incident_excluded", incident=incident_id, reason=reason)

    # ── Reporting ─────────────────────────────────────────────────────────────

    def generate_report(
        self,
        tenant_id:      str,
        sla_target_pct: float,
        period_start:   datetime | None = None,
        period_end:     datetime | None = None,
    ) -> SLAReport:
        """
        Generate an SLA compliance report for the specified period.
        Defaults to the previous calendar month if no period given.
        """
        if period_start is None:
            now = datetime.now(timezone.utc)
            period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if now.month == 1:
                period_start = period_start.replace(year=now.year - 1, month=12)
            else:
                period_start = period_start.replace(month=now.month - 1)
        if period_end is None:
            period_end = datetime.now(timezone.utc)

        # Fetch health checks for the period
        def _norm(ts: datetime) -> datetime:
            return ts.replace(tzinfo=None) if ts.tzinfo else ts

        ps_n = _norm(period_start)
        pe_n = _norm(period_end)

        checks = (
            self.db.query(HealthCheck)
            .filter(HealthCheck.tenant_id == tenant_id)
            .all()
        )
        checks = [
            c for c in checks
            if ps_n <= _norm(c.checked_at) <= pe_n
        ]

        total = len(checks)
        up    = sum(1 for c in checks if c.is_up)
        down  = total - up

        uptime_pct = (up / total * 100) if total > 0 else 100.0
        response_times = [c.response_ms for c in checks if c.is_up and c.response_ms > 0]
        response_times.sort()

        avg_ms = sum(response_times) / len(response_times) if response_times else 0.0
        p95_ms = response_times[int(len(response_times) * 0.95)] if response_times else 0.0
        p99_ms = response_times[int(len(response_times) * 0.99)] if response_times else 0.0

        # Error rate: 5xx responses
        error_checks = [c for c in checks if c.status_code >= 500]
        error_rate = len(error_checks) / total * 100 if total > 0 else 0.0

        # Fetch non-excluded incidents
        incidents = (
            self.db.query(SLAIncident)
            .filter(
                SLAIncident.tenant_id == tenant_id,
                SLAIncident.is_excluded == False,
            )
            .all()
        )
        incidents_in_period = [
            i for i in incidents
            if ps_n <= _norm(i.started_at) <= pe_n
        ]
        open_inc = sum(1 for i in incidents_in_period if i.is_open)
        downtime_min = sum(i.duration_minutes for i in incidents_in_period if not i.is_open)

        breached = uptime_pct < sla_target_pct
        credit = calculate_credit_pct(uptime_pct, sla_target_pct)

        report = SLAReport(
            tenant_id=tenant_id,
            period_start=period_start,
            period_end=period_end,
            sla_target_pct=sla_target_pct,
            uptime_pct=round(uptime_pct, 4),
            total_checks=total,
            checks_up=up,
            checks_down=down,
            downtime_minutes=round(downtime_min, 2),
            incident_count=len(incidents_in_period),
            avg_response_ms=round(avg_ms, 1),
            p95_response_ms=round(p95_ms, 1),
            p99_response_ms=round(p99_ms, 1),
            error_rate_pct=round(error_rate, 4),
            sla_breached=breached,
            credit_pct=credit,
            open_incidents=open_inc,
        )

        log.info("sla.report_generated", tenant=tenant_id,
                 uptime=report.uptime_pct, breached=breached, credit=credit)
        return report

    def portfolio_report(
        self,
        tenant_sla_map: dict[str, float],
        period_start: datetime | None = None,
        period_end: datetime | None   = None,
    ) -> list[SLAReport]:
        """Generate reports for all tenants in the portfolio."""
        return [
            self.generate_report(tid, sla_pct, period_start, period_end)
            for tid, sla_pct in tenant_sla_map.items()
        ]

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_open_incidents(self, tenant_id: str | None = None) -> list[SLAIncident]:
        q = self.db.query(SLAIncident).filter(SLAIncident.resolved_at.is_(None))
        if tenant_id:
            q = q.filter(SLAIncident.tenant_id == tenant_id)
        return q.all()

    def recent_checks(self, tenant_id: str, n: int = 50) -> list[HealthCheck]:
        return (
            self.db.query(HealthCheck)
            .filter_by(tenant_id=tenant_id)
            .order_by(HealthCheck.checked_at.desc())
            .limit(n)
            .all()
        )
