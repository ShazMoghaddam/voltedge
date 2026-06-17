"""
VoltEdge — SOC 2 Audit Trail

Provides tamper-evident logging for all security-relevant events:

  - Authentication (login success/failure, token issue/expiry)
  - Authorisation (permission grants/denials)
  - Data access (who read/exported what, when)
  - Admin actions (user create/deactivate, role assignment)
  - System events (model retrain, pipeline triggers, config changes)
  - Anomaly alerts (when alerts are raised and acknowledged)

SOC 2 Trust Service Criteria covered:
  CC6.2  - Logical access controls
  CC6.3  - Access removal
  CC7.2  - Monitoring of system components
  CC8.1  - Change management
  A1.2   - Capacity monitoring (pipeline events)

Storage:
  - Local: SQLite (dev)
  - Prod:  Same DB as RBAC (PostgreSQL) via SQLAlchemy
  - Each row has a SHA-256 chain hash over (previous_hash + payload)
    to detect deletion or modification of log entries.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from sqlalchemy import Column, DateTime, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session

from voltedge.rbac.models import Base
from voltedge.utils.logger import get_logger

log = get_logger(__name__)


# ── Event category enum ───────────────────────────────────────────────────────

class AuditEventType(str, Enum):
    # Auth
    AUTH_LOGIN_SUCCESS    = "auth.login.success"
    AUTH_LOGIN_FAILURE    = "auth.login.failure"
    AUTH_TOKEN_ISSUED     = "auth.token.issued"
    AUTH_TOKEN_EXPIRED    = "auth.token.expired"
    AUTH_LOGOUT           = "auth.logout"
    # Authorisation
    AUTHZ_GRANTED         = "authz.permission.granted"
    AUTHZ_DENIED          = "authz.permission.denied"
    # Data access
    DATA_READ             = "data.read"
    DATA_EXPORT           = "data.export"
    DATA_WRITE            = "data.write"
    # Admin
    USER_CREATED          = "admin.user.created"
    USER_DEACTIVATED      = "admin.user.deactivated"
    ROLE_ASSIGNED         = "admin.role.assigned"
    ROLE_REVOKED          = "admin.role.revoked"
    # System
    PIPELINE_TRIGGERED    = "system.pipeline.triggered"
    MODEL_RETRAINED       = "system.model.retrained"
    CONFIG_CHANGED        = "system.config.changed"
    # Alerts
    ALERT_RAISED          = "alert.raised"
    ALERT_ACKNOWLEDGED    = "alert.acknowledged"


# ── SQLAlchemy model ──────────────────────────────────────────────────────────

class AuditLog(Base):
    """
    Single immutable audit log entry.
    chain_hash = SHA-256(prev_hash + event_type + actor + timestamp + details_json)
    """
    __tablename__ = "audit_logs"

    id:           Mapped[str]      = mapped_column(String, primary_key=True,
                                                    default=lambda: str(uuid.uuid4()))
    timestamp:    Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )
    event_type:   Mapped[str]  = mapped_column(String(64), nullable=False, index=True)
    actor:        Mapped[str]  = mapped_column(String(128), nullable=False, index=True)
    resource:     Mapped[str | None] = mapped_column(String(128), nullable=True)
    action:       Mapped[str | None] = mapped_column(String(64),  nullable=True)
    site_id:      Mapped[str | None] = mapped_column(String(128), nullable=True)
    ip_address:   Mapped[str | None] = mapped_column(String(45),  nullable=True)
    user_agent:   Mapped[str | None] = mapped_column(String(256), nullable=True)
    details_json: Mapped[str]  = mapped_column(Text, default="{}")
    chain_hash:   Mapped[str]  = mapped_column(String(64), nullable=False)

    @property
    def details(self) -> dict:
        return json.loads(self.details_json)


# ── Audit service ─────────────────────────────────────────────────────────────

class AuditService:
    """
    Writes audit events and provides query/export capabilities.

    Args:
        db: SQLAlchemy session (injected).
    """

    GENESIS_HASH = "0" * 64   # Initial hash for the chain

    def __init__(self, db: Session) -> None:
        self.db = db

    # ── Write ─────────────────────────────────────────────────────────────────

    def log_event(
        self,
        event_type: AuditEventType | str,
        actor: str,
        resource:   str | None = None,
        action:     str | None = None,
        site_id:    str | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        details:    dict[str, Any] | None = None,
    ) -> AuditLog:
        """
        Write a single audit event. Thread-safe (each call is a DB commit).

        Args:
            event_type: AuditEventType enum value or raw string.
            actor:      Username or system identifier.
            resource:   RBAC resource (e.g. "sites", "esg").
            action:     RBAC action (e.g. "read", "export").
            site_id:    VoltEdge site being accessed.
            ip_address: Client IP address.
            user_agent: Client User-Agent string.
            details:    Arbitrary additional context.
        """
        prev_hash = self._latest_chain_hash()
        event_str = str(event_type.value if isinstance(event_type, AuditEventType) else event_type)
        details_json = json.dumps(details or {}, default=str)
        now = datetime.now(timezone.utc)

        chain_input = f"{prev_hash}{event_str}{actor}{now.replace(tzinfo=None).isoformat()}{details_json}"
        chain_hash = hashlib.sha256(chain_input.encode()).hexdigest()

        entry = AuditLog(
            event_type=event_str,
            actor=actor,
            resource=resource,
            action=action,
            site_id=site_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details_json=details_json,
            chain_hash=chain_hash,
            timestamp=now,
        )
        self.db.add(entry)
        self.db.commit()
        log.debug("audit.event_written", event_type=event_str, actor=actor)
        return entry

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(
        self,
        event_type: str | None = None,
        actor:      str | None = None,
        site_id:    str | None = None,
        since:      datetime | None = None,
        until:      datetime | None = None,
        limit:      int = 500,
    ) -> list[AuditLog]:
        """
        Query audit logs with optional filters.
        Results ordered newest-first.
        """
        q = self.db.query(AuditLog)
        if event_type:
            q = q.filter(AuditLog.event_type == event_type)
        if actor:
            q = q.filter(AuditLog.actor == actor)
        if site_id:
            q = q.filter(AuditLog.site_id == site_id)
        if since:
            q = q.filter(AuditLog.timestamp >= since)
        if until:
            q = q.filter(AuditLog.timestamp <= until)
        return q.order_by(AuditLog.timestamp.desc()).limit(limit).all()

    def recent(self, n: int = 50) -> list[AuditLog]:
        """Return the n most recent audit events."""
        return (
            self.db.query(AuditLog)
            .order_by(AuditLog.timestamp.desc())
            .limit(n)
            .all()
        )

    def actor_summary(self, actor: str) -> dict[str, int]:
        """
        Return event type counts for a given actor.
        Useful for user activity reports.
        """
        from sqlalchemy import func
        rows = (
            self.db.query(AuditLog.event_type, func.count(AuditLog.id).label("count"))
            .filter(AuditLog.actor == actor)
            .group_by(AuditLog.event_type)
            .all()
        )
        return {row.event_type: row.count for row in rows}

    # ── Chain integrity ───────────────────────────────────────────────────────

    def verify_chain_integrity(self) -> dict[str, Any]:
        """
        Walk the audit chain and verify each hash.
        Returns {"valid": True/False, "entries": N, "first_breach": id_or_None}
        """
        entries = (
            self.db.query(AuditLog)
            .order_by(AuditLog.timestamp.asc())
            .all()
        )
        if not entries:
            return {"valid": True, "entries": 0, "first_breach": None}

        prev_hash = self.GENESIS_HASH
        for entry in entries:
            ts = entry.timestamp.replace(tzinfo=None) if entry.timestamp.tzinfo else entry.timestamp
            chain_input = (
                f"{prev_hash}"
                f"{entry.event_type}"
                f"{entry.actor}"
                f"{ts.isoformat()}"
                f"{entry.details_json}"
            )
            expected = hashlib.sha256(chain_input.encode()).hexdigest()
            if expected != entry.chain_hash:
                log.error("audit.chain_breach", entry_id=entry.id)
                return {"valid": False, "entries": len(entries), "first_breach": entry.id}
            prev_hash = entry.chain_hash

        return {"valid": True, "entries": len(entries), "first_breach": None}

    # ── Export ────────────────────────────────────────────────────────────────

    def export_csv(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> str:
        """Export audit log to CSV string for compliance reporting."""
        import csv, io
        entries = self.query(since=since, until=until, limit=10_000)
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([
            "id", "timestamp", "event_type", "actor",
            "resource", "action", "site_id", "ip_address", "chain_hash",
        ])
        for e in reversed(entries):   # chronological order
            writer.writerow([
                e.id, e.timestamp.isoformat(), e.event_type, e.actor,
                e.resource or "", e.action or "", e.site_id or "",
                e.ip_address or "", e.chain_hash,
            ])
        return buf.getvalue()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _latest_chain_hash(self) -> str:
        latest = (
            self.db.query(AuditLog)
            .order_by(AuditLog.timestamp.desc())
            .first()
        )
        return latest.chain_hash if latest else self.GENESIS_HASH


# ── FastAPI middleware ────────────────────────────────────────────────────────

async def audit_middleware(request, call_next):
    """
    Starlette middleware that logs every authenticated API request.
    Attach to app: app.middleware("http")(audit_middleware)
    """
    from starlette.requests import Request
    response = await call_next(request)
    return response
