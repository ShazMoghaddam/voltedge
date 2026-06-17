"""
VoltEdge OEM — White-Label Branding System

Allows OEM partners (Shell, BP, Eni, TotalEnergies) to deploy
VoltEdge under their own brand with custom:

  - Product name, logo URL, tagline
  - Colour palette (primary, secondary, accent, dark)
  - Custom domain (e.g. energyiq.shell.com)
  - Permitted feature set (feature flags per tenant)
  - Custom ESG carbon intensity factors (partner-specific grid data)
  - CSS theme overrides for the Dash dashboard

Branding is resolved at request time from:
  1. Request host header  → tenant lookup
  2. X-Tenant-ID header   → explicit override (API clients)
  3. Default (VoltEdge)   → if no match

Stored in SQLite (dev) / PostgreSQL (prod) via SQLAlchemy.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Boolean, Column, DateTime, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session

from voltedge.utils.logger import get_logger

log = get_logger(__name__)


class BrandingBase(DeclarativeBase):
    pass


# ── Database model ────────────────────────────────────────────────────────────

class TenantBranding(BrandingBase):
    """
    Per-tenant OEM branding configuration.
    One row per partner/customer.
    """
    __tablename__ = "tenant_branding"

    id:           Mapped[str] = mapped_column(String, primary_key=True,
                                               default=lambda: str(uuid.uuid4()))
    tenant_id:    Mapped[str] = mapped_column(String, unique=True, nullable=False)
    product_name: Mapped[str] = mapped_column(String, default="VoltEdge")
    tagline:      Mapped[str] = mapped_column(String, default="Enterprise Energy Intelligence")
    logo_url:     Mapped[str] = mapped_column(String, default="")
    favicon_url:  Mapped[str] = mapped_column(String, default="")
    custom_domain: Mapped[str | None] = mapped_column(String, nullable=True)

    # Colours (hex)
    color_primary:   Mapped[str] = mapped_column(String, default="#00d4ff")
    color_secondary: Mapped[str] = mapped_column(String, default="#0d0d1a")
    color_accent:    Mapped[str] = mapped_column(String, default="#7fff6b")
    color_text:      Mapped[str] = mapped_column(String, default="#e0e0e0")

    # Carbon intensity override (kg CO2e/kWh)
    carbon_intensity: Mapped[float | None] = mapped_column(default=None)

    # Serialised feature flags and CSS overrides
    feature_flags_json: Mapped[str] = mapped_column(Text, default="{}")
    css_overrides_json: Mapped[str] = mapped_column(Text, default="{}")

    is_active:    Mapped[bool]     = mapped_column(Boolean, default=True)
    created_at:   Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (UniqueConstraint("tenant_id", name="uq_tenant_id"),)

    @property
    def feature_flags(self) -> dict[str, bool]:
        return json.loads(self.feature_flags_json)

    @property
    def css_overrides(self) -> dict[str, str]:
        return json.loads(self.css_overrides_json)

    def has_feature(self, feature: str) -> bool:
        return self.feature_flags.get(feature, False)


# ── Python dataclass view (immutable, for passing around) ─────────────────────

@dataclass(frozen=True)
class BrandingProfile:
    """
    Resolved, immutable branding snapshot.
    Built from TenantBranding and used throughout the request lifecycle.
    """
    tenant_id:     str
    product_name:  str         = "VoltEdge"
    tagline:       str         = "Enterprise Energy Intelligence"
    logo_url:      str         = ""
    custom_domain: str         = ""
    color_primary:   str       = "#00d4ff"
    color_secondary: str       = "#0d0d1a"
    color_accent:    str       = "#7fff6b"
    color_text:      str       = "#e0e0e0"
    carbon_intensity: float | None = None
    feature_flags: dict[str, bool] = field(default_factory=dict)
    css_overrides: dict[str, str]  = field(default_factory=dict)

    @classmethod
    def from_db(cls, branding: TenantBranding) -> "BrandingProfile":
        return cls(
            tenant_id=branding.tenant_id,
            product_name=branding.product_name,
            tagline=branding.tagline,
            logo_url=branding.logo_url,
            custom_domain=branding.custom_domain or "",
            color_primary=branding.color_primary,
            color_secondary=branding.color_secondary,
            color_accent=branding.color_accent,
            color_text=branding.color_text,
            carbon_intensity=branding.carbon_intensity,
            feature_flags=branding.feature_flags,
            css_overrides=branding.css_overrides,
        )

    @classmethod
    def default(cls) -> "BrandingProfile":
        return cls(tenant_id="voltedge")

    def to_css_vars(self) -> str:
        """
        Emit a CSS :root block with all brand colour variables.
        Inject into Dash layout <style> tag for white-label theming.
        """
        overrides = self.css_overrides
        lines = [":root {"]
        lines.append(f"  --ve-color-primary:   {overrides.get('color_primary',   self.color_primary)};")
        lines.append(f"  --ve-color-secondary: {overrides.get('color_secondary', self.color_secondary)};")
        lines.append(f"  --ve-color-accent:    {overrides.get('color_accent',    self.color_accent)};")
        lines.append(f"  --ve-color-text:      {overrides.get('color_text',      self.color_text)};")
        lines.append("}")
        return "\n".join(lines)

    def dashboard_title(self) -> str:
        return f"{self.product_name} | Energy Intelligence"


# ── Branding service ──────────────────────────────────────────────────────────

# Built-in OEM partner catalogue (pre-seeded for demos)
BUILTIN_TENANTS: list[dict[str, Any]] = [
    {
        "tenant_id":    "voltedge",
        "product_name": "VoltEdge",
        "tagline":      "Enterprise Energy Intelligence",
        "color_primary": "#00d4ff",
        "color_secondary": "#0d0d1a",
        "color_accent":  "#7fff6b",
        "feature_flags": {
            "forecast": True, "anomaly": True, "esg": True,
            "optimizer": True, "lstm": True, "pdf_export": True,
            "multi_site": True, "erp_integration": True,
        },
    },
    {
        "tenant_id":    "shell-energy",
        "product_name": "Shell EnergyIQ",
        "tagline":      "Powering the Energy Transition",
        "custom_domain": "energyiq.shell.com",
        "logo_url":     "https://cdn.shell.com/logo.svg",
        "color_primary":  "#FBCE07",   # Shell yellow
        "color_secondary": "#DD1D21",  # Shell red
        "color_accent":   "#ffffff",
        "carbon_intensity": 0.195,      # Shell's reported 2023 intensity
        "feature_flags": {
            "forecast": True, "anomaly": True, "esg": True,
            "optimizer": True, "lstm": True, "pdf_export": True,
            "multi_site": True, "erp_integration": True, "shell_trading": True,
        },
    },
    {
        "tenant_id":    "bp-energy",
        "product_name": "bp EnergyOS",
        "tagline":      "Reimagining Energy",
        "custom_domain": "energyos.bp.com",
        "logo_url":     "https://cdn.bp.com/logo.svg",
        "color_primary":  "#009900",   # bp green
        "color_secondary": "#ffffff",
        "color_accent":   "#009900",
        "feature_flags": {
            "forecast": True, "anomaly": True, "esg": True,
            "optimizer": True, "lstm": True, "pdf_export": True,
            "multi_site": True, "erp_integration": True,
        },
    },
    {
        "tenant_id":    "eni-analytics",
        "product_name": "Eni Energy Analytics",
        "tagline":      "Sustainable Energy, Sustainable Future",
        "custom_domain": "analytics.eni.com",
        "color_primary":  "#FFCB07",   # Eni yellow
        "color_secondary": "#1A1A1A",
        "color_accent":   "#FFCB07",
        "feature_flags": {
            "forecast": True, "anomaly": True, "esg": True,
            "optimizer": True, "lstm": False,  "pdf_export": True,
            "multi_site": True, "erp_integration": False,
        },
    },
]


class BrandingService:
    """CRUD + resolution for tenant branding."""

    def __init__(self, db: Session) -> None:
        self.db = db

    # ── Seed ──────────────────────────────────────────────────────────────────

    def seed_builtin_tenants(self) -> None:
        """Idempotent: create built-in OEM tenants if not present."""
        for t in BUILTIN_TENANTS:
            existing = self.db.query(TenantBranding).filter_by(
                tenant_id=t["tenant_id"]
            ).first()
            if not existing:
                flags = t.pop("feature_flags", {})
                branding = TenantBranding(
                    **t,
                    feature_flags_json=json.dumps(flags),
                )
                t["feature_flags"] = flags  # restore for idempotency
                self.db.add(branding)
        self.db.commit()
        log.info("branding.seeded", tenants=len(BUILTIN_TENANTS))

    # ── Resolve ───────────────────────────────────────────────────────────────

    def resolve(
        self,
        tenant_id: str | None = None,
        host: str | None = None,
    ) -> BrandingProfile:
        """
        Resolve branding for a request.

        Priority:
          1. tenant_id (from X-Tenant-ID header or JWT claim)
          2. host (domain-based lookup)
          3. Default VoltEdge branding
        """
        if tenant_id:
            row = self.db.query(TenantBranding).filter_by(
                tenant_id=tenant_id, is_active=True
            ).first()
            if row:
                return BrandingProfile.from_db(row)

        if host:
            # Strip port
            domain = host.split(":")[0]
            row = self.db.query(TenantBranding).filter_by(
                custom_domain=domain, is_active=True
            ).first()
            if row:
                return BrandingProfile.from_db(row)

        return BrandingProfile.default()

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def create(
        self,
        tenant_id: str,
        product_name: str,
        **kwargs,
    ) -> TenantBranding:
        if self.db.query(TenantBranding).filter_by(tenant_id=tenant_id).first():
            raise ValueError(f"Tenant '{tenant_id}' already exists.")
        flags = kwargs.pop("feature_flags", {})
        css   = kwargs.pop("css_overrides", {})
        branding = TenantBranding(
            tenant_id=tenant_id,
            product_name=product_name,
            feature_flags_json=json.dumps(flags),
            css_overrides_json=json.dumps(css),
            **kwargs,
        )
        self.db.add(branding)
        self.db.commit()
        self.db.refresh(branding)
        log.info("branding.created", tenant=tenant_id)
        return branding

    def update_feature_flag(self, tenant_id: str, feature: str, enabled: bool) -> None:
        row = self.db.query(TenantBranding).filter_by(tenant_id=tenant_id).first()
        if not row:
            raise ValueError(f"Tenant '{tenant_id}' not found.")
        flags = row.feature_flags
        flags[feature] = enabled
        row.feature_flags_json = json.dumps(flags)
        self.db.commit()
        log.info("branding.flag_updated", tenant=tenant_id, feature=feature, enabled=enabled)

    def get(self, tenant_id: str) -> TenantBranding | None:
        return self.db.query(TenantBranding).filter_by(tenant_id=tenant_id).first()

    def list_tenants(self) -> list[TenantBranding]:
        return self.db.query(TenantBranding).filter_by(is_active=True).all()

    def deactivate(self, tenant_id: str) -> None:
        row = self.db.query(TenantBranding).filter_by(tenant_id=tenant_id).first()
        if row:
            row.is_active = False
            self.db.commit()
            log.info("branding.deactivated", tenant=tenant_id)
