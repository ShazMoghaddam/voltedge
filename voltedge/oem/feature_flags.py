"""
VoltEdge OEM — Feature Flag System

Runtime feature toggles that work across three dimensions:
  1. Global  — platform-wide defaults (e.g. new feature rolling out)
  2. Tier    — enabled/disabled per subscription tier
  3. Tenant  — individual overrides per tenant (A/B tests, early access)

Resolution order (highest priority first):
  tenant override → tier default → global default → False

Flags are evaluated at request time — no restart required.
Supports gradual rollouts (0–100% of eligible tenants).

Built-in flags:
  - forecast_v2_model      New ensemble forecasting model (rolling out to Enterprise first)
  - lstm_streaming         Real-time LSTM inference (Enterprise EA)
  - esg_scope3             Scope 3 emissions reporting (EA)
  - marketplace_checkout   In-app subscription upgrade
  - dark_mode              UI dark mode (all tiers)
  - advanced_heatmaps      Enhanced heatmap visualisations
  - erp_sap                SAP IS-U integration
  - erp_oracle             Oracle Utilities integration
  - multi_region_failover  Automatic regional failover
  - pdf_export             ESG PDF export
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from voltedge.utils.logger import get_logger

log = get_logger(__name__)


# ── Tier enum (mirrors oem.tenants) ──────────────────────────────────────────

class Tier(str, Enum):
    STARTER      = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE   = "enterprise"
    ALL          = "all"         # All tiers


# ── Flag definition ───────────────────────────────────────────────────────────

@dataclass
class FlagDefinition:
    """
    Definition of a feature flag.

    Args:
        key:           Unique identifier (snake_case).
        description:   Human-readable description.
        default:       Default value if no override exists.
        tier_defaults: Per-tier enable/disable overrides.
        rollout_pct:   Gradual rollout — percentage of tenants enabled (0–100).
        tags:          Categorisation tags.
    """
    key:           str
    description:   str
    default:       bool                 = False
    tier_defaults: dict[Tier, bool]     = field(default_factory=dict)
    rollout_pct:   float                = 100.0
    tags:          list[str]            = field(default_factory=list)

    def is_enabled_for_tier(self, tier: Tier) -> bool:
        return self.tier_defaults.get(tier, self.default)

    def is_in_rollout(self, tenant_id: str) -> bool:
        """Deterministic rollout: hash tenant_id to a 0–100 bucket."""
        if self.rollout_pct >= 100.0:
            return True
        if self.rollout_pct <= 0.0:
            return False
        bucket = int(hashlib.md5(f"{self.key}:{tenant_id}".encode()).hexdigest(), 16) % 100
        return bucket < self.rollout_pct


# ── Built-in flag catalogue ───────────────────────────────────────────────────

FEATURE_FLAGS: dict[str, FlagDefinition] = {
    "forecast_v2_model": FlagDefinition(
        key="forecast_v2_model",
        description="Ensemble GBM+LSTM forecasting model (v2)",
        default=False,
        tier_defaults={
            Tier.STARTER: False, Tier.PROFESSIONAL: False, Tier.ENTERPRISE: True,
        },
        tags=["ml", "forecasting"],
    ),
    "lstm_streaming": FlagDefinition(
        key="lstm_streaming",
        description="Real-time LSTM inference (<1s response)",
        default=False,
        tier_defaults={Tier.ENTERPRISE: True},
        rollout_pct=25.0,   # 25% of Enterprise tenants in EA
        tags=["ml", "lstm", "early_access"],
    ),
    "esg_scope3": FlagDefinition(
        key="esg_scope3",
        description="Scope 3 supply chain emissions reporting",
        default=False,
        tier_defaults={Tier.PROFESSIONAL: False, Tier.ENTERPRISE: False},
        rollout_pct=0.0,    # Not released yet
        tags=["esg", "coming_soon"],
    ),
    "marketplace_checkout": FlagDefinition(
        key="marketplace_checkout",
        description="In-app subscription upgrade via AWS Marketplace",
        default=True,
        tier_defaults={Tier.ALL: True},
        tags=["billing", "marketplace"],
    ),
    "dark_mode": FlagDefinition(
        key="dark_mode",
        description="Dashboard dark mode UI",
        default=True,
        tier_defaults={Tier.ALL: True},
        tags=["ui"],
    ),
    "advanced_heatmaps": FlagDefinition(
        key="advanced_heatmaps",
        description="Enhanced heatmap visualisations with drill-down",
        default=False,
        tier_defaults={Tier.PROFESSIONAL: True, Tier.ENTERPRISE: True},
        tags=["ui", "dashboard"],
    ),
    "erp_sap": FlagDefinition(
        key="erp_sap",
        description="SAP IS-U / PM ERP integration",
        default=False,
        tier_defaults={Tier.ENTERPRISE: True},
        tags=["erp", "integration"],
    ),
    "erp_oracle": FlagDefinition(
        key="erp_oracle",
        description="Oracle Utilities CC&B integration",
        default=False,
        tier_defaults={Tier.ENTERPRISE: True},
        tags=["erp", "integration"],
    ),
    "multi_region_failover": FlagDefinition(
        key="multi_region_failover",
        description="Automatic regional failover routing",
        default=False,
        tier_defaults={Tier.ENTERPRISE: True},
        tags=["infrastructure", "ha"],
    ),
    "pdf_export": FlagDefinition(
        key="pdf_export",
        description="ESG PDF report export",
        default=False,
        tier_defaults={Tier.PROFESSIONAL: True, Tier.ENTERPRISE: True},
        tags=["esg", "reporting"],
    ),
}


# ── Feature flag service ──────────────────────────────────────────────────────

class FeatureFlagService:
    """
    Evaluates feature flags for a given tenant + tier context.

    Stores tenant overrides in memory (production: use Redis or DB).
    """

    def __init__(
        self,
        catalogue: dict[str, FlagDefinition] | None = None,
    ) -> None:
        self._flags:    dict[str, FlagDefinition] = catalogue or FEATURE_FLAGS
        self._overrides: dict[str, dict[str, bool]] = {}  # {tenant_id: {flag: bool}}

    # ── Evaluation ────────────────────────────────────────────────────────────

    def is_enabled(
        self,
        flag_key:  str,
        tenant_id: str,
        tier:      Tier = Tier.STARTER,
    ) -> bool:
        """
        Resolve a feature flag for a tenant.

        Resolution order:
          1. Tenant-specific override (highest priority)
          2. Rollout bucket check (if flag has rollout_pct < 100)
          3. Tier default
          4. Global default
        """
        flag = self._flags.get(flag_key)
        if flag is None:
            log.warning("feature_flags.unknown_key", key=flag_key)
            return False

        # 1. Tenant override
        tenant_overrides = self._overrides.get(tenant_id, {})
        if flag_key in tenant_overrides:
            return tenant_overrides[flag_key]

        # 2. Rollout gate (before tier check)
        if flag.rollout_pct < 100.0:
            if not flag.is_in_rollout(tenant_id):
                return False

        # 3. Tier default
        if tier in flag.tier_defaults:
            return flag.tier_defaults[tier]
        if Tier.ALL in flag.tier_defaults:
            return flag.tier_defaults[Tier.ALL]

        # 4. Global default
        return flag.default

    def get_all_flags(
        self, tenant_id: str, tier: Tier = Tier.STARTER
    ) -> dict[str, bool]:
        """Return the full flag map for a tenant."""
        return {
            key: self.is_enabled(key, tenant_id, tier)
            for key in self._flags
        }

    # ── Overrides ─────────────────────────────────────────────────────────────

    def set_override(self, tenant_id: str, flag_key: str, enabled: bool) -> None:
        """Set a tenant-specific flag override."""
        if flag_key not in self._flags:
            raise ValueError(f"Unknown flag: '{flag_key}'")
        if tenant_id not in self._overrides:
            self._overrides[tenant_id] = {}
        self._overrides[tenant_id][flag_key] = enabled
        log.info("feature_flags.override_set",
                 tenant=tenant_id, flag=flag_key, enabled=enabled)

    def clear_override(self, tenant_id: str, flag_key: str) -> None:
        """Remove a tenant-specific override (reverts to tier/global default)."""
        if tenant_id in self._overrides:
            self._overrides[tenant_id].pop(flag_key, None)

    def clear_all_overrides(self, tenant_id: str) -> None:
        """Remove all overrides for a tenant."""
        self._overrides.pop(tenant_id, None)

    # ── Admin / inspection ────────────────────────────────────────────────────

    def list_flags(self, tag: str | None = None) -> list[FlagDefinition]:
        flags = list(self._flags.values())
        if tag:
            flags = [f for f in flags if tag in f.tags]
        return flags

    def flag_summary(self) -> dict[str, Any]:
        return {
            "total_flags": len(self._flags),
            "enabled_globally": sum(1 for f in self._flags.values() if f.default),
            "rollout_flags": sum(1 for f in self._flags.values() if f.rollout_pct < 100),
            "ea_flags": sum(1 for f in self._flags.values() if "early_access" in f.tags),
        }

    def flags_for_tier(self, tier: Tier) -> dict[str, bool]:
        """Return which flags are enabled for a given tier (no tenant context)."""
        result = {}
        for key, flag in self._flags.items():
            if tier in flag.tier_defaults:
                result[key] = flag.tier_defaults[tier]
            elif Tier.ALL in flag.tier_defaults:
                result[key] = flag.tier_defaults[Tier.ALL]
            else:
                result[key] = flag.default
        return result
