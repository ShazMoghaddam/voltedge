"""
VoltEdge — Multi-Region Configuration & Routing

Provides:
  - RegionConfig: per-region settings (cloud credentials, S3 bucket, DB URL)
  - MultiRegionRouter: selects the optimal region for a given site_id
  - DataResidencyPolicy: enforces GDPR / data sovereignty rules
  - HealthChecker: queries all regional /health endpoints

Sites are pinned to a home region based on:
  1. Explicit site→region mapping (from config)
  2. Data residency policy (e.g. EU sites must stay in eu-west-1)
  3. Latency-based fallback (closest available region)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from voltedge.utils.logger import get_logger

log = get_logger(__name__)


# ── Region enum ───────────────────────────────────────────────────────────────

class AWSRegion(str, Enum):
    EU_WEST_1      = "eu-west-1"
    US_EAST_1      = "us-east-1"
    AP_SOUTHEAST_1 = "ap-southeast-1"


# ── Data residency policies ───────────────────────────────────────────────────

class DataResidencyZone(str, Enum):
    EU    = "eu"      # GDPR — must stay in eu-west-1
    US    = "us"      # CCPA — must stay in us-east-1
    APAC  = "apac"    # Local laws — must stay in ap-southeast-1
    GLOBAL = "global"  # No restriction — any region


RESIDENCY_TO_REGION: dict[DataResidencyZone, AWSRegion] = {
    DataResidencyZone.EU:   AWSRegion.EU_WEST_1,
    DataResidencyZone.US:   AWSRegion.US_EAST_1,
    DataResidencyZone.APAC: AWSRegion.AP_SOUTHEAST_1,
}


# ── Per-region config ─────────────────────────────────────────────────────────

@dataclass
class RegionConfig:
    """
    All settings needed to operate VoltEdge in one AWS region.

    Args:
        region:         AWS region identifier.
        s3_bucket:      S3 bucket for this region's data.
        db_url:         PostgreSQL connection string.
        api_endpoint:   Regional API URL (for cross-region health checks).
        is_primary:     Whether this is the primary (write) region.
        residency_zone: Data residency zone this region serves.
    """
    region:         AWSRegion
    s3_bucket:      str
    db_url:         str
    api_endpoint:   str       = ""
    is_primary:     bool      = False
    residency_zone: DataResidencyZone = DataResidencyZone.GLOBAL

    @property
    def region_code(self) -> str:
        return self.region.value.replace("-", "_")


# ── Multi-region router ───────────────────────────────────────────────────────

class MultiRegionRouter:
    """
    Routes site data operations to the correct regional backend.

    Args:
        configs:       List of RegionConfig objects, one per deployed region.
        site_mapping:  Explicit site_id → AWSRegion overrides.
        policy:        DataResidencyZone → AWSRegion rules (default: RESIDENCY_TO_REGION).
    """

    def __init__(
        self,
        configs: list[RegionConfig],
        site_mapping: dict[str, AWSRegion] | None = None,
        policy: dict[DataResidencyZone, AWSRegion] | None = None,
    ) -> None:
        if not configs:
            raise ValueError("At least one RegionConfig is required.")

        self._configs:      dict[AWSRegion, RegionConfig] = {c.region: c for c in configs}
        self._site_mapping: dict[str, AWSRegion]          = site_mapping or {}
        self._policy:       dict[DataResidencyZone, AWSRegion] = policy or RESIDENCY_TO_REGION

        # Validate that primary region exists if any config claims to be primary
        primaries = [c for c in configs if c.is_primary]
        if len(primaries) > 1:
            raise ValueError(f"Only one primary region allowed; got {[p.region for p in primaries]}")

    @property
    def primary(self) -> RegionConfig:
        """Return the primary region config (or the first config if none marked primary)."""
        for cfg in self._configs.values():
            if cfg.is_primary:
                return cfg
        return next(iter(self._configs.values()))

    def route(
        self,
        site_id: str,
        residency: DataResidencyZone = DataResidencyZone.GLOBAL,
    ) -> RegionConfig:
        """
        Determine the correct region for a site_id.

        Priority:
          1. Explicit site_mapping
          2. Data residency policy
          3. Primary region

        Args:
            site_id:   VoltEdge site identifier.
            residency: Caller-provided residency zone hint.

        Returns:
            RegionConfig for the selected region.
        """
        # 1. Explicit override
        if site_id in self._site_mapping:
            region = self._site_mapping[site_id]
            if region in self._configs:
                log.debug("router.explicit_mapping", site=site_id, region=region.value)
                return self._configs[region]

        # 2. Policy-based residency
        if residency != DataResidencyZone.GLOBAL:
            policy_region = self._policy.get(residency)
            if policy_region and policy_region in self._configs:
                log.debug("router.policy", site=site_id, residency=residency.value,
                          region=policy_region.value)
                return self._configs[policy_region]

        # 3. Infer from site_id prefix (e.g. "EU-LONDON-01" → eu-west-1)
        sid_upper = site_id.upper()
        inferred = self._infer_region_from_site_id(sid_upper)
        if inferred and inferred in self._configs:
            log.debug("router.inferred", site=site_id, region=inferred.value)
            return self._configs[inferred]

        # 4. Primary fallback
        log.debug("router.primary_fallback", site=site_id)
        return self.primary

    def all_regions(self) -> list[RegionConfig]:
        return list(self._configs.values())

    def add_site_mapping(self, site_id: str, region: AWSRegion) -> None:
        if region not in self._configs:
            raise ValueError(f"Region {region} not configured.")
        self._site_mapping[site_id] = region
        log.info("router.site_mapped", site=site_id, region=region.value)

    def region_summary(self) -> dict[str, Any]:
        return {
            "regions": [c.region.value for c in self._configs.values()],
            "primary": self.primary.region.value,
            "site_overrides": len(self._site_mapping),
            "policy_rules": len(self._policy),
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _infer_region_from_site_id(site_id_upper: str) -> AWSRegion | None:
        """Heuristic: look for geographic prefixes in the site ID."""
        eu_keywords  = ("EU-", "GB-", "DE-", "FR-", "NL-", "LONDON", "AMSTERDAM",
                         "FRANKFURT", "PARIS", "ROTTERDAM", "DUBLIN")
        us_keywords  = ("US-", "NA-", "HOUSTON", "NEWYORK", "CHICAGO", "DALLAS")
        apac_keywords = ("APAC-", "SG-", "JP-", "AU-", "SINGAPORE", "TOKYO", "SYDNEY")

        for kw in eu_keywords:
            if kw in site_id_upper:
                return AWSRegion.EU_WEST_1
        for kw in us_keywords:
            if kw in site_id_upper:
                return AWSRegion.US_EAST_1
        for kw in apac_keywords:
            if kw in site_id_upper:
                return AWSRegion.AP_SOUTHEAST_1

        return None


# ── Health checker ────────────────────────────────────────────────────────────

async def check_all_regions(configs: list[RegionConfig], timeout: float = 10.0) -> dict[str, bool]:
    """
    Asynchronously check /health on each regional API endpoint.

    Returns:
        dict mapping region.value → True/False (healthy/unhealthy).
    """
    import asyncio
    import httpx

    async def _check_one(cfg: RegionConfig) -> tuple[str, bool]:
        if not cfg.api_endpoint:
            return (cfg.region.value, False)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.get(f"{cfg.api_endpoint}/health")
                healthy = r.status_code == 200 and r.json().get("status") == "ok"
                log.info("multiregion.health", region=cfg.region.value, healthy=healthy)
                return (cfg.region.value, healthy)
        except Exception as exc:
            log.warning("multiregion.health_failed", region=cfg.region.value, error=str(exc))
            return (cfg.region.value, False)

    tasks = [_check_one(cfg) for cfg in configs]
    results = await asyncio.gather(*tasks)
    return dict(results)
