"""
VoltEdge Carbon — Credit Registry Integration

Connects to voluntary carbon market registries to:
  - Verify renewable energy certificates (RECs / GOOs)
  - Issue carbon credits against verified ESG reductions
  - Track credit retirement for Scope 2 market-based reporting
  - Query current credit prices for valuation

Supported registries:
  Verra VCS     — Verified Carbon Standard (largest voluntary market)
  Gold Standard — Premium-quality credits with SDG co-benefits
  IREC          — International REC Standard (renewable energy)
  REGO          — UK Renewable Energy Guarantees of Origin

Architecture:
  BaseRegistry (ABC) — defines the interface
  VerraRegistry      — Verra API v1 (api.verra.org)
  GoldStandardRegistry
  MockRegistry       — deterministic test double; no HTTP

All registry adapters use dry_run=True in tests — no real API calls.
Credits are issued when VoltEdge verifies renewable consumption
against metered ESG data and submits a claim to the registry.
"""

from __future__ import annotations

import hashlib
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import httpx

from voltedge.utils.logger import get_logger

log = get_logger(__name__)


# ── Enums ─────────────────────────────────────────────────────────────────────

class CreditStatus(str, Enum):
    PENDING   = "pending"    # Claim submitted, awaiting verification
    ISSUED    = "issued"     # Credit issued and held in account
    RETIRED   = "retired"    # Credit retired against Scope 2 claim
    CANCELLED = "cancelled"  # Claim rejected or withdrawn


class RegistryType(str, Enum):
    VERRA        = "verra"
    GOLD_STANDARD = "gold_standard"
    IREC         = "irec"
    REGO         = "rego"
    MOCK         = "mock"


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class CarbonCredit:
    """
    One tonne of CO₂e reduced / avoided / removed, represented as a credit.

    Fields follow the Verra VCS schema; Gold Standard uses the same
    logical structure with different field names mapped at parse time.
    """
    credit_id:      str            # Registry-assigned serial number
    registry:       RegistryType
    project_id:     str            # Registry project ID
    project_name:   str
    vintage_year:   int            # Year the reduction occurred
    quantity_tco2e: float          # Tonnes CO₂e this credit represents
    status:         CreditStatus   = CreditStatus.ISSUED
    methodology:    str            = ""
    country:        str            = ""
    sdg_goals:      list[int]      = field(default_factory=list)  # UN SDGs
    issued_at:      datetime       = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    retired_at:     datetime | None = None
    retired_by:     str            = ""   # Entity retiring the credit
    price_usd:      float | None   = None

    @property
    def is_retired(self) -> bool:
        return self.status == CreditStatus.RETIRED

    def retire(self, entity: str) -> "CarbonCredit":
        """Return a new retired credit (credits are immutable after issuance)."""
        from dataclasses import replace
        return replace(
            self,
            status=CreditStatus.RETIRED,
            retired_at=datetime.now(timezone.utc),
            retired_by=entity,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "credit_id":      self.credit_id,
            "registry":       self.registry.value,
            "project_id":     self.project_id,
            "project_name":   self.project_name,
            "vintage_year":   self.vintage_year,
            "quantity_tco2e": self.quantity_tco2e,
            "status":         self.status.value,
            "methodology":    self.methodology,
            "country":        self.country,
            "sdg_goals":      self.sdg_goals,
            "issued_at":      self.issued_at.isoformat(),
            "retired_at":     self.retired_at.isoformat() if self.retired_at else None,
            "retired_by":     self.retired_by,
            "price_usd":      self.price_usd,
        }


@dataclass
class CreditClaim:
    """
    A claim submitted to the registry for credit issuance.
    Links VoltEdge ESG data to a registry submission.
    """
    claim_id:        str  = field(default_factory=lambda: str(uuid.uuid4()))
    site_id:         str  = ""
    period_start:    datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    period_end:      datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    kwh_renewable:   float = 0.0
    co2e_avoided_t:  float = 0.0    # tCO₂e
    project_id:      str  = ""
    registry:        RegistryType = RegistryType.MOCK
    evidence_hash:   str  = ""      # SHA-256 of the underlying ESG data
    status:          CreditStatus = CreditStatus.PENDING
    issued_credit_id: str | None = None

    def evidence_digest(self, data: str) -> "CreditClaim":
        """Attach a tamper-evident hash of the supporting ESG evidence."""
        from dataclasses import replace
        return replace(
            self,
            evidence_hash=hashlib.sha256(data.encode()).hexdigest()
        )


# ── Base registry ─────────────────────────────────────────────────────────────

class BaseRegistry(ABC):
    """Abstract carbon credit registry adapter."""

    registry_type: RegistryType = RegistryType.MOCK

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run

    @abstractmethod
    def submit_claim(self, claim: CreditClaim) -> CreditClaim:
        """Submit a credit claim. Returns updated claim with registry response."""

    @abstractmethod
    def get_credit(self, credit_id: str) -> CarbonCredit | None:
        """Retrieve a specific credit by its serial number."""

    @abstractmethod
    def list_credits(
        self,
        project_id: str | None = None,
        status: CreditStatus | None = None,
    ) -> list[CarbonCredit]:
        """List credits, optionally filtered by project or status."""

    @abstractmethod
    def retire_credit(self, credit_id: str, entity: str) -> CarbonCredit:
        """Retire a credit against a Scope 2 claim."""

    @abstractmethod
    def get_credit_price(self, project_id: str) -> float | None:
        """Return the current spot price per tCO₂e (USD)."""


# ── Mock registry (tests + demos) ─────────────────────────────────────────────

class MockRegistry(BaseRegistry):
    """
    In-memory registry for testing and demos.
    Deterministic: same claim → same credit_id every time.
    No HTTP calls made.
    """

    registry_type = RegistryType.MOCK

    MOCK_PROJECTS = {
        "VCS-001": {
            "name":        "Brazilian Amazon REDD+",
            "methodology": "VM0015",
            "country":     "BR",
            "sdg_goals":   [13, 15, 1],
            "price_usd":   14.50,
        },
        "GS-002": {
            "name":        "Kenya Clean Cookstoves",
            "methodology": "AMS-II.G",
            "country":     "KE",
            "sdg_goals":   [3, 7, 13],
            "price_usd":   22.80,
        },
        "IREC-003": {
            "name":        "UK Offshore Wind",
            "methodology": "IREC-RE",
            "country":     "GB",
            "sdg_goals":   [7, 13],
            "price_usd":   8.20,
        },
        "REGO-004": {
            "name":        "Scottish Hydro Power",
            "methodology": "REGO-REG",
            "country":     "GB",
            "sdg_goals":   [7],
            "price_usd":   6.50,
        },
    }

    DEFAULT_PROJECT = "IREC-003"

    def __init__(self) -> None:
        super().__init__(dry_run=True)
        self._credits: dict[str, CarbonCredit] = {}
        self._claims:  dict[str, CreditClaim]  = {}

    def submit_claim(self, claim: CreditClaim) -> CreditClaim:
        if claim.co2e_avoided_t <= 0:
            raise ValueError(f"co2e_avoided_t must be > 0, got {claim.co2e_avoided_t}")

        project_id = claim.project_id or self.DEFAULT_PROJECT
        project    = self.MOCK_PROJECTS.get(project_id, self.MOCK_PROJECTS[self.DEFAULT_PROJECT])

        # Deterministic credit_id for reproducibility
        credit_id = "MOCK-" + hashlib.sha256(
            f"{claim.claim_id}{project_id}{claim.co2e_avoided_t}".encode()
        ).hexdigest()[:12].upper()

        credit = CarbonCredit(
            credit_id=credit_id,
            registry=RegistryType.MOCK,
            project_id=project_id,
            project_name=project["name"],
            vintage_year=datetime.now(timezone.utc).year,
            quantity_tco2e=claim.co2e_avoided_t,
            status=CreditStatus.ISSUED,
            methodology=project["methodology"],
            country=project["country"],
            sdg_goals=project["sdg_goals"],
            price_usd=project["price_usd"],
        )
        self._credits[credit_id] = credit

        from dataclasses import replace
        updated_claim = replace(
            claim,
            status=CreditStatus.ISSUED,
            issued_credit_id=credit_id,
        )
        self._claims[claim.claim_id] = updated_claim
        log.info("mock_registry.credit_issued",
                 credit_id=credit_id, tco2e=credit.quantity_tco2e)
        return updated_claim

    def get_credit(self, credit_id: str) -> CarbonCredit | None:
        return self._credits.get(credit_id)

    def list_credits(
        self,
        project_id: str | None = None,
        status: CreditStatus | None = None,
    ) -> list[CarbonCredit]:
        credits = list(self._credits.values())
        if project_id:
            credits = [c for c in credits if c.project_id == project_id]
        if status:
            credits = [c for c in credits if c.status == status]
        return credits

    def retire_credit(self, credit_id: str, entity: str) -> CarbonCredit:
        credit = self._credits.get(credit_id)
        if not credit:
            raise ValueError(f"Credit '{credit_id}' not found.")
        if credit.is_retired:
            raise ValueError(f"Credit '{credit_id}' is already retired.")
        retired = credit.retire(entity)
        self._credits[credit_id] = retired
        log.info("mock_registry.credit_retired",
                 credit_id=credit_id, entity=entity)
        return retired

    def get_credit_price(self, project_id: str) -> float | None:
        project = self.MOCK_PROJECTS.get(project_id)
        return project["price_usd"] if project else None

    def portfolio_value(self) -> dict[str, Any]:
        """Total portfolio value of all issued (non-retired) credits."""
        active = [c for c in self._credits.values()
                  if c.status == CreditStatus.ISSUED]
        total_tco2e = sum(c.quantity_tco2e for c in active)
        total_value = sum(
            c.quantity_tco2e * (c.price_usd or 0) for c in active
        )
        return {
            "active_credits":  len(active),
            "total_tco2e":     round(total_tco2e, 4),
            "total_value_usd": round(total_value, 2),
            "by_project": {
                pid: round(sum(c.quantity_tco2e for c in active if c.project_id == pid), 4)
                for pid in {c.project_id for c in active}
            },
        }


# ── Verra registry (production) ───────────────────────────────────────────────

class VerraRegistry(BaseRegistry):
    """
    Verra VCS registry adapter.
    https://registry.verra.org/app/search/VCS

    API reference: https://api.verra.org/v1/docs
    All calls require an API key (set VERRA_API_KEY env var).
    Use dry_run=True for local development.
    """

    registry_type = RegistryType.VERRA
    BASE_URL      = "https://api.verra.org/v1"

    def __init__(self, api_key: str = "", dry_run: bool = False) -> None:
        super().__init__(dry_run=dry_run)
        self._api_key = api_key

    def submit_claim(self, claim: CreditClaim) -> CreditClaim:
        if self.dry_run:
            log.info("verra.submit_claim.dry_run", claim_id=claim.claim_id)
            from dataclasses import replace
            return replace(claim, status=CreditStatus.PENDING)

        with httpx.Client(timeout=30) as client:
            resp = client.post(
                f"{self.BASE_URL}/credits/claim",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "projectId":   claim.project_id,
                    "quantity":    claim.co2e_avoided_t,
                    "periodStart": claim.period_start.isoformat(),
                    "periodEnd":   claim.period_end.isoformat(),
                    "evidenceHash": claim.evidence_hash,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            from dataclasses import replace
            return replace(claim, status=CreditStatus.PENDING,
                           issued_credit_id=data.get("claimId"))

    def get_credit(self, credit_id: str) -> CarbonCredit | None:
        if self.dry_run:
            return None
        with httpx.Client(timeout=30) as client:
            resp = client.get(
                f"{self.BASE_URL}/credits/{credit_id}",
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return self._parse_verra_credit(resp.json())

    def list_credits(self, project_id=None, status=None) -> list[CarbonCredit]:
        if self.dry_run:
            return []
        params: dict = {}
        if project_id:
            params["projectId"] = project_id
        if status:
            params["status"] = status.value
        with httpx.Client(timeout=30) as client:
            resp = client.get(
                f"{self.BASE_URL}/credits",
                headers={"Authorization": f"Bearer {self._api_key}"},
                params=params,
            )
            resp.raise_for_status()
            return [self._parse_verra_credit(c) for c in resp.json().get("items", [])]

    def retire_credit(self, credit_id: str, entity: str) -> CarbonCredit:
        if self.dry_run:
            raise RuntimeError("Cannot retire credits in dry_run mode.")
        with httpx.Client(timeout=30) as client:
            resp = client.post(
                f"{self.BASE_URL}/credits/{credit_id}/retire",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"retiredBy": entity},
            )
            resp.raise_for_status()
            return self._parse_verra_credit(resp.json())

    def get_credit_price(self, project_id: str) -> float | None:
        if self.dry_run:
            return None
        with httpx.Client(timeout=30) as client:
            resp = client.get(
                f"{self.BASE_URL}/projects/{project_id}/price",
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json().get("spotPriceUsd")

    @staticmethod
    def _parse_verra_credit(data: dict) -> CarbonCredit:
        return CarbonCredit(
            credit_id=data["serialNumber"],
            registry=RegistryType.VERRA,
            project_id=data.get("projectId", ""),
            project_name=data.get("projectName", ""),
            vintage_year=int(data.get("vintageYear", 0)),
            quantity_tco2e=float(data.get("quantity", 0)),
            status=CreditStatus(data.get("status", "issued")),
            methodology=data.get("methodology", ""),
            country=data.get("country", ""),
            sdg_goals=data.get("sdgGoals", []),
        )
