"""
VoltEdge Carbon — Credit Issuance & Portfolio Management

Bridges VoltEdge's ESG data layer with carbon registries.
Takes verified energy data, calculates avoided emissions,
and submits claims for carbon credit issuance.

Workflow:
  1. Pull processed energy data for a site + period
  2. Compute renewable fraction and avoided tCO₂e
  3. Build a CreditClaim with tamper-evident evidence hash
  4. Submit to the chosen registry
  5. Track issued credits in the portfolio ledger

Credit accounting follows GHG Protocol Scope 2 market-based method:
  avoided_tco2e = kwh_renewable × grid_emission_factor_kg_per_kwh / 1000

Price valuation uses registry spot prices where available,
falling back to a configurable default market price.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from voltedge.carbon.registry import (
    BaseRegistry, CarbonCredit, CreditClaim,
    CreditStatus, MockRegistry, RegistryType,
)
from voltedge.utils.logger import get_logger

log = get_logger(__name__)

# Default voluntary carbon market price (USD per tCO₂e)
# Based on BloombergNEF / Ecosystem Marketplace 2024 data
DEFAULT_MARKET_PRICE_USD = 15.0


@dataclass
class IssuanceResult:
    """Outcome of a credit issuance request."""
    claim:          CreditClaim
    credits_issued: list[CarbonCredit] = field(default_factory=list)
    tco2e_issued:   float = 0.0
    portfolio_value_usd: float = 0.0
    success:        bool  = False
    message:        str   = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id":           self.claim.claim_id,
            "site_id":            self.claim.site_id,
            "status":             self.claim.status.value,
            "tco2e_issued":       self.tco2e_issued,
            "portfolio_value_usd": self.portfolio_value_usd,
            "credit_ids":         [c.credit_id for c in self.credits_issued],
            "success":            self.success,
            "message":            self.message,
        }


class CreditIssuanceService:
    """
    Issues carbon credits from verified VoltEdge energy data.

    Args:
        registry:              Registry adapter (MockRegistry for dev/test).
        default_project_id:    Default registry project to issue against.
        renewable_fraction:    What fraction of consumption is renewable (0-1).
        grid_ef_kg_per_kwh:    Grid emission factor for avoided emissions calc.
        market_price_usd:      Fallback credit price if registry has no quote.
    """

    def __init__(
        self,
        registry:           BaseRegistry | None = None,
        default_project_id: str   = "IREC-003",
        renewable_fraction: float = 0.30,
        grid_ef_kg_per_kwh: float = 0.207,
        market_price_usd:   float = DEFAULT_MARKET_PRICE_USD,
    ) -> None:
        self._registry    = registry or MockRegistry()
        self._project_id  = default_project_id
        self._ren_frac    = min(1.0, max(0.0, renewable_fraction))
        self._grid_ef     = grid_ef_kg_per_kwh
        self._market_price = market_price_usd
        self._ledger:  list[CreditClaim]  = []
        self._credits: dict[str, CarbonCredit] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def issue_from_data(
        self,
        df:                 pd.DataFrame,
        site_id:            str,
        period_start:       datetime | None = None,
        period_end:         datetime | None = None,
        renewable_fraction: float | None = None,
        project_id:         str | None = None,
    ) -> IssuanceResult:
        """
        Issue credits from a processed energy DataFrame.

        Args:
            df:                 Processed DataFrame with 'kwh' and 'timestamp'.
            site_id:            Site identifier.
            period_start/end:   Override period (defaults to df timestamp range).
            renewable_fraction: Override for this claim (defaults to class default).
            project_id:         Registry project to issue against.

        Returns:
            IssuanceResult with the issued credit(s).
        """
        if df.empty or "kwh" not in df.columns:
            return IssuanceResult(
                claim=CreditClaim(site_id=site_id),
                success=False,
                message="No energy data available.",
            )

        ren_frac   = renewable_fraction if renewable_fraction is not None else self._ren_frac
        project_id = project_id or self._project_id

        total_kwh      = float(df["kwh"].sum())
        kwh_renewable  = total_kwh * ren_frac
        co2e_avoided_t = kwh_renewable * self._grid_ef / 1000.0  # kg → tCO₂e

        if co2e_avoided_t < 0.001:
            return IssuanceResult(
                claim=CreditClaim(site_id=site_id),
                success=False,
                message=(
                    f"Avoided emissions ({co2e_avoided_t:.6f} tCO₂e) below "
                    "minimum issuance threshold of 0.001 tCO₂e."
                ),
            )

        # Derive period from data if not specified
        if period_start is None and "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], utc=True)
            period_start = ts.min().to_pydatetime()
            period_end   = ts.max().to_pydatetime()
        period_start = period_start or datetime.now(timezone.utc)
        period_end   = period_end   or datetime.now(timezone.utc)

        # Build tamper-evident evidence hash
        evidence_data = json.dumps({
            "site_id":      site_id,
            "total_kwh":    round(total_kwh, 4),
            "kwh_renewable": round(kwh_renewable, 4),
            "ren_fraction": ren_frac,
            "grid_ef":      self._grid_ef,
            "period_start": period_start.isoformat(),
            "period_end":   period_end.isoformat(),
        }, sort_keys=True)

        claim = CreditClaim(
            site_id=site_id,
            period_start=period_start,
            period_end=period_end,
            kwh_renewable=round(kwh_renewable, 4),
            co2e_avoided_t=round(co2e_avoided_t, 6),
            project_id=project_id,
            registry=self._registry.registry_type,
        ).evidence_digest(evidence_data)

        try:
            claim = self._registry.submit_claim(claim)
        except Exception as exc:
            log.error("credits.submit_failed", site=site_id, error=str(exc))
            return IssuanceResult(
                claim=claim, success=False, message=str(exc)
            )

        # Fetch issued credit
        issued_credits: list[CarbonCredit] = []
        if claim.issued_credit_id:
            credit = self._registry.get_credit(claim.issued_credit_id)
            if credit:
                price = (
                    self._registry.get_credit_price(project_id)
                    or self._market_price
                )
                from dataclasses import replace
                credit = replace(credit, price_usd=price)
                issued_credits.append(credit)
                self._credits[credit.credit_id] = credit

        self._ledger.append(claim)

        portfolio_value = sum(
            c.quantity_tco2e * (c.price_usd or self._market_price)
            for c in self._credits.values()
            if c.status == CreditStatus.ISSUED
        )

        log.info(
            "credits.issued",
            site=site_id,
            tco2e=co2e_avoided_t,
            credit_ids=[c.credit_id for c in issued_credits],
        )

        return IssuanceResult(
            claim=claim,
            credits_issued=issued_credits,
            tco2e_issued=round(co2e_avoided_t, 6),
            portfolio_value_usd=round(portfolio_value, 2),
            success=claim.status in (CreditStatus.ISSUED, CreditStatus.PENDING),
            message=f"Issued {len(issued_credits)} credit(s) for {co2e_avoided_t:.4f} tCO₂e.",
        )

    def retire_for_scope2(self, credit_id: str, entity: str) -> CarbonCredit:
        """
        Retire a credit to offset Scope 2 market-based emissions.
        Returns the retired credit.
        """
        retired = self._registry.retire_credit(credit_id, entity)
        if credit_id in self._credits:
            self._credits[credit_id] = retired
        log.info("credits.retired_for_scope2",
                 credit_id=credit_id, entity=entity)
        return retired

    def bulk_issue(
        self,
        site_data: dict[str, pd.DataFrame],
        renewable_fraction: float | None = None,
        project_id:         str | None = None,
    ) -> list[IssuanceResult]:
        """Issue credits for multiple sites in one operation."""
        return [
            self.issue_from_data(
                df=df,
                site_id=site_id,
                renewable_fraction=renewable_fraction,
                project_id=project_id,
            )
            for site_id, df in site_data.items()
        ]

    # ── Portfolio ─────────────────────────────────────────────────────────────

    def portfolio_summary(self) -> dict[str, Any]:
        """Return a summary of all credits in the ledger."""
        issued   = [c for c in self._credits.values()
                    if c.status == CreditStatus.ISSUED]
        retired  = [c for c in self._credits.values()
                    if c.status == CreditStatus.RETIRED]

        return {
            "total_claims":       len(self._ledger),
            "issued_credits":     len(issued),
            "retired_credits":    len(retired),
            "total_tco2e_issued": round(sum(c.quantity_tco2e for c in issued), 4),
            "total_tco2e_retired": round(sum(c.quantity_tco2e for c in retired), 4),
            "portfolio_value_usd": round(
                sum(c.quantity_tco2e * (c.price_usd or self._market_price)
                    for c in issued), 2
            ),
            "credits": [c.to_dict() for c in list(issued) + list(retired)],
        }

    def avoided_emissions_kg(self, kwh_renewable: float) -> float:
        """Convert renewable kWh to avoided kg CO₂e."""
        return round(kwh_renewable * self._grid_ef, 4)

    def credit_value_usd(self, tco2e: float, project_id: str | None = None) -> float:
        """Estimate USD value of credits for a given tCO₂e quantity."""
        price = (
            self._registry.get_credit_price(project_id or self._project_id)
            or self._market_price
        )
        return round(tco2e * price, 2)
