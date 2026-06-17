"""
VoltEdge Demand Response — Event Data Models

Grid demand response (DR) events are instructions from grid operators
to reduce or shift load during periods of high demand or low supply.
Participating sites earn revenue for reducing load on demand.

Event types:
  EMERGENCY     — Grid stability risk; mandatory reduction (Triad avoidance UK)
  ECONOMIC      — Price signal; optional reduction with financial incentive
  RELIABILITY   — Capacity shortfall; voluntary reduction requested
  TEST          — Planned DR test; site must demonstrate response capability
  BASELINE      — Metering period to establish baseline consumption

Revenue model (UK Capacity Market / Demand Side Response):
  Revenue = reduced_kwh × clearing_price_per_kwh
  Clearing prices range from £0.05/kWh (economic) to £2.50/kWh (emergency)

Data sources:
  UK: National Grid ESO (Balancing Mechanism + DSBR)
  EU: ENTSO-E Transparency Platform
  US: PJM, CAISO, MISO via OpenADR 2.0

This module defines the canonical event model used across all grid connectors.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class DREventType(str, Enum):
    EMERGENCY    = "emergency"
    ECONOMIC     = "economic"
    RELIABILITY  = "reliability"
    TEST         = "test"
    BASELINE     = "baseline"


class DREventStatus(str, Enum):
    UPCOMING    = "upcoming"     # Notified, not yet started
    ACTIVE      = "active"       # Event window is live
    COMPLETED   = "completed"    # Event window closed
    CANCELLED   = "cancelled"    # Event cancelled before start


class SiteResponseStatus(str, Enum):
    OPTED_IN    = "opted_in"
    OPTED_OUT   = "opted_out"
    RESPONDING  = "responding"   # Actively reducing load
    COMPLETED   = "completed"    # Response window closed, awaiting settlement
    FAILED      = "failed"       # Did not achieve target reduction


@dataclass
class DREvent:
    """
    A grid demand response event instruction.

    Fields follow OpenADR 2.0 schema extended with VoltEdge-specific fields.
    """
    event_id:             str
    grid_operator:        str             # "NESO", "PJM", "CAISO", "ENTSOE"
    event_type:           DREventType
    status:               DREventStatus
    start_time:           datetime
    end_time:             datetime
    notification_time:    datetime
    target_reduction_kw:  float           # Fleet-level reduction target
    clearing_price_kwh:   float           # Revenue per kWh reduced
    currency:             str   = "GBP"
    grid_region:          str   = "GB"
    mandatory:            bool  = False   # True for EMERGENCY events
    min_response_kw:      float = 0.0    # Minimum site participation
    max_response_kw:      float = 0.0    # Maximum site participation
    penalty_per_kwh:      float = 0.0    # Penalty for failing mandatory event
    raw_payload:          dict[str, Any] = field(default_factory=dict)

    @property
    def duration_minutes(self) -> float:
        return (self.end_time - self.start_time).total_seconds() / 60

    @property
    def is_active(self) -> bool:
        now = datetime.now(timezone.utc)
        start = self.start_time if self.start_time.tzinfo else self.start_time.replace(tzinfo=timezone.utc)
        end   = self.end_time   if self.end_time.tzinfo   else self.end_time.replace(tzinfo=timezone.utc)
        return start <= now <= end

    @property
    def potential_revenue(self) -> float:
        """Maximum revenue if full target reduction is achieved."""
        hours = self.duration_minutes / 60
        return round(self.target_reduction_kw * hours * self.clearing_price_kwh, 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id":            self.event_id,
            "grid_operator":       self.grid_operator,
            "event_type":          self.event_type.value,
            "status":              self.status.value,
            "start_time":          self.start_time.isoformat(),
            "end_time":            self.end_time.isoformat(),
            "duration_minutes":    self.duration_minutes,
            "target_reduction_kw": self.target_reduction_kw,
            "clearing_price_kwh":  self.clearing_price_kwh,
            "currency":            self.currency,
            "mandatory":           self.mandatory,
            "potential_revenue":   self.potential_revenue,
        }


@dataclass
class SiteResponse:
    """
    A site's participation record for a specific DR event.
    Tracks opt-in, actual reduction, and settlement.
    """
    response_id:       str      = field(default_factory=lambda: str(uuid.uuid4())[:8])
    event_id:          str      = ""
    site_id:           str      = ""
    status:            SiteResponseStatus = SiteResponseStatus.OPTED_IN
    committed_kw:      float    = 0.0    # Reduction committed by operator
    actual_reduction_kw: float  = 0.0    # Measured reduction vs baseline
    baseline_kw:       float    = 0.0    # Baseline consumption during event
    response_kw:       float    = 0.0    # Actual consumption during event
    revenue_earned:    float    = 0.0
    penalty_charged:   float    = 0.0
    opted_in_at:       datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at:      datetime | None = None
    notes:             str      = ""

    @property
    def reduction_pct(self) -> float:
        if self.baseline_kw == 0:
            return 0.0
        return round(self.actual_reduction_kw / self.baseline_kw * 100, 1)

    @property
    def net_revenue(self) -> float:
        return round(self.revenue_earned - self.penalty_charged, 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "response_id":          self.response_id,
            "event_id":             self.event_id,
            "site_id":              self.site_id,
            "status":               self.status.value,
            "committed_kw":         self.committed_kw,
            "actual_reduction_kw":  self.actual_reduction_kw,
            "baseline_kw":          self.baseline_kw,
            "response_kw":          self.response_kw,
            "reduction_pct":        self.reduction_pct,
            "revenue_earned":       self.revenue_earned,
            "penalty_charged":      self.penalty_charged,
            "net_revenue":          self.net_revenue,
        }
