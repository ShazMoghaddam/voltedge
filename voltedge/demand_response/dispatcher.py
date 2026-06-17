"""
VoltEdge Demand Response — Load Shedding Dispatcher

Receives DR events from grid operators and orchestrates the site's
response: calculates optimal load reduction, issues shed commands,
measures actual reduction, and computes settlement revenue.

Load shedding prioritisation (least-to-most critical loads):
  1. EV charging stations        — defer, reschedule off-peak
  2. HVAC pre-cooling/pre-heating — reduce setpoint by 2°C
  3. Lighting non-critical zones — dim or switch off
  4. Industrial batch processes  — pause if safe to do so
  5. Compressed air systems      — reduce pressure setpoint
  6. Refrigeration non-critical  — raise setpoint by 2°C
  ⚠  Core production / safety systems — NEVER curtailed

Revenue calculation:
  revenue = actual_reduction_kwh × clearing_price_per_kwh
  penalty = max(0, committed_reduction_kwh - actual_kwh) × penalty_per_kwh

Settlement is calculated at event close using metered data from storage.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from voltedge.demand_response.events import (
    DREvent, DREventStatus, DREventType, SiteResponse, SiteResponseStatus,
)
from voltedge.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class ShedAction:
    """A single load curtailment action issued to a site asset."""
    action_id:    str
    asset_type:   str          # "ev_charging", "hvac", "lighting", "batch_process"
    site_id:      str
    command:      str          # "off", "reduce_50pct", "setpoint_adjust"
    target_kw:    float        # Expected kW reduction from this action
    priority:     int          # Lower = shed first
    reversible:   bool = True  # Can be un-shed when event ends
    issued_at:    datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id":  self.action_id,
            "asset_type": self.asset_type,
            "site_id":    self.site_id,
            "command":    self.command,
            "target_kw":  self.target_kw,
            "priority":   self.priority,
            "reversible": self.reversible,
        }


# Load shed asset catalogue (priority order)
SHEDDABLE_ASSETS = [
    {"type": "ev_charging",      "priority": 1, "typical_kw": 22.0,  "command": "off"},
    {"type": "hvac_precool",     "priority": 2, "typical_kw": 15.0,  "command": "setpoint_adjust"},
    {"type": "lighting_noncore", "priority": 3, "typical_kw": 8.0,   "command": "reduce_50pct"},
    {"type": "batch_process",    "priority": 4, "typical_kw": 50.0,  "command": "pause"},
    {"type": "compressed_air",   "priority": 5, "typical_kw": 12.0,  "command": "setpoint_adjust"},
    {"type": "refrigeration_nc", "priority": 6, "typical_kw": 10.0,  "command": "setpoint_adjust"},
]


class DemandResponseDispatcher:
    """
    Orchestrates a site's demand response participation.

    Args:
        store:                Storage backend to read baseline consumption.
        site_assets:          Optional override for sheddable asset capacities.
        auto_respond:         Automatically opt into economic events.
        min_revenue_per_event: Minimum expected revenue to auto-opt-in.
    """

    def __init__(
        self,
        store:                  Any = None,
        site_assets:            dict[str, float] | None = None,
        auto_respond:           bool  = True,
        min_revenue_per_event:  float = 50.0,
    ) -> None:
        self._store           = store
        self._assets          = site_assets or {}
        self._auto_respond    = auto_respond
        self._min_revenue     = min_revenue_per_event
        self._responses:      dict[str, SiteResponse] = {}
        self._shed_actions:   dict[str, list[ShedAction]] = {}

    # ── Event handling ────────────────────────────────────────────────────────

    def evaluate_event(self, event: DREvent, site_id: str) -> dict[str, Any]:
        """
        Evaluate whether to opt into a DR event and calculate expected revenue.

        Returns a dict with recommendation, expected_revenue, and shed_plan.
        """
        potential_rev  = event.potential_revenue
        shed_plan      = self._plan_shed(event.target_reduction_kw, site_id)
        feasible_kw    = sum(a.target_kw for a in shed_plan)
        can_commit     = feasible_kw >= event.min_response_kw

        recommendation = (
            "opt_in"  if (can_commit and (
                event.mandatory or
                (self._auto_respond and potential_rev >= self._min_revenue)
            )) else
            "opt_out"
        )

        return {
            "event_id":         event.event_id,
            "site_id":          site_id,
            "recommendation":   recommendation,
            "feasible_kw":      round(feasible_kw, 1),
            "target_kw":        event.target_reduction_kw,
            "expected_revenue": potential_rev,
            "currency":         event.currency,
            "shed_plan":        [a.to_dict() for a in shed_plan],
            "mandatory":        event.mandatory,
        }

    def opt_in(self, event: DREvent, site_id: str, committed_kw: float) -> SiteResponse:
        """Register the site's participation in a DR event."""
        response = SiteResponse(
            event_id=event.event_id,
            site_id=site_id,
            status=SiteResponseStatus.OPTED_IN,
            committed_kw=committed_kw,
        )
        self._responses[f"{event.event_id}:{site_id}"] = response
        shed_plan = self._plan_shed(committed_kw, site_id)
        self._shed_actions[f"{event.event_id}:{site_id}"] = shed_plan
        log.info("dr.opted_in", site=site_id, event_id=event.event_id,
                 committed_kw=committed_kw)
        return response

    def opt_out(self, event: DREvent, site_id: str, reason: str = "") -> SiteResponse:
        """Register the site's opt-out from a DR event."""
        response = SiteResponse(
            event_id=event.event_id,
            site_id=site_id,
            status=SiteResponseStatus.OPTED_OUT,
            notes=reason,
        )
        self._responses[f"{event.event_id}:{site_id}"] = response
        log.info("dr.opted_out", site=site_id, event_id=event.event_id, reason=reason)
        return response

    def get_shed_actions(self, event_id: str, site_id: str) -> list[ShedAction]:
        """Return the ordered list of shed actions for an event + site."""
        return self._shed_actions.get(f"{event_id}:{site_id}", [])

    def settle_event(
        self,
        event:   DREvent,
        site_id: str,
        actual_df: pd.DataFrame | None = None,
    ) -> SiteResponse:
        """
        Settle a completed DR event:
        1. Read metered data during event window (from store or actual_df)
        2. Compare vs baseline
        3. Calculate revenue and any penalties
        4. Update response record
        """
        key      = f"{event.event_id}:{site_id}"
        response = self._responses.get(key)
        if not response:
            raise ValueError(f"No response record for event={event.event_id} site={site_id}")

        if response.status == SiteResponseStatus.OPTED_OUT:
            return response

        # Get metered data during event window
        df = actual_df
        if df is None and self._store:
            df = self._store.read(site_id, layer="processed", days=1)

        baseline_kw, response_kw = self._compute_baseline_and_response(df, event)
        actual_reduction = max(0.0, baseline_kw - response_kw)
        duration_h = event.duration_minutes / 60.0

        revenue = actual_reduction * duration_h * event.clearing_price_kwh

        penalty = 0.0
        if event.penalty_per_kwh > 0:
            shortfall_kw = max(0.0, response.committed_kw - actual_reduction)
            penalty = shortfall_kw * duration_h * event.penalty_per_kwh

        from dataclasses import replace
        settled = replace(
            response,
            status=SiteResponseStatus.COMPLETED,
            actual_reduction_kw=round(actual_reduction, 2),
            baseline_kw=round(baseline_kw, 2),
            response_kw=round(response_kw, 2),
            revenue_earned=round(revenue, 2),
            penalty_charged=round(penalty, 2),
            completed_at=datetime.now(timezone.utc),
        )
        self._responses[key] = settled
        log.info(
            "dr.settled",
            site=site_id, event_id=event.event_id,
            reduction_kw=actual_reduction,
            revenue=revenue, penalty=penalty,
        )
        return settled

    # ── Portfolio ─────────────────────────────────────────────────────────────

    def portfolio_revenue(self) -> dict[str, Any]:
        """Aggregate DR revenue and performance across all settled responses."""
        completed = [
            r for r in self._responses.values()
            if r.status == SiteResponseStatus.COMPLETED
        ]
        total_revenue = sum(r.revenue_earned for r in completed)
        total_penalty = sum(r.penalty_charged for r in completed)
        avg_reduction = (
            sum(r.reduction_pct for r in completed) / len(completed)
            if completed else 0.0
        )
        return {
            "events_responded":    len(completed),
            "total_revenue":       round(total_revenue, 2),
            "total_penalty":       round(total_penalty, 2),
            "net_revenue":         round(total_revenue - total_penalty, 2),
            "avg_reduction_pct":   round(avg_reduction, 1),
            "by_site": {
                site: round(sum(r.revenue_earned for r in completed if r.site_id == site), 2)
                for site in {r.site_id for r in completed}
            },
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _plan_shed(self, target_kw: float, site_id: str) -> list[ShedAction]:
        """Build an ordered list of shed actions to reach target_kw."""
        remaining = target_kw
        actions:  list[ShedAction] = []
        import uuid as _uuid

        for asset in SHEDDABLE_ASSETS:
            if remaining <= 0:
                break
            # Use site-specific capacity if available, else typical
            capacity = self._assets.get(f"{site_id}:{asset['type']}", asset["typical_kw"])
            shed_kw  = min(capacity, remaining)
            if shed_kw < 0.5:
                continue
            actions.append(ShedAction(
                action_id=str(_uuid.uuid4())[:8],
                asset_type=asset["type"],
                site_id=site_id,
                command=asset["command"],
                target_kw=round(shed_kw, 1),
                priority=asset["priority"],
            ))
            remaining -= shed_kw

        return actions

    @staticmethod
    def _compute_baseline_and_response(
        df: pd.DataFrame | None, event: DREvent
    ) -> tuple[float, float]:
        """
        Extract baseline (same-weekday average) and event-window consumption.
        Returns (baseline_kw, response_kw).
        """
        if df is None or df.empty or "kwh" not in df.columns:
            return 0.0, 0.0

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Event window
        start = event.start_time if event.start_time.tzinfo else event.start_time.replace(tzinfo=timezone.utc)
        end   = event.end_time   if event.end_time.tzinfo   else event.end_time.replace(tzinfo=timezone.utc)

        event_df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
        response_kw = float(event_df["kwh"].mean()) if not event_df.empty else 0.0

        # Baseline: average of same hours on prior weekdays
        event_dow   = start.weekday()
        event_hours = set(range(start.hour, end.hour + 1))
        prior = df[
            (df["timestamp"] < start) &
            (df["timestamp"].dt.weekday == event_dow) &
            (df["timestamp"].dt.hour.isin(event_hours))
        ]
        baseline_kw = float(prior["kwh"].mean()) if not prior.empty else response_kw

        return baseline_kw, response_kw
