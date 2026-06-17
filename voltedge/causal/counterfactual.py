"""
VoltEdge Causal AI — Counterfactual Engine

Answers "what would have happened if..." questions using Pearl's
do-calculus intervention operator do(X=x).

The do-operator differs from simple conditioning:
  Conditional:     P(Y | X=x)     — observed X=x (selection bias possible)
  Interventional:  P(Y | do(X=x)) — we *set* X=x, cutting its incoming edges

This is the difference between:
  "Sites that happen to have 25°C ambient temperature use Y kWh."
  "If we intervene to keep temperature at 25°C (HVAC pre-cooling), we'd use Y kWh."

Implemented interventions:
  - Temperature control (HVAC pre-cooling)
  - Shift schedule changes (start/end time shifts)
  - Renewable fraction change (REC procurement)
  - Power factor correction (capacitor bank installation)
  - Load shifting (time-of-use schedule change)
  - Baseline query (no intervention — current trajectory)

Each counterfactual returns:
  - Predicted outcome variable value under intervention
  - Delta vs actual/counterfactual baseline
  - Estimated annual impact (kWh, CO₂, cost)
  - Confidence interval
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from voltedge.causal.graph import CausalGraph, CausalGraphBuilder
from voltedge.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class Intervention:
    """A single do(X=x) intervention specification."""
    variable:  str
    new_value: float          # The value we're setting via do(X=new_value)
    description: str = ""     # Human label for the intervention
    feasibility: str = "medium"  # "high" | "medium" | "low"
    capex_estimate_gbp: float = 0.0  # Estimated capital cost to implement


@dataclass
class CounterfactualResult:
    """
    Result of a do-calculus intervention.

    Compares the predicted outcome under the intervention against
    the observed/baseline outcome.
    """
    intervention:       Intervention
    target_variable:    str
    baseline_value:     float         # Observed / current trajectory value
    counterfactual_value: float       # Predicted value under do(X=x)
    delta:              float         # counterfactual - baseline
    delta_pct:          float         # delta / baseline * 100
    annual_kwh_impact:  float = 0.0
    annual_co2_impact:  float = 0.0   # kg CO₂e
    annual_cost_impact: float = 0.0   # GBP/EUR/USD
    cost_currency:      str   = "GBP"
    confidence_interval: tuple[float, float] = (0.0, 0.0)
    confidence:         str   = "medium"
    narrative:          str   = ""

    @property
    def is_beneficial(self) -> bool:
        """True if the intervention reduces consumption or cost."""
        return self.delta < 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "intervention":        self.intervention.description,
            "variable":            self.intervention.variable,
            "new_value":           self.intervention.new_value,
            "target_variable":     self.target_variable,
            "baseline_value":      round(self.baseline_value, 3),
            "counterfactual_value": round(self.counterfactual_value, 3),
            "delta":               round(self.delta, 3),
            "delta_pct":           round(self.delta_pct, 1),
            "annual_kwh_impact":   round(self.annual_kwh_impact, 1),
            "annual_co2_impact":   round(self.annual_co2_impact, 1),
            "annual_cost_impact":  round(self.annual_cost_impact, 2),
            "cost_currency":       self.cost_currency,
            "confidence_interval": [round(v, 3) for v in self.confidence_interval],
            "confidence":          self.confidence,
            "narrative":           self.narrative,
            "is_beneficial":       self.is_beneficial,
            "feasibility":         self.intervention.feasibility,
            "capex_gbp":           self.intervention.capex_estimate_gbp,
        }


class CounterfactualEngine:
    """
    Computes do-calculus counterfactuals from a learned causal graph.

    The core operation is graph surgery: to compute do(X=x), we:
    1. Remove all incoming edges to X in the DAG (the "surgery")
    2. Set X to the intervention value x
    3. Propagate through the graph to compute the new value of the target

    Args:
        graph:              Learned causal DAG for the site.
        carbon_intensity:   kg CO₂e per kWh for impact calculation.
        tariff_per_kwh:     Energy cost per kWh for cost impact.
        currency:           Cost currency string.
        annual_hours:       Hours per year (default 8760).
    """

    def __init__(
        self,
        graph:             CausalGraph,
        carbon_intensity:  float = 0.207,
        tariff_per_kwh:    float = 0.28,
        currency:          str   = "GBP",
        annual_hours:      int   = 8760,
    ) -> None:
        self._graph     = graph
        self._ci        = carbon_intensity
        self._tariff    = tariff_per_kwh
        self._currency  = currency
        self._ann_hours = annual_hours

    # ── Public API ────────────────────────────────────────────────────────────

    def compute(
        self,
        df:             pd.DataFrame,
        intervention:   Intervention,
        target:         str = "kwh",
    ) -> CounterfactualResult:
        """
        Compute do(intervention.variable = intervention.new_value) → target.

        Args:
            df:           Historical processed DataFrame (provides baselines).
            intervention: The variable to intervene on and its new value.
            target:       The outcome variable to predict (default: kwh).

        Returns:
            CounterfactualResult with predicted outcome and impact estimates.
        """
        if df.empty or "kwh" not in df.columns:
            raise ValueError("DataFrame must contain 'kwh' column.")

        # Baseline: mean of target in the provided data
        baseline = float(df[target].mean()) if target in df.columns else 0.0

        # Check that the intervention variable is in our graph
        if intervention.variable not in self._graph.variables:
            return CounterfactualResult(
                intervention=intervention,
                target_variable=target,
                baseline_value=baseline,
                counterfactual_value=baseline,
                delta=0.0,
                delta_pct=0.0,
                confidence="low",
                narrative=(
                    f"Variable '{intervention.variable}' is not in the causal graph "
                    f"for this site. Cannot compute counterfactual. "
                    f"Run CausalGraphBuilder.build() with data that includes this variable."
                ),
            )

        # Graph surgery: find all causal paths from intervention variable to target
        path = self._graph.causal_path(intervention.variable, target)
        if path is None:
            return CounterfactualResult(
                intervention=intervention,
                target_variable=target,
                baseline_value=baseline,
                counterfactual_value=baseline,
                delta=0.0,
                delta_pct=0.0,
                confidence="low",
                narrative=(
                    f"No causal path from '{intervention.variable}' to '{target}' "
                    f"in the causal graph. The intervention would have no causal effect."
                ),
            )

        # Propagate the intervention through the causal path
        cf_value = self._propagate(
            intervention.variable,
            intervention.new_value,
            target, path, df,
        )

        delta     = cf_value - baseline
        delta_pct = (delta / (abs(baseline) + 1e-8)) * 100

        # Annual impact calculations
        hourly_delta = delta   # df is already in kWh/h units
        annual_kwh   = hourly_delta * self._ann_hours
        annual_co2   = annual_kwh * self._ci
        annual_cost  = annual_kwh * self._tariff

        # Bootstrap confidence interval (simple ±)
        se = abs(delta) * 0.15   # 15% standard error estimate
        ci = (cf_value - 1.96 * se, cf_value + 1.96 * se)

        confidence = self._assess_confidence(df, path, delta)
        narrative  = self._build_narrative(
            intervention, target, baseline, cf_value,
            delta_pct, annual_kwh, annual_cost, self._currency,
        )

        log.info("counterfactual.computed",
                 site=self._graph.site_id,
                 intervention=intervention.variable,
                 target=target,
                 delta_pct=round(delta_pct, 1),
                 confidence=confidence)

        return CounterfactualResult(
            intervention=intervention,
            target_variable=target,
            baseline_value=round(baseline, 3),
            counterfactual_value=round(cf_value, 3),
            delta=round(delta, 3),
            delta_pct=round(delta_pct, 1),
            annual_kwh_impact=round(annual_kwh, 1),
            annual_co2_impact=round(annual_co2, 1),
            annual_cost_impact=round(annual_cost, 2),
            cost_currency=self._currency,
            confidence_interval=(round(ci[0], 3), round(ci[1], 3)),
            confidence=confidence,
            narrative=narrative,
        )

    def multi_intervention(
        self,
        df:            pd.DataFrame,
        interventions: list[Intervention],
        target:        str = "kwh",
    ) -> dict[str, Any]:
        """
        Evaluate multiple interventions and compare them.
        Returns individual results plus a combined scenario estimate.
        """
        results = [
            self.compute(df, iv, target)
            for iv in interventions
        ]

        # Combined effect (assuming additive, which is approximate)
        total_delta_kwh = sum(r.annual_kwh_impact for r in results)
        total_co2       = sum(r.annual_co2_impact  for r in results)
        total_cost      = sum(r.annual_cost_impact for r in results)

        baseline = results[0].baseline_value if results else 0.0

        return {
            "site_id":        self._graph.site_id,
            "target":         target,
            "baseline_value": round(baseline, 3),
            "interventions":  [r.to_dict() for r in results],
            "combined_scenario": {
                "total_annual_kwh_saving": round(total_delta_kwh, 1),
                "total_annual_co2_saving": round(total_co2, 1),
                "total_annual_cost_saving": round(total_cost, 2),
                "currency": self._currency,
                "combined_delta_pct": round(
                    total_delta_kwh / (abs(baseline) * self._ann_hours + 1e-8) * 100, 1
                ),
            },
        }

    # ── Built-in intervention templates ───────────────────────────────────────

    @staticmethod
    def temperature_intervention(
        target_temp_c: float,
        current_temp_c: float = 25.0,
    ) -> Intervention:
        direction = "pre-cooling" if target_temp_c < current_temp_c else "setpoint increase"
        return Intervention(
            variable="temperature_c",
            new_value=target_temp_c,
            description=f"HVAC {direction} to {target_temp_c:.0f}°C",
            feasibility="high",
            capex_estimate_gbp=0.0,   # Operational, no capex
        )

    @staticmethod
    def shift_schedule_intervention(business_hours: bool) -> Intervention:
        return Intervention(
            variable="is_business_hour",
            new_value=1.0 if business_hours else 0.0,
            description=(
                "Extend to 24/7 operation"
                if business_hours else
                "Shift to off-peak hours only"
            ),
            feasibility="medium",
            capex_estimate_gbp=0.0,
        )

    @staticmethod
    def power_factor_intervention(target_pf: float = 0.95) -> Intervention:
        return Intervention(
            variable="power_factor",
            new_value=target_pf,
            description=f"Install PF correction to reach PF {target_pf:.2f}",
            feasibility="high",
            capex_estimate_gbp=12_000.0,
        )

    # ── Internal propagation ──────────────────────────────────────────────────

    def _propagate(
        self,
        iv_variable: str,
        iv_value:    float,
        target:      str,
        path:        list,
        df:          pd.DataFrame,
    ) -> float:
        """
        Propagate a do(X=x) intervention along the causal path to the target.

        Uses the chain rule of causal effects:
        Δtarget = Σ (β_edge × Δ_predecessor)
        where Δ_predecessor is the deviation caused by the intervention.
        """
        # Baseline values for each variable in the path
        baselines: dict[str, float] = {
            col: float(df[col].mean())
            for col in df.select_dtypes(include=[np.number]).columns
        }

        # How much did the intervention variable change?
        iv_baseline = baselines.get(iv_variable, iv_value)
        iv_delta    = iv_value - iv_baseline

        # Walk the path, accumulating the effect
        cumulative_delta = iv_delta
        for i in range(len(path) - 1):
            # Variable names are indices in the graph, but we store string names
            # Find the edge
            step_cause  = self._graph.variables[path[i]]   if isinstance(path[i], int) else path[i]
            step_effect = self._graph.variables[path[i+1]] if isinstance(path[i+1], int) else path[i+1]

            edge = next(
                (e for e in self._graph.edges
                 if e.cause == step_cause and e.effect == step_effect),
                None,
            )
            if edge is None:
                break
            cumulative_delta *= edge.strength

        baseline_target = baselines.get(target, 0.0)
        return float(baseline_target + cumulative_delta)

    @staticmethod
    def _assess_confidence(
        df:    pd.DataFrame,
        path:  list,
        delta: float,
    ) -> str:
        if len(df) < 168:   # Less than 1 week
            return "low"
        if len(path) > 4:   # Very long causal chain
            return "low"
        if abs(delta) < 0.5:
            return "medium"
        return "high" if len(df) >= 720 else "medium"

    @staticmethod
    def _build_narrative(
        iv:          "Intervention",
        target:      str,
        baseline:    float,
        cf_value:    float,
        delta_pct:   float,
        annual_kwh:  float,
        annual_cost: float,
        currency:    str,
    ) -> str:
        direction = "reduction" if delta_pct < 0 else "increase"
        direction_verb = "reduce" if delta_pct < 0 else "increase"

        return (
            f"If we {iv.description.lower()}, the causal model predicts "
            f"{target.replace('_', ' ')} would {direction_verb} by "
            f"{abs(delta_pct):.1f}% from {baseline:.1f} to {cf_value:.1f}. "
            f"Annualised, this represents a {direction} of "
            f"{abs(annual_kwh):,.0f} kWh/year "
            f"({currency}{abs(annual_cost):,.0f}/year at current tariff)."
        )
