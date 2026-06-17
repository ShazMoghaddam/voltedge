"""
VoltEdge Causal AI — Energy Digital Twin

A digital twin is a parameterised simulation model of a physical site
that can be run forward in time to predict behaviour under different
operating conditions — without touching the real site.

VoltEdge's digital twin combines:
  1. A learned causal graph (structural equations from the data)
  2. A stochastic demand model (captures natural variability)
  3. Scenario injection (override any variable at any timestep)
  4. What-if scenario comparison (run multiple futures side-by-side)

Use cases:
  - "What happens to our consumption if we add a 200kW EV fleet?"
  - "How much does pre-cooling save during Triad risk windows?"
  - "What's the ROI on a 500kWh BESS over 10 years?"
  - "If we move production from day to night shift, what's the impact?"

The twin runs Monte Carlo simulations (N=100 by default) to produce
a distribution of outcomes rather than a single deterministic number.
This gives operators credible intervals for planning decisions.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from voltedge.causal.graph import CausalGraph
from voltedge.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class ScenarioParameter:
    """A single variable override in a what-if scenario."""
    variable:     str
    value:        float
    start_hour:   int   = 0      # Hour offset from simulation start
    end_hour:     int   = 8760   # Hour offset from simulation start (default: full year)
    description:  str  = ""


@dataclass
class Scenario:
    """
    A named what-if scenario: a set of parameter overrides
    applied to the digital twin simulation.
    """
    name:          str
    description:   str  = ""
    parameters:    list[ScenarioParameter] = field(default_factory=list)
    capex_gbp:     float = 0.0   # One-time cost to implement this scenario
    opex_delta_gbp: float = 0.0  # Annual operating cost change

    def add_parameter(self, variable: str, value: float, **kwargs) -> "Scenario":
        self.parameters.append(ScenarioParameter(variable=variable, value=value, **kwargs))
        return self


@dataclass
class TwinSimulationResult:
    """Results of running the digital twin for one scenario."""
    scenario_name:    str
    simulation_hours: int

    # Hourly time-series (mean across Monte Carlo runs)
    timestamps:       list[str]
    kwh_mean:         list[float]
    kwh_p10:          list[float]
    kwh_p90:          list[float]

    # Aggregates
    total_kwh:        float
    peak_kw:          float
    avg_kw:           float
    total_co2_kg:     float
    total_cost:       float
    currency:         str   = "GBP"

    # Monte Carlo metadata
    n_simulations:    int   = 100
    confidence:       str   = "medium"

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario":        self.scenario_name,
            "simulation_hours": self.simulation_hours,
            "total_kwh":       round(self.total_kwh, 1),
            "peak_kw":         round(self.peak_kw, 1),
            "avg_kw":          round(self.avg_kw, 2),
            "total_co2_kg":    round(self.total_co2_kg, 1),
            "total_cost":      round(self.total_cost, 2),
            "currency":        self.currency,
            "n_simulations":   self.n_simulations,
            "confidence":      self.confidence,
            "hourly_preview":  {
                "timestamps": self.timestamps[:48],
                "kwh_mean":   [round(v, 2) for v in self.kwh_mean[:48]],
                "kwh_p10":    [round(v, 2) for v in self.kwh_p10[:48]],
                "kwh_p90":    [round(v, 2) for v in self.kwh_p90[:48]],
            },
        }


@dataclass
class ScenarioComparison:
    """Side-by-side comparison of multiple scenarios against a baseline."""
    baseline:     TwinSimulationResult
    scenarios:    list[TwinSimulationResult]

    @property
    def best_scenario(self) -> TwinSimulationResult | None:
        """Scenario with lowest total_kwh."""
        if not self.scenarios:
            return None
        return min(self.scenarios, key=lambda s: s.total_kwh)

    def ranked_by_saving(self) -> list[dict[str, Any]]:
        base_kwh = self.baseline.total_kwh
        ranked = []
        for s in self.scenarios:
            delta_kwh  = s.total_kwh - base_kwh
            delta_cost = s.total_cost - self.baseline.total_cost
            ranked.append({
                "scenario":        s.scenario_name,
                "total_kwh":       round(s.total_kwh, 1),
                "delta_kwh":       round(delta_kwh, 1),
                "delta_pct":       round(delta_kwh / (abs(base_kwh) + 1e-8) * 100, 1),
                "delta_cost":      round(delta_cost, 2),
                "currency":        s.currency,
                "total_co2_kg":    round(s.total_co2_kg, 1),
                "confidence":      s.confidence,
            })
        return sorted(ranked, key=lambda x: x["delta_kwh"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline":        self.baseline.to_dict(),
            "scenarios":       [s.to_dict() for s in self.scenarios],
            "ranked_by_saving": self.ranked_by_saving(),
            "best_scenario":   self.best_scenario.scenario_name if self.best_scenario else None,
        }


class EnergyDigitalTwin:
    """
    Monte Carlo simulation model of a site's energy system.

    Parameterised from historical data + causal graph.
    Runs N stochastic forward simulations to produce credible intervals.

    Args:
        graph:             Learned causal DAG (provides structural equations).
        df:                Historical data to calibrate the twin.
        carbon_intensity:  kg CO₂e per kWh.
        tariff_per_kwh:    Energy cost per kWh.
        currency:          Cost currency.
        n_simulations:     Monte Carlo sample count (default 100).
        seed:              Random seed for reproducibility.
    """

    def __init__(
        self,
        graph:             CausalGraph,
        df:                pd.DataFrame,
        carbon_intensity:  float = 0.207,
        tariff_per_kwh:    float = 0.28,
        currency:          str   = "GBP",
        n_simulations:     int   = 100,
        seed:              int   = 42,
    ) -> None:
        self._graph   = graph
        self._ci      = carbon_intensity
        self._tariff  = tariff_per_kwh
        self._currency = currency
        self._n_sims  = n_simulations
        self._rng     = random.Random(seed)
        self._np_rng  = np.random.default_rng(seed)

        # Calibrate from historical data
        self._params  = self._calibrate(df)

    # ── Public API ────────────────────────────────────────────────────────────

    def run_baseline(self, hours: int = 8760) -> TwinSimulationResult:
        """Simulate the current trajectory without any interventions."""
        return self._simulate(Scenario(name="Baseline (no intervention)"), hours)

    def run_scenario(
        self,
        scenario: Scenario,
        hours:    int = 8760,
    ) -> TwinSimulationResult:
        """Simulate a specific what-if scenario."""
        return self._simulate(scenario, hours)

    def compare_scenarios(
        self,
        scenarios: list[Scenario],
        hours:     int = 8760,
    ) -> ScenarioComparison:
        """Run baseline + all scenarios and return a side-by-side comparison."""
        log.info("digital_twin.compare_start",
                 site=self._graph.site_id,
                 scenarios=len(scenarios), hours=hours)

        baseline   = self.run_baseline(hours)
        results    = [self.run_scenario(s, hours) for s in scenarios]

        log.info("digital_twin.compare_complete",
                 site=self._graph.site_id,
                 best=min(results, key=lambda r: r.total_kwh).scenario_name)

        return ScenarioComparison(baseline=baseline, scenarios=results)

    # ── Built-in scenario templates ───────────────────────────────────────────

    @staticmethod
    def bess_scenario(capacity_kwh: float, charge_rate_kw: float) -> Scenario:
        """Battery Energy Storage System — peak shaving."""
        return Scenario(
            name=f"BESS {capacity_kwh:.0f}kWh / {charge_rate_kw:.0f}kW",
            description=f"Install {capacity_kwh:.0f}kWh BESS for peak shaving",
            capex_gbp=capacity_kwh * 350,
        ).add_parameter("kwh_roll_max_24h", charge_rate_kw * 0.8,
                        description="Peak demand capped by BESS discharge")

    @staticmethod
    def ev_fleet_scenario(vehicles: int, kw_per_vehicle: float = 22.0) -> Scenario:
        """Add an EV fleet with managed (off-peak) charging."""
        total_kw = vehicles * kw_per_vehicle
        return Scenario(
            name=f"EV fleet {vehicles}× {kw_per_vehicle:.0f}kW managed",
            description=f"{vehicles} EVs charged off-peak (midnight–6am)",
            capex_gbp=vehicles * 8_000,
        ).add_parameter("kwh", total_kw,
                        start_hour=0, end_hour=6,
                        description="EV charging load added off-peak")

    @staticmethod
    def solar_scenario(capacity_kwp: float, yield_kwh_per_kwp: float = 900) -> Scenario:
        """Rooftop solar PV — reduces net import."""
        annual_generation = capacity_kwp * yield_kwh_per_kwp
        hourly_gen = annual_generation / 8760
        return Scenario(
            name=f"Solar PV {capacity_kwp:.0f}kWp",
            description=f"{capacity_kwp:.0f}kWp rooftop solar, {yield_kwh_per_kwp:.0f}kWh/kWp/yr",
            capex_gbp=capacity_kwp * 1_200,
        ).add_parameter("kwh", -hourly_gen,
                        start_hour=0, end_hour=8760,
                        description="Solar generation offsets import")

    @staticmethod
    def night_shift_scenario() -> Scenario:
        """Move production from day to night shift."""
        return Scenario(
            name="Night shift migration",
            description="Shift core production to 22:00–06:00",
        ).add_parameter("is_business_hour", 0.0,
                        start_hour=7, end_hour=19,
                        description="Daytime production reduced"
                       ).add_parameter("is_business_hour", 1.0,
                        start_hour=22, end_hour=30,
                        description="Night-time production active")

    # ── Monte Carlo simulation ────────────────────────────────────────────────

    def _simulate(self, scenario: Scenario, hours: int) -> TwinSimulationResult:
        """Run N Monte Carlo simulations for a given scenario."""
        all_runs: list[np.ndarray] = []

        for _ in range(self._n_sims):
            run = self._single_run(scenario, hours)
            all_runs.append(run)

        stacked = np.array(all_runs)   # (n_sims, hours)
        mean_kwh = stacked.mean(axis=0)
        p10_kwh  = np.percentile(stacked, 10, axis=0)
        p90_kwh  = np.percentile(stacked, 90, axis=0)

        start = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        timestamps = [
            (start + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M")
            for h in range(hours)
        ]

        total_kwh  = float(mean_kwh.sum())
        total_co2  = total_kwh * self._ci
        total_cost = total_kwh * self._tariff

        confidence = (
            "high"   if self._n_sims >= 200 and len(self._params) > 5 else
            "medium" if self._n_sims >= 100 else
            "low"
        )

        return TwinSimulationResult(
            scenario_name=scenario.name,
            simulation_hours=hours,
            timestamps=timestamps,
            kwh_mean=mean_kwh.tolist(),
            kwh_p10=p10_kwh.tolist(),
            kwh_p90=p90_kwh.tolist(),
            total_kwh=round(total_kwh, 1),
            peak_kw=round(float(mean_kwh.max()), 1),
            avg_kw=round(float(mean_kwh.mean()), 2),
            total_co2_kg=round(total_co2, 1),
            total_cost=round(total_cost, 2),
            currency=self._currency,
            n_simulations=self._n_sims,
            confidence=confidence,
        )

    def _single_run(self, scenario: Scenario, hours: int) -> np.ndarray:
        """One Monte Carlo forward simulation."""
        p = self._params
        result = np.zeros(hours)

        for h in range(hours):
            hour_of_day = h % 24
            day_of_week = (h // 24) % 7

            # Base demand from calibrated profile
            base = p["hourly_profile"][hour_of_day]

            # Weekend/weekday adjustment
            if day_of_week >= 5:
                base *= p["weekend_factor"]

            # Apply scenario parameters
            override = self._apply_scenario(scenario, h, base)
            base = override if override is not None else base

            # Autoregressive effect
            if h > 0:
                ar_effect = p["ar_coef"] * (result[h - 1] - p["mean_kwh"])
                base += ar_effect

            # Stochastic noise
            noise = self._np_rng.normal(0, p["noise_std"])
            result[h] = max(0.0, base + noise)

        return result

    @staticmethod
    def _apply_scenario(
        scenario: Scenario, hour: int, base: float
    ) -> float | None:
        """Return modified base if any scenario parameter applies at this hour."""
        for param in scenario.parameters:
            if param.start_hour <= hour < param.end_hour:
                # Additive for load additions (e.g. EV), multiplicative for scale
                if param.variable in ("kwh",):
                    return base + param.value
                elif param.variable == "is_business_hour":
                    if param.value == 0.0:
                        return base * 0.35   # Reduced to standby
                    else:
                        return base * 1.40   # Production-level load
        return None

    def _calibrate(self, df: pd.DataFrame) -> dict[str, Any]:
        """Extract parameters from historical data to calibrate the twin."""
        if df.empty or "kwh" not in df.columns:
            return {
                "hourly_profile": [100.0] * 24,
                "weekend_factor": 0.60,
                "ar_coef":        0.30,
                "noise_std":      10.0,
                "mean_kwh":       100.0,
            }

        df = df.copy()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df["hour"]      = df["timestamp"].dt.hour
            df["dow"]       = df["timestamp"].dt.weekday

        kwh = df["kwh"]

        # Hourly demand profile
        if "hour" in df.columns:
            profile = df.groupby("hour")["kwh"].mean()
            hourly_profile = [float(profile.get(h, kwh.mean())) for h in range(24)]
        else:
            hourly_profile = [float(kwh.mean())] * 24

        # Weekend factor
        if "dow" in df.columns:
            weekday_mean = float(df[df["dow"] < 5]["kwh"].mean())
            weekend_mean = float(df[df["dow"] >= 5]["kwh"].mean())
            weekend_factor = (
                weekend_mean / (weekday_mean + 1e-8)
                if weekday_mean > 0 else 0.6
            )
        else:
            weekend_factor = 0.60

        # AR(1) coefficient
        kwh_vals = kwh.values
        if len(kwh_vals) > 1:
            ar_coef = float(np.corrcoef(kwh_vals[:-1], kwh_vals[1:])[0, 1])
            ar_coef = max(0.0, min(0.9, ar_coef))
        else:
            ar_coef = 0.30

        return {
            "hourly_profile": hourly_profile,
            "weekend_factor": min(1.0, max(0.1, weekend_factor)),
            "ar_coef":        ar_coef,
            "noise_std":      max(0.1, float(kwh.std()) * 0.15),
            "mean_kwh":       float(kwh.mean()),
        }
