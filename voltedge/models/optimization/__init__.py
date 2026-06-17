"""
VoltEdge — Cost & Load Optimization Model

Analyses energy consumption patterns and recommends actionable
load-shifting strategies to minimise peak demand charges and
tariff-period costs.

Approach:
  - Time-of-Use (ToU) tariff modelling
  - Peak shaving recommendations via threshold analysis
  - Flexible load identification (non-critical loads that can shift)
  - Payback estimation for each recommendation

Usage:
    from voltedge.models.optimization import CostOptimizer
    optimizer = CostOptimizer(site_id="SITE-01", tariff="UK_HALF_HOURLY")
    report = optimizer.analyse(df)
    for rec in report.recommendations:
        print(rec.description, rec.estimated_annual_saving_gbp)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from voltedge.utils.logger import get_logger

log = get_logger(__name__)


# ── Tariff definitions ────────────────────────────────────────────────────────

TARIFFS: dict[str, dict[str, Any]] = {
    "UK_HALF_HOURLY": {
        "currency": "GBP",
        "peak_hours": list(range(16, 20)),        # 16:00–19:59 (Triad risk window)
        "shoulder_hours": list(range(7, 16)),
        "off_peak_hours": list(range(0, 7)) + [20, 21, 22, 23],
        "peak_rate_per_kwh": 0.38,
        "shoulder_rate_per_kwh": 0.22,
        "off_peak_rate_per_kwh": 0.09,
        "demand_charge_per_kw_month": 12.50,      # £/kW of monthly peak
    },
    "EU_INDUSTRIAL": {
        "currency": "EUR",
        "peak_hours": list(range(8, 20)),
        "shoulder_hours": list(range(6, 8)) + [20, 21],
        "off_peak_hours": list(range(0, 6)) + [22, 23],
        "peak_rate_per_kwh": 0.28,
        "shoulder_rate_per_kwh": 0.16,
        "off_peak_rate_per_kwh": 0.07,
        "demand_charge_per_kw_month": 9.80,
    },
    "US_COMMERCIAL": {
        "currency": "USD",
        "peak_hours": list(range(9, 21)),
        "shoulder_hours": list(range(7, 9)) + [21, 22],
        "off_peak_hours": list(range(0, 7)) + [23],
        "peak_rate_per_kwh": 0.18,
        "shoulder_rate_per_kwh": 0.12,
        "off_peak_rate_per_kwh": 0.06,
        "demand_charge_per_kw_month": 15.00,
    },
}


# ── Output data model ─────────────────────────────────────────────────────────

@dataclass
class OptimizationRecommendation:
    category: str                        # "peak_shaving" | "load_shifting" | "power_factor"
    description: str
    estimated_annual_saving: float       # In tariff currency
    currency: str
    effort: str                          # "low" | "medium" | "high"
    payback_months: float | None = None
    affected_hours: list[int] = field(default_factory=list)
    kwh_shift_potential: float = 0.0


@dataclass
class OptimizationReport:
    site_id: str
    tariff_name: str
    analysis_period_days: int
    current_annual_cost: float
    optimised_annual_cost: float
    currency: str
    peak_demand_kw: float
    avg_power_factor: float | None
    recommendations: list[OptimizationRecommendation]

    @property
    def total_potential_saving(self) -> float:
        return sum(r.estimated_annual_saving for r in self.recommendations)

    @property
    def saving_pct(self) -> float:
        if self.current_annual_cost == 0:
            return 0.0
        return self.total_potential_saving / self.current_annual_cost * 100


# ── Optimizer ─────────────────────────────────────────────────────────────────

class CostOptimizer:
    """
    Analyses processed energy DataFrames against a ToU tariff and
    generates prioritised cost-reduction recommendations.

    Args:
        site_id:       Site identifier.
        tariff:        Tariff name (see TARIFFS dict above).
        capex_per_kwh: Optional BESS capital cost (£/kWh) for payback calc.
    """

    def __init__(
        self,
        site_id: str,
        tariff: str = "UK_HALF_HOURLY",
        capex_per_kwh: float = 350.0,
    ) -> None:
        self.site_id = site_id
        self.tariff_name = tariff
        self.tariff = TARIFFS.get(tariff, TARIFFS["UK_HALF_HOURLY"])
        self.capex_per_kwh = capex_per_kwh

    # ── Public API ────────────────────────────────────────────────────────────

    def analyse(self, df: pd.DataFrame) -> OptimizationReport:
        """
        Run the full optimisation analysis pipeline.

        Args:
            df: Processed DataFrame with `timestamp` and `kwh` columns.

        Returns:
            OptimizationReport with recommendations sorted by saving (desc).
        """
        if df.empty or "kwh" not in df.columns:
            raise ValueError("DataFrame must have a 'kwh' column.")

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["hour"] = df["timestamp"].dt.hour
        df = df.sort_values("timestamp").reset_index(drop=True)

        period_days = max(1, (df["timestamp"].max() - df["timestamp"].min()).days)

        log.info("optimizer.analyse_start", site=self.site_id,
                 tariff=self.tariff_name, days=period_days)

        current_cost = self._compute_energy_cost(df)
        annual_scale = 365 / period_days
        annual_current = current_cost * annual_scale

        recommendations: list[OptimizationRecommendation] = []

        # Analysis modules
        recommendations += self._peak_shaving_analysis(df, annual_scale)
        recommendations += self._load_shifting_analysis(df, annual_scale)
        recommendations += self._power_factor_analysis(df, annual_scale)
        recommendations += self._off_peak_opportunity_analysis(df, annual_scale)

        # Sort by saving descending
        recommendations.sort(key=lambda r: r.estimated_annual_saving, reverse=True)

        total_saving = sum(r.estimated_annual_saving for r in recommendations)
        optimised_annual = max(0.0, annual_current - total_saving)
        peak_kw = float(df["kwh"].max())
        avg_pf = float(df["power_factor"].mean()) if "power_factor" in df.columns else None

        report = OptimizationReport(
            site_id=self.site_id,
            tariff_name=self.tariff_name,
            analysis_period_days=period_days,
            current_annual_cost=round(annual_current, 2),
            optimised_annual_cost=round(optimised_annual, 2),
            currency=self.tariff["currency"],
            peak_demand_kw=round(peak_kw, 2),
            avg_power_factor=round(avg_pf, 3) if avg_pf else None,
            recommendations=recommendations,
        )

        log.info(
            "optimizer.analyse_complete",
            site=self.site_id,
            current_annual_cost=report.current_annual_cost,
            potential_saving=report.total_potential_saving,
            saving_pct=round(report.saving_pct, 1),
            recommendations=len(recommendations),
        )
        return report

    # ── Analysis modules ──────────────────────────────────────────────────────

    def _compute_energy_cost(self, df: pd.DataFrame) -> float:
        """Compute actual energy cost over the data period using ToU tariff."""
        t = self.tariff
        peak_mask     = df["hour"].isin(t["peak_hours"])
        shoulder_mask = df["hour"].isin(t["shoulder_hours"])
        off_peak_mask = df["hour"].isin(t["off_peak_hours"])

        cost = (
            df.loc[peak_mask,     "kwh"].sum() * t["peak_rate_per_kwh"] +
            df.loc[shoulder_mask, "kwh"].sum() * t["shoulder_rate_per_kwh"] +
            df.loc[off_peak_mask, "kwh"].sum() * t["off_peak_rate_per_kwh"]
        )
        # Demand charge: peak kW × monthly charge × 12
        monthly_peak_kw = df["kwh"].max()
        demand_cost = monthly_peak_kw * t["demand_charge_per_kw_month"] * 12
        return float(cost + demand_cost / (365 / max(1, len(df) / 24)))

    def _peak_shaving_analysis(
        self, df: pd.DataFrame, annual_scale: float
    ) -> list[OptimizationRecommendation]:
        """Estimate savings from shaving peak demand by 15% via BESS."""
        peak_kw = float(df["kwh"].max())
        target_reduction_kw = peak_kw * 0.15
        t = self.tariff

        # Demand charge saving
        demand_saving = target_reduction_kw * t["demand_charge_per_kw_month"] * 12
        # Energy cost saving: avoided peak-rate hours
        peak_hours_df = df[df["hour"].isin(t["peak_hours"])]
        avg_peak_kwh = float(peak_hours_df["kwh"].mean()) if not peak_hours_df.empty else 0
        energy_saving_period = avg_peak_kwh * 0.1 * len(peak_hours_df) * t["peak_rate_per_kwh"]
        energy_saving_annual = energy_saving_period * annual_scale

        total_annual_saving = demand_saving + energy_saving_annual
        bess_size_kwh = target_reduction_kw * 2  # 2h discharge
        capex = bess_size_kwh * self.capex_per_kwh
        payback = (capex / total_annual_saving * 12) if total_annual_saving > 0 else None

        return [OptimizationRecommendation(
            category="peak_shaving",
            description=(
                f"Deploy BESS ({bess_size_kwh:.0f} kWh) to shave peak demand by "
                f"{target_reduction_kw:.1f} kW during peak tariff windows "
                f"({min(t['peak_hours'])}:00–{max(t['peak_hours'])+1}:00)."
            ),
            estimated_annual_saving=round(total_annual_saving, 2),
            currency=t["currency"],
            effort="high",
            payback_months=round(payback, 1) if payback else None,
            affected_hours=t["peak_hours"],
            kwh_shift_potential=round(bess_size_kwh, 1),
        )]

    def _load_shifting_analysis(
        self, df: pd.DataFrame, annual_scale: float
    ) -> list[OptimizationRecommendation]:
        """Identify peak-hour loads that could shift to off-peak."""
        t = self.tariff
        peak_df     = df[df["hour"].isin(t["peak_hours"])]
        off_peak_df = df[df["hour"].isin(t["off_peak_hours"])]

        if peak_df.empty or off_peak_df.empty:
            return []

        # Shiftable = loads above the off-peak mean during peak hours
        baseline_kwh  = float(off_peak_df["kwh"].mean())
        peak_excess   = (peak_df["kwh"] - baseline_kwh).clip(lower=0)
        shiftable_kwh = float(peak_excess.mean()) * len(peak_df)

        rate_diff = t["peak_rate_per_kwh"] - t["off_peak_rate_per_kwh"]
        period_saving = shiftable_kwh * rate_diff * 0.3  # Assume 30% is actually shiftable
        annual_saving = period_saving * annual_scale

        return [OptimizationRecommendation(
            category="load_shifting",
            description=(
                f"Shift ~{shiftable_kwh * 0.3:.0f} kWh of flexible loads "
                f"(HVAC pre-cooling, EV charging, batch processes) "
                f"from peak ({min(t['peak_hours'])}–{max(t['peak_hours'])+1}h) "
                f"to off-peak ({t['off_peak_rate_per_kwh']:.2f} {t['currency']}/kWh)."
            ),
            estimated_annual_saving=round(annual_saving, 2),
            currency=t["currency"],
            effort="medium",
            affected_hours=t["peak_hours"],
            kwh_shift_potential=round(shiftable_kwh * 0.3, 1),
        )]

    def _power_factor_analysis(
        self, df: pd.DataFrame, annual_scale: float
    ) -> list[OptimizationRecommendation]:
        """Flag poor power factor and estimate penalty savings."""
        if "power_factor" not in df.columns:
            return []

        avg_pf = float(df["power_factor"].mean())
        if avg_pf >= 0.92:
            return []

        # Reactive power penalty proxy: ~1–3% of bill per 0.01 PF below 0.92
        gap = max(0, 0.92 - avg_pf)
        penalty_pct = gap * 100 * 0.015  # 1.5% bill per 0.01 PF deficit
        t = self.tariff
        energy_cost = df["kwh"].sum() * t["peak_rate_per_kwh"]  # Rough proxy
        annual_saving = energy_cost * penalty_pct * annual_scale

        return [OptimizationRecommendation(
            category="power_factor",
            description=(
                f"Install power factor correction capacitors. Current avg PF = {avg_pf:.3f} "
                f"(target ≥0.92). Penalty estimated at {penalty_pct*100:.1f}% of energy bill."
            ),
            estimated_annual_saving=round(annual_saving, 2),
            currency=t["currency"],
            effort="medium",
            affected_hours=list(range(24)),
        )]

    def _off_peak_opportunity_analysis(
        self, df: pd.DataFrame, annual_scale: float
    ) -> list[OptimizationRecommendation]:
        """Flag weekend/night hours that are underutilised for batch workloads."""
        t = self.tariff
        weekend_df = df[df["timestamp"].dt.weekday >= 5] if "timestamp" in df.columns else pd.DataFrame()

        if weekend_df.empty:
            return []

        off_peak_weekend_avg = float(weekend_df[weekend_df["hour"].isin(t["off_peak_hours"])]["kwh"].mean())
        overall_avg = float(df["kwh"].mean())

        if off_peak_weekend_avg >= overall_avg * 0.6:
            return []  # Weekend utilisation already reasonable

        potential_kwh = (overall_avg - off_peak_weekend_avg) * len(weekend_df) * 0.2
        saving = potential_kwh * (t["peak_rate_per_kwh"] - t["off_peak_rate_per_kwh"])

        return [OptimizationRecommendation(
            category="scheduling",
            description=(
                "Reschedule batch IT jobs, data backups, and maintenance tasks "
                "to weekend off-peak hours. Current weekend off-peak utilisation "
                f"is {off_peak_weekend_avg:.1f} kWh vs site avg {overall_avg:.1f} kWh."
            ),
            estimated_annual_saving=round(saving * annual_scale, 2),
            currency=t["currency"],
            effort="low",
            affected_hours=t["off_peak_hours"],
        )]
