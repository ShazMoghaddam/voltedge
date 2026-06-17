"""
VoltEdge — ESG metrics and carbon intensity calculations.

Computes Scope 2 emissions (market-based and location-based),
energy intensity KPIs, and ESG-ready compliance metrics aligned
with GHG Protocol and GRI 302.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd


# ── Emission factors (kgCO2e per kWh) ────────────────────────────────────────
# Source: IEA 2023 / DESNZ UK Grid Intensity
GRID_EMISSION_FACTORS: dict[str, float] = {
    "GB": 0.207,    # UK — 2023 average
    "DE": 0.385,
    "FR": 0.052,    # France — high nuclear share
    "US": 0.386,
    "CN": 0.581,
    "NL": 0.270,
    "DEFAULT": 0.350,
}


@dataclass
class ESGMetrics:
    """Computed ESG snapshot for a site / time window."""

    site_id: str
    period_start: datetime
    period_end: datetime
    total_kwh: float
    total_co2_kg: float
    carbon_intensity_kgco2_per_kwh: float
    energy_intensity_kwh_per_unit: float | None  # Unit depends on context (m², revenue, etc.)
    renewable_fraction: float          # 0–1
    peak_demand_kw: float
    avg_power_factor: float | None
    gri_302_1: float                   # GRI 302-1: Energy consumption within org (GJ)
    scope_2_location_based_tco2e: float
    scope_2_market_based_tco2e: float  # Uses supplier-specific emission factor if known

    @property
    def gj_total(self) -> float:
        return self.total_kwh * 0.0036  # kWh → GJ


class ESGCalculator:
    """
    Compute ESG metrics from processed energy DataFrames.
    """

    def __init__(
        self,
        country_code: str = "GB",
        floor_area_m2: float | None = None,
        supplier_emission_factor: float | None = None,
        renewable_kwh: float = 0.0,
    ) -> None:
        self.grid_factor = GRID_EMISSION_FACTORS.get(
            country_code.upper(), GRID_EMISSION_FACTORS["DEFAULT"]
        )
        self.floor_area_m2 = floor_area_m2
        self.supplier_factor = supplier_emission_factor or self.grid_factor
        self.renewable_kwh = renewable_kwh

    def compute(self, df: pd.DataFrame, site_id: str) -> ESGMetrics:
        """
        Compute a full ESG snapshot from a processed DataFrame.

        Args:
            df:       Processed DataFrame with `timestamp` and `kwh` columns.
            site_id:  Site identifier for the output object.
        """
        if df.empty or "kwh" not in df.columns:
            raise ValueError("DataFrame must contain a 'kwh' column.")

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        total_kwh = df["kwh"].sum()
        period_start = df["timestamp"].min().to_pydatetime()
        period_end = df["timestamp"].max().to_pydatetime()

        total_co2_kg = total_kwh * self.grid_factor
        renewable_fraction = min(1.0, self.renewable_kwh / total_kwh) if total_kwh > 0 else 0.0

        # Scope 2 (location-based): uses grid average factor
        scope2_location = (total_kwh * self.grid_factor) / 1000  # tCO2e

        # Scope 2 (market-based): uses supplier/contract factor, net of RECs
        net_kwh = max(0.0, total_kwh - self.renewable_kwh)
        scope2_market = (net_kwh * self.supplier_factor) / 1000  # tCO2e

        peak_demand = df["kwh"].max()
        avg_pf = df["power_factor"].mean() if "power_factor" in df.columns else None

        energy_intensity = (
            total_kwh / self.floor_area_m2 if self.floor_area_m2 else None
        )

        return ESGMetrics(
            site_id=site_id,
            period_start=period_start,
            period_end=period_end,
            total_kwh=round(total_kwh, 2),
            total_co2_kg=round(total_co2_kg, 2),
            carbon_intensity_kgco2_per_kwh=round(self.grid_factor, 4),
            energy_intensity_kwh_per_unit=round(energy_intensity, 2) if energy_intensity else None,
            renewable_fraction=round(renewable_fraction, 4),
            peak_demand_kw=round(peak_demand, 2),
            avg_power_factor=round(avg_pf, 3) if avg_pf else None,
            gri_302_1=round(total_kwh * 0.0036, 3),
            scope_2_location_based_tco2e=round(scope2_location, 4),
            scope_2_market_based_tco2e=round(scope2_market, 4),
        )

    def summarise_sites(
        self, site_metrics: list[ESGMetrics]
    ) -> dict[str, float]:
        """Roll up ESG metrics across multiple sites for portfolio-level reporting."""
        return {
            "total_kwh": sum(m.total_kwh for m in site_metrics),
            "total_co2_kg": sum(m.total_co2_kg for m in site_metrics),
            "scope_2_location_tco2e": sum(m.scope_2_location_based_tco2e for m in site_metrics),
            "scope_2_market_tco2e": sum(m.scope_2_market_based_tco2e for m in site_metrics),
            "gri_302_1_gj": sum(m.gri_302_1 for m in site_metrics),
            "avg_renewable_fraction": (
                sum(m.renewable_fraction for m in site_metrics) / len(site_metrics)
                if site_metrics else 0.0
            ),
        }
