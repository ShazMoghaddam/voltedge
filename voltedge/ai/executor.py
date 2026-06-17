"""
VoltEdge AI — Tool Executor

Executes tool calls returned by Claude and formats results
back as structured dicts for the API tool_result block.

Each execute_* method:
  - Pulls real data from VoltEdge's service layer
  - Returns a plain dict (JSON-serialisable, no pandas/numpy)
  - Never raises — returns {"error": "..."} on failure so the
    conversation continues gracefully even if a site has no data
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from voltedge.ai.context_builder import ContextBuilder
from voltedge.utils.logger import get_logger

log = get_logger(__name__)


class ToolExecutor:
    """
    Routes tool calls to VoltEdge's data layer and returns
    JSON-serialisable result dicts.

    Args:
        store:   Storage backend (LocalStore / S3Store / etc.)
        sites:   Known site IDs (used to populate portfolio tools)
        context_builder: Optional pre-built ContextBuilder instance.
    """

    def __init__(
        self,
        store:            Any,
        sites:            list[str],
        context_builder:  ContextBuilder | None = None,
        carbon_intensity: float = 0.207,
        tariff_per_kwh:   float = 0.28,
        currency:         str   = "GBP",
    ) -> None:
        self._store  = store
        self._sites  = sites
        self._ctx    = context_builder or ContextBuilder(
            store=store,
            carbon_intensity=carbon_intensity,
            tariff_per_kwh=tariff_per_kwh,
            currency=currency,
        )

    # ── Dispatch ──────────────────────────────────────────────────────────────

    def execute(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a tool call to the appropriate executor method."""
        try:
            dispatch = {
                "get_site_summary":      self._get_site_summary,
                "get_portfolio_summary": self._get_portfolio_summary,
                "get_anomalies":         self._get_anomalies,
                "get_forecast":          self._get_forecast,
                "get_esg_metrics":       self._get_esg_metrics,
                "get_optimization_tips": self._get_optimization_tips,
                "compare_sites":         self._compare_sites,
                "get_peak_hours":        self._get_peak_hours,
                "get_root_cause":        self._get_root_cause,
                "run_counterfactual":    self._run_counterfactual,
                "simulate_scenario":     self._simulate_scenario,
            }
            fn = dispatch.get(tool_name)
            if fn is None:
                return {"error": f"Unknown tool: {tool_name}"}
            return fn(**tool_input)
        except Exception as exc:
            log.error("tool_executor.error", tool=tool_name, error=str(exc))
            return {"error": str(exc), "tool": tool_name}

    # ── Tool implementations ──────────────────────────────────────────────────

    def _get_site_summary(self, site_id: str, hours: int = 24) -> dict:
        snap = self._ctx.build_site(site_id, hours)
        return {
            "site_id":              snap.site_id,
            "period_hours":         snap.period_hours,
            "snapshot_time":        snap.snapshot_time,
            "total_kwh":            snap.total_kwh,
            "avg_kwh_per_hour":     snap.avg_kwh_per_hour,
            "peak_kwh":             snap.peak_kwh,
            "peak_at":              snap.peak_at,
            "min_kwh":              snap.min_kwh,
            "trend_direction":      snap.trend_direction,
            "pct_change":           snap.pct_change_vs_prev_period,
            "co2_kg":               snap.co2_kg,
            "carbon_intensity":     snap.carbon_intensity,
            "scope2_tco2e":         snap.scope2_tco2e,
            "anomaly_count":        snap.anomaly_count,
            "anomalies":            snap.anomaly_details,
            "forecast_24h_kwh":     snap.forecast_24h_kwh,
            "forecast_available":   snap.forecast_available,
            "estimated_cost":       snap.estimated_cost,
            "cost_currency":        snap.cost_currency,
        }

    def _get_portfolio_summary(self, hours: int = 24) -> dict:
        portfolio = self._ctx.build_portfolio(self._sites, hours)
        return {
            "period_hours":      portfolio.period_hours,
            "snapshot_time":     portfolio.snapshot_time,
            "site_count":        len(portfolio.sites),
            "total_kwh":         portfolio.total_kwh,
            "total_co2_kg":      portfolio.total_co2_kg,
            "total_anomalies":   portfolio.total_anomalies,
            "highest_consumer":  portfolio.highest_consumer,
            "lowest_consumer":   portfolio.lowest_consumer,
            "sites": [
                {
                    "site_id":     s.site_id,
                    "total_kwh":   s.total_kwh,
                    "co2_kg":      s.co2_kg,
                    "anomalies":   s.anomaly_count,
                    "trend":       s.trend_direction,
                    "pct_change":  s.pct_change_vs_prev_period,
                }
                for s in sorted(portfolio.sites, key=lambda x: x.total_kwh, reverse=True)
            ],
        }

    def _get_anomalies(
        self,
        site_id:   str | None = None,
        hours:     int = 24,
        min_score: float = 0.70,
    ) -> dict:
        sites = [site_id] if site_id else self._sites
        all_anomalies: list[dict] = []

        for sid in sites:
            snap = self._ctx.build_site(sid, hours)
            for a in snap.anomaly_details:
                if a.get("score", 0) >= min_score:
                    all_anomalies.append({**a, "site_id": sid})

        all_anomalies.sort(key=lambda x: x.get("score", 0), reverse=True)

        return {
            "period_hours":   hours,
            "sites_checked":  len(sites),
            "anomaly_count":  len(all_anomalies),
            "min_score":      min_score,
            "anomalies":      all_anomalies[:20],   # cap at 20 for context
        }

    def _get_forecast(self, site_id: str, horizon_hours: int = 24) -> dict:
        try:
            import pandas as pd
            from voltedge.models.forecasting import DemandForecaster

            df = self._store.read(site_id, layer="processed", days=90)
            if df.empty or len(df) < 48:
                return {"error": f"Insufficient data for {site_id} — need ≥48h of processed data"}

            forecaster = DemandForecaster(site_id=site_id, horizon_hours=horizon_hours)
            metrics    = forecaster.train(df)
            forecast   = forecaster.predict(df)

            points = forecast.head(min(horizon_hours, 24)).to_dict(orient="records")
            for p in points:
                if hasattr(p.get("timestamp"), "isoformat"):
                    p["timestamp"] = p["timestamp"].isoformat()

            return {
                "site_id":        site_id,
                "horizon_hours":  horizon_hours,
                "model_mae":      metrics.get("mae"),
                "model_r2":       metrics.get("r2"),
                "total_forecast_kwh": round(
                    sum(p["kwh_forecast"] for p in points), 1
                ),
                "avg_hourly_kwh": round(
                    sum(p["kwh_forecast"] for p in points) / max(len(points), 1), 1
                ),
                "forecast_points": points[:12],   # first 12h for context brevity
            }
        except Exception as exc:
            return {"error": f"Forecast failed for {site_id}: {exc}"}

    def _get_esg_metrics(
        self,
        site_id: str | None = None,
        days:    int = 30,
        country: str = "GB",
    ) -> dict:
        try:
            from voltedge.esg.metrics import ESGCalculator

            calc  = ESGCalculator(country_code=country)
            sites = [site_id] if site_id else self._sites
            results = []

            for sid in sites:
                df = self._store.read(sid, layer="processed", days=days)
                if df.empty:
                    continue
                m = calc.compute(df, sid)
                results.append({
                    "site_id":                  m.site_id,
                    "total_kwh":                m.total_kwh,
                    "total_co2_kg":             m.total_co2_kg,
                    "scope2_location_tco2e":    m.scope_2_location_based_tco2e,
                    "scope2_market_tco2e":      m.scope_2_market_based_tco2e,
                    "gri_302_1_gj":             m.gri_302_1,
                    "carbon_intensity":         m.carbon_intensity_kgco2_per_kwh,
                    "renewable_fraction":       m.renewable_fraction,
                    "peak_demand_kw":           m.peak_demand_kw,
                    "period_days":              days,
                    "country":                  country,
                })

            if not results:
                return {"error": "No ESG data available for the requested sites/period"}

            portfolio_totals = {
                "total_kwh":      sum(r["total_kwh"] for r in results),
                "total_co2_kg":   sum(r["total_co2_kg"] for r in results),
                "scope2_location_tco2e_total": sum(r["scope2_location_tco2e"] for r in results),
                "scope2_market_tco2e_total":   sum(r["scope2_market_tco2e"] for r in results),
            }

            return {
                "period_days":      days,
                "country":          country,
                "sites":            results,
                "portfolio_totals": portfolio_totals,
            }
        except Exception as exc:
            return {"error": f"ESG calculation failed: {exc}"}

    def _get_optimization_tips(self, site_id: str, tariff: str = "UK_HALF_HOURLY") -> dict:
        try:
            from voltedge.models.optimization import CostOptimizer

            df = self._store.read(site_id, layer="processed", days=30)
            if df.empty:
                return {"error": f"No data for {site_id}"}

            optimizer = CostOptimizer(site_id=site_id, tariff=tariff)
            report    = optimizer.analyse(df)

            return {
                "site_id":               report.site_id,
                "tariff":                report.tariff_name,
                "current_annual_cost":   report.current_annual_cost,
                "optimised_annual_cost": report.optimised_annual_cost,
                "total_saving":          round(report.total_potential_saving, 2),
                "saving_pct":            round(report.saving_pct, 1),
                "currency":              report.currency,
                "peak_demand_kw":        report.peak_demand_kw,
                "recommendations": [
                    {
                        "category":    r.category,
                        "description": r.description[:200],
                        "annual_saving": r.estimated_annual_saving,
                        "effort":      r.effort,
                        "payback_months": r.payback_months,
                    }
                    for r in report.recommendations[:5]
                ],
            }
        except Exception as exc:
            return {"error": f"Optimisation failed for {site_id}: {exc}"}

    def _compare_sites(self, site_a: str, site_b: str, hours: int = 24) -> dict:
        snap_a = self._ctx.build_site(site_a, hours)
        snap_b = self._ctx.build_site(site_b, hours)

        def _pct_diff(a, b):
            if b == 0:
                return None
            return round((a - b) / b * 100, 1)

        return {
            "period_hours": hours,
            "site_a": {
                "site_id":     snap_a.site_id,
                "total_kwh":   snap_a.total_kwh,
                "co2_kg":      snap_a.co2_kg,
                "anomalies":   snap_a.anomaly_count,
                "trend":       snap_a.trend_direction,
                "cost":        snap_a.estimated_cost,
            },
            "site_b": {
                "site_id":     snap_b.site_id,
                "total_kwh":   snap_b.total_kwh,
                "co2_kg":      snap_b.co2_kg,
                "anomalies":   snap_b.anomaly_count,
                "trend":       snap_b.trend_direction,
                "cost":        snap_b.estimated_cost,
            },
            "comparison": {
                "kwh_diff_pct":  _pct_diff(snap_a.total_kwh, snap_b.total_kwh),
                "co2_diff_pct":  _pct_diff(snap_a.co2_kg, snap_b.co2_kg),
                "higher_consumer": (
                    snap_a.site_id if snap_a.total_kwh >= snap_b.total_kwh
                    else snap_b.site_id
                ),
                "more_anomalies": (
                    snap_a.site_id if snap_a.anomaly_count >= snap_b.anomaly_count
                    else snap_b.site_id
                ),
            },
        }

    def _get_peak_hours(self, site_id: str, days: int = 30) -> dict:
        import pandas as pd

        df = self._store.read(site_id, layer="processed", days=days)
        if df.empty or "kwh" not in df.columns:
            return {"error": f"No data for {site_id}"}

        if "timestamp" not in df.columns:
            return {"error": "No timestamp column"}

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["hour"]      = df["timestamp"].dt.hour
        df["day_name"]  = df["timestamp"].dt.day_name()

        by_hour = df.groupby("hour")["kwh"].mean().round(2)
        by_day  = df.groupby("day_name")["kwh"].mean().round(2)

        peak_hour     = int(by_hour.idxmax())
        low_hour      = int(by_hour.idxmin())
        peak_day      = by_day.idxmax()
        low_day       = by_day.idxmin()

        return {
            "site_id":      site_id,
            "period_days":  days,
            "peak_hour":    peak_hour,
            "peak_hour_avg_kwh": float(by_hour[peak_hour]),
            "low_hour":     low_hour,
            "low_hour_avg_kwh": float(by_hour[low_hour]),
            "peak_day":     peak_day,
            "peak_day_avg_kwh": float(by_day[peak_day]),
            "low_day":      low_day,
            "hourly_profile": by_hour.to_dict(),
            "daily_profile":  by_day.to_dict(),
        }


# ── Causal AI tool implementations (v8.0) ─────────────────────────────────────

    def _get_root_cause(
        self,
        site_id:           str,
        anomaly_timestamp: str,
        anomaly_variable:  str = "kwh",
    ) -> dict:
        try:
            from voltedge.causal.graph import CausalGraphBuilder
            from voltedge.causal.root_cause import RootCauseAnalyser

            df = self._store.read(site_id, layer="processed", days=30)
            if df.empty:
                return {"error": f"No data for {site_id}"}

            graph    = CausalGraphBuilder(min_samples=48).build(df, site_id)
            analyser = RootCauseAnalyser(graph)
            report   = analyser.analyse(df, anomaly_timestamp, anomaly_variable)
            return report.to_dict()
        except Exception as exc:
            return {"error": f"Root-cause analysis failed: {exc}"}

    def _run_counterfactual(
        self,
        site_id:                str,
        intervention_variable:  str,
        intervention_value:     float,
        description:            str = "",
    ) -> dict:
        try:
            from voltedge.causal.graph import CausalGraphBuilder
            from voltedge.causal.counterfactual import (
                CounterfactualEngine, Intervention,
            )

            df = self._store.read(site_id, layer="processed", days=30)
            if df.empty:
                return {"error": f"No data for {site_id}"}

            graph  = CausalGraphBuilder(min_samples=48).build(df, site_id)
            engine = CounterfactualEngine(graph)
            iv     = Intervention(
                variable=intervention_variable,
                new_value=intervention_value,
                description=description or f"Set {intervention_variable}={intervention_value}",
            )
            result = engine.compute(df, iv)
            return result.to_dict()
        except Exception as exc:
            return {"error": f"Counterfactual failed: {exc}"}

    def _simulate_scenario(
        self,
        site_id:          str,
        scenario_type:    str,
        scenario_params:  dict | None = None,
        simulation_hours: int = 8760,
    ) -> dict:
        try:
            from voltedge.causal.graph import CausalGraphBuilder
            from voltedge.causal.digital_twin import EnergyDigitalTwin

            df = self._store.read(site_id, layer="processed", days=90)
            if df.empty:
                return {"error": f"No data for {site_id}"}

            graph  = CausalGraphBuilder(min_samples=48).build(df, site_id)
            twin   = EnergyDigitalTwin(graph=graph, df=df, n_simulations=30)
            params = scenario_params or {}

            scenario_map = {
                "baseline":    lambda: twin.run_baseline(hours=simulation_hours),
                "solar_pv":    lambda: twin.run_scenario(
                    EnergyDigitalTwin.solar_scenario(
                        params.get("capacity_kwp", 100),
                        params.get("yield_kwh_per_kwp", 900),
                    ), hours=simulation_hours),
                "bess":        lambda: twin.run_scenario(
                    EnergyDigitalTwin.bess_scenario(
                        params.get("capacity_kwh", 500),
                        params.get("charge_rate_kw", 200),
                    ), hours=simulation_hours),
                "ev_fleet":    lambda: twin.run_scenario(
                    EnergyDigitalTwin.ev_fleet_scenario(
                        params.get("vehicles", 10),
                        params.get("kw_per_vehicle", 22),
                    ), hours=simulation_hours),
                "night_shift": lambda: twin.run_scenario(
                    EnergyDigitalTwin.night_shift_scenario(),
                    hours=simulation_hours),
            }

            fn = scenario_map.get(scenario_type)
            if fn is None:
                return {"error": f"Unknown scenario: {scenario_type}"}

            result   = fn()
            baseline = twin.run_baseline(hours=simulation_hours)

            d = result.to_dict()
            d["vs_baseline"] = {
                "delta_kwh":    round(result.total_kwh - baseline.total_kwh, 1),
                "delta_co2_kg": round(result.total_co2_kg - baseline.total_co2_kg, 1),
                "delta_cost":   round(result.total_cost - baseline.total_cost, 2),
                "delta_pct":    round(
                    (result.total_kwh - baseline.total_kwh) /
                    (abs(baseline.total_kwh) + 1e-8) * 100, 1
                ),
            }
            return d
        except Exception as exc:
            return {"error": f"Simulation failed: {exc}"}
