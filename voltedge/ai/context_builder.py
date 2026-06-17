"""
VoltEdge AI — Context Builder

Assembles structured, token-efficient context snapshots from VoltEdge's
data layer for injection into the LLM. This is the RAG layer — it turns
raw time-series data into concise natural language summaries the model
can reason over without hitting context limits.

Design principles:
  - Never pass raw DataFrames to the LLM — summarise them
  - Always include units and timestamps
  - Cap total context at ~4000 tokens (well within claude-sonnet limits)
  - Deterministic output: same data = same context (for cache-friendliness)

Context sections assembled:
  1. Site inventory (name, type, status)
  2. Recent consumption summary (24h / 7d stats)
  3. Active anomalies (score, label, timestamp)
  4. ESG snapshot (Scope 2, carbon intensity)
  5. Latest forecast (next 24h)
  6. Top cost-saving opportunities

Usage:
    builder = ContextBuilder(store=LocalStore(), sites=["LONDON-01"])
    ctx = builder.build(site_id="LONDON-01", hours=24)
    print(ctx.to_prompt_text())
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from voltedge.utils.logger import get_logger

log = get_logger(__name__)

# Approx token budget per section (chars ÷ 4 ≈ tokens)
MAX_CONTEXT_CHARS = 12_000


@dataclass
class SiteSnapshot:
    """Distilled one-site context for LLM injection."""
    site_id:          str
    snapshot_time:    str
    period_hours:     int

    # Consumption
    total_kwh:        float = 0.0
    avg_kwh_per_hour: float = 0.0
    peak_kwh:         float = 0.0
    peak_at:          str   = ""
    min_kwh:          float = 0.0

    # Trend
    pct_change_vs_prev_period: float = 0.0
    trend_direction:           str   = "stable"  # up / down / stable

    # Anomalies
    anomaly_count:    int         = 0
    anomaly_details:  list[dict]  = field(default_factory=list)

    # ESG
    co2_kg:           float = 0.0
    carbon_intensity: float = 0.0
    scope2_tco2e:     float = 0.0

    # Forecast
    forecast_24h_kwh:  float = 0.0
    forecast_available: bool = False

    # Cost
    estimated_cost:   float = 0.0
    cost_currency:    str   = "GBP"

    # Raw for tool responses
    raw_stats:        dict[str, Any] = field(default_factory=dict)

    def to_prompt_text(self) -> str:
        """Render as a concise prompt-ready text block."""
        lines = [
            f"=== Site: {self.site_id} ===",
            f"Snapshot: {self.snapshot_time} UTC | Period: last {self.period_hours}h",
            "",
            "[ Consumption ]",
            f"  Total:   {self.total_kwh:,.1f} kWh",
            f"  Average: {self.avg_kwh_per_hour:,.1f} kWh/h",
            f"  Peak:    {self.peak_kwh:,.1f} kWh at {self.peak_at}",
            f"  Min:     {self.min_kwh:,.1f} kWh",
            f"  Trend:   {self.trend_direction} "
            f"({self.pct_change_vs_prev_period:+.1f}% vs prior {self.period_hours}h)",
            "",
            "[ ESG / Emissions ]",
            f"  CO₂:             {self.co2_kg:,.1f} kg",
            f"  Carbon intensity: {self.carbon_intensity:.4f} kg CO₂/kWh",
            f"  Scope 2 (loc):   {self.scope2_tco2e:.4f} tCO₂e",
            "",
        ]

        if self.anomaly_count > 0:
            lines += [
                "[ Anomalies Detected ]",
                f"  Count: {self.anomaly_count}",
            ]
            for a in self.anomaly_details[:5]:
                lines.append(
                    f"  - {a.get('timestamp','?')[:16]}: "
                    f"score={a.get('score',0):.3f}, label={a.get('label','?')}"
                )
            lines.append("")
        else:
            lines += ["[ Anomalies ] None detected in this period", ""]

        if self.forecast_available:
            lines += [
                "[ 24h Forecast ]",
                f"  Projected consumption: {self.forecast_24h_kwh:,.1f} kWh",
                "",
            ]

        if self.estimated_cost > 0:
            lines += [
                "[ Estimated Cost ]",
                f"  {self.estimated_cost:,.2f} {self.cost_currency} "
                f"(based on {self.period_hours}h)",
                "",
            ]

        return "\n".join(lines)


@dataclass
class PortfolioSnapshot:
    """Multi-site aggregated context."""
    sites:            list[SiteSnapshot]
    snapshot_time:    str
    period_hours:     int
    total_kwh:        float = 0.0
    total_co2_kg:     float = 0.0
    total_anomalies:  int   = 0
    highest_consumer: str   = ""
    lowest_consumer:  str   = ""

    def to_prompt_text(self) -> str:
        lines = [
            "=== VoltEdge Portfolio Overview ===",
            f"Snapshot: {self.snapshot_time} UTC | Period: last {self.period_hours}h",
            f"Sites monitored: {len(self.sites)}",
            "",
            "[ Portfolio Totals ]",
            f"  Total consumption: {self.total_kwh:,.1f} kWh",
            f"  Total CO₂:         {self.total_co2_kg:,.1f} kg",
            f"  Total anomalies:   {self.total_anomalies}",
            f"  Highest consumer:  {self.highest_consumer}",
            f"  Lowest consumer:   {self.lowest_consumer}",
            "",
            "[ Per-Site Summary ]",
        ]
        for s in sorted(self.sites, key=lambda x: x.total_kwh, reverse=True):
            anom = f"  ⚠ {s.anomaly_count} anomalies" if s.anomaly_count else ""
            lines.append(
                f"  {s.site_id:<35} {s.total_kwh:>10,.1f} kWh"
                f"  {s.pct_change_vs_prev_period:>+6.1f}%{anom}"
            )
        lines.append("")
        return "\n".join(lines)


class ContextBuilder:
    """
    Builds LLM-ready context from VoltEdge's data layer.

    Args:
        store:             Storage backend (LocalStore / S3Store).
        carbon_intensity:  Default kg CO₂/kWh if ESGCalculator unavailable.
        tariff_per_kwh:    Default electricity cost for cost estimates.
        currency:          Cost currency string.
    """

    def __init__(
        self,
        store:              Any,
        carbon_intensity:   float = 0.207,
        tariff_per_kwh:     float = 0.28,
        currency:           str   = "GBP",
    ) -> None:
        self._store   = store
        self._ci      = carbon_intensity
        self._tariff  = tariff_per_kwh
        self._currency = currency

    # ── Public API ────────────────────────────────────────────────────────────

    def build_site(self, site_id: str, hours: int = 24) -> SiteSnapshot:
        """Build a context snapshot for one site."""
        df = self._store.read(site_id, layer="processed", days=max(1, hours // 24 + 1))

        snap = SiteSnapshot(
            site_id=site_id,
            snapshot_time=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            period_hours=hours,
        )

        if df.empty or "kwh" not in df.columns:
            log.warning("context.no_data", site=site_id)
            return snap

        # Filter to requested window
        df = self._filter_hours(df, hours)
        if df.empty:
            return snap

        kwh = df["kwh"]
        snap.total_kwh        = round(float(kwh.sum()), 2)
        snap.avg_kwh_per_hour = round(float(kwh.mean()), 2)
        snap.peak_kwh         = round(float(kwh.max()), 2)
        snap.min_kwh          = round(float(kwh.min()), 2)

        # Peak timestamp
        if "timestamp" in df.columns:
            peak_idx = kwh.idxmax()
            peak_ts  = pd.to_datetime(df.loc[peak_idx, "timestamp"])
            snap.peak_at = peak_ts.strftime("%Y-%m-%d %H:%M")

        # Trend vs prior period
        prior_df = self._filter_hours(
            self._store.read(site_id, layer="processed", days=max(1, hours // 24 * 2 + 1)),
            hours * 2
        )
        if not prior_df.empty:
            prior_period = self._filter_hours_offset(prior_df, hours, hours)
            if not prior_period.empty:
                prior_total = float(prior_period["kwh"].sum())
                if prior_total > 0:
                    snap.pct_change_vs_prev_period = round(
                        (snap.total_kwh - prior_total) / prior_total * 100, 1
                    )
        snap.trend_direction = (
            "up"     if snap.pct_change_vs_prev_period > 5  else
            "down"   if snap.pct_change_vs_prev_period < -5 else
            "stable"
        )

        # ESG
        snap.co2_kg           = round(snap.total_kwh * self._ci, 2)
        snap.carbon_intensity = self._ci
        snap.scope2_tco2e     = round(snap.co2_kg / 1000, 6)

        # Cost estimate
        snap.estimated_cost = round(snap.total_kwh * self._tariff, 2)
        snap.cost_currency  = self._currency

        # Anomalies from processed columns
        snap.anomaly_count, snap.anomaly_details = self._extract_anomalies(df)

        # Forecast
        snap.forecast_24h_kwh, snap.forecast_available = self._quick_forecast(df)

        snap.raw_stats = {
            "rows": len(df),
            "hours_coverage": hours,
            "kwh_std": round(float(kwh.std()), 2),
        }

        return snap

    def build_portfolio(self, site_ids: list[str], hours: int = 24) -> PortfolioSnapshot:
        """Build a multi-site aggregated snapshot."""
        snapshots = [self.build_site(sid, hours) for sid in site_ids]
        total_kwh  = sum(s.total_kwh for s in snapshots)
        total_co2  = sum(s.co2_kg for s in snapshots)
        total_anom = sum(s.anomaly_count for s in snapshots)

        by_kwh = sorted(snapshots, key=lambda s: s.total_kwh)
        return PortfolioSnapshot(
            sites=snapshots,
            snapshot_time=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            period_hours=hours,
            total_kwh=round(total_kwh, 2),
            total_co2_kg=round(total_co2, 2),
            total_anomalies=total_anom,
            highest_consumer=by_kwh[-1].site_id if by_kwh else "",
            lowest_consumer=by_kwh[0].site_id  if by_kwh else "",
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _filter_hours(df: pd.DataFrame, hours: int) -> pd.DataFrame:
        if "timestamp" not in df.columns or df.empty:
            return df
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)
        return df[df["timestamp"] >= cutoff].reset_index(drop=True)

    @staticmethod
    def _filter_hours_offset(df: pd.DataFrame, end_hours: int, start_hours: int) -> pd.DataFrame:
        if "timestamp" not in df.columns or df.empty:
            return df
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        now = pd.Timestamp.now(tz="UTC")
        return df[
            (df["timestamp"] >= now - pd.Timedelta(hours=start_hours)) &
            (df["timestamp"] <  now - pd.Timedelta(hours=end_hours))
        ].reset_index(drop=True)

    @staticmethod
    def _extract_anomalies(df: pd.DataFrame) -> tuple[int, list[dict]]:
        if "anomaly_score" not in df.columns:
            return 0, []
        flagged = df[df.get("is_anomaly", df["anomaly_score"] > 0.75) == True]
        details = []
        for _, row in flagged.head(10).iterrows():
            details.append({
                "timestamp": str(row.get("timestamp", ""))[:19],
                "score":     round(float(row.get("anomaly_score", 0)), 3),
                "label":     str(row.get("anomaly_label", "anomaly")),
                "kwh":       round(float(row.get("kwh", 0)), 2),
            })
        return len(flagged), details

    @staticmethod
    def _quick_forecast(df: pd.DataFrame) -> tuple[float, bool]:
        """Naive 24h forecast: average of same hour from last 7 days × 24."""
        if "kwh" not in df.columns or len(df) < 48:
            return 0.0, False
        avg = float(df["kwh"].tail(24).mean())
        return round(avg * 24, 1), True
