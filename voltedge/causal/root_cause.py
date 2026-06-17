"""
VoltEdge Causal AI — Root-Cause Analyser

Given an anomaly event and a causal graph, traces backwards through
the DAG to identify which upstream variables most likely *caused*
the anomaly — not just which ones were correlated with it.

Algorithm:
  1. Find the anomaly in the time-series (timestamp + variable)
  2. Walk the causal graph backwards from the anomalous variable
  3. Score each ancestor by: causal strength × deviation magnitude
     at the time of the anomaly
  4. Rank causes by this combined score
  5. Return a structured explanation with the causal chain

Output (natural language + structured):
  "LONDON-FACTORY-01 consumption spiked 42% at 14:00 on 15 May.
   Root cause: ambient temperature rose to 31°C (+8°C above baseline),
   driving HVAC load increase [strength: 0.67, p<0.001].
   Contributing factor: shift change at 14:00 added production load
   [strength: 0.34, p<0.01]."
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from voltedge.causal.graph import CausalEdge, CausalGraph
from voltedge.utils.logger import get_logger

log = get_logger(__name__)

# How many standard deviations above normal qualifies as "significant deviation"
DEVIATION_THRESHOLD = 1.5


@dataclass
class CausalFactor:
    """One contributing causal factor in a root-cause explanation."""
    variable:      str
    causal_strength: float      # Beta coefficient from the graph
    deviation_z:   float        # Standard deviations from normal at anomaly time
    contribution_score: float   # strength × |deviation|
    direction:     str          # "increase" or "decrease"
    value_at_anomaly: float | None = None
    baseline_mean:    float | None = None
    mechanism:        str      = ""
    lag_hours:        int      = 0
    depth:            int      = 1   # 1 = direct cause, 2 = cause of cause, etc.

    def to_dict(self) -> dict[str, Any]:
        return {
            "variable":         self.variable,
            "causal_strength":  round(self.causal_strength, 4),
            "deviation_z":      round(self.deviation_z, 4),
            "contribution_score": round(self.contribution_score, 4),
            "direction":        self.direction,
            "value_at_anomaly": self.value_at_anomaly,
            "baseline_mean":    self.baseline_mean,
            "mechanism":        self.mechanism,
            "lag_hours":        self.lag_hours,
            "depth":            self.depth,
        }


@dataclass
class RootCauseReport:
    """
    Full root-cause analysis for one anomaly event.

    Provides:
      - Structured causal factors (ranked by contribution score)
      - Natural language narrative
      - Confidence rating
      - Recommended actions
    """
    site_id:           str
    anomaly_timestamp: str
    anomaly_variable:  str     = "kwh"
    anomaly_value:     float   = 0.0
    baseline_value:    float   = 0.0
    deviation_pct:     float   = 0.0
    anomaly_label:     str     = ""

    causal_factors:    list[CausalFactor] = field(default_factory=list)
    narrative:         str   = ""
    confidence:        str   = "medium"   # "high" | "medium" | "low"
    recommendations:   list[str] = field(default_factory=list)

    @property
    def primary_cause(self) -> CausalFactor | None:
        return self.causal_factors[0] if self.causal_factors else None

    @property
    def top_3_causes(self) -> list[CausalFactor]:
        return self.causal_factors[:3]

    def to_dict(self) -> dict[str, Any]:
        return {
            "site_id":           self.site_id,
            "anomaly_timestamp": self.anomaly_timestamp,
            "anomaly_variable":  self.anomaly_variable,
            "anomaly_value":     round(self.anomaly_value, 3),
            "baseline_value":    round(self.baseline_value, 3),
            "deviation_pct":     round(self.deviation_pct, 1),
            "anomaly_label":     self.anomaly_label,
            "causal_factors":    [f.to_dict() for f in self.causal_factors],
            "narrative":         self.narrative,
            "confidence":        self.confidence,
            "recommendations":   self.recommendations,
            "primary_cause":     self.primary_cause.to_dict() if self.primary_cause else None,
        }


class RootCauseAnalyser:
    """
    Traces anomalies back to their causal origins using a CausalGraph.

    Args:
        graph:          Learned causal DAG for the site.
        max_depth:      How many causal hops back to trace (default 3).
        min_contribution: Minimum contribution score to include a factor.
    """

    def __init__(
        self,
        graph:            CausalGraph,
        max_depth:        int   = 3,
        min_contribution: float = 0.05,
    ) -> None:
        self._graph    = graph
        self._max_depth = max_depth
        self._min_contr = min_contribution

    # ── Public API ────────────────────────────────────────────────────────────

    def analyse(
        self,
        df:                pd.DataFrame,
        anomaly_timestamp: str | datetime,
        anomaly_variable:  str = "kwh",
        anomaly_label:     str = "",
        window_hours:      int = 168,
    ) -> RootCauseReport:
        """
        Analyse the root cause of an anomaly.

        Args:
            df:                Processed DataFrame containing the anomaly period.
            anomaly_timestamp: When the anomaly occurred.
            anomaly_variable:  Which variable was anomalous (default: kwh).
            anomaly_label:     Label from AnomalyDetector (optional).
            window_hours:      Baseline window in hours for computing normal.

        Returns:
            RootCauseReport with ranked causal factors and narrative.
        """
        ts = pd.to_datetime(anomaly_timestamp, utc=True)
        df  = df.copy()

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values("timestamp").reset_index(drop=True)

        # Extract the anomaly row
        anom_mask = df["timestamp"] == ts
        if not anom_mask.any():
            # Find closest timestamp
            df["_tdelta"] = (df["timestamp"] - ts).abs()
            anom_idx = df["_tdelta"].idxmin()
            df.drop(columns="_tdelta", inplace=True)
        else:
            anom_idx = df[anom_mask].index[0]

        anom_row = df.iloc[anom_idx]

        # Compute baseline statistics (prior window_hours)
        baseline_df = df.iloc[max(0, anom_idx - window_hours): anom_idx]
        baselines   = baseline_df.mean(numeric_only=True)
        std_devs    = baseline_df.std(numeric_only=True).replace(0, 1e-8)

        anom_value    = float(anom_row.get(anomaly_variable, 0))
        baseline_val  = float(baselines.get(anomaly_variable, anom_value))
        deviation_pct = (
            (anom_value - baseline_val) / (abs(baseline_val) + 1e-8) * 100
        )

        log.info("root_cause.analyse_start",
                 site=self._graph.site_id,
                 timestamp=str(ts)[:16],
                 variable=anomaly_variable,
                 deviation_pct=round(deviation_pct, 1))

        # Walk causal graph backwards from anomaly_variable
        factors = self._trace_causes(
            anomaly_variable, anom_row, baselines, std_devs, depth=1
        )

        # Filter and rank
        factors = [f for f in factors if f.contribution_score >= self._min_contr]
        factors.sort(key=lambda f: f.contribution_score, reverse=True)

        # Generate narrative and recommendations
        narrative = self._build_narrative(
            self._graph.site_id, ts, anomaly_variable,
            anom_value, baseline_val, deviation_pct, factors, anomaly_label,
        )
        recommendations = self._build_recommendations(factors, anomaly_label)
        confidence = self._assess_confidence(factors, len(df))

        report = RootCauseReport(
            site_id=self._graph.site_id,
            anomaly_timestamp=str(ts)[:19],
            anomaly_variable=anomaly_variable,
            anomaly_value=anom_value,
            baseline_value=baseline_val,
            deviation_pct=round(deviation_pct, 1),
            anomaly_label=anomaly_label,
            causal_factors=factors,
            narrative=narrative,
            confidence=confidence,
            recommendations=recommendations,
        )

        log.info("root_cause.analyse_complete",
                 site=self._graph.site_id,
                 factors=len(factors),
                 primary=factors[0].variable if factors else "none",
                 confidence=confidence)
        return report

    def analyse_period(
        self,
        df:               pd.DataFrame,
        start:            str | datetime,
        end:              str | datetime,
        anomaly_variable: str = "kwh",
        top_n:            int = 5,
    ) -> list[RootCauseReport]:
        """
        Analyse all anomalies in a time period. Returns reports for each.
        """
        df = df.copy()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        start_ts = pd.to_datetime(start, utc=True)
        end_ts   = pd.to_datetime(end,   utc=True)
        window   = df[
            (df["timestamp"] >= start_ts) &
            (df["timestamp"] <= end_ts)
        ]

        if "anomaly_score" in df.columns:
            flagged = window[window["anomaly_score"] > 0.75]
        else:
            # Flag rows > 2 std from mean
            mu  = df[anomaly_variable].mean()
            sig = df[anomaly_variable].std() + 1e-8
            flagged = window[abs(window[anomaly_variable] - mu) / sig > 2.0]

        reports = []
        for _, row in flagged.head(top_n).iterrows():
            try:
                report = self.analyse(
                    df, row["timestamp"], anomaly_variable,
                    anomaly_label=str(row.get("anomaly_label", "")),
                )
                reports.append(report)
            except Exception as exc:
                log.warning("root_cause.period_error", error=str(exc))

        return reports

    # ── Internal ──────────────────────────────────────────────────────────────

    def _trace_causes(
        self,
        variable:  str,
        anom_row:  pd.Series,
        baselines: pd.Series,
        std_devs:  pd.Series,
        depth:     int,
    ) -> list[CausalFactor]:
        """Recursively walk the causal graph, collecting factors."""
        if depth > self._max_depth:
            return []

        factors: list[CausalFactor] = []
        causes = self._graph.get_causes(variable)

        for edge in causes:
            cause_var = edge.cause

            # Compute deviation at anomaly time
            if cause_var not in anom_row.index:
                continue

            value_at = float(anom_row[cause_var])
            baseline = float(baselines.get(cause_var, value_at))
            std      = float(std_devs.get(cause_var, 1.0))
            z_score  = (value_at - baseline) / std

            if abs(z_score) < 0.5 and depth > 1:
                continue   # Prune low-deviation ancestors

            contribution = abs(edge.strength) * abs(z_score)

            factors.append(CausalFactor(
                variable=cause_var,
                causal_strength=edge.strength,
                deviation_z=round(z_score, 3),
                contribution_score=round(contribution, 4),
                direction="increase" if z_score > 0 else "decrease",
                value_at_anomaly=round(value_at, 3),
                baseline_mean=round(baseline, 3),
                mechanism=edge.mechanism,
                lag_hours=edge.lag_hours,
                depth=depth,
            ))

            # Recurse
            if abs(z_score) >= DEVIATION_THRESHOLD:
                deeper = self._trace_causes(
                    cause_var, anom_row, baselines, std_devs, depth + 1
                )
                factors.extend(deeper)

        return factors

    @staticmethod
    def _build_narrative(
        site_id:       str,
        timestamp:     "pd.Timestamp",
        variable:      str,
        anom_value:    float,
        baseline:      float,
        deviation_pct: float,
        factors:       list[CausalFactor],
        label:         str,
    ) -> str:
        direction = "above" if deviation_pct > 0 else "below"
        ts_str    = timestamp.strftime("%H:%M on %d %b %Y")

        lines = [
            f"{site_id}: {variable} was {abs(deviation_pct):.1f}% {direction} "
            f"baseline at {ts_str} ({anom_value:.1f} vs {baseline:.1f} normal)."
        ]

        if label and label not in ("normal", "anomaly"):
            lines.append(f"Anomaly classification: {label.replace('_', ' ')}.")

        if not factors:
            # Build an interpretive message using what we do know about the anomaly
            direction_word = "below" if deviation_pct < 0 else "above"
            pct_abs = abs(deviation_pct)

            if label and "drop" in label:
                context = (
                    "consistent with planned downtime, a weekend schedule, "
                    "or a maintenance window"
                )
                action = "Check the shift plan and confirm equipment was intentionally offline."
            elif label and "spike" in label:
                context = (
                    "consistent with an unplanned equipment start-up, "
                    "a fault condition, or an unmeasured load coming online"
                )
                action = "Investigate whether unscheduled equipment start-up or fault condition occurred at this timestamp."
            else:
                context = (
                    "consistent with planned downtime, shift schedule changes, "
                    "or sensor coverage gaps for this window"
                )
                action = "Log the event and monitor for recurrence over the next 7 days."

            lines.append(
                f"No dominant causal pattern identified in the causal graph — "
                f"{pct_abs:.1f}% {direction_word} baseline is {context}. "
                f"{action}"
            )
        else:
            primary = factors[0]
            dev_desc = (
                f"{abs(primary.deviation_z):.1f} standard deviations "
                f"{'above' if primary.deviation_z > 0 else 'below'} baseline"
            )
            lines.append(
                f"Primary cause: {primary.variable.replace('_', ' ')} "
                f"was {dev_desc} at the time of the anomaly "
                f"({primary.mechanism}). "
                f"Causal strength: {abs(primary.causal_strength):.3f}."
            )

            if len(factors) > 1:
                secondary = factors[1]
                lines.append(
                    f"Contributing factor: {secondary.variable.replace('_', ' ')} "
                    f"deviation of {abs(secondary.deviation_z):.1f}σ "
                    f"({secondary.mechanism})."
                )

        return " ".join(lines)

    @staticmethod
    def _build_recommendations(
        factors: list[CausalFactor],
        label:   str,
    ) -> list[str]:
        recs: list[str] = []

        for f in factors[:3]:
            var = f.variable
            if "temperature" in var:
                recs.append(
                    "Review HVAC set-points and pre-cooling schedule; "
                    "consider demand response pre-cooling during off-peak tariff window."
                )
            elif "power_factor" in var:
                recs.append(
                    "Schedule power factor correction capacitor inspection; "
                    "low PF increases reactive power draw and grid charges."
                )
            elif "voltage" in var:
                recs.append(
                    "Log voltage event with DNO (Distribution Network Operator); "
                    "persistent voltage deviation may indicate network fault."
                )
            elif "business_hour" in var or "shift" in var:
                recs.append(
                    "Review shift-start load sequencing; "
                    "stagger equipment start-up to reduce peak demand charge."
                )

        if "flat_line" in label:
            recs.append(
                "Sensor reporting flat-line readings — check meter connectivity "
                "and communication link before treating as a genuine load event."
            )
        if "sudden_spike" in label:
            recs.append(
                "Investigate whether unscheduled equipment start-up or fault "
                "condition occurred at this timestamp."
            )

        if not recs:
            recs.append(
                "No specific remediation identified. "
                "Log the event and monitor for recurrence over the next 7 days."
            )

        return recs[:4]

    @staticmethod
    def _assess_confidence(factors: list[CausalFactor], n_samples: int) -> str:
        if n_samples < 72 or not factors:
            return "low"
        top_score = factors[0].contribution_score if factors else 0
        if top_score > 0.5 and factors[0].causal_strength != 0:
            return "high"
        if top_score > 0.2:
            return "medium"
        return "low"
