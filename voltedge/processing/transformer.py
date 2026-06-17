"""
VoltEdge — Data transformation and feature engineering.

Takes validated raw DataFrames from ingestion and produces
ML-ready feature sets with temporal, statistical, and derived features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from voltedge.utils.logger import get_logger

log = get_logger(__name__)


class EnergyTransformer:
    """
    Stateless transformer: raw sensor DataFrame → feature-engineered DataFrame.

    All methods are pure functions that return new DataFrames; the original
    is never mutated. Call `transform()` to run the full pipeline, or invoke
    individual stages for custom pipelines.
    """

    # ── Public API ────────────────────────────────────────────────────────────

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full transformation pipeline end-to-end."""
        if df.empty:
            log.warning("transformer.empty_input")
            return df

        log.info("transformer.start", rows=len(df))

        df = self._parse_timestamps(df)
        df = self._clean(df)
        df = self._add_temporal_features(df)
        df = self._add_rolling_features(df)
        df = self._add_lag_features(df)
        df = self._add_derived_features(df)

        log.info("transformer.done", rows=len(df), cols=len(df.columns))
        return df

    # ── Pipeline stages ───────────────────────────────────────────────────────

    def _parse_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove / interpolate invalid readings:
        - Negative kWh values → NaN
        - Extreme outliers (>4 IQR from median) → NaN
        - Interpolate gaps up to 3 consecutive NaN hours
        """
        df = df.copy()

        # Negative energy is physically impossible
        df.loc[df["kwh"] < 0, "kwh"] = np.nan

        # IQR-based outlier removal
        q1, q3 = df["kwh"].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 4 * iqr, q3 + 4 * iqr
        outlier_mask = (df["kwh"] < lower) | (df["kwh"] > upper)
        outlier_count = outlier_mask.sum()
        if outlier_count:
            log.debug("transformer.outliers_removed", count=int(outlier_count))
        df.loc[outlier_mask, "kwh"] = np.nan

        # Time-aware interpolation — requires DatetimeIndex; restore RangeIndex after
        df = df.set_index("timestamp")
        df["kwh"] = df["kwh"].interpolate(method="time", limit=3)
        df = df.reset_index()

        # Forward-fill remaining NaN in auxiliary columns
        for col in ["voltage_v", "current_a", "power_factor", "temperature_c"]:
            if col in df.columns:
                df[col] = df[col].ffill().bfill()

        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        ts = df["timestamp"]

        df["hour"] = ts.dt.hour
        df["day_of_week"] = ts.dt.dayofweek          # 0=Mon, 6=Sun
        df["day_of_year"] = ts.dt.dayofyear
        df["month"] = ts.dt.month
        df["week_of_year"] = ts.dt.isocalendar().week.astype(int)
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_business_hour"] = (
            (df["hour"].between(8, 18)) & (df["is_weekend"] == 0)
        ).astype(int)

        # Cyclic encoding — prevents the model treating 23:00 → 00:00 as a gap
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        return df

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        kwh = df["kwh"]

        for window in (3, 6, 12, 24):
            df[f"kwh_roll_mean_{window}h"] = kwh.rolling(window, min_periods=1).mean()
            df[f"kwh_roll_std_{window}h"] = kwh.rolling(window, min_periods=1).std()

        df["kwh_roll_max_24h"] = kwh.rolling(24, min_periods=1).max()
        df["kwh_roll_min_24h"] = kwh.rolling(24, min_periods=1).min()

        return df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for lag in (1, 2, 3, 6, 12, 24, 48, 168):  # hours back; 168 = 1 week
            df[f"kwh_lag_{lag}h"] = df["kwh"].shift(lag)
        return df

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Apparent power (kVA) — relevant for demand charges
        if "voltage_v" in df.columns and "current_a" in df.columns:
            df["apparent_power_kva"] = (df["voltage_v"] * df["current_a"]) / 1000

        # Power factor quality flag
        if "power_factor" in df.columns:
            df["pf_poor"] = (df["power_factor"] < 0.85).astype(int)

        # Delta from previous reading — useful for anomaly detection
        df["kwh_delta"] = df["kwh"].diff()
        df["kwh_pct_change"] = df["kwh"].pct_change().replace([np.inf, -np.inf], np.nan)

        return df
