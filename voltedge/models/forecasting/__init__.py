"""
VoltEdge — Demand Forecasting Model

Gradient Boosting regressor predicting energy consumption N hours ahead.
Uses temporal cyclic features + lag features from EnergyTransformer.

Usage:
    from voltedge.models.forecasting import DemandForecaster
    model = DemandForecaster(site_id="SITE-01", horizon_hours=24)
    metrics = model.train(df)
    forecast_df = model.predict(df.tail(48))
    model.save(metrics)
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from voltedge.utils.logger import get_logger

log = get_logger(__name__)

FEATURE_COLUMNS: list[str] = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_weekend", "is_business_hour",
    "kwh_lag_1h", "kwh_lag_2h", "kwh_lag_3h", "kwh_lag_6h",
    "kwh_lag_12h", "kwh_lag_24h", "kwh_lag_48h", "kwh_lag_168h",
    "kwh_roll_mean_3h", "kwh_roll_mean_6h", "kwh_roll_mean_12h",
    "kwh_roll_mean_24h", "kwh_roll_std_24h",
    "kwh_roll_max_24h", "kwh_roll_min_24h",
]

TARGET_COLUMN = "kwh"


class DemandForecaster:
    """
    Hourly energy demand forecaster via Gradient Boosting Regression.

    Lifecycle: instantiate → train(df) → predict(df) → save() / load()
    """

    model_name = "demand_forecaster"
    model_version = "1.0.0"

    def __init__(
        self,
        site_id: str,
        horizon_hours: int = 24,
        artifact_dir: Path | str = "data/models",
    ) -> None:
        self.site_id = site_id
        self.horizon_hours = horizon_hours
        self._artifact_dir = Path(artifact_dir) / self.model_name / site_id
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        self._model: GradientBoostingRegressor | None = None
        self._scaler: StandardScaler | None = None
        self._feature_cols: list[str] = []

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame) -> dict[str, float]:
        """Fit the model. Returns evaluation metrics dict."""
        df = self._prepare(df)
        self._feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

        if not self._feature_cols:
            raise ValueError("No feature columns found. Run EnergyTransformer first.")

        df["target"] = df[TARGET_COLUMN].shift(-self.horizon_hours)
        df = df.dropna(subset=["target"] + self._feature_cols)

        X = df[self._feature_cols].values
        y = df["target"].values

        log.info("forecaster.training_start", site=self.site_id, rows=len(df))

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_maes: list[float] = []
        for _, (tr_idx, val_idx) in enumerate(tscv.split(X_scaled)):
            tmp = GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=5,
                min_samples_leaf=10, subsample=0.8, random_state=42,
            )
            tmp.fit(X_scaled[tr_idx], y[tr_idx])
            cv_maes.append(mean_absolute_error(y[val_idx], tmp.predict(X_scaled[val_idx])))

        # Final model on full dataset
        self._model = GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            min_samples_leaf=10, subsample=0.8, random_state=42,
        )
        self._model.fit(X_scaled, y)

        y_hat = self._model.predict(X_scaled)
        metrics = {
            "mae": round(mean_absolute_error(y, y_hat), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y, y_hat))), 4),
            "r2": round(r2_score(y, y_hat), 4),
            "mape": round(float(np.mean(np.abs((y - y_hat) / (np.abs(y) + 1e-8))) * 100), 4),
            "cv_mae_mean": round(float(np.mean(cv_maes)), 4),
            "cv_mae_std": round(float(np.std(cv_maes)), 4),
            "training_rows": int(len(df)),
        }
        log.info("forecaster.training_complete", site=self.site_id, **metrics)
        return metrics

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a DataFrame with columns:
        timestamp, kwh_forecast, lower_bound, upper_bound, horizon_h
        """
        if self._model is None:
            raise RuntimeError("Call train() or load() first.")

        df = self._prepare(df)
        available = [c for c in self._feature_cols if c in df.columns]
        X_raw = df[available].tail(1).values

        X_full = np.zeros((1, len(self._feature_cols)))
        for i, col in enumerate(self._feature_cols):
            if col in available:
                X_full[0, i] = X_raw[0, available.index(col)]

        X = self._scaler.transform(X_full) if self._scaler else X_full
        base = float(self._model.predict(X)[0])
        std = base * 0.08  # ~8% uncertainty band

        last_ts = pd.to_datetime(df["timestamp"].iloc[-1], utc=True)
        return pd.DataFrame([
            {
                "timestamp": last_ts + pd.Timedelta(hours=h),
                "kwh_forecast": max(0.0, round(base, 3)),
                "lower_bound": max(0.0, round(base - 1.96 * std, 3)),
                "upper_bound": max(0.0, round(base + 1.96 * std, 3)),
                "horizon_h": h,
            }
            for h in range(1, self.horizon_hours + 1)
        ])

    # ── Feature importance ────────────────────────────────────────────────────

    def feature_importance(self) -> pd.DataFrame:
        if not self._model:
            raise RuntimeError("Model not trained.")
        return (
            pd.DataFrame({"feature": self._feature_cols,
                          "importance": self._model.feature_importances_})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, metrics: dict[str, float] | None = None) -> None:
        if not self._model:
            raise RuntimeError("No model to save.")
        with open(self._artifact_dir / "model.pkl", "wb") as f:
            pickle.dump(self._model, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self._artifact_dir / "scaler.pkl", "wb") as f:
            pickle.dump(self._scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
        meta: dict[str, Any] = {
            "model_name": self.model_name, "model_version": self.model_version,
            "site_id": self.site_id, "horizon_hours": self.horizon_hours,
            "feature_columns": self._feature_cols, "metrics": metrics or {},
        }
        (self._artifact_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        log.info("forecaster.saved", site=self.site_id)

    def load(self) -> None:
        with open(self._artifact_dir / "model.pkl", "rb") as f:
            self._model = pickle.load(f)
        with open(self._artifact_dir / "scaler.pkl", "rb") as f:
            self._scaler = pickle.load(f)
        meta = json.loads((self._artifact_dir / "metadata.json").read_text())
        self._feature_cols = meta.get("feature_columns", FEATURE_COLUMNS)
        self.horizon_hours = meta.get("horizon_hours", self.horizon_hours)
        log.info("forecaster.loaded", site=self.site_id)

    @staticmethod
    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values("timestamp").reset_index(drop=True)
        return df.dropna(subset=[TARGET_COLUMN])
