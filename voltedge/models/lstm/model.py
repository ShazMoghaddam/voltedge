"""
VoltEdge — LSTM Sequence Forecaster

Bidirectional LSTM that ingests a sliding window of hourly readings
and forecasts demand up to 168 hours ahead. Complements the GBM
DemandForecaster: use GBM for fast, explainable forecasts; use LSTM
when you need to capture long-range temporal dependencies (seasonal
patterns beyond 168-lag features).

Architecture:
    Input  →  [Bi-LSTM(64)]  →  Dropout(0.2)  →  [LSTM(32)]  →
    Dense(16, relu)  →  Dense(horizon_hours)

Training strategy:
    - Sliding window dataset construction
    - EarlyStopping on val_loss (patience=10)
    - ReduceLROnPlateau on val_loss (factor=0.5, patience=5)
    - 80/20 chronological train/val split (no shuffling — time series)

Usage:
    from voltedge.models.lstm import LSTMForecaster

    model = LSTMForecaster(site_id="LONDON-FACTORY-01", lookback=48, horizon=24)
    metrics = model.train(df)
    forecast_df = model.predict(df)
    model.save()

    # Later:
    model2 = LSTMForecaster(site_id="LONDON-FACTORY-01")
    model2.load()
    forecast_df = model2.predict(df)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # silence TF boot noise

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks as keras_callbacks
from sklearn.preprocessing import MinMaxScaler

from voltedge.utils.logger import get_logger

log = get_logger(__name__)

# Features used for sequence input (must be present after EnergyTransformer)
SEQUENCE_FEATURES: list[str] = [
    "kwh",
    "hour_sin", "hour_cos",
    "dow_sin",  "dow_cos",
    "is_weekend", "is_business_hour",
    "kwh_roll_mean_24h",
    "kwh_roll_std_24h",
]


class LSTMForecaster:
    """
    Bidirectional LSTM energy demand forecaster.

    Args:
        site_id:       Site identifier (used for artifact namespacing).
        lookback:      Number of past hours fed into the sequence (default 48).
        horizon:       Hours ahead to predict (default 24).
        artifact_dir:  Directory for model artifacts.
        epochs:        Maximum training epochs (early stopping may end sooner).
        batch_size:    Training batch size.
    """

    model_name    = "lstm_forecaster"
    model_version = "1.0.0"

    def __init__(
        self,
        site_id: str,
        lookback:     int  = 48,
        horizon:      int  = 24,
        artifact_dir: str | Path = "data/models",
        epochs:       int  = 100,
        batch_size:   int  = 32,
    ) -> None:
        self.site_id      = site_id
        self.lookback     = lookback
        self.horizon      = horizon
        self.epochs       = epochs
        self.batch_size   = batch_size

        self._artifact_dir = Path(artifact_dir) / self.model_name / site_id
        self._artifact_dir.mkdir(parents=True, exist_ok=True)

        self._model:   keras.Model   | None = None
        self._scaler:  MinMaxScaler  | None = None
        self._features: list[str]           = []

    # ── Public API ────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame) -> dict[str, float]:
        """
        Build sliding-window dataset, train Bi-LSTM, return metrics.

        Returns dict with: mae, rmse, mape, val_mae, val_loss, epochs_trained.
        """
        df = self._prepare(df)
        self._features = [f for f in SEQUENCE_FEATURES if f in df.columns]

        if not self._features:
            raise ValueError("No sequence features found. Run EnergyTransformer first.")
        if len(df) < self.lookback + self.horizon + 10:
            raise ValueError(
                f"Need ≥{self.lookback + self.horizon + 10} rows, got {len(df)}."
            )

        log.info("lstm.train_start", site=self.site_id,
                 rows=len(df), lookback=self.lookback, horizon=self.horizon,
                 features=len(self._features))

        # Scale features to [0, 1]
        self._scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = self._scaler.fit_transform(df[self._features].values)

        # Build sliding windows
        X, y = self._make_windows(scaled)

        # Chronological 80/20 split
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        log.info("lstm.dataset", train=len(X_train), val=len(X_val))

        # Build model
        self._model = self._build_model(
            n_features=len(self._features), horizon=self.horizon
        )

        # Callbacks
        cb = [
            keras_callbacks.EarlyStopping(
                monitor="val_loss", patience=10,
                restore_best_weights=True, verbose=0,
            ),
            keras_callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5,
                min_lr=1e-5, verbose=0,
            ),
        ]

        history = self._model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=cb,
            verbose=0,
            shuffle=False,   # Preserve temporal order within batches
        )

        epochs_trained = len(history.history["loss"])

        # Evaluate on validation set
        y_hat_val = self._model.predict(X_val, verbose=0)
        # Inverse-transform kwh predictions only (index 0 = kwh in SEQUENCE_FEATURES)
        y_true_kwh = self._inverse_kwh(y_val)
        y_pred_kwh = self._inverse_kwh(y_hat_val)

        mae  = float(np.mean(np.abs(y_true_kwh - y_pred_kwh)))
        rmse = float(np.sqrt(np.mean((y_true_kwh - y_pred_kwh) ** 2)))
        mape = float(np.mean(np.abs(
            (y_true_kwh - y_pred_kwh) / (np.abs(y_true_kwh) + 1e-8)
        )) * 100)

        metrics = {
            "mae":           round(mae, 4),
            "rmse":          round(rmse, 4),
            "mape":          round(mape, 4),
            "val_loss":      round(float(history.history["val_loss"][-1]), 6),
            "val_mae":       round(float(history.history.get("val_mae", [0])[-1]), 4),
            "epochs_trained": int(epochs_trained),
            "training_rows": int(len(df)),
            "lookback":      self.lookback,
            "horizon":       self.horizon,
        }
        log.info("lstm.train_complete", site=self.site_id, **metrics)
        return metrics

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a multi-step forecast from the most recent `lookback` rows.

        Returns DataFrame: timestamp, kwh_forecast, lower_bound, upper_bound, horizon_h.
        """
        if self._model is None:
            raise RuntimeError("Call train() or load() first.")

        df = self._prepare(df)
        available = [f for f in self._features if f in df.columns]

        if len(df) < self.lookback:
            raise ValueError(
                f"Need ≥{self.lookback} rows for prediction, got {len(df)}."
            )

        # Take last `lookback` rows
        window = df[available].tail(self.lookback).values
        window_scaled = self._scaler.transform(window)
        X = window_scaled[np.newaxis, :, :]   # (1, lookback, n_features)

        # Monte Carlo dropout for uncertainty — run N forward passes
        predictions = []
        for _ in range(30):
            pred = self._model(X, training=True).numpy()   # training=True keeps dropout active
            predictions.append(np.atleast_1d(self._inverse_kwh(pred)))

        preds = np.array(predictions)   # (30, horizon)
        mean_forecast = preds.mean(axis=0)
        std_forecast  = preds.std(axis=0)

        last_ts = pd.to_datetime(df["timestamp"].iloc[-1], utc=True)

        return pd.DataFrame([
            {
                "timestamp":    last_ts + pd.Timedelta(hours=int(h + 1)),
                "kwh_forecast": max(0.0, round(float(mean_forecast[h]), 3)),
                "lower_bound":  max(0.0, round(float(mean_forecast[h] - 1.96 * std_forecast[h]), 3)),
                "upper_bound":  max(0.0, round(float(mean_forecast[h] + 1.96 * std_forecast[h]), 3)),
                "horizon_h":    int(h + 1),
            }
            for h in range(self.horizon)
        ])

    def compare_with_gbm(
        self, df: pd.DataFrame, gbm_forecast: pd.DataFrame
    ) -> dict[str, Any]:
        """
        Compare LSTM forecast against a GBM forecast on the same horizon.
        Returns a dict with side-by-side metrics and ensemble average.
        """
        lstm_forecast = self.predict(df)

        shared_h = min(len(lstm_forecast), len(gbm_forecast))
        lstm_kwh = lstm_forecast["kwh_forecast"].values[:shared_h]
        gbm_kwh  = gbm_forecast["kwh_forecast"].values[:shared_h]
        ensemble = (lstm_kwh + gbm_kwh) / 2

        return {
            "horizon_hours": shared_h,
            "lstm_mean_forecast": round(float(lstm_kwh.mean()), 3),
            "gbm_mean_forecast":  round(float(gbm_kwh.mean()),  3),
            "ensemble_mean":      round(float(ensemble.mean()), 3),
            "lstm_ci_width_mean": round(float(
                (lstm_forecast["upper_bound"] - lstm_forecast["lower_bound"]).mean()
            ), 3),
            "gbm_ci_width_mean": round(float(
                (gbm_forecast["upper_bound"] - gbm_forecast["lower_bound"]).mean()
            ), 3),
            "ensemble_forecast": ensemble.tolist(),
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, metrics: dict[str, float] | None = None) -> None:
        if self._model is None:
            raise RuntimeError("No model to save.")

        keras_path = self._artifact_dir / "model.keras"
        self._model.save(str(keras_path))

        import pickle
        with open(self._artifact_dir / "scaler.pkl", "wb") as f:
            import pickle as pkl
            pkl.dump(self._scaler, f, protocol=pkl.HIGHEST_PROTOCOL)

        meta: dict[str, Any] = {
            "model_name":    self.model_name,
            "model_version": self.model_version,
            "site_id":       self.site_id,
            "lookback":      self.lookback,
            "horizon":       self.horizon,
            "features":      self._features,
            "metrics":       metrics or {},
        }
        (self._artifact_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        log.info("lstm.saved", site=self.site_id, path=str(self._artifact_dir))

    def load(self) -> None:
        keras_path = self._artifact_dir / "model.keras"
        self._model = keras.models.load_model(str(keras_path))

        import pickle
        with open(self._artifact_dir / "scaler.pkl", "rb") as f:
            self._scaler = pickle.load(f)

        meta = json.loads((self._artifact_dir / "metadata.json").read_text())
        self.lookback   = meta["lookback"]
        self.horizon    = meta["horizon"]
        self._features  = meta["features"]
        log.info("lstm.loaded", site=self.site_id)

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_model(n_features: int, horizon: int) -> keras.Model:
        """Bidirectional LSTM with MC Dropout for uncertainty quantification."""
        model = keras.Sequential([
            keras.layers.Input(shape=(None, n_features)),
            keras.layers.Bidirectional(
                keras.layers.LSTM(64, return_sequences=True)
            ),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(32, return_sequences=False),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(horizon),
        ], name="voltedge_lstm")

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="huber",   # Robust to outliers vs MSE
            metrics=["mae"],
        )
        return model

    def _make_windows(
        self, scaled: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Construct (X, y) sliding window arrays from scaled data."""
        X, y = [], []
        kwh_idx = self._features.index("kwh") if "kwh" in self._features else 0

        for i in range(len(scaled) - self.lookback - self.horizon):
            X.append(scaled[i : i + self.lookback])
            # Target: kwh for the next `horizon` steps
            future = scaled[i + self.lookback : i + self.lookback + self.horizon, kwh_idx]
            y.append(future)

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def _inverse_kwh(self, scaled_kwh: np.ndarray) -> np.ndarray:
        """
        Inverse-transform kwh-only predictions back to original scale.
        Pads zeros for other features so MinMaxScaler can invert correctly.
        """
        kwh_idx   = self._features.index("kwh") if "kwh" in self._features else 0
        n_feat    = len(self._features)
        n_samples = scaled_kwh.shape[0]

        # Handle 1D or 2D input
        flat = scaled_kwh.reshape(n_samples, -1)
        out  = np.zeros((n_samples * flat.shape[1], n_feat), dtype=np.float32)
        out[:, kwh_idx] = flat.reshape(-1)

        inv = self._scaler.inverse_transform(out)
        return inv[:, kwh_idx].reshape(n_samples, -1).squeeze()

    @staticmethod
    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values("timestamp").reset_index(drop=True)
        return df.dropna(subset=["kwh"])
