"""
VoltEdge — Anomaly Detection Model

Isolation Forest for unsupervised anomaly detection in energy consumption.
Detects: sudden spikes, flat-line sensor failures, power factor degradation.

Usage:
    from voltedge.models.anomaly import AnomalyDetector
    model = AnomalyDetector(site_id="SITE-01")
    metrics = model.train(df)
    results_df = model.predict(df)    # Adds 'anomaly_score' and 'is_anomaly' columns
    model.save()
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from voltedge.utils.logger import get_logger

log = get_logger(__name__)

# Features most diagnostic of anomalous energy behaviour
ANOMALY_FEATURES: list[str] = [
    "kwh",
    "kwh_delta",
    "kwh_pct_change",
    "kwh_roll_std_24h",
    "kwh_roll_mean_24h",
    "hour_sin", "hour_cos",
    "is_weekend",
    "is_business_hour",
]

OPTIONAL_FEATURES: list[str] = [
    "power_factor",
    "voltage_v",
    "apparent_power_kva",
    "pf_poor",
]


class AnomalyDetector:
    """
    Isolation Forest anomaly detector for energy consumption streams.

    Scores each reading with an anomaly_score (-1 = anomaly, 1 = normal in sklearn).
    We remap to a 0–1 probability-like score for dashboard display.

    Args:
        site_id:         Site identifier for artifact namespacing.
        contamination:   Expected fraction of anomalies in training data (default 5%).
        artifact_dir:    Where to persist model artifacts.
    """

    model_name = "anomaly_detector"
    model_version = "1.0.0"

    def __init__(
        self,
        site_id: str,
        contamination: float = 0.05,
        artifact_dir: Path | str = "data/models",
    ) -> None:
        self.site_id = site_id
        self.contamination = contamination
        self._artifact_dir = Path(artifact_dir) / self.model_name / site_id
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        self._model: IsolationForest | None = None
        self._scaler: RobustScaler | None = None
        self._feature_cols: list[str] = []

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame) -> dict[str, float]:
        """
        Fit the anomaly detector on historical data.
        Returns a metrics dict (precision proxy via known injected anomalies if available).
        """
        df = self._prepare(df)
        self._feature_cols = [
            c for c in ANOMALY_FEATURES + OPTIONAL_FEATURES if c in df.columns
        ]

        if not self._feature_cols:
            raise ValueError("No usable feature columns. Run EnergyTransformer first.")

        df_clean = df.dropna(subset=self._feature_cols)
        X = df_clean[self._feature_cols].values

        log.info(
            "anomaly.training_start",
            site=self.site_id,
            rows=len(df_clean),
            features=len(self._feature_cols),
            contamination=self.contamination,
        )

        # RobustScaler is resistant to outliers themselves during fit
        self._scaler = RobustScaler()
        X_scaled = self._scaler.fit_transform(X)

        self._model = IsolationForest(
            n_estimators=200,
            contamination=self.contamination,
            max_features=min(len(self._feature_cols), 8),
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
        )
        self._model.fit(X_scaled)

        # Evaluate against injected anomaly labels if available
        metrics: dict[str, float] = {
            "training_rows": float(len(df_clean)),
            "features": float(len(self._feature_cols)),
            "contamination": self.contamination,
        }

        if "metadata" in df_clean.columns:
            # Check for injected_anomaly flags from simulator
            try:
                has_label = df_clean["metadata"].apply(
                    lambda m: m.get("injected_anomaly", False) if isinstance(m, dict) else False
                )
                if has_label.any():
                    scores = self._model.decision_function(X_scaled)
                    # Anomalies have negative scores; convert to binary predictions
                    predicted = (scores < np.percentile(scores, self.contamination * 100)).astype(int)
                    actual = has_label.astype(int).values
                    tp = int(((predicted == 1) & (actual == 1)).sum())
                    fp = int(((predicted == 1) & (actual == 0)).sum())
                    fn = int(((predicted == 0) & (actual == 1)).sum())
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    metrics.update({
                        "precision": round(precision, 4),
                        "recall": round(recall, 4),
                        "f1": round(2 * precision * recall / (precision + recall + 1e-8), 4),
                        "true_positives": float(tp),
                        "false_positives": float(fp),
                    })
            except Exception:
                pass  # Labels not available — skip supervised metrics

        log.info("anomaly.training_complete", site=self.site_id, **{
            k: v for k, v in metrics.items() if isinstance(v, float)
        })
        return metrics

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score each reading. Adds columns:
            anomaly_score  (0.0 = normal → 1.0 = highly anomalous)
            is_anomaly     (bool — True if exceeds threshold)
            anomaly_label  (human-readable reason string)
        """
        if self._model is None:
            raise RuntimeError("Call train() or load() first.")

        df = self._prepare(df).copy()
        available = [c for c in self._feature_cols if c in df.columns]
        df_feat = df.dropna(subset=available)
        X = df_feat[available].values

        # Pad to full feature width
        if available != self._feature_cols:
            X_full = np.zeros((len(X), len(self._feature_cols)))
            for i, col in enumerate(self._feature_cols):
                if col in available:
                    X_full[:, i] = X[:, available.index(col)]
            X = X_full

        X_scaled = self._scaler.transform(X) if self._scaler else X

        # decision_function: negative = anomaly; normalize to 0–1
        raw_scores = self._model.decision_function(X_scaled)
        min_s, max_s = raw_scores.min(), raw_scores.max()
        normalized = 1 - (raw_scores - min_s) / (max_s - min_s + 1e-8)

        sklearn_labels = self._model.predict(X_scaled)  # 1=normal, -1=anomaly

        df_feat = df_feat.copy()
        df_feat["anomaly_score"] = np.round(normalized, 4)
        df_feat["is_anomaly"] = sklearn_labels == -1
        df_feat["anomaly_label"] = df_feat.apply(
            self._classify_anomaly, axis=1
        )

        return df_feat

    # ── Anomaly classification ────────────────────────────────────────────────

    def _classify_anomaly(self, row: pd.Series) -> str:
        """Provide a human-readable label for flagged anomalies."""
        if not row.get("is_anomaly", False):
            return "normal"

        labels: list[str] = []

        if "kwh_pct_change" in row and abs(row["kwh_pct_change"]) > 1.5:
            labels.append("sudden_spike" if row["kwh_pct_change"] > 0 else "sudden_drop")

        if "kwh_roll_std_24h" in row and row.get("kwh_roll_std_24h", 1) < 0.1:
            labels.append("flat_line_sensor")

        if "power_factor" in row and pd.notna(row["power_factor"]) and row["power_factor"] < 0.80:
            labels.append("poor_power_factor")

        if "kwh" in row and "kwh_roll_mean_24h" in row:
            ratio = row["kwh"] / (row["kwh_roll_mean_24h"] + 1e-8)
            if ratio > 3.0:
                labels.append("extreme_consumption")
            elif ratio < 0.1 and row.get("is_business_hour", 0) == 1:
                labels.append("unexpected_low_load")

        return ", ".join(labels) if labels else "anomaly"

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, metrics: dict[str, float] | None = None) -> None:
        if not self._model:
            raise RuntimeError("No model to save.")
        with open(self._artifact_dir / "model.pkl", "wb") as f:
            pickle.dump(self._model, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self._artifact_dir / "scaler.pkl", "wb") as f:
            pickle.dump(self._scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
        meta = {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "site_id": self.site_id,
            "contamination": self.contamination,
            "feature_columns": self._feature_cols,
            "metrics": metrics or {},
        }
        (self._artifact_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        log.info("anomaly.saved", site=self.site_id)

    def load(self) -> None:
        with open(self._artifact_dir / "model.pkl", "rb") as f:
            self._model = pickle.load(f)
        with open(self._artifact_dir / "scaler.pkl", "rb") as f:
            self._scaler = pickle.load(f)
        meta = json.loads((self._artifact_dir / "metadata.json").read_text())
        self._feature_cols = meta.get("feature_columns", ANOMALY_FEATURES)
        self.contamination = meta.get("contamination", self.contamination)
        log.info("anomaly.loaded", site=self.site_id)

    @staticmethod
    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values("timestamp").reset_index(drop=True)
        return df
