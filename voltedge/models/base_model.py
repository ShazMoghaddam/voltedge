"""
VoltEdge — Base ML model interface.

All VoltEdge models (forecasting, anomaly detection, optimisation) inherit
from BaseModel. This enforces a consistent train/predict/save/load lifecycle
and integrates with the model registry for versioning.
"""

from __future__ import annotations

import abc
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from config.settings import settings
from voltedge.utils.logger import get_logger

log = get_logger(__name__)


class ModelMetadata:
    """Lightweight metadata persisted alongside every model artifact."""

    def __init__(
        self,
        model_name: str,
        model_version: str,
        site_id: str,
        trained_at: datetime | None = None,
        metrics: dict[str, float] | None = None,
        feature_columns: list[str] | None = None,
    ) -> None:
        self.model_name = model_name
        self.model_version = model_version
        self.site_id = site_id
        self.trained_at = trained_at or datetime.now(timezone.utc)
        self.metrics = metrics or {}
        self.feature_columns = feature_columns or []

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "site_id": self.site_id,
            "trained_at": self.trained_at.isoformat(),
            "metrics": self.metrics,
            "feature_columns": self.feature_columns,
        }

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))


class BaseVoltEdgeModel(abc.ABC):
    """
    Abstract base for all VoltEdge ML models.

    Lifecycle:
        model = MyModel(site_id="SITE-01")
        model.train(df_train)
        predictions = model.predict(df_features)
        model.save()
        # Later:
        model.load()
    """

    model_name: str = "base"
    model_version: str = "0.1.0"

    def __init__(self, site_id: str) -> None:
        self.site_id = site_id
        self._model: Any = None
        self._metadata: ModelMetadata | None = None
        self._artifact_dir = settings.ml.artifact_path / self.model_name / site_id
        self._artifact_dir.mkdir(parents=True, exist_ok=True)

    # ── Required interface ────────────────────────────────────────────────────

    @abc.abstractmethod
    def train(self, df: pd.DataFrame) -> dict[str, float]:
        """
        Fit the model on historical data.
        Returns a dict of evaluation metrics (e.g. {"mae": 12.3, "rmse": 18.1}).
        """
        ...

    @abc.abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions from feature data.
        Returns a DataFrame with a `prediction` column (plus confidence bounds
        where applicable).
        """
        ...

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, metrics: dict[str, float] | None = None) -> Path:
        """Pickle the fitted model + write metadata JSON."""
        if self._model is None:
            raise RuntimeError("Cannot save: model has not been trained.")

        artifact_path = self._artifact_dir / "model.pkl"
        with open(artifact_path, "wb") as fh:
            pickle.dump(self._model, fh, protocol=pickle.HIGHEST_PROTOCOL)

        meta = ModelMetadata(
            model_name=self.model_name,
            model_version=self.model_version,
            site_id=self.site_id,
            metrics=metrics or {},
        )
        meta.save(self._artifact_dir / "metadata.json")
        self._metadata = meta

        log.info(
            "model.saved",
            model=self.model_name,
            site=self.site_id,
            path=str(artifact_path),
        )
        return artifact_path

    def load(self) -> None:
        """Load a previously saved model artifact."""
        artifact_path = self._artifact_dir / "model.pkl"
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"No saved model for {self.model_name}/{self.site_id} "
                f"at {artifact_path}"
            )

        with open(artifact_path, "rb") as fh:
            self._model = pickle.load(fh)

        meta_path = self._artifact_dir / "metadata.json"
        if meta_path.exists():
            data = json.loads(meta_path.read_text())
            self._metadata = ModelMetadata(**{
                k: v for k, v in data.items()
                if k != "trained_at"
            })
            self._metadata.trained_at = datetime.fromisoformat(data["trained_at"])

        log.info("model.loaded", model=self.model_name, site=self.site_id)

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    @property
    def metadata(self) -> ModelMetadata | None:
        return self._metadata
