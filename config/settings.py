"""
VoltEdge — Central Configuration
Settings loaded from environment variables / .env file.
Never hard-code secrets. Never commit .env.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class CloudProvider(str, Enum):
    AWS = "aws"
    AZURE = "azure"
    LOCAL = "local"


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class VoltEdgeSettings(BaseSettings):
    """Single source of truth for all runtime config."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────────
    app_name: str = "VoltEdge"
    app_version: str = "1.0.0"
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False

    # ── Cloud ────────────────────────────────────────────────────────────────
    cloud_provider: CloudProvider = CloudProvider.LOCAL
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_region: str = "eu-west-1"
    aws_s3_bucket: str = "voltedge-data"
    azure_storage_connection_string: str | None = None
    azure_container_name: str = "voltedge-data"

    # ── Data Ingestion ───────────────────────────────────────────────────────
    ingestion_interval_seconds: int = 60
    max_sites: int = 50
    api_timeout_seconds: int = 30

    # ── ML Models ───────────────────────────────────────────────────────────
    forecast_horizon_hours: int = 24
    anomaly_contamination: float = 0.05
    model_retrain_hours: int = 168

    # ── Dashboard ────────────────────────────────────────────────────────────
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8050
    dashboard_debug: bool = True

    # ── ESG ──────────────────────────────────────────────────────────────────
    carbon_intensity_kg_per_kwh: float = 0.207
    esg_report_currency: str = "USD"

    # ── Logging ──────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_format: str = "console"   # "console" (dev) | "json" (prod)

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return v.upper()

    # ── Computed paths (properties, not fields — avoids Pydantic None clash) ─
    @property
    def base_dir(self) -> Path:
        return Path(__file__).resolve().parent.parent

    @property
    def data_dir(self) -> Path:
        p = self.base_dir / "data"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def raw_data_dir(self) -> Path:
        p = self.data_dir / "raw"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def processed_data_dir(self) -> Path:
        p = self.data_dir / "processed"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def model_artifacts_dir(self) -> Path:
        p = self.data_dir / "models"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION

    @property
    def is_development(self) -> bool:
        return self.environment == Environment.DEVELOPMENT

    @property
    def use_cloud(self) -> bool:
        return self.cloud_provider != CloudProvider.LOCAL


@lru_cache(maxsize=1)
def get_settings() -> VoltEdgeSettings:
    """Cached settings singleton. Call get_settings.cache_clear() in tests."""
    return VoltEdgeSettings()


settings = get_settings()
