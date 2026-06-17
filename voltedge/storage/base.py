"""
VoltEdge — Storage abstraction layer.

All persistence operations go through the BaseStore interface.
Swap LocalStore for S3Store or AzureBlobStore without changing
any upstream code.

Storage layout (Hive-style partitioning for cloud compatibility):
    {prefix}/site_id={site_id}/year={yyyy}/month={mm}/day={dd}/{uuid}.parquet
"""

from __future__ import annotations

import abc
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from voltedge.utils.logger import get_logger

log = get_logger(__name__)


def _partition_path(site_id: str, ts: datetime | None = None) -> str:
    ts = ts or datetime.now(timezone.utc)
    return (
        f"site_id={site_id}/"
        f"year={ts.year:04d}/"
        f"month={ts.month:02d}/"
        f"day={ts.day:02d}"
    )


# ── Abstract interface ─────────────────────────────────────────────────────────

class BaseStore(abc.ABC):

    @abc.abstractmethod
    def write(self, df: pd.DataFrame, site_id: str, layer: str = "raw") -> str:
        """Persist a DataFrame. Returns the URI/path written to."""
        ...

    @abc.abstractmethod
    def read(self, site_id: str, layer: str = "raw", days: int = 7) -> pd.DataFrame:
        """Read recent data for a site, merging across partitions."""
        ...

    @abc.abstractmethod
    def list_sites(self) -> list[str]:
        """Return all site IDs that have data in this store."""
        ...


# ── Local filesystem implementation ───────────────────────────────────────────

class LocalStore(BaseStore):
    """
    Stores data as Parquet files partitioned by site/date.
    Suitable for development and AWS-free-tier demos.
    Swap for S3Store in production with zero code changes upstream.
    """

    def __init__(self, base_path: Path | str = "data/processed") -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def write(self, df: pd.DataFrame, site_id: str, layer: str = "raw") -> str:
        if df.empty:
            log.warning("store.write_skipped", reason="empty_dataframe", site=site_id)
            return ""

        partition = _partition_path(site_id)
        dest_dir = self.base_path / layer / partition
        dest_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{uuid.uuid4().hex[:8]}.parquet"
        dest = dest_dir / filename
        df.to_parquet(dest, index=False, compression="snappy")

        log.info("store.written", path=str(dest), rows=len(df), layer=layer, site=site_id)
        return str(dest)

    def read(self, site_id: str, layer: str = "raw", days: int = 7) -> pd.DataFrame:
        site_dir = self.base_path / layer / f"site_id={site_id}"
        if not site_dir.exists():
            log.warning("store.no_data", site=site_id, layer=layer)
            return pd.DataFrame()

        files = sorted(site_dir.rglob("*.parquet"))
        if not files:
            return pd.DataFrame()

        # Load and concatenate — simple for free-tier volumes
        frames = [pd.read_parquet(f) for f in files]
        df = pd.concat(frames, ignore_index=True)

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
            df = df[df["timestamp"] >= cutoff]
            df = df.sort_values("timestamp").reset_index(drop=True)

        log.info("store.read", site=site_id, rows=len(df), days=days)
        return df

    def list_sites(self) -> list[str]:
        raw_dir = self.base_path / "raw"
        if not raw_dir.exists():
            return []
        return [
            p.name.replace("site_id=", "")
            for p in raw_dir.iterdir()
            if p.is_dir() and p.name.startswith("site_id=")
        ]
