"""
VoltEdge — AzureBlobStore

Drop-in replacement for LocalStore backed by Azure Blob Storage.
Identical BaseStore interface; same Hive partition layout as S3Store.

Free-tier notes:
  - Azure Free Account: 5 GB LRS Blob storage for 12 months
  - Use Azurite emulator locally: azurite --skipApiVersionCheck

Usage:
    from voltedge.storage.azure_store import AzureBlobStore

    # Connection string (from Azure portal)
    store = AzureBlobStore(
        connection_string="DefaultEndpointsProtocol=https;AccountName=...;...",
        container="voltedge-data",
    )
    # OR — Azurite dev emulator:
    store = AzureBlobStore(
        connection_string="UseDevelopmentStorage=true",
        container="voltedge-data",
    )
"""

from __future__ import annotations

import io
import uuid
from datetime import timedelta
from typing import TYPE_CHECKING

import pandas as pd

from voltedge.storage.base import BaseStore, _partition_path
from voltedge.utils.logger import get_logger

log = get_logger(__name__)

# Azure SDK is an optional dependency — guard import so the rest of VoltEdge
# still works when the azure package isn't installed (e.g. on AWS deployments).
try:
    from azure.storage.blob import (
        BlobServiceClient,
        ContainerClient,
        generate_blob_sas,
        BlobSasPermissions,
    )
    from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
    _AZURE_AVAILABLE = True
except ImportError:
    _AZURE_AVAILABLE = False


def _require_azure() -> None:
    if not _AZURE_AVAILABLE:
        raise ImportError(
            "azure-storage-blob is not installed. "
            "Run: pip install azure-storage-blob"
        )


class AzureBlobStore(BaseStore):
    """
    Azure Blob Storage-backed Parquet store.

    Args:
        connection_string: Azure storage connection string or
                           "UseDevelopmentStorage=true" for Azurite.
        container:         Blob container name (created if absent).
        prefix:            Optional blob name prefix (e.g. "voltedge/").
    """

    def __init__(
        self,
        connection_string: str,
        container: str = "voltedge-data",
        prefix: str = "",
    ) -> None:
        _require_azure()
        self.container_name = container
        self.prefix = prefix.rstrip("/") + "/" if prefix else ""

        self._service: BlobServiceClient = BlobServiceClient.from_connection_string(
            connection_string
        )
        self._container: ContainerClient = self._service.get_container_client(container)
        self._ensure_container()

    # ── BaseStore interface ───────────────────────────────────────────────────

    def write(self, df: pd.DataFrame, site_id: str, layer: str = "raw") -> str:
        if df.empty:
            log.warning("azure_store.write_skipped", reason="empty_dataframe", site=site_id)
            return ""

        partition = _partition_path(site_id)
        filename = f"{uuid.uuid4().hex[:8]}.parquet"
        blob_name = f"{self.prefix}{layer}/{partition}/{filename}"

        buf = io.BytesIO()
        df.to_parquet(buf, index=False, compression="snappy", engine="pyarrow")
        buf.seek(0)

        self._container.upload_blob(name=blob_name, data=buf, overwrite=True)

        uri = f"azure://{self.container_name}/{blob_name}"
        log.info("azure_store.written", uri=uri, rows=len(df), layer=layer, site=site_id)
        return uri

    def read(self, site_id: str, layer: str = "raw", days: int = 7) -> pd.DataFrame:
        prefix = f"{self.prefix}{layer}/site_id={site_id}/"
        blobs = list(self._container.list_blobs(name_starts_with=prefix))

        if not blobs:
            log.warning("azure_store.no_data", site=site_id, layer=layer)
            return pd.DataFrame()

        frames: list[pd.DataFrame] = []
        for blob in blobs:
            if not blob.name.endswith(".parquet"):
                continue
            try:
                data = self._container.download_blob(blob.name).readall()
                frames.append(pd.read_parquet(io.BytesIO(data), engine="pyarrow"))
            except Exception as exc:
                log.warning("azure_store.read_error", blob=blob.name, error=str(exc))

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
            df = df[df["timestamp"] >= cutoff]
            df = df.sort_values("timestamp").reset_index(drop=True)

        log.info("azure_store.read", site=site_id, rows=len(df), days=days)
        return df

    def list_sites(self) -> list[str]:
        prefix = f"{self.prefix}raw/"
        blobs = self._container.list_blobs(name_starts_with=prefix)
        sites: set[str] = set()
        for blob in blobs:
            # Extract site_id from: raw/site_id=X/year=.../...
            parts = blob.name[len(prefix):].split("/")
            if parts and parts[0].startswith("site_id="):
                sites.add(parts[0].replace("site_id=", ""))
        return sorted(sites)

    # ── Azure-specific helpers ────────────────────────────────────────────────

    def delete_site_data(self, site_id: str, layer: str = "raw") -> int:
        prefix = f"{self.prefix}{layer}/site_id={site_id}/"
        blobs = list(self._container.list_blobs(name_starts_with=prefix))
        count = 0
        for blob in blobs:
            self._container.delete_blob(blob.name)
            count += 1
        log.info("azure_store.deleted", site=site_id, layer=layer, count=count)
        return count

    def storage_summary(self) -> dict:
        blobs = list(self._container.list_blobs(name_starts_with=self.prefix))
        total_bytes = sum(b.size or 0 for b in blobs)
        return {
            "container": self.container_name,
            "total_objects": len(blobs),
            "total_mb": round(total_bytes / 1_048_576, 3),
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _ensure_container(self) -> None:
        try:
            self._container.create_container()
            log.info("azure_store.container_created", container=self.container_name)
        except ResourceExistsError:
            pass
