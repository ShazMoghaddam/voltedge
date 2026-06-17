"""
VoltEdge — AWS S3 Storage Backend

Drop-in replacement for LocalStore. Uses the same BaseStore interface
so all upstream code (pipeline, CLI, tests) works without modification.

Storage layout mirrors LocalStore (Hive-partitioned Parquet):
    s3://{bucket}/{prefix}/site_id={site_id}/year={Y}/month={M}/day={D}/{uuid}.parquet

Free-tier usage notes:
    - S3 Free Tier: 5 GB storage, 20k GET, 2k PUT per month
    - Parquet + Snappy compression keeps file sizes very small (~50kB per day/site)
    - Use LocalStore for development; S3Store for staging/production
"""

from __future__ import annotations

import io
import uuid
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from voltedge.storage.base import BaseStore, _partition_path
from voltedge.utils.logger import get_logger

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

log = get_logger(__name__)


class S3Store(BaseStore):
    """
    AWS S3-backed store using Parquet + Hive-style partitioning.

    Args:
        bucket:     S3 bucket name (must already exist or auto_create=True).
        prefix:     Key prefix inside the bucket (default: "voltedge").
        region:     AWS region (default: "eu-west-1").
        auto_create: Create the bucket if it doesn't exist (useful for LocalStack).
        endpoint_url: Override endpoint — set to "http://localhost:4566" for LocalStack.
        aws_access_key_id / aws_secret_access_key: Optional credential override.
    """

    def __init__(
        self,
        bucket: str = "voltedge-data",
        prefix: str = "voltedge",
        region: str = "eu-west-1",
        auto_create: bool = False,
        endpoint_url: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix
        self.region = region
        self._endpoint = endpoint_url

        session = boto3.Session(
            aws_access_key_id=aws_access_key_id or "test",
            aws_secret_access_key=aws_secret_access_key or "test",
            region_name=region,
        )
        self._s3: S3Client = session.client(
            "s3",
            endpoint_url=endpoint_url,
        )

        if auto_create:
            self._ensure_bucket()

    # ── BaseStore interface ────────────────────────────────────────────────────

    def write(self, df: pd.DataFrame, site_id: str, layer: str = "raw") -> str:
        """Serialize df to Parquet in memory and PUT to S3. Returns the S3 URI."""
        if df.empty:
            log.warning("s3_store.write_skipped", reason="empty_dataframe", site=site_id)
            return ""

        partition = _partition_path(site_id)
        filename = f"{uuid.uuid4().hex[:8]}.parquet"
        key = str(PurePosixPath(self.prefix) / layer / partition / filename)

        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False, compression="snappy", engine="pyarrow")
        buffer.seek(0)

        self._s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=buffer.getvalue(),
            ContentType="application/octet-stream",
        )

        uri = f"s3://{self.bucket}/{key}"
        log.info("s3_store.written", uri=uri, rows=len(df), layer=layer, site=site_id)
        return uri

    def read(self, site_id: str, layer: str = "raw", days: int = 7) -> pd.DataFrame:
        """
        List all Parquet objects under site_id prefix and read the recent `days`.
        Uses list_objects_v2 pagination for large buckets.
        """
        site_prefix = str(PurePosixPath(self.prefix) / layer / f"site_id={site_id}")

        keys = self._list_keys(site_prefix)
        if not keys:
            log.warning("s3_store.no_data", site=site_id, layer=layer)
            return pd.DataFrame()

        frames: list[pd.DataFrame] = []
        for key in keys:
            try:
                obj = self._s3.get_object(Bucket=self.bucket, Key=key)
                buf = io.BytesIO(obj["Body"].read())
                frames.append(pd.read_parquet(buf, engine="pyarrow"))
            except Exception as exc:
                log.warning("s3_store.read_error", key=key, error=str(exc))

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
            df = df[df["timestamp"] >= cutoff].sort_values("timestamp").reset_index(drop=True)

        log.info("s3_store.read", site=site_id, rows=len(df), keys=len(keys))
        return df

    def list_sites(self) -> list[str]:
        """Return all site IDs that have data in this bucket/prefix."""
        layer_prefix = str(PurePosixPath(self.prefix) / "raw") + "/"
        paginator = self._s3.get_paginator("list_objects_v2")
        site_ids: set[str] = set()

        for page in paginator.paginate(Bucket=self.bucket, Prefix=layer_prefix, Delimiter="/"):
            for cp in page.get("CommonPrefixes", []):
                part = cp["Prefix"].rstrip("/").split("/")[-1]
                if part.startswith("site_id="):
                    site_ids.add(part.replace("site_id=", ""))

        return sorted(site_ids)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _list_keys(self, prefix: str) -> list[str]:
        """Page through all objects under a prefix, return list of keys."""
        paginator = self._s3.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".parquet"):
                    keys.append(obj["Key"])
        return keys

    def _ensure_bucket(self) -> None:
        """Create the S3 bucket if it doesn't exist."""
        try:
            self._s3.head_bucket(Bucket=self.bucket)
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchBucket"):
                if self.region == "us-east-1":
                    self._s3.create_bucket(Bucket=self.bucket)
                else:
                    self._s3.create_bucket(
                        Bucket=self.bucket,
                        CreateBucketConfiguration={"LocationConstraint": self.region},
                    )
                log.info("s3_store.bucket_created", bucket=self.bucket, region=self.region)
            else:
                raise

    def bucket_size_bytes(self) -> int:
        """Estimate total bytes stored (useful for free-tier monitoring)."""
        paginator = self._s3.get_paginator("list_objects_v2")
        total = 0
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                total += obj.get("Size", 0)
        return total
