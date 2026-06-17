"""
Tests for S3Store — uses moto to mock AWS S3 without real credentials or network.
"""

from __future__ import annotations

import asyncio
import pytest
import pandas as pd
from moto import mock_aws

from voltedge.ingestion.simulators import SimulatedSiteConnector
from voltedge.processing.transformer import EnergyTransformer
from voltedge.storage.s3_store import S3Store
from voltedge.storage.base import LocalStore


BUCKET = "test-voltedge-bucket"
REGION = "us-east-1"


def _make_store() -> S3Store:
    """Create an S3Store pointing at the moto mock."""
    return S3Store(bucket=BUCKET, prefix="voltedge", region=REGION, auto_create=True)


def _make_df(site_id: str = "S3-TEST-01", hours: int = 48) -> pd.DataFrame:
    connector = SimulatedSiteConnector(site_id, "factory", hours=hours, seed=1)
    result = asyncio.run(connector.fetch())
    return EnergyTransformer().transform(result.data)


# ── Write tests ───────────────────────────────────────────────────────────────

@mock_aws
def test_s3_write_returns_uri():
    store = _make_store()
    df = _make_df()
    uri = store.write(df, site_id="S3-TEST-01", layer="raw")
    assert uri.startswith("s3://")
    assert "S3-TEST-01" in uri
    assert uri.endswith(".parquet")


@mock_aws
def test_s3_write_empty_df_returns_empty_string():
    store = _make_store()
    result = store.write(pd.DataFrame(), site_id="EMPTY-SITE", layer="raw")
    assert result == ""


@mock_aws
def test_s3_write_both_layers():
    store = _make_store()
    df = _make_df()
    raw_uri = store.write(df, site_id="DUAL-SITE", layer="raw")
    proc_uri = store.write(df, site_id="DUAL-SITE", layer="processed")
    assert "raw" in raw_uri
    assert "processed" in proc_uri


# ── Read tests ────────────────────────────────────────────────────────────────

@mock_aws
def test_s3_read_returns_dataframe():
    store = _make_store()
    df = _make_df()
    store.write(df, site_id="READ-SITE", layer="processed")
    result = store.read("READ-SITE", layer="processed", days=30)
    assert not result.empty
    assert "kwh" in result.columns


@mock_aws
def test_s3_read_row_count_matches_write():
    store = _make_store()
    df = _make_df(hours=72)
    store.write(df, site_id="COUNT-SITE", layer="processed")
    result = store.read("COUNT-SITE", layer="processed", days=30)
    assert len(result) == len(df)


@mock_aws
def test_s3_read_missing_site_returns_empty():
    store = _make_store()
    result = store.read("NONEXISTENT-SITE", layer="raw", days=7)
    assert result.empty


@mock_aws
def test_s3_read_timestamp_sorted():
    store = _make_store()
    df = _make_df(hours=100)
    store.write(df, site_id="SORT-SITE", layer="processed")
    result = store.read("SORT-SITE", layer="processed", days=30)
    ts = pd.to_datetime(result["timestamp"])
    assert (ts.diff().dropna() >= pd.Timedelta(0)).all()


@mock_aws
def test_s3_read_multiple_writes_merged():
    """Two separate writes to same site should be read back as one combined DataFrame."""
    store = _make_store()
    df1 = _make_df("MERGE-SITE", hours=24)
    df2 = _make_df("MERGE-SITE", hours=24)
    store.write(df1, site_id="MERGE-SITE", layer="raw")
    store.write(df2, site_id="MERGE-SITE", layer="raw")
    result = store.read("MERGE-SITE", layer="raw", days=30)
    assert len(result) == len(df1) + len(df2)


# ── list_sites tests ──────────────────────────────────────────────────────────

@mock_aws
def test_s3_list_sites_empty():
    store = _make_store()
    assert store.list_sites() == []


@mock_aws
def test_s3_list_sites_returns_written_sites():
    store = _make_store()
    for site in ["SITE-A", "SITE-B", "SITE-C"]:
        df = _make_df(site, hours=12)
        store.write(df, site_id=site, layer="raw")
    sites = store.list_sites()
    assert set(sites) == {"SITE-A", "SITE-B", "SITE-C"}


@mock_aws
def test_s3_list_sites_sorted():
    store = _make_store()
    for site in ["ZSITE", "ASITE", "MSITE"]:
        store.write(_make_df(site, hours=12), site_id=site, layer="raw")
    assert store.list_sites() == sorted(["ZSITE", "ASITE", "MSITE"])


# ── API compatibility tests ───────────────────────────────────────────────────

@mock_aws
def test_s3_store_is_base_store_subclass():
    from voltedge.storage.base import BaseStore
    store = _make_store()
    assert isinstance(store, BaseStore)


@mock_aws
def test_s3_store_interface_matches_local_store():
    """S3Store and LocalStore must expose identical public methods."""
    s3_methods = {m for m in dir(S3Store) if not m.startswith("_")}
    local_methods = {m for m in dir(LocalStore) if not m.startswith("_")}
    core = {"write", "read", "list_sites"}
    assert core.issubset(s3_methods)
    assert core.issubset(local_methods)


@mock_aws
def test_s3_bucket_size_bytes():
    store = _make_store()
    df = _make_df(hours=24)
    store.write(df, site_id="SIZE-SITE", layer="raw")
    size = store.bucket_size_bytes()
    assert size > 0


# ── Pipeline integration ──────────────────────────────────────────────────────

@mock_aws
def test_pipeline_with_s3_store():
    """Full pipeline run using S3Store instead of LocalStore."""
    import asyncio
    from voltedge.core.pipeline import EnergyPipeline

    store = _make_store()
    pipeline = EnergyPipeline(store=store)
    pipeline.register_connector(
        "PIPELINE-S3", SimulatedSiteConnector("PIPELINE-S3", "office", hours=48, seed=5)
    )
    summary = asyncio.run(pipeline.run_all())
    assert summary.success
    assert summary.total_records > 0

    # Verify data was actually written to S3
    sites = store.list_sites()
    assert "PIPELINE-S3" in sites
