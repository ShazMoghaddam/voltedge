"""Tests for AWS Marketplace metering — all dry_run, no AWS calls."""

from __future__ import annotations

from datetime import datetime, timezone
import pytest

from voltedge.oem.marketplace import (
    DIMENSIONS, MarketplaceMeteringService, MeteringResult, UsageRecord,
)

TS = datetime(2026, 5, 15, 12, 0, 0, tzinfo=timezone.utc)
CUST = "aws-cust-abc123"


# ── UsageRecord ───────────────────────────────────────────────────────────────

def test_usage_record_auto_transaction_id():
    r = UsageRecord(CUST, "active_sites", 5, TS)
    assert len(r.transaction_id) == 32


def test_usage_record_deterministic_id():
    r1 = UsageRecord(CUST, "active_sites", 5, TS)
    r2 = UsageRecord(CUST, "active_sites", 5, TS)
    assert r1.transaction_id == r2.transaction_id


def test_usage_record_different_hours_different_ids():
    ts2 = TS.replace(hour=13)
    r1 = UsageRecord(CUST, "active_sites", 5, TS)
    r2 = UsageRecord(CUST, "active_sites", 5, ts2)
    assert r1.transaction_id != r2.transaction_id


def test_usage_record_valid():
    r = UsageRecord(CUST, "active_sites", 5)
    assert r.is_valid


def test_usage_record_invalid_dimension():
    r = UsageRecord(CUST, "invalid_dim", 5)
    assert not r.is_valid


def test_usage_record_negative_quantity():
    r = UsageRecord(CUST, "active_sites", -1)
    assert not r.is_valid


def test_usage_record_zero_quantity_valid():
    r = UsageRecord(CUST, "active_sites", 0)
    assert r.is_valid


def test_all_dimensions_valid():
    for dim in DIMENSIONS:
        r = UsageRecord(CUST, dim, 1)
        assert r.is_valid, f"Dimension {dim} should be valid"


# ── resolve_customer (dry_run) ────────────────────────────────────────────────

def test_resolve_customer_returns_dict():
    svc = MarketplaceMeteringService(dry_run=True)
    result = svc.resolve_customer("fake-registration-token")
    assert "customer_id" in result
    assert "product_code" in result


def test_resolve_customer_id_is_deterministic():
    svc = MarketplaceMeteringService(dry_run=True)
    r1 = svc.resolve_customer("same-token")
    r2 = svc.resolve_customer("same-token")
    assert r1["customer_id"] == r2["customer_id"]


def test_resolve_customer_different_tokens_different_ids():
    svc = MarketplaceMeteringService(dry_run=True)
    r1 = svc.resolve_customer("token-A")
    r2 = svc.resolve_customer("token-B")
    assert r1["customer_id"] != r2["customer_id"]


# ── meter_usage (dry_run) ─────────────────────────────────────────────────────

def test_meter_usage_returns_success():
    svc = MarketplaceMeteringService(dry_run=True)
    record = UsageRecord(CUST, "active_sites", 5, TS)
    result = svc.meter_usage(record)
    assert result["Status"] == "Success"
    assert result["DryRun"] is True


def test_meter_usage_invalid_record_raises():
    svc = MarketplaceMeteringService(dry_run=True)
    bad = UsageRecord(CUST, "not_a_dimension", 5)
    with pytest.raises(ValueError, match="Invalid"):
        svc.meter_usage(bad)


# ── batch_meter_usage (dry_run) ───────────────────────────────────────────────

def test_batch_empty_list_succeeds():
    svc = MarketplaceMeteringService(dry_run=True)
    result = svc.batch_meter_usage([])
    assert result.success is True
    assert result.records_submitted == 0


def test_batch_all_valid():
    svc = MarketplaceMeteringService(dry_run=True)
    records = [UsageRecord(CUST, dim, 1, TS) for dim in DIMENSIONS]
    result = svc.batch_meter_usage(records)
    assert result.success is True
    assert result.records_submitted == len(DIMENSIONS)
    assert result.records_failed == 0


def test_batch_with_invalid_records():
    svc = MarketplaceMeteringService(dry_run=True)
    records = [
        UsageRecord(CUST, "active_sites", 5, TS),
        UsageRecord(CUST, "bad_dim", 5, TS),   # invalid
        UsageRecord(CUST, "api_calls", 3, TS),
    ]
    result = svc.batch_meter_usage(records)
    assert result.records_submitted == 2
    assert result.records_failed == 1


def test_batch_success_rate():
    svc = MarketplaceMeteringService(dry_run=True)
    records = [
        UsageRecord(CUST, "active_sites", 1, TS),
        UsageRecord(CUST, "active_sites", 2, TS),
    ]
    result = svc.batch_meter_usage(records)
    assert result.success_rate == 1.0


def test_batch_zero_success_rate_when_all_fail():
    svc = MarketplaceMeteringService(dry_run=True)
    records = [UsageRecord(CUST, "bad", 1, TS)]
    result = svc.batch_meter_usage(records)
    assert result.records_submitted == 0
    assert result.records_failed == 1


def test_batch_is_dry_run():
    svc = MarketplaceMeteringService(dry_run=True)
    result = svc.batch_meter_usage([UsageRecord(CUST, "active_sites", 1, TS)])
    assert result.dry_run is True


# ── build_hourly_records ──────────────────────────────────────────────────────

def test_build_hourly_records_returns_list():
    usage = [{"customer_id": CUST, "sites_count": 3, "api_calls_hour": 500}]
    records = MarketplaceMeteringService.build_hourly_records(usage, hour=TS)
    assert len(records) > 0


def test_build_records_sites_dimension():
    usage = [{"customer_id": CUST, "sites_count": 7}]
    records = MarketplaceMeteringService.build_hourly_records(usage, hour=TS)
    site_recs = [r for r in records if r.dimension == "active_sites"]
    assert len(site_recs) == 1
    assert site_recs[0].quantity == 7


def test_build_records_api_calls_rounded_up():
    usage = [{"customer_id": CUST, "api_calls_hour": 1001}]  # 1001 → 2 thousands
    records = MarketplaceMeteringService.build_hourly_records(usage, hour=TS)
    api_recs = [r for r in records if r.dimension == "api_calls"]
    assert api_recs[0].quantity == 2


def test_build_records_api_calls_minimum_1_thousand():
    usage = [{"customer_id": CUST, "api_calls_hour": 1}]  # 1 call → min 1 thousand
    records = MarketplaceMeteringService.build_hourly_records(usage, hour=TS)
    api_recs = [r for r in records if r.dimension == "api_calls"]
    assert api_recs[0].quantity == 1


def test_build_records_data_gb_rounded_up():
    usage = [{"customer_id": CUST, "data_gb_hour": 1.1}]  # 1.1 → 2 GB
    records = MarketplaceMeteringService.build_hourly_records(usage, hour=TS)
    data_recs = [r for r in records if r.dimension == "data_gb"]
    assert data_recs[0].quantity == 2


def test_build_records_zero_usage_omitted():
    usage = [{"customer_id": CUST, "sites_count": 0, "api_calls_hour": 0}]
    records = MarketplaceMeteringService.build_hourly_records(usage, hour=TS)
    assert len(records) == 0


def test_build_records_missing_customer_omitted():
    usage = [{"sites_count": 5}]  # no customer_id
    records = MarketplaceMeteringService.build_hourly_records(usage, hour=TS)
    assert len(records) == 0


def test_build_records_multi_tenant():
    usage = [
        {"customer_id": "C1", "sites_count": 3},
        {"customer_id": "C2", "sites_count": 7},
        {"customer_id": "C3", "sites_count": 1},
    ]
    records = MarketplaceMeteringService.build_hourly_records(usage, hour=TS)
    customer_ids = {r.customer_identifier for r in records}
    assert customer_ids == {"C1", "C2", "C3"}


def test_build_records_all_four_dimensions():
    usage = [{
        "customer_id": CUST,
        "sites_count": 5,
        "api_calls_hour": 2000,
        "data_gb_hour": 3.5,
        "ml_calls_hour": 1500,
    }]
    records = MarketplaceMeteringService.build_hourly_records(usage, hour=TS)
    dims = {r.dimension for r in records}
    assert dims == {"active_sites", "api_calls", "data_gb", "ml_predictions"}


def test_build_records_uses_current_hour_when_not_specified():
    usage = [{"customer_id": CUST, "sites_count": 1}]
    records = MarketplaceMeteringService.build_hourly_records(usage)
    assert records[0].timestamp.minute == 0
    assert records[0].timestamp.second == 0
