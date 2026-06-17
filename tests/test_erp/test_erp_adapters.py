"""Tests for SAP and Oracle ERP adapters."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from voltedge.erp.base import BaseERPAdapter, ERPFetchResult, ERPReading
from voltedge.erp.sap_adapter import SAPAdapter
from voltedge.erp.oracle_adapter import OracleAdapter


SITE = "ERP-TEST-SITE"


# ── ERPReading dataclass ──────────────────────────────────────────────────────

def test_erp_reading_defaults():
    r = ERPReading(
        site_id=SITE, meter_id="M1",
        timestamp=datetime.now(timezone.utc), kwh=100.0,
    )
    assert r.erp_source == "generic"
    assert r.currency == "USD"


# ── Base adapter ──────────────────────────────────────────────────────────────

def test_base_adapter_is_abstract():
    with pytest.raises(TypeError):
        BaseERPAdapter(site_id="X")  # type: ignore


def test_fetch_result_success_rate_all_valid():
    import pandas as pd
    r = ERPFetchResult(
        source="test", site_id="X",
        records_fetched=10, records_valid=10, records_invalid=0,
        data=pd.DataFrame(),
    )
    assert r.success_rate == 1.0


def test_fetch_result_success_rate_zero():
    import pandas as pd
    r = ERPFetchResult(
        source="test", site_id="X",
        records_fetched=0, records_valid=0, records_invalid=0,
        data=pd.DataFrame(),
    )
    assert r.success_rate == 0.0


# ── SAP Adapter — mock mode ───────────────────────────────────────────────────

@pytest.fixture
def sap_mock():
    return SAPAdapter(SITE, use_mock=True, days_back=2)


def test_sap_mock_fetch_returns_result(sap_mock):
    result = asyncio.run(sap_mock.fetch())
    assert isinstance(result, ERPFetchResult)
    assert result.source == "sap_is_u"
    assert result.site_id == SITE


def test_sap_mock_produces_48_records(sap_mock):
    result = asyncio.run(sap_mock.fetch())
    assert result.records_fetched == 48
    assert result.records_valid == 48


def test_sap_mock_dataframe_has_kwh(sap_mock):
    result = asyncio.run(sap_mock.fetch())
    assert "kwh" in result.data.columns


def test_sap_mock_kwh_all_positive(sap_mock):
    result = asyncio.run(sap_mock.fetch())
    assert (result.data["kwh"] >= 0).all()


def test_sap_mock_has_cost_centre(sap_mock):
    result = asyncio.run(sap_mock.fetch())
    assert "cost_centre" in result.data.columns
    assert (result.data["cost_centre"] != "").all()


def test_sap_mock_currency_is_eur(sap_mock):
    result = asyncio.run(sap_mock.fetch())
    assert (result.data["currency"] == "EUR").all()


def test_sap_success_rate_one(sap_mock):
    result = asyncio.run(sap_mock.fetch())
    assert result.success_rate == 1.0


# ── SAP Adapter — inject mode ─────────────────────────────────────────────────

def _sap_record(epoch_ms: int, kwh: float = 100.0) -> dict:
    return {
        "MeterID": "SAP-M-01",
        "FunctionalLocation": "FL-001",
        "ReadingDate": f"/Date({epoch_ms})/",
        "ReadingValue": str(kwh),
        "UnitOfMeasure": "KWH",
        "CostCentre": "CC-001",
        "Currency": "EUR",
        "Amount": str(kwh * 0.22),
    }


def test_sap_parse_epoch_date():
    adapter = SAPAdapter(SITE)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    reading = adapter._parse(_sap_record(now_ms, kwh=250.0))
    assert reading.kwh == 250.0
    assert reading.currency == "EUR"


def test_sap_parse_mwh_unit():
    adapter = SAPAdapter(SITE)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    rec = _sap_record(now_ms, kwh=1.5)
    rec["UnitOfMeasure"] = "MWH"
    reading = adapter._parse(rec)
    assert reading.kwh == 1500.0


def test_sap_inject_records():
    adapter = SAPAdapter(SITE)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    adapter.inject_records([_sap_record(now_ms, 200.0)])
    result = asyncio.run(adapter.fetch())
    assert result.records_valid == 1
    assert result.data["kwh"].iloc[0] == 200.0


def test_sap_negative_kwh_rejected():
    adapter = SAPAdapter(SITE)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    adapter.inject_records([_sap_record(now_ms, -50.0)])
    result = asyncio.run(adapter.fetch())
    assert result.records_invalid == 1
    assert result.records_valid == 0


def test_sap_fl_mapping_applied():
    adapter = SAPAdapter(SITE, fl_mapping={SITE: "1000-FL-LONDON-01"}, use_mock=True)
    fl = adapter.fl_mapping.get(SITE)
    assert fl == "1000-FL-LONDON-01"


# ── Oracle Adapter — mock mode ────────────────────────────────────────────────

@pytest.fixture
def oracle_mock():
    return OracleAdapter(SITE, use_mock=True, days_back=2)


def test_oracle_mock_fetch_returns_result(oracle_mock):
    result = asyncio.run(oracle_mock.fetch())
    assert result.source == "oracle_utilities"
    assert result.site_id == SITE


def test_oracle_mock_produces_48_records(oracle_mock):
    result = asyncio.run(oracle_mock.fetch())
    assert result.records_fetched == 48


def test_oracle_mock_dataframe_has_kwh(oracle_mock):
    result = asyncio.run(oracle_mock.fetch())
    assert "kwh" in result.data.columns


def test_oracle_mock_kwh_positive(oracle_mock):
    result = asyncio.run(oracle_mock.fetch())
    assert (result.data["kwh"] >= 0).all()


def test_oracle_mock_currency_is_usd(oracle_mock):
    result = asyncio.run(oracle_mock.fetch())
    assert (result.data["currency"] == "USD").all()


def test_oracle_success_rate_one(oracle_mock):
    result = asyncio.run(oracle_mock.fetch())
    assert result.success_rate == 1.0


# ── Oracle Adapter — inject mode ─────────────────────────────────────────────

def _oracle_record(ts: datetime, kwh: float = 100.0) -> dict:
    return {
        "meter_id":         "ORC-M-01",
        "site_id":          SITE,
        "reading_datetime": ts.isoformat(),
        "energy_kwh":       kwh,
        "unit_of_measure":  "KWH",
        "cost_centre":      "CC-002",
        "currency":         "USD",
        "cost_amount":      kwh * 0.18,
    }


def test_oracle_parse_iso_timestamp():
    adapter = OracleAdapter(SITE)
    now = datetime.now(timezone.utc)
    reading = adapter._parse(_oracle_record(now, kwh=175.0))
    assert reading.kwh == 175.0
    assert reading.currency == "USD"


def test_oracle_parse_mwh_unit():
    adapter = OracleAdapter(SITE)
    now = datetime.now(timezone.utc)
    rec = _oracle_record(now, kwh=2.0)
    rec["unit_of_measure"] = "MWH"
    reading = adapter._parse(rec)
    assert reading.kwh == 2000.0


def test_oracle_inject_and_fetch():
    adapter = OracleAdapter(SITE)
    now = datetime.now(timezone.utc)
    adapter.inject_records([_oracle_record(now, 300.0)])
    result = asyncio.run(adapter.fetch())
    assert result.records_valid == 1
    assert result.data["kwh"].iloc[0] == 300.0


def test_oracle_negative_kwh_rejected():
    adapter = OracleAdapter(SITE)
    now = datetime.now(timezone.utc)
    adapter.inject_records([_oracle_record(now, -10.0)])
    result = asyncio.run(adapter.fetch())
    assert result.records_invalid == 1


# ── Mixed valid/invalid ───────────────────────────────────────────────────────

def test_sap_mixed_valid_invalid():
    adapter = SAPAdapter(SITE)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    adapter.inject_records([
        _sap_record(now_ms, 100.0),
        _sap_record(now_ms, -5.0),   # invalid
        _sap_record(now_ms, 200.0),
    ])
    result = asyncio.run(adapter.fetch())
    assert result.records_valid == 2
    assert result.records_invalid == 1
    assert result.success_rate == pytest.approx(2 / 3)


def test_oracle_mixed_valid_invalid():
    adapter = OracleAdapter(SITE)
    now = datetime.now(timezone.utc)
    adapter.inject_records([
        _oracle_record(now, 100.0),
        _oracle_record(now, -1.0),   # invalid
    ])
    result = asyncio.run(adapter.fetch())
    assert result.records_valid == 1
    assert result.records_invalid == 1


# ── Source names ──────────────────────────────────────────────────────────────

def test_sap_source_name():
    assert SAPAdapter(SITE).source_name == "sap_is_u"


def test_oracle_source_name():
    assert OracleAdapter(SITE).source_name == "oracle_utilities"
