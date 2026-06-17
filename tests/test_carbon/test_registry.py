"""Tests for carbon credit registry."""
from __future__ import annotations
from datetime import datetime, timezone
import pytest
from voltedge.carbon.registry import (
    CarbonCredit, CreditClaim, CreditStatus, MockRegistry,
    RegistryType, VerraRegistry,
)

TS = datetime(2026, 5, 1, tzinfo=timezone.utc)


def _claim(**kwargs):
    defaults = dict(
        site_id="SITE-01", project_id="IREC-003",
        period_start=TS, period_end=TS,
        kwh_renewable=1000.0, co2e_avoided_t=0.207,
    )
    return CreditClaim(**{**defaults, **kwargs})


# ── CarbonCredit ──────────────────────────────────────────────────────────────

def test_credit_is_not_retired_by_default():
    c = CarbonCredit("C1", RegistryType.MOCK, "P1", "Test",
                     2026, 1.0, CreditStatus.ISSUED)
    assert not c.is_retired

def test_credit_retire_returns_new():
    c = CarbonCredit("C1", RegistryType.MOCK, "P1", "Test",
                     2026, 1.0, CreditStatus.ISSUED)
    retired = c.retire("Shell plc")
    assert retired.is_retired
    assert retired.retired_by == "Shell plc"
    assert c.status == CreditStatus.ISSUED   # original unchanged

def test_credit_to_dict_has_keys():
    c = CarbonCredit("C1", RegistryType.MOCK, "P1", "Test",
                     2026, 1.0, CreditStatus.ISSUED)
    d = c.to_dict()
    for k in ("credit_id","registry","quantity_tco2e","status","vintage_year"):
        assert k in d

def test_credit_retire_sets_timestamp():
    c = CarbonCredit("C1", RegistryType.MOCK, "P1", "Test",
                     2026, 1.0, CreditStatus.ISSUED)
    retired = c.retire("Test Entity")
    assert retired.retired_at is not None


# ── CreditClaim ───────────────────────────────────────────────────────────────

def test_claim_has_id():
    c = _claim()
    assert len(c.claim_id) == 36   # UUID4

def test_claim_evidence_digest():
    c = _claim().evidence_digest("some evidence data")
    assert len(c.evidence_hash) == 64    # SHA-256 hex

def test_claim_evidence_deterministic():
    c1 = _claim().evidence_digest("same data")
    c2 = _claim().evidence_digest("same data")
    assert c1.evidence_hash == c2.evidence_hash

def test_claim_different_data_different_hash():
    c1 = _claim().evidence_digest("data A")
    c2 = _claim().evidence_digest("data B")
    assert c1.evidence_hash != c2.evidence_hash


# ── MockRegistry ──────────────────────────────────────────────────────────────

@pytest.fixture
def registry():
    return MockRegistry()


def test_submit_claim_returns_issued(registry):
    claim = _claim()
    updated = registry.submit_claim(claim)
    assert updated.status == CreditStatus.ISSUED


def test_submit_claim_sets_credit_id(registry):
    claim = _claim()
    updated = registry.submit_claim(claim)
    assert updated.issued_credit_id is not None
    assert updated.issued_credit_id.startswith("MOCK-")


def test_submit_claim_zero_co2e_raises(registry):
    with pytest.raises(ValueError):
        registry.submit_claim(_claim(co2e_avoided_t=0.0))


def test_submit_claim_negative_co2e_raises(registry):
    with pytest.raises(ValueError):
        registry.submit_claim(_claim(co2e_avoided_t=-1.0))


def test_get_credit_after_submit(registry):
    claim = _claim()
    updated = registry.submit_claim(claim)
    credit = registry.get_credit(updated.issued_credit_id)
    assert credit is not None
    assert credit.quantity_tco2e == pytest.approx(0.207)


def test_get_missing_credit_returns_none(registry):
    assert registry.get_credit("NONEXISTENT-ID") is None


def test_list_credits_empty(registry):
    assert registry.list_credits() == []


def test_list_credits_after_submit(registry):
    registry.submit_claim(_claim())
    credits = registry.list_credits()
    assert len(credits) == 1


def test_list_credits_filter_by_status(registry):
    claim = _claim()
    updated = registry.submit_claim(claim)
    issued  = registry.list_credits(status=CreditStatus.ISSUED)
    retired = registry.list_credits(status=CreditStatus.RETIRED)
    assert len(issued)  == 1
    assert len(retired) == 0


def test_retire_credit(registry):
    updated = registry.submit_claim(_claim())
    retired = registry.retire_credit(updated.issued_credit_id, "Shell plc")
    assert retired.is_retired
    assert retired.retired_by == "Shell plc"


def test_retire_same_credit_twice_raises(registry):
    updated = registry.submit_claim(_claim())
    registry.retire_credit(updated.issued_credit_id, "Entity A")
    with pytest.raises(ValueError):
        registry.retire_credit(updated.issued_credit_id, "Entity B")


def test_retire_unknown_credit_raises(registry):
    with pytest.raises(ValueError):
        registry.retire_credit("FAKE-ID", "Entity")


def test_get_credit_price(registry):
    price = registry.get_credit_price("IREC-003")
    assert price is not None
    assert price > 0


def test_get_credit_price_unknown_project(registry):
    price = registry.get_credit_price("UNKNOWN-PROJ")
    assert price is None


def test_deterministic_credit_id(registry):
    c1 = registry.submit_claim(_claim())
    registry2 = MockRegistry()
    c2 = registry2.submit_claim(_claim(claim_id=c1.claim_id[:-1] + "X"))
    # Different claim_id → different credit_id
    assert c1.issued_credit_id != c2.issued_credit_id


def test_portfolio_value(registry):
    registry.submit_claim(_claim(co2e_avoided_t=2.0))
    registry.submit_claim(_claim(co2e_avoided_t=3.0))
    pv = registry.portfolio_value()
    assert pv["active_credits"] == 2
    assert pv["total_tco2e"] == pytest.approx(5.0)
    assert pv["total_value_usd"] > 0


def test_portfolio_value_retired_not_counted(registry):
    updated = registry.submit_claim(_claim(co2e_avoided_t=1.0))
    registry.retire_credit(updated.issued_credit_id, "Corp")
    pv = registry.portfolio_value()
    assert pv["active_credits"] == 0


# ── VerraRegistry (dry_run) ───────────────────────────────────────────────────

def test_verra_dry_run_submit_returns_pending():
    reg   = VerraRegistry(dry_run=True)
    claim = _claim()
    updated = reg.submit_claim(claim)
    assert updated.status == CreditStatus.PENDING


def test_verra_dry_run_list_returns_empty():
    reg = VerraRegistry(dry_run=True)
    assert reg.list_credits() == []


def test_verra_dry_run_get_returns_none():
    reg = VerraRegistry(dry_run=True)
    assert reg.get_credit("any-id") is None


# ── CreditIssuanceService ─────────────────────────────────────────────────────

# (these live here since they depend on fixtures from this file)
import asyncio
from voltedge.carbon.credits import CreditIssuanceService, IssuanceResult
from voltedge.ingestion.simulators import SimulatedSiteConnector
from voltedge.processing.transformer import EnergyTransformer


@pytest.fixture(scope="module")
def sample_df():
    connector = SimulatedSiteConnector("CRED-SITE", "factory", hours=200, seed=9)
    result = asyncio.run(connector.fetch())
    return EnergyTransformer().transform(result.data)


@pytest.fixture
def issuance_svc():
    return CreditIssuanceService(
        registry=MockRegistry(),
        renewable_fraction=0.30,
        grid_ef_kg_per_kwh=0.207,
    )


def test_issue_from_data_succeeds(issuance_svc, sample_df):
    r = issuance_svc.issue_from_data(sample_df, "CRED-SITE")
    assert r.success is True


def test_issue_from_data_tco2e_positive(issuance_svc, sample_df):
    r = issuance_svc.issue_from_data(sample_df, "CRED-SITE")
    assert r.tco2e_issued > 0


def test_issue_from_data_credit_issued(issuance_svc, sample_df):
    r = issuance_svc.issue_from_data(sample_df, "CRED-SITE")
    assert len(r.credits_issued) >= 1


def test_issue_from_data_portfolio_value_positive(issuance_svc, sample_df):
    r = issuance_svc.issue_from_data(sample_df, "CRED-SITE")
    assert r.portfolio_value_usd > 0


def test_issue_from_empty_df_fails(issuance_svc):
    import pandas as pd
    r = issuance_svc.issue_from_data(pd.DataFrame(), "EMPTY-SITE")
    assert r.success is False


def test_issue_zero_renewable_fails(issuance_svc, sample_df):
    r = issuance_svc.issue_from_data(sample_df, "CRED-SITE", renewable_fraction=0.0)
    assert r.success is False
    assert "threshold" in r.message


def test_retire_for_scope2(issuance_svc, sample_df):
    r = issuance_svc.issue_from_data(sample_df, "CRED-SITE")
    credit_id = r.credits_issued[0].credit_id
    retired = issuance_svc.retire_for_scope2(credit_id, "Shell plc")
    assert retired.is_retired


def test_portfolio_summary_structure(issuance_svc, sample_df):
    issuance_svc.issue_from_data(sample_df, "CRED-SITE")
    summary = issuance_svc.portfolio_summary()
    for k in ("total_claims", "issued_credits", "total_tco2e_issued"):
        assert k in summary


def test_avoided_emissions_calculation(issuance_svc):
    kg = issuance_svc.avoided_emissions_kg(1000.0)
    assert kg == pytest.approx(207.0)   # 1000 kWh × 0.207 kg/kWh


def test_credit_value_usd(issuance_svc):
    val = issuance_svc.credit_value_usd(1.0, "IREC-003")
    assert val > 0


def test_bulk_issue(issuance_svc, sample_df):
    results = issuance_svc.bulk_issue({"SITE-A": sample_df, "SITE-B": sample_df})
    assert len(results) == 2
    assert all(r.success for r in results)


def test_result_to_dict(issuance_svc, sample_df):
    r = issuance_svc.issue_from_data(sample_df, "CRED-SITE")
    d = r.to_dict()
    for k in ("claim_id", "site_id", "tco2e_issued", "success"):
        assert k in d
