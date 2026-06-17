"""Tests for the feature flag system."""

from __future__ import annotations

import pytest
from voltedge.oem.feature_flags import (
    FEATURE_FLAGS, FeatureFlagService, FlagDefinition, Tier,
)


@pytest.fixture
def svc():
    return FeatureFlagService()


# ── Flag catalogue ────────────────────────────────────────────────────────────

def test_catalogue_has_expected_flags():
    for key in ("forecast_v2_model", "lstm_streaming", "pdf_export",
                "erp_sap", "dark_mode", "esg_scope3"):
        assert key in FEATURE_FLAGS, f"Missing flag: {key}"


def test_all_flags_have_descriptions():
    for key, flag in FEATURE_FLAGS.items():
        assert flag.description, f"Flag {key} has no description"


def test_all_flags_have_keys_matching_dict():
    for key, flag in FEATURE_FLAGS.items():
        assert flag.key == key


# ── Tier defaults ─────────────────────────────────────────────────────────────

def test_enterprise_has_erp_sap(svc):
    assert svc.is_enabled("erp_sap", "corp-A", Tier.ENTERPRISE)


def test_starter_no_erp_sap(svc):
    assert not svc.is_enabled("erp_sap", "corp-B", Tier.STARTER)


def test_professional_has_pdf_export(svc):
    assert svc.is_enabled("pdf_export", "corp-C", Tier.PROFESSIONAL)


def test_starter_no_pdf_export(svc):
    assert not svc.is_enabled("pdf_export", "corp-D", Tier.STARTER)


def test_enterprise_forecast_v2(svc):
    assert svc.is_enabled("forecast_v2_model", "corp-E", Tier.ENTERPRISE)


def test_starter_no_forecast_v2(svc):
    assert not svc.is_enabled("forecast_v2_model", "corp-F", Tier.STARTER)


def test_dark_mode_all_tiers(svc):
    for tier in (Tier.STARTER, Tier.PROFESSIONAL, Tier.ENTERPRISE):
        assert svc.is_enabled("dark_mode", "any-corp", tier)


def test_esg_scope3_not_released(svc):
    """esg_scope3 has rollout_pct=0.0 — no one gets it yet."""
    for tier in (Tier.STARTER, Tier.PROFESSIONAL, Tier.ENTERPRISE):
        assert not svc.is_enabled("esg_scope3", "any-corp", tier)


# ── Tenant overrides ──────────────────────────────────────────────────────────

def test_override_enables_for_starter(svc):
    svc.set_override("early-adopter", "erp_sap", True)
    assert svc.is_enabled("erp_sap", "early-adopter", Tier.STARTER)


def test_override_disables_for_enterprise(svc):
    svc.set_override("corp-G", "dark_mode", False)
    assert not svc.is_enabled("dark_mode", "corp-G", Tier.ENTERPRISE)


def test_override_takes_priority_over_tier(svc):
    # Enterprise gets erp_sap by default
    svc.set_override("enterprise-no-sap", "erp_sap", False)
    assert not svc.is_enabled("erp_sap", "enterprise-no-sap", Tier.ENTERPRISE)


def test_clear_override_reverts_to_tier(svc):
    svc.set_override("corp-H", "erp_sap", True)
    svc.clear_override("corp-H", "erp_sap")
    # Starter shouldn't have it anymore
    assert not svc.is_enabled("erp_sap", "corp-H", Tier.STARTER)


def test_clear_all_overrides(svc):
    svc.set_override("corp-I", "erp_sap", True)
    svc.set_override("corp-I", "pdf_export", True)
    svc.clear_all_overrides("corp-I")
    assert not svc.is_enabled("erp_sap", "corp-I", Tier.STARTER)
    assert not svc.is_enabled("pdf_export", "corp-I", Tier.STARTER)


def test_set_override_unknown_flag_raises(svc):
    with pytest.raises(ValueError, match="Unknown flag"):
        svc.set_override("corp-J", "nonexistent_flag", True)


# ── get_all_flags ─────────────────────────────────────────────────────────────

def test_get_all_flags_returns_all(svc):
    flags = svc.get_all_flags("corp-K", Tier.ENTERPRISE)
    assert set(flags.keys()) == set(FEATURE_FLAGS.keys())


def test_get_all_flags_enterprise_has_erp(svc):
    flags = svc.get_all_flags("corp-K", Tier.ENTERPRISE)
    assert flags["erp_sap"] is True
    assert flags["erp_oracle"] is True


def test_get_all_flags_starter_no_erp(svc):
    flags = svc.get_all_flags("corp-L", Tier.STARTER)
    assert flags["erp_sap"] is False
    assert flags["pdf_export"] is False


# ── Rollout ───────────────────────────────────────────────────────────────────

def test_rollout_deterministic():
    svc = FeatureFlagService()
    r1 = svc.is_enabled("lstm_streaming", "shell-energy", Tier.ENTERPRISE)
    r2 = svc.is_enabled("lstm_streaming", "shell-energy", Tier.ENTERPRISE)
    assert r1 == r2


def test_rollout_different_tenants_may_differ():
    """With 25% rollout, some Enterprise tenants should be in, some out."""
    svc = FeatureFlagService()
    results = [
        svc.is_enabled("lstm_streaming", f"tenant-{i:04d}", Tier.ENTERPRISE)
        for i in range(100)
    ]
    # With 25% rollout we expect roughly 25 enabled — at least some True and some False
    assert any(results), "Expected at least some tenants to have lstm_streaming"
    assert not all(results), "Expected not all tenants to have lstm_streaming"


def test_rollout_0pct_disabled_for_all():
    svc = FeatureFlagService()
    results = [
        svc.is_enabled("esg_scope3", f"t-{i}", Tier.ENTERPRISE)
        for i in range(20)
    ]
    assert not any(results)


def test_rollout_100pct_enabled_for_all():
    svc = FeatureFlagService()
    results = [
        svc.is_enabled("marketplace_checkout", f"t-{i}", Tier.STARTER)
        for i in range(20)
    ]
    assert all(results)


# ── Unknown flag ──────────────────────────────────────────────────────────────

def test_unknown_flag_returns_false(svc):
    result = svc.is_enabled("does_not_exist", "corp-X", Tier.ENTERPRISE)
    assert result is False


# ── flags_for_tier ────────────────────────────────────────────────────────────

def test_flags_for_enterprise_tier(svc):
    flags = svc.flags_for_tier(Tier.ENTERPRISE)
    assert flags["erp_sap"] is True
    assert flags["pdf_export"] is True


def test_flags_for_starter_tier(svc):
    flags = svc.flags_for_tier(Tier.STARTER)
    assert flags["erp_sap"] is False
    assert flags["pdf_export"] is False
    assert flags["dark_mode"] is True


# ── list_flags / summary ──────────────────────────────────────────────────────

def test_list_flags_returns_all(svc):
    flags = svc.list_flags()
    assert len(flags) == len(FEATURE_FLAGS)


def test_list_flags_by_tag(svc):
    erp_flags = svc.list_flags(tag="erp")
    assert all("erp" in f.tags for f in erp_flags)
    assert len(erp_flags) >= 2


def test_flag_summary_structure(svc):
    summary = svc.flag_summary()
    assert "total_flags" in summary
    assert "enabled_globally" in summary
    assert "rollout_flags" in summary
    assert summary["total_flags"] == len(FEATURE_FLAGS)
