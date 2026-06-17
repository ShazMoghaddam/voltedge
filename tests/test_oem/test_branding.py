"""Tests for OEM white-label branding system."""

from __future__ import annotations

import json
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from voltedge.oem.branding import (
    BrandingBase, BrandingProfile, BrandingService,
    BUILTIN_TENANTS, TenantBranding,
)


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    BrandingBase.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def svc(db):
    s = BrandingService(db)
    s.seed_builtin_tenants()
    return s


# ── Seed ──────────────────────────────────────────────────────────────────────

def test_seed_creates_builtin_tenants(svc, db):
    tenants = db.query(TenantBranding).all()
    assert len(tenants) == len(BUILTIN_TENANTS)


def test_seed_is_idempotent(svc):
    svc.seed_builtin_tenants()
    svc.seed_builtin_tenants()
    tenants = svc.list_tenants()
    assert len(tenants) == len(BUILTIN_TENANTS)


def test_shell_tenant_has_yellow_primary(svc):
    shell = svc.get("shell-energy")
    assert shell is not None
    assert shell.color_primary == "#FBCE07"


def test_bp_tenant_has_green_primary(svc):
    bp = svc.get("bp-energy")
    assert bp.color_primary == "#009900"


def test_shell_has_custom_domain(svc):
    shell = svc.get("shell-energy")
    assert shell.custom_domain == "energyiq.shell.com"


def test_eni_lstm_disabled(svc):
    eni = svc.get("eni-analytics")
    assert not eni.has_feature("lstm")


def test_shell_lstm_enabled(svc):
    shell = svc.get("shell-energy")
    assert shell.has_feature("lstm")


# ── BrandingProfile ───────────────────────────────────────────────────────────

def test_profile_from_db(svc):
    shell = svc.get("shell-energy")
    profile = BrandingProfile.from_db(shell)
    assert profile.product_name == "Shell EnergyIQ"
    assert profile.tenant_id == "shell-energy"
    assert profile.color_primary == "#FBCE07"


def test_profile_default():
    profile = BrandingProfile.default()
    assert profile.tenant_id == "voltedge"
    assert profile.product_name == "VoltEdge"


def test_profile_dashboard_title():
    profile = BrandingProfile.default()
    assert "VoltEdge" in profile.dashboard_title()


def test_profile_shell_dashboard_title(svc):
    shell = svc.get("shell-energy")
    profile = BrandingProfile.from_db(shell)
    assert "Shell EnergyIQ" in profile.dashboard_title()


def test_profile_css_vars_contains_primary(svc):
    shell = svc.get("shell-energy")
    profile = BrandingProfile.from_db(shell)
    css = profile.to_css_vars()
    assert "#FBCE07" in css
    assert "--ve-color-primary" in css


def test_profile_css_vars_root_block():
    profile = BrandingProfile.default()
    css = profile.to_css_vars()
    assert css.startswith(":root {")
    assert css.endswith("}")


def test_profile_carbon_intensity_override(svc):
    shell = svc.get("shell-energy")
    profile = BrandingProfile.from_db(shell)
    assert profile.carbon_intensity == 0.195


def test_profile_default_no_carbon_override():
    profile = BrandingProfile.default()
    assert profile.carbon_intensity is None


# ── Resolution ────────────────────────────────────────────────────────────────

def test_resolve_by_tenant_id(svc):
    profile = svc.resolve(tenant_id="shell-energy")
    assert profile.product_name == "Shell EnergyIQ"


def test_resolve_by_host_domain(svc):
    profile = svc.resolve(host="energyiq.shell.com")
    assert profile.product_name == "Shell EnergyIQ"


def test_resolve_by_host_with_port(svc):
    profile = svc.resolve(host="energyiq.shell.com:8050")
    assert profile.product_name == "Shell EnergyIQ"


def test_resolve_unknown_returns_default(svc):
    profile = svc.resolve(tenant_id="unknown-corp")
    assert profile.tenant_id == "voltedge"


def test_resolve_no_args_returns_default(svc):
    profile = svc.resolve()
    assert profile.product_name == "VoltEdge"


def test_resolve_tenant_id_takes_priority_over_host(svc):
    profile = svc.resolve(tenant_id="bp-energy", host="energyiq.shell.com")
    assert profile.tenant_id == "bp-energy"


# ── CRUD ──────────────────────────────────────────────────────────────────────

def test_create_tenant(svc):
    t = svc.create(
        tenant_id="total-energy",
        product_name="TotalEnergies EIQ",
        tagline="Together, let's be energy positive",
        color_primary="#F0304B",
        feature_flags={"forecast": True, "anomaly": False},
    )
    assert t.tenant_id == "total-energy"
    assert t.product_name == "TotalEnergies EIQ"


def test_create_duplicate_raises(svc):
    with pytest.raises(ValueError, match="already exists"):
        svc.create("shell-energy", "Duplicate Shell")


def test_update_feature_flag_enables(svc):
    svc.update_feature_flag("eni-analytics", "lstm", True)
    eni = svc.get("eni-analytics")
    assert eni.has_feature("lstm")


def test_update_feature_flag_disables(svc):
    svc.update_feature_flag("shell-energy", "lstm", False)
    shell = svc.get("shell-energy")
    assert not shell.has_feature("lstm")


def test_update_unknown_tenant_raises(svc):
    with pytest.raises(ValueError, match="not found"):
        svc.update_feature_flag("ghost-corp", "forecast", True)


def test_deactivate_tenant(svc):
    svc.deactivate("eni-analytics")
    tenants = svc.list_tenants()
    ids = {t.tenant_id for t in tenants}
    assert "eni-analytics" not in ids


def test_list_tenants_only_active(svc):
    svc.deactivate("bp-energy")
    tenants = svc.list_tenants()
    assert all(t.is_active for t in tenants)


def test_deactivated_tenant_not_resolved(svc):
    svc.deactivate("bp-energy")
    profile = svc.resolve(tenant_id="bp-energy")
    assert profile.tenant_id == "voltedge"


def test_feature_flags_json_roundtrip(svc):
    t = svc.create("roundtrip-co", "RT Co",
                   feature_flags={"x": True, "y": False, "z": True})
    row = svc.get("roundtrip-co")
    flags = row.feature_flags
    assert flags["x"] is True
    assert flags["y"] is False
    assert flags["z"] is True
