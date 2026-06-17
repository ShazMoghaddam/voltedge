"""Tests for MultiRegionRouter and DataResidencyPolicy."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from voltedge.core.multiregion import (
    AWSRegion, DataResidencyZone, MultiRegionRouter, RegionConfig,
    check_all_regions,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _eu_config(primary: bool = True) -> RegionConfig:
    return RegionConfig(
        region=AWSRegion.EU_WEST_1,
        s3_bucket="voltedge-data-eu-prod",
        db_url="postgresql://host-eu/voltedge",
        api_endpoint="https://api-eu.voltedge.io",
        is_primary=primary,
        residency_zone=DataResidencyZone.EU,
    )


def _us_config() -> RegionConfig:
    return RegionConfig(
        region=AWSRegion.US_EAST_1,
        s3_bucket="voltedge-data-us-prod",
        db_url="postgresql://host-us/voltedge",
        api_endpoint="https://api-us.voltedge.io",
        residency_zone=DataResidencyZone.US,
    )


def _apac_config() -> RegionConfig:
    return RegionConfig(
        region=AWSRegion.AP_SOUTHEAST_1,
        s3_bucket="voltedge-data-apac-prod",
        db_url="postgresql://host-apac/voltedge",
        api_endpoint="https://api-apac.voltedge.io",
        residency_zone=DataResidencyZone.APAC,
    )


@pytest.fixture
def three_region_router():
    return MultiRegionRouter([_eu_config(primary=True), _us_config(), _apac_config()])


# ── RegionConfig ──────────────────────────────────────────────────────────────

def test_region_config_region_code():
    cfg = _eu_config()
    assert cfg.region_code == "eu_west_1"


def test_region_config_us_code():
    cfg = _us_config()
    assert cfg.region_code == "us_east_1"


# ── Router construction ───────────────────────────────────────────────────────

def test_router_requires_at_least_one_config():
    with pytest.raises(ValueError):
        MultiRegionRouter([])


def test_router_rejects_multiple_primaries():
    with pytest.raises(ValueError, match="primary"):
        MultiRegionRouter([_eu_config(primary=True), RegionConfig(
            region=AWSRegion.US_EAST_1,
            s3_bucket="b", db_url="u", is_primary=True
        )])


def test_router_primary_returns_primary_region(three_region_router):
    assert three_region_router.primary.region == AWSRegion.EU_WEST_1


def test_router_single_region_defaults_to_primary():
    router = MultiRegionRouter([_eu_config(primary=False)])
    assert router.primary.region == AWSRegion.EU_WEST_1


# ── Routing ───────────────────────────────────────────────────────────────────

def test_explicit_site_mapping_wins(three_region_router):
    three_region_router.add_site_mapping("SPECIAL-SITE", AWSRegion.US_EAST_1)
    cfg = three_region_router.route("SPECIAL-SITE")
    assert cfg.region == AWSRegion.US_EAST_1


def test_eu_residency_routes_to_eu(three_region_router):
    cfg = three_region_router.route("ANY-SITE", residency=DataResidencyZone.EU)
    assert cfg.region == AWSRegion.EU_WEST_1


def test_us_residency_routes_to_us(three_region_router):
    cfg = three_region_router.route("ANY-SITE", residency=DataResidencyZone.US)
    assert cfg.region == AWSRegion.US_EAST_1


def test_apac_residency_routes_to_apac(three_region_router):
    cfg = three_region_router.route("ANY-SITE", residency=DataResidencyZone.APAC)
    assert cfg.region == AWSRegion.AP_SOUTHEAST_1


def test_inferred_eu_from_site_id_prefix(three_region_router):
    cfg = three_region_router.route("EU-LONDON-FACTORY-01")
    assert cfg.region == AWSRegion.EU_WEST_1


def test_inferred_us_from_site_id_prefix(three_region_router):
    cfg = three_region_router.route("US-HOUSTON-PLANT-01")
    assert cfg.region == AWSRegion.US_EAST_1


def test_inferred_apac_from_site_id_prefix(three_region_router):
    cfg = three_region_router.route("APAC-SINGAPORE-DC-01")
    assert cfg.region == AWSRegion.AP_SOUTHEAST_1


def test_unknown_site_falls_back_to_primary(three_region_router):
    cfg = three_region_router.route("UNKNOWN-SITE-XYZ")
    assert cfg.region == AWSRegion.EU_WEST_1


def test_gb_prefix_routes_to_eu(three_region_router):
    cfg = three_region_router.route("GB-LONDON-01")
    assert cfg.region == AWSRegion.EU_WEST_1


def test_de_prefix_routes_to_eu(three_region_router):
    cfg = three_region_router.route("DE-FRANKFURT-02")
    assert cfg.region == AWSRegion.EU_WEST_1


def test_london_keyword_routes_to_eu(three_region_router):
    cfg = three_region_router.route("LONDON-FACTORY-01")
    assert cfg.region == AWSRegion.EU_WEST_1


def test_explicit_mapping_overrides_residency(three_region_router):
    """Explicit site mapping should win even when residency says EU."""
    three_region_router.add_site_mapping("EU-SITE", AWSRegion.US_EAST_1)
    cfg = three_region_router.route("EU-SITE", residency=DataResidencyZone.EU)
    assert cfg.region == AWSRegion.US_EAST_1


def test_add_unknown_region_raises():
    """Router with only EU configured should reject US as 'not configured'."""
    router = MultiRegionRouter([_eu_config()])
    with pytest.raises(ValueError, match="not configured"):
        router.add_site_mapping("X", AWSRegion.US_EAST_1)


# ── All regions ───────────────────────────────────────────────────────────────

def test_all_regions_returns_all(three_region_router):
    regions = three_region_router.all_regions()
    assert len(regions) == 3


def test_region_summary_structure(three_region_router):
    summary = three_region_router.region_summary()
    assert "regions" in summary
    assert "primary" in summary
    assert len(summary["regions"]) == 3
    assert summary["primary"] == "eu-west-1"


# ── Health checker ────────────────────────────────────────────────────────────

def test_check_all_regions_with_mock():
    configs = [_eu_config(), _us_config(), _apac_config()]

    async def mock_get(*args, **kwargs):
        mock = MagicMock()
        mock.status_code = 200
        mock.json.return_value = {"status": "ok"}
        return mock

    with patch("httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(side_effect=mock_get)
        mock_cls.return_value = mock_client

        results = asyncio.run(check_all_regions(configs))

    assert set(results.keys()) == {"eu-west-1", "us-east-1", "ap-southeast-1"}
    assert all(results.values())


def test_check_all_regions_handles_failure():
    configs = [_eu_config(), _us_config()]

    with patch("httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        import httpx
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_cls.return_value = mock_client

        results = asyncio.run(check_all_regions(configs))

    assert all(not v for v in results.values())


def test_check_region_with_empty_endpoint():
    cfg = RegionConfig(
        region=AWSRegion.EU_WEST_1,
        s3_bucket="b", db_url="u",
        api_endpoint="",   # No endpoint configured
    )
    results = asyncio.run(check_all_regions([cfg]))
    assert results["eu-west-1"] is False


# ── Deploy artefact validation ────────────────────────────────────────────────

def test_k8s_manifest_exists():
    from pathlib import Path
    manifest = Path(__file__).parent.parent.parent / "deploy" / "k8s" / "deployment.yaml"
    assert manifest.exists()


def test_terraform_main_exists():
    from pathlib import Path
    tf = Path(__file__).parent.parent.parent / "deploy" / "terraform" / "main.tf"
    assert tf.exists()


def test_deploy_script_exists():
    from pathlib import Path
    script = Path(__file__).parent.parent.parent / "deploy" / "scripts" / "deploy_multiregion.sh"
    assert script.exists()


def test_k8s_manifest_has_hpa():
    from pathlib import Path
    content = (Path(__file__).parent.parent.parent / "deploy" / "k8s" / "deployment.yaml").read_text()
    assert "HorizontalPodAutoscaler" in content


def test_k8s_manifest_has_liveness_probe():
    from pathlib import Path
    content = (Path(__file__).parent.parent.parent / "deploy" / "k8s" / "deployment.yaml").read_text()
    assert "livenessProbe" in content


def test_terraform_has_three_regions():
    from pathlib import Path
    content = (Path(__file__).parent.parent.parent / "deploy" / "terraform" / "main.tf").read_text()
    assert "eu_west_1" in content
    assert "us_east_1" in content
    assert "ap_southeast_1" in content


def test_terraform_has_s3_crr():
    from pathlib import Path
    content = (Path(__file__).parent.parent.parent / "deploy" / "terraform" / "main.tf").read_text()
    assert "replication_configuration" in content
