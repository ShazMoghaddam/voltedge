"""Tests for ContextBuilder — uses simulated store data."""
from __future__ import annotations
import asyncio
import pytest
from voltedge.ai.context_builder import ContextBuilder, SiteSnapshot, PortfolioSnapshot
from voltedge.ingestion.simulators import SimulatedSiteConnector
from voltedge.processing.transformer import EnergyTransformer
from voltedge.storage.base import LocalStore


SITE = "CB-TEST-01"


@pytest.fixture(scope="module")
def store_with_data(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cb_store")
    store = LocalStore(base_path=tmp)
    connector = SimulatedSiteConnector(SITE, "factory", hours=200, seed=42)
    result = asyncio.run(connector.fetch())
    df = EnergyTransformer().transform(result.data)
    store.write(df, site_id=SITE, layer="processed")
    return store


@pytest.fixture
def builder(store_with_data):
    return ContextBuilder(store=store_with_data)


# ── SiteSnapshot ──────────────────────────────────────────────────────────────

def test_build_site_returns_snapshot(builder):
    snap = builder.build_site(SITE, hours=24)
    assert isinstance(snap, SiteSnapshot)
    assert snap.site_id == SITE


def test_build_site_total_kwh_positive(builder):
    snap = builder.build_site(SITE, hours=24)
    assert snap.total_kwh >= 0


def test_build_site_has_co2(builder):
    snap = builder.build_site(SITE, hours=24)
    assert snap.co2_kg >= 0
    assert snap.carbon_intensity > 0


def test_build_site_trend_direction_valid(builder):
    snap = builder.build_site(SITE, hours=24)
    assert snap.trend_direction in ("up", "down", "stable")


def test_build_site_period_hours_matches(builder):
    snap = builder.build_site(SITE, hours=48)
    assert snap.period_hours == 48


def test_build_site_has_snapshot_time(builder):
    snap = builder.build_site(SITE, hours=24)
    assert len(snap.snapshot_time) > 0


def test_build_site_empty_for_unknown(store_with_data):
    builder = ContextBuilder(store=store_with_data)
    snap = builder.build_site("NO-SUCH-SITE", hours=24)
    assert snap.total_kwh == 0.0
    assert snap.site_id == "NO-SUCH-SITE"


def test_build_site_cost_estimate(builder):
    snap = builder.build_site(SITE, hours=24)
    assert snap.estimated_cost >= 0
    assert snap.cost_currency in ("GBP", "EUR", "USD")


# ── to_prompt_text ────────────────────────────────────────────────────────────

def test_prompt_text_contains_site_id(builder):
    snap = builder.build_site(SITE, hours=24)
    text = snap.to_prompt_text()
    assert SITE in text


def test_prompt_text_contains_kwh(builder):
    snap = builder.build_site(SITE, hours=24)
    text = snap.to_prompt_text()
    assert "kWh" in text


def test_prompt_text_contains_co2(builder):
    snap = builder.build_site(SITE, hours=24)
    text = snap.to_prompt_text()
    assert "CO" in text or "kg" in text


def test_prompt_text_mentions_anomalies(builder):
    snap = builder.build_site(SITE, hours=24)
    text = snap.to_prompt_text()
    assert "Anomal" in text


def test_prompt_text_under_token_budget(builder):
    snap = builder.build_site(SITE, hours=24)
    text = snap.to_prompt_text()
    # Approx tokens = chars / 4; must be under 4000 tokens
    assert len(text) < 16_000   # 4000 tokens × 4 chars/token


# ── Portfolio ─────────────────────────────────────────────────────────────────

def test_build_portfolio_returns_snapshot(store_with_data):
    connector2 = SimulatedSiteConnector("CB-TEST-02", "office", hours=100, seed=7)
    result2 = asyncio.run(connector2.fetch())
    df2 = EnergyTransformer().transform(result2.data)
    store_with_data.write(df2, site_id="CB-TEST-02", layer="processed")

    builder = ContextBuilder(store=store_with_data)
    portfolio = builder.build_portfolio([SITE, "CB-TEST-02"], hours=24)
    assert isinstance(portfolio, PortfolioSnapshot)
    assert len(portfolio.sites) == 2


def test_portfolio_total_kwh_is_sum(store_with_data):
    builder = ContextBuilder(store=store_with_data)
    portfolio = builder.build_portfolio([SITE], hours=24)
    site_total = sum(s.total_kwh for s in portfolio.sites)
    assert abs(portfolio.total_kwh - site_total) < 0.01


def test_portfolio_prompt_text(store_with_data):
    builder = ContextBuilder(store=store_with_data)
    portfolio = builder.build_portfolio([SITE], hours=24)
    text = portfolio.to_prompt_text()
    assert "Portfolio" in text
    assert SITE in text
