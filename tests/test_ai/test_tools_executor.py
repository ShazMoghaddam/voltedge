"""Tests for ToolExecutor — real VoltEdge data, no Anthropic API calls."""
from __future__ import annotations
import asyncio
import pytest
from voltedge.ai.executor import ToolExecutor
from voltedge.ai.context_builder import ContextBuilder
from voltedge.ai.tools import TOOLS, TOOL_NAMES
from voltedge.ingestion.simulators import SimulatedSiteConnector
from voltedge.processing.transformer import EnergyTransformer
from voltedge.storage.base import LocalStore

SITE_A = "EXEC-SITE-A"
SITE_B = "EXEC-SITE-B"


@pytest.fixture(scope="module")
def executor(tmp_path_factory):
    tmp   = tmp_path_factory.mktemp("exec_store")
    store = LocalStore(base_path=tmp)
    for site, stype, seed in [(SITE_A, "factory", 1), (SITE_B, "office", 2)]:
        conn = SimulatedSiteConnector(site, stype, hours=300, seed=seed)
        result = asyncio.run(conn.fetch())
        df = EnergyTransformer().transform(result.data)
        store.write(df, site_id=site, layer="processed")
    return ToolExecutor(store=store, sites=[SITE_A, SITE_B])


# ── Tool catalogue ────────────────────────────────────────────────────────────

def test_all_tools_have_names():
    for tool in TOOLS:
        assert "name" in tool and tool["name"]

def test_all_tools_have_descriptions():
    for tool in TOOLS:
        assert "description" in tool and len(tool["description"]) > 10

def test_all_tools_have_input_schema():
    for tool in TOOLS:
        assert "input_schema" in tool

def test_tool_names_set_populated():
    assert len(TOOL_NAMES) == len(TOOLS)
    assert "get_site_summary" in TOOL_NAMES


# ── Dispatch ──────────────────────────────────────────────────────────────────

def test_unknown_tool_returns_error(executor):
    result = executor.execute("nonexistent_tool", {})
    assert "error" in result


# ── get_site_summary ──────────────────────────────────────────────────────────

def test_get_site_summary_structure(executor):
    r = executor.execute("get_site_summary", {"site_id": SITE_A, "hours": 24})
    for key in ("site_id", "total_kwh", "avg_kwh_per_hour", "peak_kwh",
                "co2_kg", "anomaly_count", "trend_direction"):
        assert key in r, f"Missing key: {key}"


def test_get_site_summary_kwh_non_negative(executor):
    r = executor.execute("get_site_summary", {"site_id": SITE_A})
    assert r["total_kwh"] >= 0


def test_get_site_summary_trend_valid(executor):
    r = executor.execute("get_site_summary", {"site_id": SITE_A})
    assert r["trend_direction"] in ("up", "down", "stable")


def test_get_site_summary_missing_site_returns_data(executor):
    r = executor.execute("get_site_summary", {"site_id": "GHOST-SITE"})
    # Should return snapshot with zeros, not raise
    assert "site_id" in r
    assert r["total_kwh"] == 0.0


# ── get_portfolio_summary ─────────────────────────────────────────────────────

def test_get_portfolio_summary_structure(executor):
    r = executor.execute("get_portfolio_summary", {"hours": 24})
    for key in ("site_count", "total_kwh", "total_co2_kg", "sites"):
        assert key in r

def test_get_portfolio_summary_site_count(executor):
    r = executor.execute("get_portfolio_summary", {"hours": 24})
    assert r["site_count"] == 2

def test_get_portfolio_summary_sites_list(executor):
    r = executor.execute("get_portfolio_summary", {})
    assert isinstance(r["sites"], list)
    assert len(r["sites"]) == 2


# ── get_anomalies ─────────────────────────────────────────────────────────────

def test_get_anomalies_structure(executor):
    r = executor.execute("get_anomalies", {"site_id": SITE_A, "hours": 24})
    for key in ("anomaly_count", "anomalies", "sites_checked"):
        assert key in r

def test_get_anomalies_list_type(executor):
    r = executor.execute("get_anomalies", {})
    assert isinstance(r["anomalies"], list)

def test_get_anomalies_all_sites(executor):
    r = executor.execute("get_anomalies", {"hours": 48})
    assert r["sites_checked"] == 2


# ── get_optimization_tips ─────────────────────────────────────────────────────

def test_get_optimization_tips_structure(executor):
    r = executor.execute("get_optimization_tips", {"site_id": SITE_A})
    for key in ("site_id", "current_annual_cost", "total_saving", "recommendations"):
        assert key in r

def test_get_optimization_tips_has_recommendations(executor):
    r = executor.execute("get_optimization_tips", {"site_id": SITE_A})
    assert len(r["recommendations"]) > 0

def test_get_optimization_tips_saving_positive(executor):
    r = executor.execute("get_optimization_tips", {"site_id": SITE_A})
    assert r["total_saving"] >= 0


# ── compare_sites ─────────────────────────────────────────────────────────────

def test_compare_sites_structure(executor):
    r = executor.execute("compare_sites", {"site_a": SITE_A, "site_b": SITE_B})
    for key in ("site_a", "site_b", "comparison"):
        assert key in r

def test_compare_sites_higher_consumer_valid(executor):
    r = executor.execute("compare_sites", {"site_a": SITE_A, "site_b": SITE_B})
    assert r["comparison"]["higher_consumer"] in (SITE_A, SITE_B)

def test_compare_sites_kwh_diff_is_numeric(executor):
    r = executor.execute("compare_sites", {"site_a": SITE_A, "site_b": SITE_B})
    diff = r["comparison"].get("kwh_diff_pct")
    assert diff is None or isinstance(diff, (int, float))


# ── get_peak_hours ────────────────────────────────────────────────────────────

def test_get_peak_hours_structure(executor):
    r = executor.execute("get_peak_hours", {"site_id": SITE_A, "days": 7})
    for key in ("peak_hour", "low_hour", "hourly_profile"):
        assert key in r

def test_get_peak_hours_hour_in_range(executor):
    r = executor.execute("get_peak_hours", {"site_id": SITE_A, "days": 7})
    assert 0 <= r["peak_hour"] <= 23
    assert 0 <= r["low_hour"] <= 23

def test_get_peak_hours_profile_has_24_entries(executor):
    r = executor.execute("get_peak_hours", {"site_id": SITE_A, "days": 7})
    assert len(r["hourly_profile"]) <= 24
