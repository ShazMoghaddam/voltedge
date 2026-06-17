"""Tests for the energy digital twin."""
from __future__ import annotations
import asyncio
import pytest

from voltedge.causal.graph import CausalGraphBuilder
from voltedge.causal.digital_twin import (
    EnergyDigitalTwin, Scenario, ScenarioComparison,
    ScenarioParameter, TwinSimulationResult,
)
from voltedge.ingestion.simulators import SimulatedSiteConnector
from voltedge.processing.transformer import EnergyTransformer

SITE = "TWIN-TEST-01"
SIM_HOURS = 168   # 1 week — fast for tests


@pytest.fixture(scope="module")
def twin():
    connector = SimulatedSiteConnector(SITE, "factory", hours=400, seed=13)
    result = asyncio.run(connector.fetch())
    df = EnergyTransformer().transform(result.data)
    graph = CausalGraphBuilder(alpha=0.05, min_samples=48).build(df, SITE)
    return EnergyDigitalTwin(
        graph=graph, df=df,
        n_simulations=10,   # Fast for tests
        seed=42,
    )


@pytest.fixture(scope="module")
def baseline(twin):
    return twin.run_baseline(hours=SIM_HOURS)


# ── TwinSimulationResult ──────────────────────────────────────────────────────

def test_baseline_returns_result(baseline):
    assert isinstance(baseline, TwinSimulationResult)


def test_baseline_scenario_name(baseline):
    assert "Baseline" in baseline.scenario_name


def test_baseline_correct_hours(baseline):
    assert baseline.simulation_hours == SIM_HOURS
    assert len(baseline.kwh_mean) == SIM_HOURS


def test_baseline_kwh_non_negative(baseline):
    assert all(v >= 0 for v in baseline.kwh_mean)


def test_baseline_p10_le_mean_le_p90(baseline):
    for lo, mid, hi in zip(baseline.kwh_p10, baseline.kwh_mean, baseline.kwh_p90):
        assert lo <= mid + 0.01
        assert mid <= hi + 0.01


def test_baseline_timestamps_length(baseline):
    assert len(baseline.timestamps) == SIM_HOURS


def test_baseline_total_kwh_positive(baseline):
    assert baseline.total_kwh > 0


def test_baseline_peak_ge_avg(baseline):
    assert baseline.peak_kw >= baseline.avg_kw - 0.01


def test_baseline_to_dict_keys(baseline):
    d = baseline.to_dict()
    for k in ("scenario", "total_kwh", "peak_kw", "avg_kw",
              "total_co2_kg", "total_cost", "hourly_preview"):
        assert k in d


def test_baseline_hourly_preview_48h(baseline):
    preview = baseline.to_dict()["hourly_preview"]
    assert len(preview["timestamps"]) == 48
    assert len(preview["kwh_mean"]) == 48


# ── Scenarios ─────────────────────────────────────────────────────────────────

def test_run_solar_scenario(twin):
    scenario = EnergyDigitalTwin.solar_scenario(capacity_kwp=100)
    result = twin.run_scenario(scenario, hours=SIM_HOURS)
    assert result.total_kwh < twin.run_baseline(hours=SIM_HOURS).total_kwh + 1


def test_run_bess_scenario(twin):
    scenario = EnergyDigitalTwin.bess_scenario(500, 200)
    result = twin.run_scenario(scenario, hours=SIM_HOURS)
    assert isinstance(result, TwinSimulationResult)
    assert result.total_kwh >= 0


def test_run_ev_fleet_scenario(twin):
    scenario = EnergyDigitalTwin.ev_fleet_scenario(vehicles=10)
    result = twin.run_scenario(scenario, hours=SIM_HOURS)
    assert result.total_kwh > 0


def test_run_night_shift_scenario(twin):
    scenario = EnergyDigitalTwin.night_shift_scenario()
    result = twin.run_scenario(scenario, hours=SIM_HOURS)
    assert isinstance(result, TwinSimulationResult)


def test_custom_scenario(twin):
    scenario = Scenario(
        name="Weekend only",
        description="Run only on weekends"
    ).add_parameter("is_business_hour", 0.0)
    result = twin.run_scenario(scenario, hours=SIM_HOURS)
    assert result.total_kwh >= 0


# ── compare_scenarios ─────────────────────────────────────────────────────────

def test_compare_scenarios_returns_comparison(twin):
    scenarios = [
        EnergyDigitalTwin.solar_scenario(50),
        EnergyDigitalTwin.bess_scenario(200, 100),
    ]
    comparison = twin.compare_scenarios(scenarios, hours=SIM_HOURS)
    assert isinstance(comparison, ScenarioComparison)


def test_compare_has_baseline(twin):
    comparison = twin.compare_scenarios(
        [EnergyDigitalTwin.solar_scenario(50)], hours=SIM_HOURS
    )
    assert comparison.baseline.scenario_name == "Baseline (no intervention)"


def test_compare_ranked_by_saving(twin):
    comparison = twin.compare_scenarios(
        [EnergyDigitalTwin.solar_scenario(100),
         EnergyDigitalTwin.ev_fleet_scenario(5)],
        hours=SIM_HOURS,
    )
    ranked = comparison.ranked_by_saving()
    assert len(ranked) == 2
    for item in ranked:
        assert "delta_kwh" in item
        assert "delta_pct" in item


def test_compare_best_scenario_name(twin):
    comparison = twin.compare_scenarios(
        [EnergyDigitalTwin.solar_scenario(200)], hours=SIM_HOURS
    )
    assert comparison.best_scenario is not None
    assert isinstance(comparison.best_scenario.scenario_name, str)


def test_compare_to_dict_keys(twin):
    comparison = twin.compare_scenarios(
        [EnergyDigitalTwin.solar_scenario(50)], hours=SIM_HOURS
    )
    d = comparison.to_dict()
    assert "baseline" in d
    assert "scenarios" in d
    assert "ranked_by_saving" in d


# ── Scenario templates ────────────────────────────────────────────────────────

def test_solar_scenario_has_capex():
    s = EnergyDigitalTwin.solar_scenario(100)
    assert s.capex_gbp > 0


def test_bess_scenario_has_capex():
    s = EnergyDigitalTwin.bess_scenario(500, 200)
    assert s.capex_gbp > 0


def test_ev_scenario_has_capex():
    s = EnergyDigitalTwin.ev_fleet_scenario(10)
    assert s.capex_gbp > 0


def test_night_shift_no_capex():
    s = EnergyDigitalTwin.night_shift_scenario()
    assert s.capex_gbp == 0.0


def test_scenario_add_parameter_fluent():
    s = Scenario("Test").add_parameter("kwh", 100.0)
    assert len(s.parameters) == 1
    assert s.parameters[0].variable == "kwh"
