"""Tests for the counterfactual engine."""
from __future__ import annotations
import asyncio
import pytest
import pandas as pd

from voltedge.causal.graph import CausalGraphBuilder
from voltedge.causal.counterfactual import (
    CounterfactualEngine, CounterfactualResult, Intervention,
)
from voltedge.ingestion.simulators import SimulatedSiteConnector
from voltedge.processing.transformer import EnergyTransformer

SITE = "CF-TEST-01"


@pytest.fixture(scope="module")
def df_and_engine():
    connector = SimulatedSiteConnector(SITE, "factory", hours=400, seed=11)
    result = asyncio.run(connector.fetch())
    df = EnergyTransformer().transform(result.data)
    graph = CausalGraphBuilder(alpha=0.05, min_samples=48).build(df, SITE)
    engine = CounterfactualEngine(graph, tariff_per_kwh=0.28, currency="GBP")
    return df, engine


# ── Intervention ──────────────────────────────────────────────────────────────

def test_intervention_fields():
    iv = Intervention("temperature_c", 20.0, "Cool to 20°C", feasibility="high")
    assert iv.variable == "temperature_c"
    assert iv.new_value == 20.0


def test_temperature_intervention_template():
    iv = CounterfactualEngine.temperature_intervention(18.0, current_temp_c=25.0)
    assert iv.variable == "temperature_c"
    assert iv.new_value == 18.0
    assert "pre-cooling" in iv.description.lower()


def test_shift_schedule_intervention_off_peak():
    iv = CounterfactualEngine.shift_schedule_intervention(business_hours=False)
    assert iv.variable == "is_business_hour"
    assert iv.new_value == 0.0


def test_power_factor_intervention():
    iv = CounterfactualEngine.power_factor_intervention(target_pf=0.95)
    assert iv.variable == "power_factor"
    assert iv.new_value == 0.95
    assert iv.capex_estimate_gbp > 0


# ── compute() ─────────────────────────────────────────────────────────────────

def test_compute_returns_result(df_and_engine):
    df, engine = df_and_engine
    iv = Intervention("is_business_hour", 0.0, "Off-peak shift")
    result = engine.compute(df, iv, target="kwh")
    assert isinstance(result, CounterfactualResult)


def test_compute_baseline_is_mean_kwh(df_and_engine):
    df, engine = df_and_engine
    iv = Intervention("is_business_hour", 0.0)
    result = engine.compute(df, iv)
    expected = float(df["kwh"].mean())
    assert abs(result.baseline_value - expected) < 0.01


def test_compute_unknown_variable_low_confidence(df_and_engine):
    df, engine = df_and_engine
    iv = Intervention("unmeasured_mystery_var", 1.0)
    result = engine.compute(df, iv)
    assert result.confidence == "low"
    assert "not in the causal graph" in result.narrative


def test_compute_result_to_dict_keys(df_and_engine):
    df, engine = df_and_engine
    iv = Intervention("is_business_hour", 0.0)
    result = engine.compute(df, iv)
    d = result.to_dict()
    for k in ("intervention", "baseline_value", "counterfactual_value",
              "delta", "delta_pct", "annual_kwh_impact", "narrative"):
        assert k in d


def test_compute_annual_impacts_numeric(df_and_engine):
    df, engine = df_and_engine
    iv = Intervention("is_business_hour", 0.0)
    result = engine.compute(df, iv)
    assert isinstance(result.annual_kwh_impact, float)
    assert isinstance(result.annual_co2_impact, float)
    assert isinstance(result.annual_cost_impact, float)


def test_compute_confidence_interval_ordered(df_and_engine):
    df, engine = df_and_engine
    iv = Intervention("is_business_hour", 0.0)
    result = engine.compute(df, iv)
    lo, hi = result.confidence_interval
    assert lo <= hi


def test_compute_narrative_non_empty(df_and_engine):
    df, engine = df_and_engine
    iv = Intervention("is_business_hour", 0.0, "Move to off-peak")
    result = engine.compute(df, iv)
    assert len(result.narrative) > 20


def test_compute_empty_df_raises(df_and_engine):
    _, engine = df_and_engine
    iv = Intervention("kwh", 0.0)
    with pytest.raises(ValueError):
        engine.compute(pd.DataFrame(), iv)


def test_compute_is_beneficial_flag(df_and_engine):
    df, engine = df_and_engine
    iv = Intervention("is_business_hour", 0.0)
    result = engine.compute(df, iv)
    assert result.is_beneficial == (result.delta < 0)


# ── multi_intervention() ──────────────────────────────────────────────────────

def test_multi_intervention_structure(df_and_engine):
    df, engine = df_and_engine
    ivs = [
        Intervention("is_business_hour", 0.0, "Off-peak"),
        Intervention("is_weekend", 1.0, "Weekend mode"),
    ]
    result = engine.multi_intervention(df, ivs)
    assert "interventions" in result
    assert "combined_scenario" in result
    assert len(result["interventions"]) == 2


def test_multi_intervention_combined_delta(df_and_engine):
    df, engine = df_and_engine
    ivs = [Intervention("is_business_hour", 0.0)]
    result = engine.multi_intervention(df, ivs)
    combined = result["combined_scenario"]
    assert "total_annual_kwh_saving" in combined
    assert "total_annual_cost_saving" in combined


# ── Confidence assessment ─────────────────────────────────────────────────────

def test_low_confidence_short_df():
    c = CounterfactualEngine._assess_confidence(
        pd.DataFrame({"kwh": [1.0] * 50}), [0, 1, 2], 10.0
    )
    assert c == "low"


def test_low_confidence_long_path():
    c = CounterfactualEngine._assess_confidence(
        pd.DataFrame({"kwh": [1.0] * 500}), list(range(6)), 10.0
    )
    assert c == "low"
