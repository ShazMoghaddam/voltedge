"""Tests for root-cause analyser."""
from __future__ import annotations
import asyncio
from datetime import datetime, timedelta, timezone
import pytest
import pandas as pd

from voltedge.causal.graph import CausalGraphBuilder
from voltedge.causal.root_cause import (
    CausalFactor, RootCauseAnalyser, RootCauseReport,
)
from voltedge.ingestion.simulators import SimulatedSiteConnector
from voltedge.processing.transformer import EnergyTransformer

SITE = "RC-TEST-01"


@pytest.fixture(scope="module")
def df_and_graph():
    connector = SimulatedSiteConnector(SITE, "factory", hours=400, seed=7)
    result = asyncio.run(connector.fetch())
    df = EnergyTransformer().transform(result.data)
    graph = CausalGraphBuilder(alpha=0.05, min_samples=48).build(df, SITE)
    return df, graph


@pytest.fixture(scope="module")
def analyser(df_and_graph):
    _, graph = df_and_graph
    return RootCauseAnalyser(graph, max_depth=3, min_contribution=0.0)


@pytest.fixture(scope="module")
def report(df_and_graph, analyser):
    df, _ = df_and_graph
    ts = df["timestamp"].iloc[200]
    return analyser.analyse(df, ts, anomaly_variable="kwh")


# ── CausalFactor ──────────────────────────────────────────────────────────────

def test_factor_to_dict_keys():
    f = CausalFactor(
        variable="temperature_c", causal_strength=0.6,
        deviation_z=2.1, contribution_score=1.26,
        direction="increase",
    )
    d = f.to_dict()
    for k in ("variable", "causal_strength", "deviation_z",
              "contribution_score", "direction", "depth"):
        assert k in d


# ── RootCauseReport ───────────────────────────────────────────────────────────

def test_report_has_site_id(report):
    assert report.site_id == SITE


def test_report_has_timestamp(report):
    assert len(report.anomaly_timestamp) > 0


def test_report_has_narrative(report):
    assert isinstance(report.narrative, str)
    assert len(report.narrative) > 0


def test_report_has_recommendations(report):
    assert isinstance(report.recommendations, list)
    assert len(report.recommendations) >= 1


def test_report_confidence_valid(report):
    assert report.confidence in ("high", "medium", "low")


def test_report_to_dict_keys(report):
    d = report.to_dict()
    for k in ("site_id", "anomaly_timestamp", "narrative",
              "causal_factors", "confidence", "recommendations"):
        assert k in d


def test_report_causal_factors_sorted(report):
    scores = [f.contribution_score for f in report.causal_factors]
    assert scores == sorted(scores, reverse=True)


def test_report_primary_cause(report):
    if report.causal_factors:
        assert report.primary_cause == report.causal_factors[0]
    else:
        assert report.primary_cause is None


def test_report_top_3_causes(report):
    assert len(report.top_3_causes) <= 3


def test_report_deviation_pct_numeric(report):
    assert isinstance(report.deviation_pct, float)


# ── Analyser ──────────────────────────────────────────────────────────────────

def test_analyser_handles_missing_timestamp(df_and_graph, analyser):
    df, _ = df_and_graph
    # A timestamp that doesn't exist — should find the closest
    fake_ts = "2020-01-01T00:00:00+00:00"
    report = analyser.analyse(df, fake_ts)
    assert report.site_id == SITE


def test_analyser_narrative_contains_site(report):
    assert SITE in report.narrative


def test_analyser_narrative_contains_variable(report):
    assert "kwh" in report.narrative.lower()


def test_analyse_period_returns_list(df_and_graph, analyser):
    df, _ = df_and_graph
    start = df["timestamp"].iloc[50]
    end   = df["timestamp"].iloc[250]
    reports = analyser.analyse_period(df, start, end, top_n=3)
    assert isinstance(reports, list)
    assert len(reports) <= 3


def test_analyse_period_all_correct_site(df_and_graph, analyser):
    df, _ = df_and_graph
    start = df["timestamp"].iloc[50]
    end   = df["timestamp"].iloc[200]
    reports = analyser.analyse_period(df, start, end, top_n=5)
    assert all(r.site_id == SITE for r in reports)


# ── Confidence assessment ─────────────────────────────────────────────────────

def test_low_confidence_few_samples():
    c = RootCauseAnalyser._assess_confidence([], n_samples=10)
    assert c == "low"


def test_high_confidence_strong_factor():
    f = CausalFactor("x", 0.8, 3.0, 2.4, "increase")
    c = RootCauseAnalyser._assess_confidence([f], n_samples=500)
    assert c == "high"


# ── Recommendations ───────────────────────────────────────────────────────────

def test_recommendations_flat_line():
    recs = RootCauseAnalyser._build_recommendations([], "flat_line_sensor")
    assert any("sensor" in r.lower() or "meter" in r.lower() for r in recs)


def test_recommendations_sudden_spike():
    recs = RootCauseAnalyser._build_recommendations([], "sudden_spike")
    assert any("equipment" in r.lower() or "start" in r.lower() for r in recs)


def test_recommendations_not_empty_no_label():
    recs = RootCauseAnalyser._build_recommendations([], "")
    assert len(recs) >= 1
