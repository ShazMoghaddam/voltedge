"""Tests for CostOptimizer."""

from __future__ import annotations

import asyncio
import pytest

from voltedge.ingestion.simulators import SimulatedSiteConnector
from voltedge.models.optimization import CostOptimizer, TARIFFS
from voltedge.processing.transformer import EnergyTransformer


def _get_df(hours: int = 720):
    connector = SimulatedSiteConnector("OPT-TEST", "factory", hours=hours, seed=42)
    result = asyncio.run(connector.fetch())
    return EnergyTransformer().transform(result.data)


def test_optimizer_returns_report():
    df = _get_df()
    optimizer = CostOptimizer(site_id="OPT-TEST", tariff="UK_HALF_HOURLY")
    report = optimizer.analyse(df)
    assert report is not None
    assert report.site_id == "OPT-TEST"


def test_optimizer_has_recommendations():
    df = _get_df()
    optimizer = CostOptimizer(site_id="OPT-TEST")
    report = optimizer.analyse(df)
    assert len(report.recommendations) > 0


def test_optimizer_positive_savings():
    df = _get_df()
    optimizer = CostOptimizer(site_id="OPT-TEST")
    report = optimizer.analyse(df)
    for rec in report.recommendations:
        assert rec.estimated_annual_saving >= 0, f"Negative saving: {rec}"


def test_optimizer_recommendations_sorted_desc():
    df = _get_df()
    report = CostOptimizer("OPT-TEST").analyse(df)
    savings = [r.estimated_annual_saving for r in report.recommendations]
    assert savings == sorted(savings, reverse=True)


def test_optimizer_current_cost_positive():
    df = _get_df()
    report = CostOptimizer("OPT-TEST").analyse(df)
    assert report.current_annual_cost > 0


def test_optimizer_optimised_cost_less_than_current():
    df = _get_df()
    report = CostOptimizer("OPT-TEST").analyse(df)
    assert report.optimised_annual_cost <= report.current_annual_cost


def test_optimizer_saving_pct_in_range():
    df = _get_df()
    report = CostOptimizer("OPT-TEST").analyse(df)
    assert 0 <= report.saving_pct <= 100


def test_optimizer_eu_tariff():
    df = _get_df()
    report = CostOptimizer("OPT-TEST", tariff="EU_INDUSTRIAL").analyse(df)
    assert report.currency == "EUR"
    assert len(report.recommendations) > 0


def test_optimizer_us_tariff():
    df = _get_df()
    report = CostOptimizer("OPT-TEST", tariff="US_COMMERCIAL").analyse(df)
    assert report.currency == "USD"


def test_optimizer_peak_shaving_in_recommendations():
    df = _get_df()
    report = CostOptimizer("OPT-TEST").analyse(df)
    categories = [r.category for r in report.recommendations]
    assert "peak_shaving" in categories


def test_optimizer_recommendation_has_description():
    df = _get_df()
    report = CostOptimizer("OPT-TEST").analyse(df)
    for rec in report.recommendations:
        assert len(rec.description) > 10


def test_optimizer_report_total_saving_property():
    df = _get_df()
    report = CostOptimizer("OPT-TEST").analyse(df)
    expected = sum(r.estimated_annual_saving for r in report.recommendations)
    assert abs(report.total_potential_saving - expected) < 0.01
