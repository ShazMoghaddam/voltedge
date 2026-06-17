"""Tests for Scope 3 calculator."""
from __future__ import annotations
from datetime import datetime, timezone
import pytest
from voltedge.scope3.calculator import (
    EMISSION_FACTORS, Scope3Activity, Scope3Calculator, Scope3Category, Scope3Report,
)

ENTITY = "SHELL-UK-UPSTREAM"
NOW = datetime.now(timezone.utc)


@pytest.fixture
def calc():
    return Scope3Calculator(ENTITY)


# ── Scope3Activity ────────────────────────────────────────────────────────────

def test_activity_tco2e_calculation():
    a = Scope3Activity(
        category=Scope3Category.UPSTREAM_TRANSPORT,
        description="Road freight",
        quantity=10_000.0,    # tonne-km
        unit="tonne-km",
        emission_factor=0.0762,   # kg CO₂e/tonne-km
    )
    assert a.tco2e == pytest.approx(0.762, rel=0.01)


def test_activity_tco2e_zero_quantity():
    a = Scope3Activity(
        category=Scope3Category.BUSINESS_TRAVEL,
        description="Nil travel",
        quantity=0.0, unit="km",
        emission_factor=0.25,
    )
    assert a.tco2e == 0.0


# ── Scope3Calculator ──────────────────────────────────────────────────────────

def test_add_transport_returns_self(calc):
    result = calc.add_transport("road_transport_hgv", 50_000.0)
    assert result is calc   # fluent interface


def test_add_transport_adds_activity(calc):
    calc.add_transport("road_transport_hgv", 100_000.0)
    report = calc.generate_report()
    assert len(report.activities) == 1


def test_add_business_travel_passengers(calc):
    calc.add_business_travel("flight_long_haul", km=10_000.0, passengers=5)
    report = calc.generate_report()
    # 5 passengers × 10000km × 0.1951 / 1000 = 9.755 tCO₂e
    assert report.total_tco2e == pytest.approx(9.755, rel=0.01)


def test_add_waste(calc):
    calc.add_waste("landfill_mixed", tonnes=2.0)
    report = calc.generate_report()
    # 2t × 467 kg/t / 1000 = 0.934 tCO₂e
    assert report.total_tco2e == pytest.approx(0.934, rel=0.01)


def test_add_purchased_material(calc):
    calc.add_purchased_material("steel", kg=1000.0)
    report = calc.generate_report()
    # 1000 kg × 1.85 kg/kg / 1000 = 1.85 tCO₂e
    assert report.total_tco2e == pytest.approx(1.85, rel=0.01)


def test_add_upstream_energy(calc):
    calc.add_upstream_energy("grid_electricity", kwh=100_000.0)
    report = calc.generate_report()
    assert report.total_tco2e > 0


def test_add_custom_activity(calc):
    a = Scope3Activity(
        category=Scope3Category.INVESTMENTS,
        description="Portfolio company",
        quantity=500.0,
        unit="tonne-km",
        emission_factor=0.05,
    )
    calc.add_custom_activity(a)
    report = calc.generate_report()
    assert len(report.activities) == 1


def test_fluent_chaining(calc):
    report = (
        calc
        .add_transport("road_transport_hgv", 100_000.0)
        .add_business_travel("flight_long_haul", 20_000.0)
        .add_waste("landfill_mixed", 5.0)
        .generate_report()
    )
    assert len(report.activities) == 3
    assert report.total_tco2e > 0


def test_factor_lookup(calc):
    assert calc.factor("road_transport_hgv") == EMISSION_FACTORS["road_transport_hgv"]


def test_factor_missing_returns_none(calc):
    assert calc.factor("non_existent_mode") is None


def test_available_factors_sorted(calc):
    factors = calc.available_factors()
    assert factors == sorted(factors)


# ── Scope3Report ──────────────────────────────────────────────────────────────

def test_report_by_category_groups_correctly():
    calc = Scope3Calculator(ENTITY)
    calc.add_transport("road_transport_hgv", 100_000.0)
    calc.add_transport("sea_freight", 500_000.0)
    report = calc.generate_report()
    by_cat = report.by_category
    # Both are upstream transport
    assert len(by_cat) == 1
    assert "cat_04" in list(by_cat.keys())[0]


def test_report_top_categories_ordered():
    calc = Scope3Calculator(ENTITY)
    calc.add_transport("road_transport_hgv", 100_000.0)
    calc.add_business_travel("flight_long_haul", 50_000.0, passengers=10)
    calc.add_waste("landfill_mixed", 50.0)
    report = calc.generate_report()
    top = report.top_categories
    values = [v for _, v in top]
    assert values == sorted(values, reverse=True)


def test_report_to_dict_keys():
    calc = Scope3Calculator(ENTITY)
    calc.add_transport("road_transport_hgv", 100_000.0)
    report = calc.generate_report()
    d = report.to_dict()
    for k in ("entity_id", "total_tco2e", "by_category", "activity_count", "data_quality_mix"):
        assert k in d


def test_report_data_quality_mix():
    calc = Scope3Calculator(ENTITY)
    calc.add_transport("road_transport_hgv", 100_000.0, data_quality="calculated")
    calc.add_waste("landfill_mixed", 5.0, data_quality="estimated")
    report = calc.generate_report()
    mix = report.to_dict()["data_quality_mix"]
    assert mix["calculated"] == 1
    assert mix["estimated"]  == 1
    assert mix["measured"]   == 0


def test_report_total_is_sum_of_activities():
    calc = Scope3Calculator(ENTITY)
    calc.add_transport("road_transport_hgv", 50_000.0)
    calc.add_waste("landfill_mixed", 3.0)
    report = calc.generate_report()
    expected = sum(a.tco2e for a in report.activities)
    assert report.total_tco2e == pytest.approx(expected, rel=0.001)


def test_custom_emission_factors():
    calc = Scope3Calculator(ENTITY, custom_factors={"my_mode": 0.999})
    val = calc.factor("my_mode")
    assert val == 0.999
