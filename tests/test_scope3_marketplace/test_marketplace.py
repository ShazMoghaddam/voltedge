"""Tests for the energy data marketplace."""
from __future__ import annotations
import pytest
from voltedge.marketplace.exchange import (
    BenchmarkMetric, BenchmarkSector, DataContribution,
    EnergyMarketplace, MIN_COHORT_SIZE,
)

SITE = "LONDON-FACTORY-01"


@pytest.fixture
def market():
    return EnergyMarketplace(seed=42)


# ── DataContribution ──────────────────────────────────────────────────────────

def test_contribution_from_site_hashes_id():
    c = DataContribution.from_site(
        SITE, BenchmarkSector.UK_MANUFACTURING,
        BenchmarkMetric.CARBON_INTENSITY, 0.207, period_days=30,
    )
    assert len(c.site_hash) == 16
    assert SITE not in c.site_hash   # hash is not reversible


def test_contribution_credits_scale_with_days():
    c30 = DataContribution.from_site(
        SITE, BenchmarkSector.UK_MANUFACTURING,
        BenchmarkMetric.CARBON_INTENSITY, 0.207, period_days=30,
    )
    c90 = DataContribution.from_site(
        SITE, BenchmarkSector.UK_MANUFACTURING,
        BenchmarkMetric.CARBON_INTENSITY, 0.207, period_days=90,
    )
    assert c90.credits_earned > c30.credits_earned


def test_contribution_credits_positive():
    c = DataContribution.from_site(
        SITE, BenchmarkSector.UK_MANUFACTURING,
        BenchmarkMetric.CARBON_INTENSITY, 0.207,
    )
    assert c.credits_earned > 0


# ── EnergyMarketplace — contribute ────────────────────────────────────────────

def test_contribute_returns_credits(market):
    c = DataContribution.from_site(
        SITE, BenchmarkSector.UK_MANUFACTURING,
        BenchmarkMetric.CARBON_INTENSITY, 0.210,
    )
    credits = market.contribute(c)
    assert credits > 0


def test_contribute_accrues_credits(market):
    c = DataContribution.from_site(
        SITE, BenchmarkSector.UK_MANUFACTURING,
        BenchmarkMetric.CARBON_INTENSITY, 0.210,
    )
    market.contribute(c)
    assert market.get_credits(SITE) > 0


def test_multiple_contributions_accumulate_credits(market):
    for i in range(3):
        c = DataContribution.from_site(
            f"SITE-{i}", BenchmarkSector.UK_MANUFACTURING,
            BenchmarkMetric.CARBON_INTENSITY, 0.200 + i * 0.01,
        )
        market.contribute(c)
    for i in range(3):
        assert market.get_credits(f"SITE-{i}") > 0


def test_get_credits_zero_for_nonparticipant(market):
    assert market.get_credits("NEVER-CONTRIBUTED") == 0.0


# ── EnergyMarketplace — benchmarks ───────────────────────────────────────────

def test_benchmark_available_after_seed(market):
    bm = market.get_benchmark(
        BenchmarkSector.UK_MANUFACTURING,
        BenchmarkMetric.CARBON_INTENSITY,
    )
    assert bm is not None


def test_benchmark_has_correct_percentile_order(market):
    bm = market.get_benchmark(
        BenchmarkSector.UK_MANUFACTURING,
        BenchmarkMetric.CARBON_INTENSITY,
    )
    assert bm.percentile_10 <= bm.percentile_25 <= bm.median
    assert bm.median <= bm.percentile_75 <= bm.percentile_90


def test_benchmark_cohort_size_at_least_min(market):
    bm = market.get_benchmark(
        BenchmarkSector.UK_MANUFACTURING,
        BenchmarkMetric.CARBON_INTENSITY,
    )
    assert bm.cohort_size >= MIN_COHORT_SIZE


def test_benchmark_returns_none_for_small_cohort(market):
    bm = market.get_benchmark(
        BenchmarkSector.COLD_CHAIN,
        BenchmarkMetric.ANOMALY_RATE_PCT if hasattr(BenchmarkMetric, "ANOMALY_RATE_PCT")
        else BenchmarkMetric.PEAK_DEMAND_KW,
    )
    assert bm is None   # No contributions for cold chain


def test_benchmark_to_dict_keys(market):
    bm = market.get_benchmark(
        BenchmarkSector.UK_MANUFACTURING,
        BenchmarkMetric.CARBON_INTENSITY,
    )
    d = bm.to_dict()
    for k in ("sector", "metric", "cohort_size", "median", "mean", "best_in_class"):
        assert k in d


# ── EnergyMarketplace — rank_site ─────────────────────────────────────────────

def test_rank_site_below_median_good(market):
    result = market.rank_site(
        SITE, 0.190,   # Better than typical UK manufacturing (~0.21)
        BenchmarkSector.UK_MANUFACTURING,
        BenchmarkMetric.CARBON_INTENSITY,
        lower_is_better=True,
    )
    assert result["percentile_rank"] < 50   # In top half


def test_rank_site_above_median_poor(market):
    result = market.rank_site(
        SITE, 0.250,   # Worse than median
        BenchmarkSector.UK_MANUFACTURING,
        BenchmarkMetric.CARBON_INTENSITY,
        lower_is_better=True,
    )
    assert result["percentile_rank"] > 50


def test_rank_site_has_tier(market):
    result = market.rank_site(
        SITE, 0.185,
        BenchmarkSector.UK_MANUFACTURING,
        BenchmarkMetric.CARBON_INTENSITY,
    )
    assert result["tier"] in ("top_10", "top_25", "median", "bottom_25", "bottom_10")


def test_rank_site_no_cohort_returns_message(market):
    result = market.rank_site(
        SITE, 100.0,
        BenchmarkSector.CHEMICAL_PROCESSING,
        BenchmarkMetric.ANOMALY_RATE_PCT if hasattr(BenchmarkMetric, "ANOMALY_RATE_PCT")
        else BenchmarkMetric.ENERGY_COST_PER_KWH,
    )
    assert result["rank"] is None
    assert "message" in result


def test_rank_site_vs_median_pct(market):
    bm = market.get_benchmark(
        BenchmarkSector.UK_MANUFACTURING,
        BenchmarkMetric.CARBON_INTENSITY,
    )
    result = market.rank_site(
        SITE, bm.median,
        BenchmarkSector.UK_MANUFACTURING,
        BenchmarkMetric.CARBON_INTENSITY,
    )
    assert abs(result["vs_median_pct"]) < 1.0   # At median → ~0%


# ── available_benchmarks ──────────────────────────────────────────────────────

def test_available_benchmarks_structure(market):
    benchmarks = market.available_benchmarks()
    assert isinstance(benchmarks, list)
    for bm in benchmarks:
        for k in ("sector", "metric", "cohort_size", "available"):
            assert k in bm


# ── Percentile rank ───────────────────────────────────────────────────────────

def test_percentile_rank_0_is_best(market):
    bm = market.get_benchmark(
        BenchmarkSector.UK_MANUFACTURING,
        BenchmarkMetric.CARBON_INTENSITY,
    )
    rank = bm.percentile_rank(bm.percentile_10, lower_is_better=True)
    assert rank <= 10.0


def test_percentile_rank_100_is_worst(market):
    bm = market.get_benchmark(
        BenchmarkSector.UK_MANUFACTURING,
        BenchmarkMetric.CARBON_INTENSITY,
    )
    rank = bm.percentile_rank(bm.percentile_90, lower_is_better=True)
    assert rank >= 90.0


def test_percentile_rank_in_0_100_range(market):
    bm = market.get_benchmark(
        BenchmarkSector.UK_MANUFACTURING,
        BenchmarkMetric.CARBON_INTENSITY,
    )
    for v in [0.1, 0.2, 0.3, 0.5]:
        rank = bm.percentile_rank(v)
        assert 0 <= rank <= 100
