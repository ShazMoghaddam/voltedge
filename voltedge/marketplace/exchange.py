"""
VoltEdge Energy Data Marketplace

A privacy-preserving energy benchmarking exchange where enterprises
share anonymised energy data to gain:

  - Sector benchmarks (how does my factory compare to peers?)
  - Best-practice insights (who achieves best kWh/unit output?)
  - Carbon intensity rankings
  - Demand flexibility benchmarking

Privacy guarantees:
  - All data is k-anonymised (minimum 5 contributors per cohort)
  - Differential privacy noise added to prevent re-identification
  - No raw site data is ever shared — only aggregated statistics
  - Participation is opt-in per site per metric

Revenue model:
  Participants earn "marketplace credits" for contributing data.
  Credits can be redeemed for premium benchmark reports or
  converted to account discounts.

Benchmark cohorts:
  UK manufacturing sites > 1MW peak demand
  European data centres (PUE benchmarking)
  Oil & gas upstream facilities (global)
  Office buildings (kWh/m²)

Data contributed (anonymised):
  - Hourly kWh consumption (smoothed)
  - Peak demand kW
  - Carbon intensity
  - Anomaly rate
  - Load factor (avg/peak ratio)
"""

from __future__ import annotations

import hashlib
import math
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class BenchmarkSector(str, Enum):
    UK_MANUFACTURING    = "uk_manufacturing"
    EU_DATA_CENTRES     = "eu_data_centres"
    OIL_GAS_UPSTREAM    = "oil_gas_upstream"
    OFFICE_BUILDINGS    = "office_buildings"
    COLD_CHAIN          = "cold_chain"
    CHEMICAL_PROCESSING = "chemical_processing"


class BenchmarkMetric(str, Enum):
    KWH_PER_UNIT_OUTPUT = "kwh_per_unit_output"
    CARBON_INTENSITY    = "carbon_intensity_kgco2_per_kwh"
    PEAK_DEMAND_KW      = "peak_demand_kw"
    LOAD_FACTOR         = "load_factor"
    ANOMALY_RATE        = "anomaly_rate_pct"
    ENERGY_COST_PER_KWH = "energy_cost_per_kwh"


# Minimum contributors for a cohort to be published (k-anonymity)
MIN_COHORT_SIZE = 5

# Differential privacy noise scale (Laplace mechanism)
DP_EPSILON = 1.0


@dataclass
class DataContribution:
    """A single site's anonymised contribution to a benchmark cohort."""
    contribution_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    site_hash:       str  = ""     # SHA-256 of site_id (not reversible)
    sector:          BenchmarkSector = BenchmarkSector.UK_MANUFACTURING
    metric:          BenchmarkMetric = BenchmarkMetric.KWH_PER_UNIT_OUTPUT
    value:           float = 0.0
    period_days:     int   = 30
    contributed_at:  datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    credits_earned:  float = 1.0

    @classmethod
    def from_site(
        cls,
        site_id: str,
        sector:  BenchmarkSector,
        metric:  BenchmarkMetric,
        value:   float,
        period_days: int = 30,
    ) -> "DataContribution":
        site_hash = hashlib.sha256(site_id.encode()).hexdigest()[:16]
        credits = 1.0 + (period_days / 30) * 0.5   # More data → more credits
        return cls(
            site_hash=site_hash,
            sector=sector,
            metric=metric,
            value=value,
            period_days=period_days,
            credits_earned=round(credits, 2),
        )


@dataclass
class BenchmarkResult:
    """Aggregated benchmark statistics for a cohort."""
    sector:          BenchmarkSector
    metric:          BenchmarkMetric
    cohort_size:     int
    percentile_10:   float
    percentile_25:   float
    median:          float
    percentile_75:   float
    percentile_90:   float
    mean:            float
    best_in_class:   float
    computed_at:     datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def percentile_rank(self, site_value: float, lower_is_better: bool = True) -> float:
        """
        Return where site_value sits in the benchmark distribution (0–100).
        0 = best, 100 = worst (when lower_is_better=True).
        """
        if self.percentile_90 == self.percentile_10:
            return 50.0
        rank = (site_value - self.percentile_10) / (self.percentile_90 - self.percentile_10) * 100
        rank = max(0.0, min(100.0, rank))
        return round(rank if lower_is_better else 100 - rank, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sector":        self.sector.value,
            "metric":        self.metric.value,
            "cohort_size":   self.cohort_size,
            "percentile_10": self.percentile_10,
            "percentile_25": self.percentile_25,
            "median":        self.median,
            "percentile_75": self.percentile_75,
            "percentile_90": self.percentile_90,
            "mean":          self.mean,
            "best_in_class": self.best_in_class,
            "computed_at":   self.computed_at.isoformat(),
        }


class EnergyMarketplace:
    """
    Privacy-preserving energy data exchange.

    In production: backed by a distributed DB with zero-knowledge proofs.
    For v7.0: in-memory with differential privacy noise.
    """

    def __init__(self, dp_epsilon: float = DP_EPSILON, seed: int = 42) -> None:
        self._epsilon = dp_epsilon
        self._rng     = random.Random(seed)
        # {(sector, metric): [DataContribution]}
        self._contributions: dict[tuple, list[DataContribution]] = {}
        self._credits: dict[str, float] = {}   # {site_hash: credits}
        self._seed_synthetic_data()

    # ── Participation ─────────────────────────────────────────────────────────

    def contribute(self, contribution: DataContribution) -> float:
        """
        Submit a data contribution. Returns credits earned.
        Applies differential privacy noise before storing.
        """
        # Add Laplace DP noise to protect individual values
        noisy_value = contribution.value + self._laplace_noise(
            sensitivity=contribution.value * 0.05,   # 5% sensitivity
            epsilon=self._epsilon,
        )
        from dataclasses import replace
        noisy_contrib = replace(contribution, value=max(0.0, noisy_value))

        key = (contribution.sector, contribution.metric)
        self._contributions.setdefault(key, []).append(noisy_contrib)

        # Accrue credits
        self._credits[contribution.site_hash] = (
            self._credits.get(contribution.site_hash, 0.0) + contribution.credits_earned
        )
        return contribution.credits_earned

    def get_credits(self, site_id: str) -> float:
        site_hash = hashlib.sha256(site_id.encode()).hexdigest()[:16]
        return self._credits.get(site_hash, 0.0)

    # ── Benchmark queries ─────────────────────────────────────────────────────

    def get_benchmark(
        self,
        sector: BenchmarkSector,
        metric: BenchmarkMetric,
    ) -> BenchmarkResult | None:
        """
        Return benchmark statistics for a cohort.
        Returns None if cohort_size < MIN_COHORT_SIZE (k-anonymity).
        """
        key     = (sector, metric)
        contribs = self._contributions.get(key, [])
        if len(contribs) < MIN_COHORT_SIZE:
            return None

        values = sorted(c.value for c in contribs)
        n      = len(values)

        def pct(p):
            i = (p / 100) * (n - 1)
            lo, hi = int(i), min(int(i) + 1, n - 1)
            return round(values[lo] + (i - lo) * (values[hi] - values[lo]), 4)

        return BenchmarkResult(
            sector=sector,
            metric=metric,
            cohort_size=n,
            percentile_10=pct(10),
            percentile_25=pct(25),
            median=pct(50),
            percentile_75=pct(75),
            percentile_90=pct(90),
            mean=round(sum(values) / n, 4),
            best_in_class=values[0],
        )

    def rank_site(
        self,
        site_id: str,
        site_value: float,
        sector:  BenchmarkSector,
        metric:  BenchmarkMetric,
        lower_is_better: bool = True,
    ) -> dict[str, Any]:
        """
        Show how a site compares to the benchmark cohort.
        """
        benchmark = self.get_benchmark(sector, metric)
        if benchmark is None:
            return {
                "site_id":     site_id,
                "site_value":  site_value,
                "rank":        None,
                "message":     f"Cohort too small (< {MIN_COHORT_SIZE}). Contribute data to unlock.",
            }

        rank = benchmark.percentile_rank(site_value, lower_is_better)
        tier = (
            "top_10"    if rank <= 10  else
            "top_25"    if rank <= 25  else
            "median"    if rank <= 60  else
            "bottom_25" if rank <= 75  else
            "bottom_10"
        )

        return {
            "site_id":       site_id,
            "sector":        sector.value,
            "metric":        metric.value,
            "site_value":    site_value,
            "percentile_rank": rank,
            "tier":          tier,
            "cohort_size":   benchmark.cohort_size,
            "benchmark": {
                "p10":    benchmark.percentile_10,
                "median": benchmark.median,
                "p90":    benchmark.percentile_90,
                "best":   benchmark.best_in_class,
            },
            "vs_median_pct": round((site_value - benchmark.median) / (benchmark.median + 1e-8) * 100, 1),
        }

    def available_benchmarks(self) -> list[dict[str, Any]]:
        return [
            {
                "sector":       k[0].value,
                "metric":       k[1].value,
                "cohort_size":  len(v),
                "available":    len(v) >= MIN_COHORT_SIZE,
            }
            for k, v in self._contributions.items()
        ]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _laplace_noise(self, sensitivity: float, epsilon: float) -> float:
        """Sample Laplace noise for differential privacy."""
        scale = sensitivity / (epsilon + 1e-8)
        u = self._rng.uniform(-0.5, 0.5)
        return -scale * math.copysign(1, u) * math.log(1 - 2 * abs(u))

    def _seed_synthetic_data(self) -> None:
        """
        Seed with synthetic peer data so benchmarks are available immediately.
        Real deployments replace this with actual contributor data.
        """
        # UK manufacturing: carbon intensity (kg CO₂/kWh)
        for val in [0.185, 0.198, 0.210, 0.225, 0.240, 0.195, 0.230, 0.215, 0.205, 0.220]:
            self._contributions.setdefault(
                (BenchmarkSector.UK_MANUFACTURING, BenchmarkMetric.CARBON_INTENSITY), []
            ).append(DataContribution(
                site_hash=f"seed_{val:.3f}",
                sector=BenchmarkSector.UK_MANUFACTURING,
                metric=BenchmarkMetric.CARBON_INTENSITY,
                value=val,
            ))

        # EU data centres: PUE (load factor proxy)
        for val in [0.85, 0.78, 0.82, 0.90, 0.75, 0.88, 0.71, 0.93, 0.80, 0.77]:
            self._contributions.setdefault(
                (BenchmarkSector.EU_DATA_CENTRES, BenchmarkMetric.LOAD_FACTOR), []
            ).append(DataContribution(
                site_hash=f"seed_{val:.3f}",
                sector=BenchmarkSector.EU_DATA_CENTRES,
                metric=BenchmarkMetric.LOAD_FACTOR,
                value=val,
            ))
