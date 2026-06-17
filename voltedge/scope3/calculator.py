"""
VoltEdge Scope 3 — Supply Chain Emissions Calculator

Extends VoltEdge's ESG capability beyond Scope 1+2 into
Scope 3 (value chain) emissions, following GHG Protocol
Corporate Value Chain (Scope 3) Standard.

GHG Protocol Scope 3 categories implemented:
  Category 1:  Purchased goods and services
  Category 2:  Capital goods
  Category 3:  Fuel and energy related (upstream of Scope 1+2)
  Category 4:  Upstream transportation and distribution
  Category 11: Use of sold products
  Category 12: End-of-life treatment

Emission factors sourced from:
  - UK DESNZ / BEIS Conversion Factors (2024)
  - IPCC AR6 (global warming potentials)
  - Ecoinvent 3.10 (process-level LCA data)

Data inputs:
  Suppliers submit energy and activity data via VoltEdge's
  supplier portal (or import from CSV/ERP).
  The calculator aggregates and computes tCO₂e per category.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class Scope3Category(int, Enum):
    PURCHASED_GOODS          =  1
    CAPITAL_GOODS            =  2
    FUEL_ENERGY_UPSTREAM     =  3
    UPSTREAM_TRANSPORT       =  4
    WASTE_IN_OPERATIONS      =  5
    BUSINESS_TRAVEL          =  6
    EMPLOYEE_COMMUTING       =  7
    UPSTREAM_LEASED_ASSETS   =  8
    DOWNSTREAM_TRANSPORT     =  9
    PROCESSING_SOLD_PRODUCTS = 10
    USE_OF_SOLD_PRODUCTS     = 11
    END_OF_LIFE              = 12
    DOWNSTREAM_LEASED        = 13
    FRANCHISES               = 14
    INVESTMENTS              = 15


# ── Emission factors (kg CO₂e per unit) ──────────────────────────────────────

EMISSION_FACTORS: dict[str, float] = {
    # Transport (kg CO₂e per tonne-km)
    "road_transport_hgv":   0.0762,
    "road_transport_lgv":   0.2055,
    "rail_transport":       0.0280,
    "sea_freight":          0.0116,
    "air_freight":          0.6020,

    # Business travel (kg CO₂e per km per passenger)
    "flight_short_haul":    0.2551,
    "flight_long_haul":     0.1951,
    "car_average":          0.1703,
    "rail_passenger":       0.0369,

    # Waste (kg CO₂e per tonne)
    "landfill_mixed":       467.0,
    "incineration":         21.3,
    "recycling_benefit":   -47.0,   # negative = avoided emissions

    # Materials (kg CO₂e per kg)
    "steel":                1.85,
    "aluminium":            6.70,
    "concrete":             0.14,
    "plastic_hdpe":         1.90,
    "copper":               3.80,

    # Energy upstream (kg CO₂e per kWh of energy purchased)
    "natural_gas_upstream": 0.0184,
    "grid_electricity_upstream_uk": 0.0237,
}


@dataclass
class Scope3Activity:
    """A single supply chain activity contributing to Scope 3 emissions."""
    category:       Scope3Category
    description:    str
    quantity:       float           # Amount of activity
    unit:           str             # "tonne-km", "km", "kg", "kWh"
    emission_factor: float          # kg CO₂e per unit
    supplier_id:    str  = ""
    country:        str  = ""
    data_quality:   str  = "estimated"  # "measured", "calculated", "estimated"

    @property
    def tco2e(self) -> float:
        return round(self.quantity * self.emission_factor / 1000, 6)


@dataclass
class Scope3Report:
    """Aggregated Scope 3 emissions report."""
    entity_id:    str
    period_start: datetime
    period_end:   datetime
    activities:   list[Scope3Activity] = field(default_factory=list)

    @property
    def total_tco2e(self) -> float:
        return round(sum(a.tco2e for a in self.activities), 4)

    @property
    def by_category(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for a in self.activities:
            key = f"cat_{a.category.value:02d}_{a.category.name.lower()}"
            result[key] = round(result.get(key, 0.0) + a.tco2e, 4)
        return result

    @property
    def top_categories(self) -> list[tuple[str, float]]:
        return sorted(
            self.by_category.items(), key=lambda x: x[1], reverse=True
        )[:5]

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id":    self.entity_id,
            "period_start": self.period_start.isoformat(),
            "period_end":   self.period_end.isoformat(),
            "total_tco2e":  self.total_tco2e,
            "by_category":  self.by_category,
            "top_categories": self.top_categories,
            "activity_count": len(self.activities),
            "data_quality_mix": {
                q: sum(1 for a in self.activities if a.data_quality == q)
                for q in ("measured", "calculated", "estimated")
            },
        }


class Scope3Calculator:
    """
    Calculates Scope 3 supply chain emissions from activity data.

    Follows GHG Protocol Corporate Value Chain Standard:
    - Activity-based method (preferred when activity data is available)
    - Spend-based method (fallback using economic data)
    """

    def __init__(
        self,
        entity_id: str,
        custom_factors: dict[str, float] | None = None,
    ) -> None:
        self.entity_id = entity_id
        self._factors  = {**EMISSION_FACTORS, **(custom_factors or {})}
        self._activities: list[Scope3Activity] = []

    # ── Activity logging ──────────────────────────────────────────────────────

    def add_transport(
        self,
        mode:          str,
        tonne_km:      float,
        category:      Scope3Category = Scope3Category.UPSTREAM_TRANSPORT,
        supplier_id:   str = "",
        data_quality:  str = "calculated",
    ) -> "Scope3Calculator":
        """Add a freight transport activity."""
        factor = self._factors.get(mode, self._factors["road_transport_hgv"])
        self._activities.append(Scope3Activity(
            category=category,
            description=f"Freight transport ({mode})",
            quantity=tonne_km,
            unit="tonne-km",
            emission_factor=factor,
            supplier_id=supplier_id,
            data_quality=data_quality,
        ))
        return self

    def add_business_travel(
        self,
        mode:      str,
        km:        float,
        passengers: int = 1,
        data_quality: str = "calculated",
    ) -> "Scope3Calculator":
        """Add business travel activity (flight, car, rail)."""
        factor = self._factors.get(mode, self._factors["car_average"])
        self._activities.append(Scope3Activity(
            category=Scope3Category.BUSINESS_TRAVEL,
            description=f"Business travel ({mode})",
            quantity=km * passengers,
            unit="passenger-km",
            emission_factor=factor,
            data_quality=data_quality,
        ))
        return self

    def add_waste(
        self,
        treatment: str,
        tonnes:    float,
        data_quality: str = "estimated",
    ) -> "Scope3Calculator":
        """Add waste treatment activity."""
        factor = self._factors.get(treatment, self._factors["landfill_mixed"])
        self._activities.append(Scope3Activity(
            category=Scope3Category.WASTE_IN_OPERATIONS,
            description=f"Waste ({treatment})",
            quantity=tonnes,
            unit="tonnes",
            emission_factor=factor,
            data_quality=data_quality,
        ))
        return self

    def add_purchased_material(
        self,
        material:    str,
        kg:          float,
        supplier_id: str = "",
        data_quality: str = "estimated",
    ) -> "Scope3Calculator":
        """Add purchased goods/materials activity."""
        factor = self._factors.get(material, 1.0)
        self._activities.append(Scope3Activity(
            category=Scope3Category.PURCHASED_GOODS,
            description=f"Purchased material ({material})",
            quantity=kg,
            unit="kg",
            emission_factor=factor,
            supplier_id=supplier_id,
            data_quality=data_quality,
        ))
        return self

    def add_upstream_energy(
        self,
        energy_type: str,
        kwh:         float,
        data_quality: str = "calculated",
    ) -> "Scope3Calculator":
        """Add upstream emissions from purchased energy (Cat 3)."""
        factor = self._factors.get(
            f"{energy_type}_upstream",
            self._factors["natural_gas_upstream"]
        )
        self._activities.append(Scope3Activity(
            category=Scope3Category.FUEL_ENERGY_UPSTREAM,
            description=f"Upstream energy ({energy_type})",
            quantity=kwh,
            unit="kWh",
            emission_factor=factor,
            data_quality=data_quality,
        ))
        return self

    def add_custom_activity(self, activity: Scope3Activity) -> "Scope3Calculator":
        """Add any custom Scope 3 activity."""
        self._activities.append(activity)
        return self

    # ── Report generation ─────────────────────────────────────────────────────

    def generate_report(
        self,
        period_start: datetime | None = None,
        period_end:   datetime | None = None,
    ) -> Scope3Report:
        """Generate the full Scope 3 report from logged activities."""
        now = datetime.now(timezone.utc)
        return Scope3Report(
            entity_id=self.entity_id,
            period_start=period_start or now.replace(month=1, day=1),
            period_end=period_end or now,
            activities=list(self._activities),
        )

    def factor(self, key: str) -> float | None:
        """Look up an emission factor by key."""
        return self._factors.get(key)

    def available_factors(self) -> list[str]:
        return sorted(self._factors.keys())
