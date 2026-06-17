"""
VoltEdge AI — Tool Definitions for the Anthropic API

Defines the tools the AI assistant can call to retrieve live energy data.
These map 1-to-1 with VoltEdge's existing service layer — the assistant
never has direct DB access, only through these typed interfaces.

Tools available:
  get_site_summary       — consumption stats + anomalies for one site
  get_portfolio_summary  — all-sites overview
  get_anomalies          — recent anomaly events with labels
  get_forecast           — demand forecast for next N hours
  get_esg_metrics        — Scope 2, GRI 302-1, carbon intensity
  get_optimization_tips  — cost-saving recommendations from CostOptimizer
  compare_sites          — side-by-side site comparison
  get_peak_hours         — identify peak consumption windows

Each tool returns a plain dict that the API passes back to Claude
as a tool_result, which Claude uses to formulate its answer.
"""

from __future__ import annotations

from typing import Any

# ── Anthropic tool schema definitions ─────────────────────────────────────────

TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_site_summary",
        "description": (
            "Get a comprehensive energy consumption summary for a specific site, "
            "including total kWh, average load, peak demand, trend vs prior period, "
            "CO₂ emissions, and any detected anomalies. Use this when asked about "
            "a specific site's performance, consumption, or status."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "site_id": {
                    "type": "string",
                    "description": "The VoltEdge site identifier (e.g. 'LONDON-FACTORY-01')",
                },
                "hours": {
                    "type": "integer",
                    "description": "Lookback window in hours (default 24, max 720)",
                    "default": 24,
                    "minimum": 1,
                    "maximum": 720,
                },
            },
            "required": ["site_id"],
        },
    },
    {
        "name": "get_portfolio_summary",
        "description": (
            "Get an overview of all monitored sites, showing total consumption, "
            "CO₂ emissions, anomaly counts, and which sites are the highest/lowest "
            "consumers. Use when asked about overall fleet performance, portfolio-level "
            "ESG status, or cross-site comparisons without specifying a site."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "hours": {
                    "type": "integer",
                    "description": "Lookback window in hours (default 24)",
                    "default": 24,
                    "minimum": 1,
                    "maximum": 720,
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_anomalies",
        "description": (
            "Retrieve recent anomaly detections for one or all sites, including "
            "anomaly score, classification label (sudden_spike, flat_line_sensor, "
            "poor_power_factor, extreme_consumption), and timestamp. "
            "Use when asked about unusual readings, incidents, or equipment issues."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "site_id": {
                    "type": "string",
                    "description": "Site ID, or omit for all sites",
                },
                "hours": {
                    "type": "integer",
                    "description": "Lookback window in hours (default 24)",
                    "default": 24,
                    "minimum": 1,
                    "maximum": 168,
                },
                "min_score": {
                    "type": "number",
                    "description": "Minimum anomaly score to include (0-1, default 0.7)",
                    "default": 0.70,
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_forecast",
        "description": (
            "Get the demand forecast for a site for the next N hours, including "
            "point forecast and 95% confidence interval. Use when asked about "
            "expected consumption, load predictions, or planning for upcoming periods."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "site_id": {
                    "type": "string",
                    "description": "The site to forecast",
                },
                "horizon_hours": {
                    "type": "integer",
                    "description": "How many hours ahead to forecast (default 24, max 168)",
                    "default": 24,
                    "minimum": 1,
                    "maximum": 168,
                },
            },
            "required": ["site_id"],
        },
    },
    {
        "name": "get_esg_metrics",
        "description": (
            "Get ESG compliance metrics for a site or portfolio, including "
            "Scope 2 emissions (location-based and market-based), GRI 302-1 "
            "energy consumption in GJ, carbon intensity, and renewable fraction. "
            "Use when asked about sustainability, carbon reporting, or ESG targets."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "site_id": {
                    "type": "string",
                    "description": "Site ID, or omit for portfolio-level metrics",
                },
                "days": {
                    "type": "integer",
                    "description": "Reporting period in days (default 30)",
                    "default": 30,
                    "minimum": 1,
                    "maximum": 365,
                },
                "country": {
                    "type": "string",
                    "description": "2-letter country code for grid emission factor (default 'GB')",
                    "default": "GB",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_optimization_tips",
        "description": (
            "Get cost and energy optimisation recommendations for a site, including "
            "peak shaving opportunities, load shifting potential, power factor correction, "
            "and estimated annual savings. Use when asked about reducing costs, "
            "optimising energy use, or improving efficiency."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "site_id": {
                    "type": "string",
                    "description": "The site to analyse",
                },
                "tariff": {
                    "type": "string",
                    "description": "Tariff name: UK_HALF_HOURLY, EU_INDUSTRIAL, or US_COMMERCIAL",
                    "default": "UK_HALF_HOURLY",
                    "enum": ["UK_HALF_HOURLY", "EU_INDUSTRIAL", "US_COMMERCIAL"],
                },
            },
            "required": ["site_id"],
        },
    },
    {
        "name": "compare_sites",
        "description": (
            "Compare two sites side-by-side on consumption, emissions, anomalies, "
            "and efficiency. Use when asked to compare sites, identify the better/worse "
            "performer, or understand differences between locations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "site_a": {
                    "type": "string",
                    "description": "First site ID",
                },
                "site_b": {
                    "type": "string",
                    "description": "Second site ID",
                },
                "hours": {
                    "type": "integer",
                    "description": "Comparison period in hours (default 24)",
                    "default": 24,
                    "minimum": 1,
                    "maximum": 720,
                },
            },
            "required": ["site_a", "site_b"],
        },
    },
    {
        "name": "get_peak_hours",
        "description": (
            "Identify the peak consumption hours and days for a site, showing "
            "average load by hour of day and day of week. Use when asked about "
            "when a site consumes the most, what time to schedule maintenance, "
            "or how to plan load shifting."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "site_id": {
                    "type": "string",
                    "description": "The site to analyse",
                },
                "days": {
                    "type": "integer",
                    "description": "Historical period to analyse (default 30 days)",
                    "default": 30,
                    "minimum": 7,
                    "maximum": 365,
                },
            },
            "required": ["site_id"],
        },
    },
]

# ── Tool name registry ─────────────────────────────────────────────────────────

TOOL_NAMES: set[str] = {t["name"] for t in TOOLS}


# ── Causal AI tools (v8.0) ────────────────────────────────────────────────────

CAUSAL_TOOLS: list[dict] = [
    {
        "name": "get_root_cause",
        "description": (
            "Perform causal root-cause analysis for an anomaly at a specific site. "
            "Returns ranked causal factors with contribution scores, a narrative "
            "explanation, and recommended actions. Use when asked 'why did X happen', "
            "'what caused the spike', or 'explain this anomaly'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "site_id": {
                    "type": "string",
                    "description": "Site to analyse",
                },
                "anomaly_timestamp": {
                    "type": "string",
                    "description": "ISO-8601 timestamp of the anomaly",
                },
                "anomaly_variable": {
                    "type": "string",
                    "description": "Variable that was anomalous (default: kwh)",
                    "default": "kwh",
                },
            },
            "required": ["site_id", "anomaly_timestamp"],
        },
    },
    {
        "name": "run_counterfactual",
        "description": (
            "Run a do-calculus counterfactual: 'what would consumption be if we "
            "changed X to Y?' Returns predicted outcome, delta vs baseline, and "
            "annual kWh / CO₂ / cost impact. Use for 'what if' questions, "
            "intervention planning, and ROI estimation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "site_id": {
                    "type": "string",
                    "description": "Site to run the counterfactual for",
                },
                "intervention_variable": {
                    "type": "string",
                    "description": "Variable to intervene on (e.g. temperature_c, is_business_hour)",
                },
                "intervention_value": {
                    "type": "number",
                    "description": "The value to set the intervention variable to",
                },
                "description": {
                    "type": "string",
                    "description": "Human-readable description of the intervention",
                    "default": "",
                },
            },
            "required": ["site_id", "intervention_variable", "intervention_value"],
        },
    },
    {
        "name": "simulate_scenario",
        "description": (
            "Run a digital twin scenario simulation for a site: 'what happens over "
            "a year if we add solar / BESS / EV fleet / change shift schedule?' "
            "Returns total kWh, CO₂, cost, and peak demand under the scenario "
            "compared to the current baseline. Use for capital investment planning."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "site_id": {
                    "type": "string",
                    "description": "Site to simulate",
                },
                "scenario_type": {
                    "type": "string",
                    "description": "One of: solar_pv, bess, ev_fleet, night_shift, baseline",
                    "enum": ["solar_pv", "bess", "ev_fleet", "night_shift", "baseline"],
                },
                "scenario_params": {
                    "type": "object",
                    "description": "Scenario-specific parameters (e.g. capacity_kwp for solar_pv)",
                },
                "simulation_hours": {
                    "type": "integer",
                    "description": "Hours to simulate (default 8760 = 1 year)",
                    "default": 8760,
                },
            },
            "required": ["site_id", "scenario_type"],
        },
    },
]

# Merge into main TOOLS list
TOOLS.extend(CAUSAL_TOOLS)
TOOL_NAMES.update(t["name"] for t in CAUSAL_TOOLS)
