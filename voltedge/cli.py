"""
VoltEdge CLI — run the pipeline, start the dashboard, or trigger ML training.

Usage:
    voltedge pipeline run --sites SITE-01 SITE-02
    voltedge pipeline simulate --site-type factory --hours 168
    voltedge dashboard start
    voltedge esg report --site SITE-01 --days 30
"""

from __future__ import annotations

import asyncio

import rich_click as click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
def main() -> None:
    """VoltEdge — Enterprise Energy Intelligence Platform"""


# ── Pipeline commands ─────────────────────────────────────────────────────────

@main.group()
def pipeline() -> None:
    """Data ingestion and processing commands."""


@pipeline.command("simulate")
@click.option("--site-id", default="SITE-DEMO-01", help="Site identifier")
@click.option(
    "--site-type",
    default="factory",
    type=click.Choice(["factory", "office", "warehouse", "data_center"]),
)
@click.option("--hours", default=168, help="Hours of history to generate")
@click.option("--seed", default=42, help="Random seed for reproducibility")
def pipeline_simulate(site_id: str, site_type: str, hours: int, seed: int) -> None:
    """Generate and store simulated energy data for a site."""
    from voltedge.core.pipeline import EnergyPipeline
    from voltedge.ingestion.simulators import SimulatedSiteConnector
    from voltedge.storage.base import LocalStore

    console.print(f"[bold green]▶ Running simulation[/] site={site_id} type={site_type} hours={hours}")

    store = LocalStore()
    pipeline = EnergyPipeline(store=store)
    pipeline.register_connector(site_id, SimulatedSiteConnector(site_id, site_type, hours, seed))

    summary = asyncio.run(pipeline.run_all())

    table = Table(title="Pipeline Run Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Sites processed", str(summary.sites_processed))
    table.add_row("Total records", str(summary.total_records))
    table.add_row("Errors", str(summary.total_errors))
    table.add_row("Duration (s)", f"{summary.duration_seconds:.2f}")
    console.print(table)


# ── Dashboard command ─────────────────────────────────────────────────────────

@main.command("dashboard")
@click.option("--port", default=8050)
@click.option("--debug/--no-debug", default=True)
def dashboard(port: int, debug: bool) -> None:
    """Start the Dash dashboard."""
    console.print(f"[bold blue]🔋 Starting VoltEdge Dashboard[/] on http://localhost:{port}")
    from voltedge.dashboard.app import create_app
    app = create_app()
    app.run(debug=debug, port=port)


# ── ESG command ───────────────────────────────────────────────────────────────

@main.group()
def esg() -> None:
    """ESG reporting commands."""


@esg.command("report")
@click.option("--site", required=True)
@click.option("--days", default=30)
@click.option("--country", default="GB")
def esg_report(site: str, days: int, country: str) -> None:
    """Print an ESG summary for a site."""
    from voltedge.esg.metrics import ESGCalculator
    from voltedge.storage.base import LocalStore

    store = LocalStore()
    df = store.read(site, layer="processed", days=days)
    if df.empty:
        console.print(f"[yellow]No data found for {site}. Run pipeline simulate first.[/]")
        return

    calc = ESGCalculator(country_code=country)
    metrics = calc.compute(df, site)

    table = Table(title=f"ESG Report — {site} (last {days} days)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Total consumption", f"{metrics.total_kwh:,.1f} kWh")
    table.add_row("Total CO₂", f"{metrics.total_co2_kg:,.1f} kg")
    table.add_row("Scope 2 (location)", f"{metrics.scope_2_location_based_tco2e:.3f} tCO₂e")
    table.add_row("Scope 2 (market)", f"{metrics.scope_2_market_based_tco2e:.3f} tCO₂e")
    table.add_row("GRI 302-1", f"{metrics.gri_302_1:.2f} GJ")
    table.add_row("Peak demand", f"{metrics.peak_demand_kw:.1f} kW")
    table.add_row("Renewable fraction", f"{metrics.renewable_fraction:.1%}")
    console.print(table)
