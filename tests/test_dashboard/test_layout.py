"""Tests for dashboard layout components — no browser required."""

from __future__ import annotations
import dash
import dash_bootstrap_components as dbc


def test_layout_builds_without_error():
    from voltedge.dashboard.layouts import build_layout
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY],
                    suppress_callback_exceptions=True)
    app.layout = build_layout()
    assert app.layout is not None


def test_layout_contains_required_ids():
    from voltedge.dashboard.layouts import build_layout
    layout_str = str(build_layout())
    required_ids = [
        "site-selector", "compare-site-selector", "time-window-store",
        "refresh-interval", "consumption-chart", "anomaly-chart",
        "heatmap-chart", "esg-chart", "portfolio-chart",
        "kpi-kwh", "kpi-co2", "kpi-anom", "kpi-peak",
    ]
    for id_ in required_ids:
        assert id_ in layout_str, f"Missing component id: {id_}"


def test_kpi_card_renders():
    from voltedge.dashboard.layouts import kpi_card, COLORS
    card = kpi_card("Test", "test-id", "units", "bi-star", COLORS["accent_blue"],
                    delta_id="test-id-delta")
    assert card is not None


def test_chart_card_renders():
    from voltedge.dashboard.layouts import chart_card, COLORS
    card = chart_card("My Chart", "my-chart-id", COLORS["accent_green"], height=300)
    assert card is not None


def test_all_site_colors_defined():
    from voltedge.dashboard.layouts import SITE_COLORS, SITES
    for site in SITES:
        assert site in SITE_COLORS, f"No color defined for {site}"


def test_app_server_attribute():
    from voltedge.dashboard.app import create_app
    app = create_app()
    assert hasattr(app, "server")


def test_app_title():
    from voltedge.dashboard.app import create_app
    app = create_app()
    assert "VoltEdge" in app.title
