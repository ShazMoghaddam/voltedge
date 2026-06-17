"""Tests for the v1.0 redesigned dashboard — sidebar nav, light theme, causal AI card."""

from __future__ import annotations
import dash
import dash_bootstrap_components as dbc


def test_layout_has_sidebar_navlinks():
    from voltedge.dashboard.layouts import build_layout
    layout_str = str(build_layout())
    for nav_id in ("navlink-portfolio", "navlink-esg", "navlink-ai"):
        assert nav_id in layout_str, f"Missing nav link: {nav_id}"


def test_layout_has_page_containers():
    from voltedge.dashboard.layouts import build_layout
    layout_str = str(build_layout())
    for page_id in ("page-portfolio", "page-esg", "page-ai"):
        assert page_id in layout_str, f"Missing page container: {page_id}"


def test_layout_has_causal_insight_card():
    from voltedge.dashboard.layouts import build_layout
    layout_str = str(build_layout())
    assert "causal-insight-card" in layout_str


def test_esg_and_ai_pages_hidden_by_default():
    from voltedge.dashboard.layouts import build_esg_page, build_ai_page
    esg = build_esg_page()
    ai  = build_ai_page()
    assert esg.style.get("display") == "none"
    assert ai.style.get("display") == "none"


def test_portfolio_page_visible_by_default():
    from voltedge.dashboard.layouts import build_portfolio_page
    page = build_portfolio_page()
    style = getattr(page, "style", None)
    assert style is None or style.get("display") != "none"


def test_colors_use_light_theme():
    from voltedge.dashboard.layouts import COLORS
    assert COLORS["bg_primary"] == "#f4f6f9"
    assert COLORS["bg_card"] == "#ffffff"
    assert COLORS["accent_teal"] == "#1D9E75"


def test_site_colors_still_defined_for_all_sites():
    from voltedge.dashboard.layouts import SITE_COLORS, SITES
    for site in SITES:
        assert site in SITE_COLORS


def test_causal_insight_body_renders():
    from voltedge.dashboard.layouts import causal_insight_body
    alert = causal_insight_body("Consumption spiked 42% at 14:00.", "Review HVAC setpoints.")
    assert alert is not None


def test_causal_insight_body_without_recommendation():
    from voltedge.dashboard.layouts import causal_insight_body
    alert = causal_insight_body("No strong causal drivers identified.")
    assert alert is not None


def test_build_sidebar_renders():
    from voltedge.dashboard.layouts import build_sidebar
    sidebar = build_sidebar()
    assert sidebar is not None


def test_build_navbar_renders():
    from voltedge.dashboard.layouts import build_navbar
    navbar = build_navbar()
    assert navbar is not None


def test_app_uses_flatly_theme():
    from voltedge.dashboard.app import create_app
    app = create_app()
    assert any("flatly" in s.lower() for s in app.config.external_stylesheets)


def test_ai_panel_present_in_layout():
    from voltedge.dashboard.layouts import build_layout
    layout_str = str(build_layout())
    assert "ai-chat-history" in layout_str
    assert "ai-send-btn" in layout_str


# ── Switch page callback ───────────────────────────────────────────────────────

def test_switch_page_defaults_to_portfolio():
    """URL / or /portfolio shows portfolio, hides others."""
    from voltedge.dashboard.callbacks import switch_page
    result = switch_page("/portfolio")
    portfolio_style, esg_style, ai_style = result[0], result[1], result[2]
    assert portfolio_style == {}
    assert esg_style.get("display") == "none"
    assert ai_style.get("display") == "none"


def test_switch_page_to_esg():
    """URL /esg shows ESG page, hides others."""
    from voltedge.dashboard.callbacks import switch_page
    result = switch_page("/esg")
    portfolio_style, esg_style, ai_style = result[0], result[1], result[2]
    assert portfolio_style.get("display") == "none"
    assert esg_style == {}
    assert ai_style.get("display") == "none"


def test_switch_page_active_link_styled():
    """Active page nav link gets teal colour, others get muted."""
    from voltedge.dashboard.callbacks import switch_page, COLORS
    result = switch_page("/ai")
    portfolio_link, esg_link, ai_link = result[3], result[4], result[5]
    assert ai_link["color"] == COLORS["accent_teal"]
    assert portfolio_link["color"] == COLORS["text_muted"]
    assert esg_link["color"] == COLORS["text_muted"]


# ── Causal insight callback ───────────────────────────────────────────────────

def test_causal_insight_hidden_when_anomaly_disabled():
    from voltedge.dashboard.callbacks import update_causal_insight
    children, style = update_causal_insight(
        "LONDON-FACTORY-01", 168, 0.75, ["forecast"], 0
    )
    assert children == []
    assert style.get("display") == "none"


def test_causal_insight_runs_with_anomaly_enabled():
    from voltedge.dashboard.callbacks import update_causal_insight
    children, style = update_causal_insight(
        "LONDON-FACTORY-01", 168, 0.75, ["forecast", "anomaly"], 0
    )
    # Either hidden (no anomaly above threshold) or visible with content —
    # both are valid outcomes; the call must not raise.
    assert style.get("display") in ("none", "block")
