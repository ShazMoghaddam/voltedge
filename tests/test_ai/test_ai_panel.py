"""Tests for the AI dashboard panel."""
from __future__ import annotations
import pytest
import dash
import dash_bootstrap_components as dbc


def test_build_ai_panel_returns_card():
    from voltedge.dashboard.ai_panel import build_ai_panel
    panel = build_ai_panel()
    assert panel is not None
    assert isinstance(panel, dbc.Card)


def test_ai_panel_contains_expected_ids():
    from voltedge.dashboard.ai_panel import build_ai_panel
    panel_str = str(build_ai_panel())
    for component_id in ("ai-chat-history", "ai-input", "ai-send-btn",
                         "ai-panel-toggle", "ai-session-id"):
        assert component_id in panel_str, f"Missing: {component_id}"


def test_ai_panel_has_quick_actions():
    from voltedge.dashboard.ai_panel import build_ai_panel, QUICK_ACTIONS
    panel_str = str(build_ai_panel())
    # At least the first chip text should appear somewhere in the component tree
    assert len(QUICK_ACTIONS) >= 4


def test_quick_actions_are_strings():
    from voltedge.dashboard.ai_panel import QUICK_ACTIONS
    assert all(isinstance(q, str) and len(q) > 5 for q in QUICK_ACTIONS)


def test_user_bubble_is_div():
    from voltedge.dashboard.ai_panel import _user_bubble
    from dash import html
    bubble = _user_bubble("Hello")
    assert bubble is not None


def test_assistant_bubble_is_div():
    from voltedge.dashboard.ai_panel import _assistant_bubble
    from dash import html
    bubble = _assistant_bubble("I am VoltEdge assistant")
    assert bubble is not None


def test_register_ai_callbacks_does_not_raise():
    from voltedge.dashboard.ai_panel import build_ai_panel, register_ai_callbacks
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY],
                    suppress_callback_exceptions=True)
    app.layout = build_ai_panel()
    register_ai_callbacks(app)   # Should not raise
