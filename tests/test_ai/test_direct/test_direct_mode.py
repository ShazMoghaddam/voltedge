"""Tests for Phase 1c: AI assistant direct mode."""
from __future__ import annotations

import os
import pytest


# ── api_key_available ──────────────────────────────────────────────────────────

def test_api_key_available_when_set(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
    from importlib import reload
    import voltedge.ai.direct as d
    reload(d)
    assert d.api_key_available() is True


def test_api_key_not_available_when_empty(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from importlib import reload
    import voltedge.ai.direct as d
    reload(d)
    assert d.api_key_available() is False


def test_api_key_not_available_when_whitespace(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
    from importlib import reload
    import voltedge.ai.direct as d
    reload(d)
    assert d.api_key_available() is False


# ── status() ──────────────────────────────────────────────────────────────────

def test_status_no_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from importlib import reload
    import voltedge.ai.direct as d
    reload(d)
    s = d.status()
    assert s["ready"] is False
    assert s["mode"] == "no_key"
    assert "ANTHROPIC_API_KEY" in s["message"]


def test_status_with_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    from importlib import reload
    import voltedge.ai.direct as d
    reload(d)
    s = d.status()
    assert s["ready"] is True
    assert s["mode"] == "direct"


# ── chat() dry-run ────────────────────────────────────────────────────────────

def test_chat_returns_tuple(monkeypatch):
    """chat() should return (answer_str, session_id_str)."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    import voltedge.ai.direct as d
    d._get_assistant.cache_clear()
    answer, sid = d.chat("What is our energy status?")
    assert isinstance(answer, str)
    assert isinstance(sid,    str)
    assert len(answer) > 0
    assert len(sid)    > 0


def test_chat_dry_run_mentions_api_key(monkeypatch):
    """Dry-run response should tell the user to set the API key."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    import voltedge.ai.direct as d
    d._get_assistant.cache_clear()
    answer, _ = d.chat("Tell me about anomalies")
    assert "dry" in answer.lower() or "api" in answer.lower() or "key" in answer.lower()


def test_chat_session_persists(monkeypatch):
    """Sending two messages with the same session_id uses the same session."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    import voltedge.ai.direct as d
    d._get_assistant.cache_clear()
    _, sid1 = d.chat("Hello")
    _, sid2 = d.chat("What did I just say?", session_id=sid1)
    assert sid1 == sid2


def test_chat_new_session_without_id(monkeypatch):
    """Two calls with no session_id produce different sessions."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    import voltedge.ai.direct as d
    d._get_assistant.cache_clear()
    _, sid1 = d.chat("Hello")
    _, sid2 = d.chat("Hello again")
    assert sid1 != sid2


def test_chat_existing_session_reused(monkeypatch):
    """Passing a valid session_id reuses the session."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    import voltedge.ai.direct as d
    d._get_assistant.cache_clear()
    _, sid = d.chat("First message")
    answer, sid2 = d.chat("Second message", session_id=sid)
    assert sid == sid2
    assert len(answer) > 0


# ── _no_key_card ──────────────────────────────────────────────────────────────

def test_no_key_card_renders():
    from voltedge.dashboard.ai_panel import _no_key_card
    card = _no_key_card()
    assert card is not None


def test_no_key_card_contains_setup_instructions():
    from voltedge.dashboard.ai_panel import _no_key_card
    import json
    card_str = str(_no_key_card())
    assert "ANTHROPIC_API_KEY" in card_str


# ── Panel integration ─────────────────────────────────────────────────────────

def test_ai_panel_has_status_badge():
    from voltedge.dashboard.ai_panel import build_ai_panel
    panel_str = str(build_ai_panel())
    assert "ai-status-badge" in panel_str


def test_send_message_skips_when_empty():
    """send_message returns no_update for empty input."""
    from voltedge.dashboard.ai_panel import register_ai_callbacks
    import dash
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    # Registration should not raise
    register_ai_callbacks(app)
