"""
VoltEdge AI — Direct mode (no separate API server required).

When ANTHROPIC_API_KEY is set in the environment, the dashboard can
call the AI assistant directly from the same process — no uvicorn, no
httpx, no port 8000.

The EnergyAssistant, ToolExecutor, and all 11 tools already exist in
voltedge/ai/. This module wires them together with VoltEdge's data
layer so the dashboard callback can call chat() locally.

Session management mirrors the API: sessions are stored in an
in-memory dict on the singleton assistant instance, keyed by the
browser's session_id (stored in dcc.Store with storage_type="session").

Priority order the callback uses:
  1. ANTHROPIC_API_KEY set → direct mode (this module)
  2. API server reachable on :8000 → HTTP mode (existing behaviour)
  3. Neither → styled "set your API key" prompt
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from voltedge.utils.logger import get_logger

log = get_logger(__name__)


def api_key_available() -> bool:
    """Return True if ANTHROPIC_API_KEY is set and non-empty."""
    return bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())


@lru_cache(maxsize=1)
def _get_assistant():
    """
    Build and cache a single EnergyAssistant instance for the process.
    lru_cache(1) means this is constructed once and reused across all
    callback invocations — the assistant's session dict persists in memory.
    """
    from voltedge.ai.assistant import EnergyAssistant
    from voltedge.ai.executor import ToolExecutor
    from voltedge.storage.base import LocalStore

    SITES = [
        "LONDON-FACTORY-01",
        "DUBAI-OFFICE-01",
        "ROTTERDAM-WAREHOUSE-01",
        "FRANKFURT-DC-01",
    ]

    store    = LocalStore()
    executor = ToolExecutor(store=store, sites=SITES)

    dry_run = not api_key_available()
    if dry_run:
        log.warning("ai.direct.dry_run_mode",
                    reason="ANTHROPIC_API_KEY not set")

    assistant = EnergyAssistant(executor=executor, dry_run=dry_run)
    log.info("ai.direct.assistant_created", dry_run=dry_run)
    return assistant


def chat(message: str, session_id: Optional[str] = None) -> tuple[str, str]:
    """
    Send a message to the assistant and return (answer, session_id).

    This is the single call the dashboard's send_message callback makes.
    It handles session creation/retrieval transparently.

    Returns:
        (answer_text, session_id) — session_id may be new if none was passed.
    """
    assistant = _get_assistant()

    # Reuse existing session or let the assistant create a new one
    session = None
    if session_id:
        session = assistant.get_session(session_id)

    if session is None:
        session = assistant.new_session()

    answer = assistant.chat(message, session_id=session.session_id)
    return answer, session.session_id


def status() -> dict:
    """Return assistant status for the API status indicator in the UI."""
    key_set = api_key_available()
    if not key_set:
        return {
            "mode":    "no_key",
            "ready":   False,
            "message": "Set ANTHROPIC_API_KEY to enable the AI assistant.",
        }
    return {
        "mode":    "direct",
        "ready":   True,
        "message": "AI assistant ready (direct mode — no API server needed).",
    }
