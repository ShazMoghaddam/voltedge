"""Tests for EnergyAssistant and ConversationSession — dry_run mode, no API calls."""
from __future__ import annotations
import asyncio
import pytest
from voltedge.ai.assistant import EnergyAssistant, ConversationSession
from voltedge.ai.executor import ToolExecutor
from voltedge.ingestion.simulators import SimulatedSiteConnector
from voltedge.processing.transformer import EnergyTransformer
from voltedge.storage.base import LocalStore

SITE = "ASST-TEST-01"


@pytest.fixture(scope="module")
def assistant(tmp_path_factory):
    tmp   = tmp_path_factory.mktemp("asst_store")
    store = LocalStore(base_path=tmp)
    conn  = SimulatedSiteConnector(SITE, "factory", hours=200, seed=42)
    result = asyncio.run(conn.fetch())
    df = EnergyTransformer().transform(result.data)
    store.write(df, site_id=SITE, layer="processed")
    executor = ToolExecutor(store=store, sites=[SITE])
    return EnergyAssistant(executor=executor, dry_run=True)


# ── ConversationSession ───────────────────────────────────────────────────────

def test_session_has_id():
    s = ConversationSession()
    assert len(s.session_id) == 8

def test_session_add_user():
    s = ConversationSession()
    s.add_user("Hello")
    assert s.messages[-1]["role"] == "user"
    assert s.messages[-1]["content"] == "Hello"

def test_session_add_assistant():
    s = ConversationSession()
    s.add_assistant("Hi there")
    msg = s.messages[-1]
    assert msg["role"] == "assistant"
    assert any(b.get("text") == "Hi there" for b in msg["content"])

def test_session_add_tool_result():
    s = ConversationSession()
    s.add_tool_result("tool-id-123", {"kwh": 100})
    msg = s.messages[-1]
    assert msg["role"] == "user"
    assert msg["content"][0]["type"] == "tool_result"

def test_session_turn_count():
    s = ConversationSession()
    s.add_user("q1")
    s.add_assistant("a1")
    s.add_user("q2")
    assert s.turn_count == 2

def test_session_to_dict_keys():
    s = ConversationSession()
    d = s.to_dict()
    for k in ("session_id", "turn_count", "created_at", "last_active"):
        assert k in d

def test_two_sessions_different_ids():
    s1, s2 = ConversationSession(), ConversationSession()
    assert s1.session_id != s2.session_id


# ── EnergyAssistant (dry_run) ─────────────────────────────────────────────────

def test_assistant_creates_session(assistant):
    session = assistant.new_session()
    assert session.session_id in assistant._sessions

def test_assistant_get_session(assistant):
    session = assistant.new_session()
    found = assistant.get_session(session.session_id)
    assert found is session

def test_assistant_get_missing_session(assistant):
    assert assistant.get_session("nonexistent") is None

def test_chat_returns_string(assistant):
    response = assistant.chat("What is the energy consumption today?")
    assert isinstance(response, str)
    assert len(response) > 0

def test_chat_dry_run_mentions_dry_run(assistant):
    response = assistant.chat("Tell me about anomalies")
    assert "dry" in response.lower() or "DRY" in response

def test_chat_creates_session(assistant):
    before = len(assistant._sessions)
    assistant.chat("Hello")
    assert len(assistant._sessions) >= before + 1

def test_chat_with_session_id_reuses_session(assistant):
    session = assistant.new_session()
    assistant.chat("First message", session_id=session.session_id)
    assistant.chat("Second message", session_id=session.session_id)
    found = assistant.get_session(session.session_id)
    assert found.turn_count == 2

def test_stream_chat_yields_tokens(assistant):
    tokens = list(assistant.stream_chat("Tell me about ESG metrics"))
    assert len(tokens) > 0
    assert all(isinstance(t, str) for t in tokens)

def test_stream_chat_tokens_join_to_text(assistant):
    tokens = list(assistant.stream_chat("What is our peak demand?"))
    full   = "".join(tokens)
    assert len(full) > 0

def test_session_info_structure(assistant):
    info = assistant.session_info()
    assert "active_sessions" in info
    assert "sessions" in info
    assert isinstance(info["sessions"], list)


# ── API routes ────────────────────────────────────────────────────────────────

def test_ai_chat_route(assistant, tmp_path):
    from fastapi.testclient import TestClient
    from voltedge.api.app import create_app
    from voltedge.api.ai_routes import get_assistant

    app = create_app()
    app.dependency_overrides[get_assistant] = lambda: assistant
    client = TestClient(app)

    r = client.post("/ai/chat", json={"message": "What is the energy status?"})
    assert r.status_code == 200
    body = r.json()
    assert "response" in body
    assert "session_id" in body
    assert "turn_count" in body


def test_ai_chat_has_response_text(assistant):
    from fastapi.testclient import TestClient
    from voltedge.api.app import create_app
    from voltedge.api.ai_routes import get_assistant

    app = create_app()
    app.dependency_overrides[get_assistant] = lambda: assistant
    client = TestClient(app)

    r = client.post("/ai/chat", json={"message": "Summarise anomalies"})
    assert r.status_code == 200
    assert len(r.json()["response"]) > 0


def test_ai_sessions_endpoint(assistant):
    from fastapi.testclient import TestClient
    from voltedge.api.app import create_app
    from voltedge.api.ai_routes import get_assistant

    app = create_app()
    app.dependency_overrides[get_assistant] = lambda: assistant
    client = TestClient(app)

    r = client.get("/ai/sessions")
    assert r.status_code == 200
    assert "active_sessions" in r.json()


def test_ai_delete_session(assistant):
    from fastapi.testclient import TestClient
    from voltedge.api.app import create_app
    from voltedge.api.ai_routes import get_assistant

    app = create_app()
    app.dependency_overrides[get_assistant] = lambda: assistant
    client = TestClient(app)

    # Create a session via chat
    r = client.post("/ai/chat", json={"message": "Hello"})
    sid = r.json()["session_id"]

    # Delete it
    r = client.delete(f"/ai/sessions/{sid}")
    assert r.status_code == 204


def test_ai_delete_unknown_session_returns_404(assistant):
    from fastapi.testclient import TestClient
    from voltedge.api.app import create_app
    from voltedge.api.ai_routes import get_assistant

    app = create_app()
    app.dependency_overrides[get_assistant] = lambda: assistant
    client = TestClient(app)

    r = client.delete("/ai/sessions/nonexistent-id")
    assert r.status_code == 404


def test_ai_analyze_site_route(assistant):
    from fastapi.testclient import TestClient
    from voltedge.api.app import create_app
    from voltedge.api.ai_routes import get_assistant

    app = create_app()
    app.dependency_overrides[get_assistant] = lambda: assistant
    client = TestClient(app)

    r = client.post(f"/ai/analyze/{SITE}",
                    json={"question": "What is the energy trend?", "hours": 24})
    assert r.status_code == 200
    assert r.json()["site_id" if "site_id" in r.json() else "session_id"] is not None


def test_ai_report_route(assistant):
    from fastapi.testclient import TestClient
    from voltedge.api.app import create_app
    from voltedge.api.ai_routes import get_assistant

    app = create_app()
    app.dependency_overrides[get_assistant] = lambda: assistant
    client = TestClient(app)

    r = client.post("/ai/report", json={
        "site_ids": [SITE],
        "hours": 24,
        "focus": "anomalies",
    })
    assert r.status_code == 200
    assert "response" in r.json()
