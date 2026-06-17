"""
VoltEdge API — AI Assistant Routes

REST + streaming endpoints for the energy assistant.

Endpoints:
  POST /ai/chat              — Single-turn or multi-turn chat
  POST /ai/chat/stream       — SSE streaming chat response
  GET  /ai/sessions          — List active sessions
  GET  /ai/sessions/{id}     — Session info
  DELETE /ai/sessions/{id}   — Clear session history
  POST /ai/analyze/{site_id} — One-shot site analysis (no session state)
  POST /ai/report            — Generate a structured energy report

All endpoints accept X-Session-ID header for conversation continuity.
All analytics endpoints gate on the 'analytics:read' RBAC permission.
"""

from __future__ import annotations

import json
from typing import AsyncIterator

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from voltedge.ai.assistant import EnergyAssistant
from voltedge.ai.context_builder import ContextBuilder
from voltedge.ai.executor import ToolExecutor
from voltedge.storage.base import LocalStore
from voltedge.utils.logger import get_logger

log = get_logger(__name__)
ai_router = APIRouter(prefix="/ai", tags=["AI Assistant"])

# ── Request / response models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message:    str        = Field(..., min_length=1, max_length=2000, json_schema_extra={"example": "Why did LONDON-FACTORY-01 spike at 2pm?"})
    session_id: str | None = Field(default=None, json_schema_extra={"example": "abc12345"})


class ChatResponse(BaseModel):
    session_id: str
    response:   str
    turn_count: int


class AnalyzeRequest(BaseModel):
    question:   str  = Field(..., json_schema_extra={"example": "What is our energy trend this week?"})
    hours:      int  = Field(default=24, ge=1, le=720)


class ReportRequest(BaseModel):
    site_ids:   list[str] = Field(..., json_schema_extra={"example": ["LONDON-FACTORY-01"]})
    hours:      int       = Field(default=168, ge=1, le=720)
    focus:      str       = Field(
        default="general",
        json_schema_extra={"example": "anomalies"},
        description="One of: general, anomalies, esg, cost, forecast",
    )


# ── Dependency: assistant singleton ──────────────────────────────────────────

_store     = LocalStore()
_sites     = ["LONDON-FACTORY-01", "DUBAI-OFFICE-01",
              "ROTTERDAM-WAREHOUSE-01", "FRANKFURT-DC-01"]
_executor  = ToolExecutor(store=_store, sites=_sites)
_assistant = EnergyAssistant(executor=_executor, dry_run=True)  # dry_run until API key set


def get_assistant() -> EnergyAssistant:
    return _assistant


# ── Chat endpoints ────────────────────────────────────────────────────────────

@ai_router.post("/chat", response_model=ChatResponse)
def chat(
    body:       ChatRequest,
    assistant:  EnergyAssistant = Depends(get_assistant),
    x_session_id: str | None    = Header(default=None, alias="X-Session-ID"),
) -> ChatResponse:
    """
    Send a message to the energy assistant.
    Include X-Session-ID header to continue an existing conversation.
    """
    session_id = body.session_id or x_session_id

    response_text = assistant.chat(body.message, session_id=session_id)

    session = assistant.get_session(session_id) if session_id else None
    if session is None:
        # chat() creates a new session; find it
        all_sessions = list(assistant._sessions.values())
        session = all_sessions[-1] if all_sessions else None

    return ChatResponse(
        session_id=session.session_id if session else "unknown",
        response=response_text,
        turn_count=session.turn_count if session else 1,
    )


@ai_router.post("/chat/stream")
def chat_stream(
    body:       ChatRequest,
    assistant:  EnergyAssistant = Depends(get_assistant),
    x_session_id: str | None    = Header(default=None, alias="X-Session-ID"),
) -> StreamingResponse:
    """
    Streaming chat via Server-Sent Events.
    Each token is delivered as: data: {"token": "..."}\n\n
    Final message: data: {"done": true, "session_id": "..."}\n\n
    """
    session_id = body.session_id or x_session_id

    async def _generate() -> AsyncIterator[str]:
        session = (
            assistant.get_session(session_id) if session_id else None
        ) or assistant.new_session()

        collected = []
        for token in assistant.stream_chat(body.message, session_id=session.session_id):
            collected.append(token)
            yield f"data: {json.dumps({'token': token})}\n\n"

        yield f"data: {json.dumps({'done': True, 'session_id': session.session_id, 'turn_count': session.turn_count})}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Session management ────────────────────────────────────────────────────────

@ai_router.get("/sessions")
def list_sessions(assistant: EnergyAssistant = Depends(get_assistant)) -> dict:
    return assistant.session_info()


@ai_router.get("/sessions/{session_id}")
def get_session(
    session_id: str,
    assistant:  EnergyAssistant = Depends(get_assistant),
) -> dict:
    session = assistant.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return session.to_dict()


@ai_router.delete("/sessions/{session_id}", status_code=204)
def delete_session(
    session_id: str,
    assistant:  EnergyAssistant = Depends(get_assistant),
) -> None:
    if session_id not in assistant._sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    del assistant._sessions[session_id]
    log.info("assistant.session_deleted", session=session_id)


# ── One-shot analysis ─────────────────────────────────────────────────────────

@ai_router.post("/analyze/{site_id}", response_model=ChatResponse)
def analyze_site(
    site_id:   str,
    body:      AnalyzeRequest,
    assistant: EnergyAssistant = Depends(get_assistant),
) -> ChatResponse:
    """
    Stateless single-turn analysis for a specific site.
    Each call creates a fresh session.
    """
    prompt = (
        f"Analyse site {site_id} for the last {body.hours} hours. "
        f"{body.question}"
    )
    session = assistant.new_session(metadata={"site_id": site_id, "type": "analysis"})
    response_text = assistant.chat(prompt, session_id=session.session_id)

    return ChatResponse(
        session_id=session.session_id,
        response=response_text,
        turn_count=1,
    )


# ── Report generation ─────────────────────────────────────────────────────────

@ai_router.post("/report", response_model=ChatResponse)
def generate_report(
    body:      ReportRequest,
    assistant: EnergyAssistant = Depends(get_assistant),
) -> ChatResponse:
    """
    Generate a structured energy report for one or more sites.
    The assistant fetches live data and synthesises a comprehensive report.
    """
    focus_prompts = {
        "general":   "Provide a comprehensive energy performance summary.",
        "anomalies": "Focus on anomaly detection, root causes, and recommended actions.",
        "esg":       "Focus on Scope 2 emissions, GRI 302-1 compliance, and carbon reduction opportunities.",
        "cost":      "Focus on cost optimisation, tariff analysis, and potential savings.",
        "forecast":  "Focus on demand forecasting, peak planning, and capacity recommendations.",
    }
    focus_text = focus_prompts.get(body.focus, focus_prompts["general"])

    sites_str = ", ".join(body.site_ids)
    prompt = (
        f"Generate an energy report for the following sites: {sites_str}. "
        f"Period: last {body.hours} hours. "
        f"{focus_text} "
        f"Use the available tools to fetch current data, then provide a structured report "
        f"with findings, anomalies, ESG metrics, and actionable recommendations."
    )

    session = assistant.new_session(metadata={
        "type": "report", "sites": body.site_ids, "focus": body.focus
    })
    response_text = assistant.chat(prompt, session_id=session.session_id)

    return ChatResponse(
        session_id=session.session_id,
        response=response_text,
        turn_count=session.turn_count,
    )
