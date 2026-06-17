"""
VoltEdge AI — Energy Intelligence Assistant

Multi-turn conversational assistant powered by Claude (claude-sonnet-4-20250514).
Answers natural language questions about energy data using tool use to
fetch live context from VoltEdge's data layer.

Sample questions it handles:
  "Why did LONDON-FACTORY-01 spike at 2pm yesterday?"
  "What's our ESG compliance status across all sites?"
  "Which site is our biggest carbon emitter this month?"
  "Generate a cost optimisation report for Rotterdam."
  "Compare Dubai office vs Frankfurt DC energy consumption."
  "When is our peak demand window? Should we consider BESS?"

Architecture:
  User message
      ↓
  EnergyAssistant.chat()
      ↓
  Anthropic API (with TOOLS defined in tools.py)
      ↓  (if tool_use in response)
  ToolExecutor.execute() — real VoltEdge data
      ↓
  Anthropic API (tool_result block)
      ↓
  Final natural language answer

Session management:
  ConversationSession stores message history per session_id.
  Sessions are in-memory (use Redis in production for multi-instance).
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator

import httpx

from voltedge.ai.executor import ToolExecutor
from voltedge.ai.tools import TOOLS
from voltedge.utils.logger import get_logger

log = get_logger(__name__)

MODEL          = "claude-sonnet-4-20250514"
MAX_TOKENS     = 1024
API_URL        = "https://api.anthropic.com/v1/messages"
MAX_TOOL_TURNS = 5    # Max agentic turns per user message


SYSTEM_PROMPT = """You are VoltEdge Assistant, an expert energy intelligence advisor
embedded in the VoltEdge platform. You help operators at large industrial enterprises
(factories, offices, data centres, warehouses) understand their energy consumption,
reduce costs, and meet sustainability targets.

You have access to real-time energy data through tools. Always use the tools to fetch
actual data before making recommendations — never guess at numbers.

When answering:
- Lead with the key finding, then supporting data
- Use specific numbers (kWh, kg CO₂, %, £/€/$) from the tool results
- Flag anomalies and their potential causes clearly
- Make cost and ESG implications concrete and actionable
- Keep responses concise but complete — operators are busy

If asked about a site you don't have data for, say so clearly rather than guessing."""


# ── Conversation session ──────────────────────────────────────────────────────

@dataclass
class ConversationSession:
    """In-memory conversation history for one user session."""
    session_id:  str             = field(default_factory=lambda: str(uuid.uuid4())[:8])
    messages:    list[dict]      = field(default_factory=list)
    created_at:  datetime        = field(default_factory=lambda: datetime.now(timezone.utc))
    last_active: datetime        = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata:    dict[str, Any]  = field(default_factory=dict)

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})
        self.last_active = datetime.now(timezone.utc)

    def add_assistant(self, content: list[dict] | str) -> None:
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        self.messages.append({"role": "assistant", "content": content})
        self.last_active = datetime.now(timezone.utc)

    def add_tool_result(self, tool_use_id: str, result: dict) -> None:
        """Append a tool_result block as a user message (Anthropic protocol)."""
        self.messages.append({
            "role": "user",
            "content": [{
                "type":        "tool_result",
                "tool_use_id": tool_use_id,
                "content":     json.dumps(result, default=str),
            }],
        })

    @property
    def turn_count(self) -> int:
        return sum(1 for m in self.messages if m["role"] == "user")

    def to_dict(self) -> dict:
        return {
            "session_id":  self.session_id,
            "turn_count":  self.turn_count,
            "created_at":  self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
        }


# ── Assistant ─────────────────────────────────────────────────────────────────

class EnergyAssistant:
    """
    VoltEdge AI assistant with tool use and multi-turn conversation.

    Args:
        executor:    ToolExecutor bound to live VoltEdge data.
        model:       Anthropic model ID.
        max_tokens:  Maximum tokens per API response.
        dry_run:     If True, skip the Anthropic API and return mock responses.
    """

    def __init__(
        self,
        executor:   ToolExecutor,
        model:      str   = MODEL,
        max_tokens: int   = MAX_TOKENS,
        dry_run:    bool  = False,
    ) -> None:
        self._executor   = executor
        self._model      = model
        self._max_tokens = max_tokens
        self._dry_run    = dry_run
        self._sessions:  dict[str, ConversationSession] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def new_session(self, metadata: dict | None = None) -> ConversationSession:
        """Create and register a new conversation session."""
        session = ConversationSession(metadata=metadata or {})
        self._sessions[session.session_id] = session
        log.info("assistant.session_created", session=session.session_id)
        return session

    def get_session(self, session_id: str) -> ConversationSession | None:
        return self._sessions.get(session_id)

    def chat(self, message: str, session_id: str | None = None) -> str:
        """
        Send a message and return the assistant's text response.
        Creates a new session if session_id is None or not found.

        This is the primary single-turn / multi-turn interface.
        Tool calls are resolved automatically before returning.
        """
        session = (
            self._sessions.get(session_id)
            if session_id else None
        ) or self.new_session()

        session.add_user(message)

        if self._dry_run:
            answer = self._dry_run_response(message, session)
            session.add_assistant(answer)
            return answer

        answer = self._agentic_loop(session)
        return answer

    def stream_chat(self, message: str, session_id: str | None = None) -> Iterator[str]:
        """
        Streaming variant of chat(). Yields text tokens as they arrive.
        Tool calls are still resolved synchronously (no streaming on tool turns).
        """
        session = (
            self._sessions.get(session_id) if session_id else None
        ) or self.new_session()

        session.add_user(message)

        if self._dry_run:
            response = self._dry_run_response(message, session)
            session.add_assistant(response)
            for word in response.split():
                yield word + " "
            return

        # Non-tool turns: stream; tool turns: resolve then stream final
        yield from self._stream_agentic_loop(session)

    def session_info(self) -> dict:
        return {
            "active_sessions": len(self._sessions),
            "sessions": [s.to_dict() for s in self._sessions.values()],
        }

    # ── Agentic loop ──────────────────────────────────────────────────────────

    def _agentic_loop(self, session: ConversationSession) -> str:
        """
        Run the multi-turn tool-use loop until a text answer is produced
        or the tool turn limit is reached.
        """
        for turn in range(MAX_TOOL_TURNS + 1):
            response = self._call_api(session.messages)
            content  = response.get("content", [])
            stop_reason = response.get("stop_reason", "")

            # Append assistant's full content block to session
            session.add_assistant(content)

            if stop_reason == "end_turn":
                # Extract text from response
                return self._extract_text(content)

            if stop_reason == "tool_use":
                # Execute all tool calls and append results
                tool_blocks = [b for b in content if b.get("type") == "tool_use"]
                for block in tool_blocks:
                    result = self._executor.execute(
                        block["name"], block.get("input", {})
                    )
                    session.add_tool_result(block["id"], result)
                    log.info("assistant.tool_executed",
                             tool=block["name"], session=session.session_id)
                continue   # Back to API with tool results

            # Unexpected stop reason
            return self._extract_text(content) or "I couldn't complete that request."

        return "I reached the maximum number of reasoning steps. Please try a more specific question."

    def _stream_agentic_loop(self, session: ConversationSession) -> Iterator[str]:
        """Streaming variant — yields text tokens, resolves tools synchronously."""
        for turn in range(MAX_TOOL_TURNS + 1):
            response = self._call_api(session.messages)
            content  = response.get("content", [])
            stop_reason = response.get("stop_reason", "")

            session.add_assistant(content)

            if stop_reason == "end_turn":
                text = self._extract_text(content)
                yield text
                return

            if stop_reason == "tool_use":
                tool_blocks = [b for b in content if b.get("type") == "tool_use"]
                for block in tool_blocks:
                    result = self._executor.execute(block["name"], block.get("input", {}))
                    session.add_tool_result(block["id"], result)
                continue

            yield self._extract_text(content) or ""
            return

        yield "I reached the maximum reasoning steps. Please try a simpler question."

    # ── API call ──────────────────────────────────────────────────────────────

    def _call_api(self, messages: list[dict]) -> dict:
        """Call the Anthropic Messages API and return the parsed response."""
        headers = {
            "Content-Type":      "application/json",
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model":      self._model,
            "max_tokens": self._max_tokens,
            "system":     SYSTEM_PROMPT,
            "tools":      TOOLS,
            "messages":   messages,
        }

        log.debug("assistant.api_call", model=self._model, turns=len(messages))

        with httpx.Client(timeout=60.0) as client:
            resp = client.post(API_URL, headers=headers, json=payload)
            resp.raise_for_status()
            return resp.json()

    # ── Dry run (no API calls) ────────────────────────────────────────────────

    def _dry_run_response(self, message: str, session: ConversationSession) -> str:
        """Return a canned response for testing without real API calls."""
        lower = message.lower()
        if "anomal" in lower:
            return (
                "Based on the data, I can see anomaly detection is active. "
                "In dry_run mode I cannot fetch live data — please set dry_run=False "
                "with a valid ANTHROPIC_API_KEY to get real insights."
            )
        if "esg" in lower or "emission" in lower or "carbon" in lower:
            return (
                "ESG metrics require live data access. In dry_run mode I return "
                "this placeholder. Set dry_run=False for real Scope 2 analysis."
            )
        if "forecast" in lower or "predict" in lower:
            return (
                "Demand forecasting requires model inference. In dry_run mode "
                "I cannot run the forecaster. Set dry_run=False for live predictions."
            )
        return (
            f"[DRY RUN] Received: '{message[:80]}...'. "
            "Set dry_run=False with ANTHROPIC_API_KEY to get real AI responses "
            "backed by live VoltEdge data."
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_text(content: list[dict]) -> str:
        """Extract concatenated text from a content block list."""
        parts = [b.get("text", "") for b in content if b.get("type") == "text"]
        return " ".join(parts).strip()
