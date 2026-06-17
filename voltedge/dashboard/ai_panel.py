"""
VoltEdge Dashboard — AI Assistant Chat Panel

Fully icon-based (no emojis). Bootstrap Icons throughout.
Improved error handling: styled API-offline card with instructions.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html, no_update

QUICK_ACTIONS = [
    "Summarise anomalies from the last 24 hours",
    "What is our Scope 2 emissions status?",
    "Which site has the highest energy cost?",
    "Show me the top cost-saving recommendations",
    "Compare London factory vs Dubai office consumption",
    "What caused the spike on LONDON-FACTORY-01?",
]

# ── Icon helpers ───────────────────────────────────────────────────────────────

def _bi(name: str, style: dict | None = None) -> html.I:
    base = {"fontSize": "0.85rem", "lineHeight": "1", "flexShrink": "0"}
    return html.I(className=f"bi bi-{name}", style={**base, **(style or {})})


# ── Bubble builders ────────────────────────────────────────────────────────────

def _user_bubble(text: str) -> html.Div:
    return html.Div(
        html.Div(text, className="ai-message-user",
                 style={"maxWidth": "78%", "marginLeft": "auto",
                        "marginBottom": "10px"}),
        style={"display": "flex", "justifyContent": "flex-end"},
    )


def _assistant_bubble(text: str, is_error: bool = False) -> html.Div:
    icon = _bi("exclamation-circle",
               {"color": "var(--accent-orange, #D85A30)", "fontSize": "0.9rem"}) \
           if is_error else \
           _bi("stars", {"color": "var(--teal, #1D9E75)", "fontSize": "0.9rem"})

    body: html.Element
    if is_error:
        body = _api_error_card(text)
    else:
        body = html.Div([
            html.Div([icon], style={"marginBottom": "5px"}),
            dcc.Markdown(text, style={"fontSize": "0.83rem", "margin": "0"},
                         dangerously_allow_html=False),
        ], className="ai-message-assistant",
           style={"maxWidth": "92%", "marginBottom": "10px"})

    return html.Div(body)


def _api_error_card(raw_error: str) -> html.Div:
    """Styled error card — replaces raw exception text dump."""
    is_refused = "61" in raw_error or "refused" in raw_error.lower() or \
                 "connect" in raw_error.lower()

    if is_refused:
        title   = "API server not running"
        message = "The VoltEdge API (port 8000) is not reachable from the dashboard."
        steps   = [
            "Open a new terminal tab",
            "cd ~/Downloads/voltedge",
            "uvicorn voltedge.api.app:app --reload --host 0.0.0.0 --port 8000",
            "Then return here and try again",
        ]
    else:
        title   = "API error"
        message = f"The request failed: {raw_error[:120]}"
        steps   = ["Check the terminal where the API server is running for details."]

    return html.Div([
        html.Div([
            _bi("exclamation-circle",
                {"color": "#D85A30", "fontSize": "1rem", "marginRight": "8px"}),
            html.Span(title, style={"fontWeight": "600", "fontSize": "0.83rem",
                                     "color": "#993C1D"}),
        ], className="d-flex align-items-center mb-2"),
        html.P(message, style={"fontSize": "0.78rem", "color": "#7a3216",
                                "marginBottom": "8px"}),
        *(html.Div([
            html.Span(f"{i + 1}.", style={"color": "#993C1D", "fontWeight": "600",
                                           "marginRight": "6px", "minWidth": "14px"}),
            html.Code(step, style={"fontSize": "0.75rem",
                                    "background": "rgba(153,60,29,0.1)",
                                    "padding": "1px 6px", "borderRadius": "4px"})
            if step.startswith("uvicorn") or step.startswith("cd")
            else html.Span(step, style={"fontSize": "0.77rem", "color": "#7a3216"}),
        ], className="d-flex align-items-start mb-1")
          for i, step in enumerate(steps)),
    ], style={
        "background": "#FEF3EE",
        "border": "0.5px solid #F0997B",
        "borderRadius": "10px",
        "padding": "12px 14px",
        "maxWidth": "95%",
        "marginBottom": "10px",
    })


def _no_key_card() -> html.Div:
    """Shown when neither direct mode nor HTTP mode is available."""
    return html.Div([
        html.Div([
            _bi("key", {"color": "#1D9E75", "fontSize": "1rem", "marginRight": "8px"}),
            html.Span("Set your API key to enable the AI assistant",
                      style={"fontWeight": "600", "fontSize": "0.83rem",
                             "color": "#1a5c44"}),
        ], className="d-flex align-items-center mb-2"),
        html.P("Add ANTHROPIC_API_KEY to your environment, then restart the dashboard:",
               style={"fontSize": "0.78rem", "color": "#2a7a5a", "marginBottom": "8px"}),
        html.Div([
            html.Code("export ANTHROPIC_API_KEY=sk-ant-your-key-here",
                      style={"fontSize": "0.75rem", "background": "rgba(29,158,117,0.1)",
                             "padding": "6px 10px", "borderRadius": "6px",
                             "display": "block", "color": "#0f5c3a"}),
        ], style={"marginBottom": "8px"}),
        html.P([
            "Get your key at ",
            html.A("console.anthropic.com", href="https://console.anthropic.com",
                   target="_blank",
                   style={"color": "#1D9E75", "fontSize": "0.76rem"}),
            ". No API server needed — the dashboard connects directly.",
        ], style={"fontSize": "0.76rem", "color": "#2a7a5a", "margin": "0"}),
    ], style={
        "background": "#e8f8f2",
        "border": "0.5px solid #1D9E75",
        "borderRadius": "10px",
        "padding": "12px 14px",
        "maxWidth": "95%",
        "marginBottom": "10px",
    })


# ── Main panel layout ──────────────────────────────────────────────────────────

def build_ai_panel() -> dbc.Card:
    """Return the AI assistant panel card."""
    return dbc.Card([
        dbc.CardHeader(
            dbc.Row([
                dbc.Col([
                    _bi("stars", {"color": "#1D9E75", "fontSize": "1rem",
                                   "marginRight": "8px"}),
                    html.Span("VoltEdge AI Assistant",
                              style={"fontWeight": "600", "fontSize": "0.88rem"}),
                    dbc.Badge("BETA", color="warning", pill=True,
                              className="ms-2",
                              style={"fontSize": "0.6rem", "fontWeight": "600",
                                     "verticalAlign": "middle",
                                     "padding": "3px 8px",
                                     "lineHeight": "1.4"}),
                    html.Span(id="ai-status-badge", className="ms-1",
                              style={"verticalAlign": "middle",
                                     "display": "inline-flex",
                                     "alignItems": "center"}),
                ], className="d-flex align-items-center"),
                dbc.Col(
                    dbc.Button(
                        _bi("chevron-up", {"fontSize": "0.8rem"}),
                        id="ai-panel-toggle", size="sm", color="link",
                        style={"color": "var(--text-dim, #9aa1ad)",
                                "padding": "2px 8px", "border": "none"},
                    ),
                    width="auto",
                ),
            ], align="center"),
            style={"padding": "10px 16px", "cursor": "pointer",
                    "background": "var(--bg-card, #fff)",
                    "borderBottom": "0.5px solid var(--border, rgba(15,23,42,0.07))"},
        ),

        dbc.Collapse(
            id="ai-panel-body",
            is_open=True,
            children=dbc.CardBody([
                dcc.Store(id="ai-session-id", storage_type="session"),

                # Chat history
                html.Div(
                    id="ai-chat-history",
                    style={
                        "height": "360px",
                        "overflowY": "auto",
                        "padding": "12px",
                        "background": "var(--bg-primary, #f4f6f9)",
                        "borderRadius": "10px",
                        "marginBottom": "12px",
                    },
                    children=[
                        _assistant_bubble(
                            "Hello! I'm the VoltEdge AI assistant. "
                            "I can analyse your energy data, explain anomalies, "
                            "calculate ESG metrics, and suggest cost optimisations. "
                            "What would you like to know?"
                        )
                    ],
                ),

                # API status indicator
                html.Div(id="ai-api-status", style={"marginBottom": "8px"}),

                # Quick-action chips
                html.Div([
                    html.Span(
                        label,
                        id={"type": "ai-chip", "index": i},
                        className="ai-chip",
                        style={"display": "inline-block", "marginRight": "6px",
                                "marginBottom": "6px"},
                    )
                    for i, label in enumerate(QUICK_ACTIONS)
                ], style={"marginBottom": "10px"}),

                # Input row
                dbc.InputGroup([
                    dbc.Textarea(
                        id="ai-input",
                        placeholder="Ask anything about your energy data…",
                        rows=2,
                        style={"fontSize": "0.83rem", "resize": "none",
                                "background": "var(--bg-primary, #f4f6f9)",
                                "color": "var(--text-primary, #1a1f2b)",
                                "border": "0.5px solid var(--border, rgba(15,23,42,0.07))"},
                    ),
                    dbc.Button([
                        _bi("send", {"fontSize": "0.85rem", "marginRight": "5px"}),
                        "Send",
                    ], id="ai-send-btn", n_clicks=0,
                       style={"borderRadius": "0 10px 10px 0",
                               "fontSize": "0.83rem", "fontWeight": "500",
                               "background": "#1D9E75", "border": "none",
                               "color": "#fff", "padding": "0 18px"}),
                ]),

                # Spinner target
                dbc.Spinner(html.Div(id="ai-spinner-target"), color="success",
                            size="sm",
                            spinner_style={"width": "0.9rem", "height": "0.9rem",
                                            "marginTop": "8px"}),
            ], style={"padding": "14px 16px"}),
        ),
    ], style={
        "background": "var(--bg-card, #fff)",
        "border": "0.5px solid var(--border, rgba(15,23,42,0.07))",
        "borderRadius": "14px",
        "marginTop": "0",
    })


# ── Callbacks ──────────────────────────────────────────────────────────────────

def register_ai_callbacks(app) -> None:

    # Toggle open/closed — swap chevron icon
    @app.callback(
        Output("ai-panel-body",   "is_open"),
        Output("ai-panel-toggle", "children"),
        Input("ai-panel-toggle",  "n_clicks"),
        State("ai-panel-body",    "is_open"),
        prevent_initial_call=True,
    )
    def toggle_panel(_, is_open):
        new_open = not is_open
        icon = "chevron-up" if new_open else "chevron-down"
        return new_open, html.I(
            className=f"bi bi-{icon}",
            style={"fontSize": "0.8rem", "color": "var(--text-dim, #9aa1ad)"},
        )

    # Fill input from chips
    @app.callback(
        Output("ai-input", "value", allow_duplicate=True),
        Input({"type": "ai-chip", "index": 0}, "n_clicks"),
        Input({"type": "ai-chip", "index": 1}, "n_clicks"),
        Input({"type": "ai-chip", "index": 2}, "n_clicks"),
        Input({"type": "ai-chip", "index": 3}, "n_clicks"),
        Input({"type": "ai-chip", "index": 4}, "n_clicks"),
        Input({"type": "ai-chip", "index": 5}, "n_clicks"),
        prevent_initial_call=True,
    )
    def fill_from_chip(*_):
        from dash import ctx
        if not ctx.triggered_id:
            return no_update
        idx = ctx.triggered_id.get("index", -1)
        return QUICK_ACTIONS[idx] if 0 <= idx < len(QUICK_ACTIONS) else no_update

    # Send message
    @app.callback(
        Output("ai-chat-history",  "children"),
        Output("ai-input",         "value", allow_duplicate=True),
        Output("ai-session-id",    "data"),
        Output("ai-spinner-target","children"),
        Input("ai-send-btn",       "n_clicks"),
        State("ai-input",          "value"),
        State("ai-chat-history",   "children"),
        State("ai-session-id",     "data"),
        prevent_initial_call=True,
    )
    def send_message(_, user_text, history, session_id):
        if not user_text or not user_text.strip():
            return no_update, no_update, no_update, no_update

        history = list(history or []) + [_user_bubble(user_text.strip())]
        answer  = None
        new_sid = session_id

        # ── Priority 1: direct mode (same process, no server needed) ──────────
        try:
            from voltedge.ai.direct import chat as direct_chat, api_key_available
            if api_key_available():
                answer, new_sid = direct_chat(user_text.strip(), session_id)
        except Exception as exc:
            answer = None  # fall through to HTTP mode

        # ── Priority 2: HTTP mode (API server on :8000) ───────────────────────
        if answer is None:
            try:
                import httpx
                payload = {"message": user_text.strip()}
                if session_id:
                    payload["session_id"] = session_id
                resp = httpx.post(
                    "http://localhost:8000/ai/chat",
                    json=payload, timeout=30.0,
                )
                resp.raise_for_status()
                body    = resp.json()
                answer  = body.get("response", "No response received.")
                new_sid = body.get("session_id", session_id)
            except Exception as exc:
                http_error = str(exc)
                answer = None

        # ── Priority 3: no key, no server → clear setup prompt ────────────────
        if answer is None:
            bubble = _no_key_card()
        else:
            bubble = _assistant_bubble(answer, is_error=False)

        history = list(history) + [bubble]
        return history, "", new_sid, ""

    # ── AI status badge ──────────────────────────────────────────────────────
    # Runs once on page load to show connection mode in the panel header.
    @app.callback(
        Output("ai-status-badge", "children"),
        Input("ai-panel-body", "is_open"),
    )
    def update_ai_status(_):
        from voltedge.ai.direct import api_key_available
        import httpx

        # Check direct mode first
        if api_key_available():
            return dbc.Badge(
                "Direct", color="success", pill=True,
                style={"fontSize": "0.6rem", "fontWeight": "600",
                       "padding": "3px 8px", "lineHeight": "1.4"},
                title="Connected directly — no API server needed",
            )

        # Check if HTTP server is reachable
        try:
            httpx.get("http://localhost:8000/health", timeout=1.0)
            return dbc.Badge(
                "API", color="info", pill=True,
                style={"fontSize": "0.6rem", "fontWeight": "600",
                       "padding": "3px 8px", "lineHeight": "1.4"},
                title="Connected via API server on :8000",
            )
        except Exception:
            pass

        return dbc.Badge(
            "No key", color="warning", pill=True,
            style={"fontSize": "0.6rem", "fontWeight": "600",
                   "padding": "3px 8px", "lineHeight": "1.4"},
            title="Set ANTHROPIC_API_KEY to enable AI",
        )
