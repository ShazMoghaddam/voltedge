"""
VoltEdge Dashboard — Layout v1.2
Fixes: no emojis (SVG/Bootstrap icons only), dbc.Card→html.Div for n_clicks,
softer dark mode, unified pill controls, footer always anchored, slider box removed.
"""
from __future__ import annotations
import dash_bootstrap_components as dbc
from dash import dcc, html
from voltedge.dashboard.ai_panel import build_ai_panel

# ── Design tokens ─────────────────────────────────────────────────────────────
COLORS = {
    "bg_primary":    "#f4f6f9",
    "bg_card":       "#ffffff",
    "bg_sidebar":    "#f8f9fb",
    "border":        "rgba(15, 23, 42, 0.07)",
    "accent_teal":   "#1D9E75",
    "accent_blue":   "#378ADD",
    "accent_orange": "#D85A30",
    "accent_green":  "#639922",
    "accent_purple": "#7F77DD",
    "accent_yellow": "#EF9F27",
    "accent_red":    "#E24B4A",
    "text_primary":  "#1a1f2b",
    "text_muted":    "#6b7280",
    "text_dim":      "#9aa1ad",
}

DARK_COLORS = {
    "bg_primary":  "#161b27",
    "bg_card":     "#1e2536",
    "text_primary":"#dde3ee",
}

SITE_COLORS = {
    "LONDON-FACTORY-01":      COLORS["accent_teal"],
    "DUBAI-OFFICE-01":        COLORS["accent_blue"],
    "ROTTERDAM-WAREHOUSE-01": COLORS["accent_yellow"],
    "FRANKFURT-DC-01":        COLORS["accent_purple"],
}

CHART_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", color=COLORS["text_primary"], size=12),
    xaxis=dict(gridcolor=COLORS["border"], showgrid=True, zeroline=False),
    yaxis=dict(gridcolor=COLORS["border"], showgrid=True, zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11), orientation="h",
                yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=52, r=20, t=36, b=40),
    hovermode="x unified",
)


SITES = [
    "LONDON-FACTORY-01", "DUBAI-OFFICE-01",
    "ROTTERDAM-WAREHOUSE-01", "FRANKFURT-DC-01",
]

TOOLTIPS = {
    "kWh":    "Kilowatt-hour — energy unit. 1 kWh = 1,000W running for 1 hour.",
    "CO2e":   "CO₂ equivalent — combines all greenhouse gases into one carbon number.",
    "scope2": "GHG Protocol Scope 2 — indirect emissions from purchased electricity.",
    "gri":    "GRI 302-1 — Global Reporting Initiative standard for energy consumption.",
    "mae":    "Mean Absolute Error — average forecast error in kWh. Lower is better.",
    "anom":   "Isolation-forest anomaly score 0–1. Above threshold = flagged event.",
}

# ── Icon helpers — Bootstrap Icons (already loaded via dbc.icons.BOOTSTRAP) ──

def _icon(name: str, style: dict | None = None) -> html.I:
    """Return a Bootstrap icon <i> element. name = bi class without 'bi-'."""
    base = {"fontSize": "0.88rem", "display": "inline-block", "verticalAlign": "middle"}
    return html.I(className=f"bi bi-{name}", style={**base, **(style or {})})

# Convenience aliases
def _icon_moon()    -> html.I: return _icon("moon-fill",         {"fontSize": "0.8rem"})
def _icon_sun()     -> html.I: return _icon("sun-fill",          {"fontSize": "0.9rem"})
def _icon_refresh() -> html.I: return _icon("arrow-clockwise",   {"fontSize": "0.85rem"})
def _icon_info()    -> html.I: return _icon("info-circle",       {"fontSize": "0.78rem",
                                                                    "color": COLORS["text_dim"]})


# ── Shared helpers ────────────────────────────────────────────────────────────

def _tooltip(target_id: str, text: str) -> dbc.Tooltip:
    return dbc.Tooltip(text, target=target_id, placement="top",
                       style={"fontSize": "0.76rem", "maxWidth": "240px"})

def section_label(text: str) -> html.Div:
    return html.Div(text, className="vt-section-label", style={
        "fontSize": "0.67rem",
        "textTransform": "uppercase", "letterSpacing": "1px",
        "marginBottom": "5px", "fontWeight": "500",
    })


# ── Causal AI insight ─────────────────────────────────────────────────────────

def build_causal_insight_card() -> html.Div:
    return html.Div(id="causal-insight-card", children=[],
                    style={"display": "none"}, className="mb-3")

def causal_insight_body(narrative: str, recommendation: str = "") -> dbc.Alert:
    children = [
        html.Div([
            html.I(className="bi bi-cpu", style={
                "color": COLORS["accent_orange"], "fontSize": "1rem", "marginRight": "8px",
            }),
            html.Span("Causal AI root-cause analysis", style={
                "fontWeight": "500", "fontSize": "0.8rem", "color": COLORS["accent_orange"],
            }),
        ], className="d-flex align-items-center mb-1"),
        html.Div(narrative, style={"fontSize": "0.78rem", "color": "#993C1D", "lineHeight": "1.6"}),
    ]
    if recommendation:
        children.append(html.Div(recommendation,
                                  style={"fontSize": "0.76rem", "color": "#993C1D",
                                         "marginTop": "8px", "fontWeight": "500"}))
    return dbc.Alert(children, color="warning", style={
        "background": "#FAECE7", "border": "0.5px solid #F0997B",
        "borderRadius": "12px", "padding": "12px 14px",
    })


# ── KPI card — html.Div wrapper so n_clicks works ─────────────────────────────

def kpi_card(title: str, value_id: str, unit: str,
             icon: str, color: str, delta_id: str,
             tooltip_text: str = "") -> html.Div:
    label_id = f"{value_id}-tip"
    inner = dbc.Card(dbc.CardBody([
        html.Div([
            html.I(className=f"bi {icon}", style={"color": color, "fontSize": "1rem"}),
            html.Span(title, id=label_id, style={
                "fontSize": "0.67rem",
                "textTransform": "uppercase", "letterSpacing": "1px",
                "marginLeft": "8px", "fontWeight": "500",
                "cursor": "help" if tooltip_text else "default",
                "borderBottom": f"1px dashed {COLORS['border']}" if tooltip_text else "none",
            }),
        ], className="d-flex align-items-center"),
        html.H3("—", id=value_id, style={
            "fontWeight": "500", "margin": "10px 0 2px", "fontSize": "1.7rem",
        }),
        html.Div([
            html.Small(unit, style={"color": COLORS["text_dim"], "fontSize": "0.72rem"}),
            html.Span("", id=delta_id, className="ms-2 trend-flat"),
        ], className="d-flex align-items-center"),
    ]))
    children: list = [inner]
    if tooltip_text:
        children.append(_tooltip(label_id, tooltip_text))
    # html.Div supports n_clicks — dbc.Card does not
    return html.Div(children, id=f"{value_id}-card",
                    className="kpi-wrapper h-100", n_clicks=0)


def chart_card(title: str, graph_id: str, color: str,
               height: int = 400, controls: list | None = None) -> dbc.Card:
    header = html.Div([
        html.Span(title, style={"fontWeight": "500", "fontSize": "0.85rem"}),
        html.Div(controls or [], className="ms-auto d-flex align-items-center gap-2"),
    ], className="d-flex align-items-center mb-2")
    return dbc.Card(dbc.CardBody([
        header,
        dcc.Graph(id=graph_id, style={"height": f"{height}px"},
                  config={"displayModeBar": False, "responsive": True}),
    ]))


# ── KPI drill-down modal ──────────────────────────────────────────────────────

def build_kpi_modal() -> dbc.Modal:
    return dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle(html.Span(id="kpi-modal-title"))),
        dbc.ModalBody(html.Div(id="kpi-modal-body",
                               style={"fontSize": "0.85rem", "lineHeight": "1.7"})),
        dbc.ModalFooter(
            dbc.Button("Close", id="kpi-modal-close", size="sm",
                       color="light", style={"borderRadius": "100px"})),
    ], id="kpi-modal", size="lg", is_open=False, scrollable=True)


# ── Navbar ────────────────────────────────────────────────────────────────────

def build_navbar() -> html.Div:
    return html.Div([
        # Logo
        html.Div([
            html.I(className="bi bi-lightning-charge-fill",
                   style={"color": COLORS["accent_teal"], "fontSize": "1.2rem"}),
            html.Span("VoltEdge",
                      className="vt-brand-text",
                      style={"fontSize": "1.05rem", "fontWeight": "600",
                              "marginLeft": "8px", "letterSpacing": "-0.3px"}),
        ], className="d-flex align-items-center"),

        # Site selectors
        html.Div([
            html.Div([
                section_label("Primary site"),
                dcc.Dropdown(id="site-selector",
                             options=[{"label": s, "value": s} for s in SITES],
                             value="LONDON-FACTORY-01", clearable=False,
                             style={"fontSize": "0.8rem", "minWidth": "210px"}),
            ], style={"marginLeft": "20px"}),
            html.Div([
                section_label("Compare"),
                dcc.Dropdown(id="compare-site-selector",
                             options=[{"label": "None", "value": "none"}] +
                                     [{"label": s, "value": s} for s in SITES],
                             value="none", clearable=False,
                             style={"fontSize": "0.8rem", "minWidth": "190px"}),
            ], style={"marginLeft": "10px"}),
        ], className="d-flex align-items-end gap-2"),

        # Right side (left→right): updated time · dark/light · refresh · live
        html.Div([
            # Updated time — subtle, unobtrusive
            html.Span(id="nav-last-updated"),

            # Dark / light toggle pill
            dbc.Button([
                html.I(className="bi bi-moon-fill", id="theme-icon",
                       style={"fontSize": "0.73rem"}),
                html.Span("Dark", id="theme-label"),
            ], id="theme-toggle-btn", size="sm", color="light",
               style={"marginLeft": "12px"}),

            # Export pill — matches Refresh style exactly
            dbc.Button([_icon("download"), "Export CSV"],
                       id="export-csv-btn", size="sm", color="success",
                       outline=True, style={"marginLeft": "8px"}),
            # Refresh pill
            dbc.Button([_icon_refresh(), "Refresh"],
                       id="manual-refresh-btn", size="sm",
                       color="success", outline=True,
                       style={"marginLeft": "8px"}),

            # Live — rightmost, most prominent
            html.Span([
                html.Span(className="live-dot"),
                "Live",
            ], id="live-badge", style={"marginLeft": "8px"}),

        ], className="d-flex align-items-center ms-auto"),
    ], className="d-flex align-items-center vt-navbar")



# ── Sidebar ───────────────────────────────────────────────────────────────────

def _nav_btn(label: str, icon: str, page_id: str) -> dbc.Button:
    return dbc.Button([
        html.I(className=f"bi {icon}",
               style={"fontSize": "0.88rem", "marginRight": "10px", "minWidth": "18px"}),
        html.Span(label, className="sidebar-label"),
    ], id=f"navlink-{page_id}", color="link",
       className="sidebar-navlink",
       style={"textDecoration": "none", "border": "none",
               "fontSize": "0.82rem", "color": COLORS["text_muted"],
               "borderRadius": "10px", "padding": "8px 14px",
               "width": "100%", "textAlign": "left", "marginBottom": "2px"})


def build_sidebar() -> html.Div:
    return html.Div([
        html.Div("Overview", className="sidebar-section-label",
                 style={"fontSize": "0.64rem", "textTransform": "uppercase", "letterSpacing": "1px",
                         "padding": "12px 14px 6px", "fontWeight": "500"}),
        _nav_btn("Portfolio",    "bi-grid-1x2",  "portfolio"),
        _nav_btn("ESG & carbon", "bi-recycle",   "esg"),
        _nav_btn("AI assistant", "bi-stars",     "ai"),
    ], className="vt-sidebar", style={
        "width": "200px", "minWidth": "200px", "flexShrink": "0",
        "display": "flex", "flexDirection": "column",
        "background": "#f8f9fb",
        "borderRight": "0.5px solid rgba(15,23,42,0.08)",
        "padding": "10px 8px",
    })


# ── Control panel ─────────────────────────────────────────────────────────────

def build_control_panel() -> dbc.Card:
    anom_info_id = "anom-thresh-info"
    return dbc.Card(dbc.CardBody([
        dbc.Row([
            # Time window
            dbc.Col([
                section_label("Time window"),
                dbc.ButtonGroup([
                    dbc.Button("24h",  id="tw-24",  n_clicks=0, size="sm", color="light", outline=True),
                    dbc.Button("7d",   id="tw-168", n_clicks=1, size="sm", color="success"),
                    dbc.Button("30d",  id="tw-720", n_clicks=0, size="sm", color="light", outline=True),
                ], className="tw-group"),
            ], md=3),
            # AI layers
            dbc.Col([
                section_label("AI layers"),
                dbc.Checklist(
                    id="model-toggles",
                    options=[{"label": " Forecast",  "value": "forecast"},
                             {"label": " Anomalies", "value": "anomaly"}],
                    value=["forecast", "anomaly"],
                    inline=True, switch=True,
                    className="mt-1 model-toggle-group",
                    style={"color": COLORS["text_muted"], "fontSize": "0.82rem"},
                ),
            ], md=3),
            # Anomaly threshold — slider + our own pill value display
            dbc.Col([
                html.Div([
                    section_label("Anomaly threshold"),
                    html.Span(id=anom_info_id, style={
                        "marginLeft": "6px", "cursor": "help",
                        "color": COLORS["text_dim"],
                        "display": "inline-flex", "alignItems": "center",
                    }, children=_icon_info()),
                    _tooltip(anom_info_id, TOOLTIPS["anom"]),
                ], className="d-flex align-items-center"),
                html.Div([
                    dcc.Slider(
                        id="anomaly-threshold",
                        min=0.5, max=0.95, step=0.05, value=0.75,
                        marks={0.5: "0.5", 0.75: "0.75", 0.95: "0.95"},
                        tooltip=False,
                        allow_direct_input=False,
                        className="mt-1 flex-grow-1",
                    ),
                    # Our own pill — all visual styling lives in CSS (#threshold-value-pill)
                    html.Span("0.75", id="threshold-value-pill"),
                ], className="d-flex align-items-center mt-1"),
            ], md=5),
        ], className="g-3 align-items-center"),
    ]), style={"marginBottom": "16px"})


# ── Portfolio page ────────────────────────────────────────────────────────────

def build_portfolio_page() -> html.Div:
    mae_id = "forecast-mae-tip"
    return html.Div([
        build_control_panel(),
        build_causal_insight_card(),

        # dcc.Loading wraps KPIs + charts — spinner appears on refresh/site change
        dcc.Loading(
            id="portfolio-loading",
            type="circle",
            color=COLORS["accent_teal"],
            style={"position": "fixed", "top": "50%", "left": "50%",
                   "transform": "translate(-50%,-50%)", "zIndex": "999"},
            children=[
                dbc.Row([
                    dbc.Col(kpi_card("Total consumption", "kpi-kwh",  "kWh",
                                     "bi-lightning-charge", COLORS["accent_teal"],
                                     "kpi-kwh-delta", TOOLTIPS["kWh"]), md=3, className="kpi-col"),
                    dbc.Col(kpi_card("CO₂ emissions",     "kpi-co2",  "kg CO₂e",
                                     "bi-cloud",            COLORS["accent_blue"],
                                     "kpi-co2-delta", TOOLTIPS["CO2e"]), md=3, className="kpi-col"),
                    dbc.Col(kpi_card("Anomalies flagged", "kpi-anom", "alerts",
                                     "bi-exclamation-triangle", COLORS["accent_red"],
                                     "kpi-anom-delta", TOOLTIPS["anom"]), md=3, className="kpi-col"),
                    dbc.Col(kpi_card("Peak demand",       "kpi-peak", "kW",
                                     "bi-graph-up-arrow",   COLORS["accent_green"],
                                     "kpi-peak-delta"), md=3, className="kpi-col"),
                ], className="mb-3 g-3"),

                dbc.Row([
                    dbc.Col(chart_card(
                        "Energy consumption + 24h forecast", "consumption-chart",
                        COLORS["accent_teal"], height=400,
                        controls=[
                            html.Span(id=mae_id,
                                      style={"cursor": "help", "color": COLORS["text_dim"],
                                              "display": "inline-flex", "alignItems": "center"},
                                      children=_icon_info()),
                            _tooltip(mae_id, TOOLTIPS["mae"]),
                            dbc.Badge("", id="forecast-mae-badge", color="light",
                                      style={"fontSize": "0.68rem", "color": COLORS["text_muted"],
                                             "border": f"0.5px solid {COLORS['border']}",
                                             "borderRadius": "100px"}),
                        ]
                    ), md=8),
                    dbc.Col(chart_card(
                        "Anomaly score stream", "anomaly-chart",
                        COLORS["accent_yellow"], height=400,
                    ), md=4),
                ], className="mb-3 g-3"),

                dbc.Row([
                    dbc.Col(chart_card(
                        "Consumption heatmap — hour × weekday", "heatmap-chart",
                        COLORS["accent_green"], height=340,
                    ), md=6),
                    dbc.Col(chart_card(
                        "Site comparison", "comparison-chart",
                        COLORS["accent_purple"], height=340,
                    ), md=6),
                ], className="mb-3 g-3"),

                dbc.Row([
                    dbc.Col(chart_card(
                        "Portfolio — all sites (daily kWh)", "portfolio-chart",
                        COLORS["accent_teal"], height=260,
                    )),
                ], className="mb-3 g-3"),
            ],
        ),

        build_kpi_modal(),
    ], id="page-portfolio")


def build_esg_page() -> html.Div:
    s2_id = "esg-scope2-info"
    gr_id  = "esg-gri-info"
    return html.Div([
        # ESG KPI summary row — reuse kpi_card helper so dark mode works
        dbc.Row([
            dbc.Col(kpi_card("Scope 2 (location)", "esg-kpi-scope2", "tCO₂e",
                             "bi-cloud", COLORS["accent_blue"], "esg-kpi-scope2-d"), md=3),
            dbc.Col(kpi_card("Carbon intensity", "esg-kpi-intensity", "kg CO₂/kWh",
                             "bi-activity", COLORS["accent_teal"], "esg-kpi-intensity-d"), md=3),
            dbc.Col(kpi_card("GRI 302-1 energy", "esg-kpi-gri", "GJ",
                             "bi-sun", COLORS["accent_yellow"], "esg-kpi-gri-d"), md=3),
            dbc.Col(kpi_card("DR savings (est.)", "esg-kpi-dr", "period",
                             "bi-currency-pound", COLORS["accent_green"], "esg-kpi-dr-d"), md=3),
        ], className="mb-3 g-3"),

        # Charts row 1
        dbc.Row([
            dbc.Col([
                chart_card("Scope 2 emissions — daily (location vs market-based)", "esg-chart",
                           COLORS["accent_orange"], height=340),
                html.Div([
                    html.Span(id=s2_id,
                              style={"cursor": "help", "color": COLORS["text_muted"],
                                     "fontSize": "0.72rem", "marginRight": "4px",
                                     "display": "inline-flex", "alignItems": "center", "gap": "3px"},
                              children=[_icon_info(), " Scope 2"]),
                    _tooltip(s2_id, TOOLTIPS["scope2"]),
                    html.Span(" · ", style={"color": COLORS["text_dim"], "margin": "0 6px"}),
                    html.Span(id=gr_id,
                              style={"cursor": "help", "color": COLORS["text_muted"],
                                     "fontSize": "0.72rem",
                                     "display": "inline-flex", "alignItems": "center", "gap": "3px"},
                              children=[_icon_info(), " GRI 302-1"]),
                    _tooltip(gr_id, TOOLTIPS["gri"]),
                ], style={"padding": "6px 4px"}),
            ], md=6),
            dbc.Col(chart_card(
                "Carbon intensity — rolling 7-day average (kgCO₂/kWh)", "esg-intensity-chart",
                COLORS["accent_teal"], height=340,
            ), md=6),
        ], className="mb-3 g-3"),

        # Charts row 2
        dbc.Row([
            dbc.Col(chart_card(
                "Estimated demand response savings — daily (£)", "esg-dr-chart",
                COLORS["accent_green"], height=300,
            ), md=6),
            dbc.Col(chart_card(
                "Forecast feature importance", "feature-importance-chart",
                COLORS["accent_purple"], height=300,
            ), md=6),
        ], className="mb-3 g-3"),
    ], id="page-esg", style={"display": "none"})


def build_ai_page() -> html.Div:
    return html.Div([build_ai_panel()],
                    id="page-ai", style={"display": "none"})


# ── Full layout ───────────────────────────────────────────────────────────────

def build_layout() -> html.Div:
    return html.Div([
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="site-data-store"),
        dcc.Store(id="time-window-store",  data=168),
        dcc.Store(id="theme-store", data="light", storage_type="local"),
        html.Div(id="theme-dom-sync", style={"display": "none"}),
        dcc.Store(id="active-page-store",  data="portfolio"),
        dcc.Interval(id="refresh-interval", interval=60_000, n_intervals=0),
        dcc.Download(id="download-csv"),

        build_navbar(),

        # Body row: sidebar + content — inline styles ensure layout even before CSS loads
        html.Div([
            build_sidebar(),
            html.Div([
                html.Div([
                    build_portfolio_page(),
                    build_esg_page(),
                    build_ai_page(),
                    dbc.Offcanvas(
                        id="anomaly-drawer", title="Anomaly detail",
                        placement="end", is_open=False,
                        children=[html.Div(id="anomaly-drawer-content")],
                        style={"width": "440px"},
                    ),
                ], className="vt-pages", style={"flex": "1"}),

                html.Footer([
                    html.Div("VoltEdge v1.0.0  ·  Enterprise Energy Intelligence",
                             style={"marginBottom": "3px"}),
                    html.Div([
                        "Designed and developed by ",
                        html.A("Shaz Moghaddam",
                               href="https://shazmoghaddam.github.io/",
                               target="_blank", rel="noopener noreferrer"),
                        "  ·  London",
                    ], style={"marginBottom": "3px"}),
                    html.Div([
                        html.A("Portfolio",
                               href="https://shazmoghaddam.github.io/",
                               target="_blank", rel="noopener noreferrer",
                               style={"margin": "0 8px"}),
                        "·",
                        html.A("GitHub",
                               href="https://github.com/ShazMoghaddam",
                               target="_blank", rel="noopener noreferrer",
                               style={"margin": "0 8px"}),
                        "·",
                        html.A("LinkedIn",
                               href="https://www.linkedin.com/in/shazmoghaddam/",
                               target="_blank", rel="noopener noreferrer",
                               style={"margin": "0 8px"}),
                    ]),
                ], className="vt-footer", style={
                    "textAlign": "center",
                    "fontSize": "0.69rem",
                    "color": "#9aa1ad",
                    "padding": "16px 0 20px",
                    "borderTop": "0.5px solid rgba(15,23,42,0.07)",
                    "marginTop": "24px",
                }),
            ], className="vt-content", style={
                "flex": "1", "minWidth": "0",
                "padding": "20px",
                "display": "flex", "flexDirection": "column",
            }),
        ], className="vt-body-row", style={
            "display": "flex", "flexDirection": "row",
            "alignItems": "stretch", "flex": "1",
        }),
    ], id="vt-root", style={
        "minHeight": "100vh", "display": "flex", "flexDirection": "column",
        "background": "#f3f5f8",
    })
