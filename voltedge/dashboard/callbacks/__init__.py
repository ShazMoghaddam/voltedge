"""
VoltEdge Dashboard — Callbacks
All Dash callbacks are registered here and imported by app.py.
Separated from layout to keep each module focused and testable.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, ctx, no_update

from voltedge.dashboard.layouts import CHART_BASE, COLORS, SITE_COLORS, SITES
from voltedge.models.anomaly import AnomalyDetector
from voltedge.models.forecasting import DemandForecaster
from voltedge.processing.transformer import EnergyTransformer

SITE_TYPES = {
    "LONDON-FACTORY-01":      "factory",
    "DUBAI-OFFICE-01":        "office",
    "ROTTERDAM-WAREHOUSE-01": "warehouse",
    "FRANKFURT-DC-01":        "data_center",
}

GRID_FACTOR = 0.207  # kg CO₂e / kWh (UK 2023)


# ── Data helpers ──────────────────────────────────────────────────────────────

def _load_df(site_id: str, hours: int) -> pd.DataFrame:
    """
    Load site readings from SQLite (primary) or simulator (fallback).
    The DB loader returns raw rows; the transformer adds ML features.
    """
    from voltedge.db.loader import load_site_readings, _fallback

    raw = load_site_readings(site_id, hours)
    if raw.empty:
        return pd.DataFrame()

    try:
        return EnergyTransformer().transform(raw)
    except Exception:
        return pd.DataFrame()


# ── Data cache — avoids redundant DB+transform on every callback ──────────────
import threading as _threading
from datetime import timedelta as _timedelta

_df_cache: dict[tuple, tuple["pd.DataFrame", "datetime"]] = {}
_df_lock  = _threading.Lock()
_DF_TTL   = _timedelta(minutes=5)


def _load_df_cached(site_id: str, hours: int) -> pd.DataFrame:
    """Thin in-memory cache over _load_df. TTL = 5 minutes.

    update_dashboard calls _load_df up to 6× per trigger (main df,
    prior-period df, portfolio chart × 4 sites). With a 5-minute TTL
    the DB is queried once per site per window and the result reused
    for the rest of that callback chain and any rapid re-triggers.
    """
    key = (site_id, hours)
    now = datetime.now(timezone.utc)
    with _df_lock:
        if key in _df_cache:
            df, cached_at = _df_cache[key]
            if now - cached_at < _DF_TTL:
                return df

    df = _load_df(site_id, hours)
    with _df_lock:
        _df_cache[key] = (df, now)
    return df



def _empty_fig(message: str = "No data") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font=dict(color=COLORS["text_muted"], size=14))
    return fig.update_layout(**CHART_BASE)


def _placeholder_fig(title: str, detail: str = "", icon: str = "") -> go.Figure:
    """
    Styled placeholder for when a model or data source is unavailable.
    More informative than _empty_fig — shows a title and a detail line.
    """
    fig = go.Figure()
    text = f"<b>{title}</b>" + (f"<br><span style='font-size:11px'>{detail}</span>" if detail else "")
    fig.add_annotation(
        text=text, x=0.5, y=0.5, xref="paper", yref="paper",
        showarrow=False, align="center",
        font=dict(color=COLORS["text_muted"], size=13),
    )
    return fig.update_layout(**CHART_BASE)


def _rgba(hex_color: str, alpha: float = 0.08) -> str:
    """Convert a '#RRGGBB' hex color to an 'rgba(r,g,b,a)' string for fillcolor."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# ── Time window toggle ────────────────────────────────────────────────────────

@callback(
    Output("time-window-store", "data"),
    Output("tw-24",  "color"),
    Output("tw-168", "color"),
    Output("tw-720", "color"),
    Output("tw-24",  "outline"),
    Output("tw-168", "outline"),
    Output("tw-720", "outline"),
    Input("tw-24",  "n_clicks"),
    Input("tw-168", "n_clicks"),
    Input("tw-720", "n_clicks"),
    prevent_initial_call=True,
)
def update_time_window(_a, _b, _c):
    triggered = ctx.triggered_id
    mapping = {"tw-24": 24, "tw-168": 168, "tw-720": 720}
    hours = mapping.get(triggered, 168)
    def _color(btn_id):
        return "success" if mapping[btn_id] == hours else "light"
    def _outline(btn_id):
        return mapping[btn_id] != hours
    return (
        hours,
        _color("tw-24"),  _color("tw-168"),  _color("tw-720"),
        _outline("tw-24"), _outline("tw-168"), _outline("tw-720"),
    )


# ── Main chart update ─────────────────────────────────────────────────────────

@callback(
    # KPIs
    Output("kpi-kwh",  "children"),
    Output("kpi-co2",  "children"),
    Output("kpi-anom", "children"),
    Output("kpi-peak", "children"),
    # Trend indicators
    Output("kpi-kwh-delta",  "children"),
    Output("kpi-co2-delta",  "children"),
    Output("kpi-anom-delta", "children"),
    Output("kpi-peak-delta", "children"),
    Output("kpi-kwh-delta",  "className"),
    Output("kpi-co2-delta",  "className"),
    Output("kpi-anom-delta", "className"),
    Output("kpi-peak-delta", "className"),
    # Charts
    Output("consumption-chart",        "figure"),
    Output("anomaly-chart",            "figure"),
    Output("heatmap-chart",            "figure"),
    Output("comparison-chart",         "figure"),
    Output("esg-chart",                "figure"),
    Output("feature-importance-chart", "figure"),
    Output("portfolio-chart",          "figure"),
    # Misc
    Output("nav-last-updated",   "children"),
    Output("forecast-mae-badge", "children"),
    # Inputs
    Input("site-selector",        "value"),
    Input("compare-site-selector","value"),
    Input("time-window-store",    "data"),
    Input("model-toggles",        "value"),
    Input("anomaly-threshold",    "value"),
    Input("refresh-interval",     "n_intervals"),
    Input("manual-refresh-btn",   "n_clicks"),
)
def update_dashboard(
    site_id, compare_id, hours, model_toggles, threshold, _interval, _refresh
):
    hours = hours or 168
    model_toggles = model_toggles or []
    color = SITE_COLORS.get(site_id, COLORS["accent_blue"])

    df = _load_df_cached(site_id, hours)
    _no_trend = ("", "trend-flat")
    if df.empty:
        empty = _empty_fig()
        return ("—", "—", "—", "—",
                "", "", "", "",
                "trend-flat","trend-flat","trend-flat","trend-flat",
                empty, empty, empty, empty, empty, empty, empty,
                "No data", "")

    ts  = df["timestamp"]
    kwh = df["kwh"]

    # ── KPIs ─────────────────────────────────────────────────────────────────
    today = pd.Timestamp.now(tz="UTC").date()
    today_mask = pd.to_datetime(ts).dt.date == today
    today_kwh  = float(kwh[today_mask].sum()) if today_mask.any() else float(kwh.sum())
    today_co2  = today_kwh * GRID_FACTOR
    peak_kw    = float(kwh.max())

    # ── Trend: compare to prior period ───────────────────────────────────────
    def _trend(current: float, prior: float) -> tuple[str, str]:
        """Return (label, css_class) for a trend indicator."""
        if prior <= 0:
            return "", "trend-flat"
        pct = (current - prior) / prior * 100
        if abs(pct) < 1:
            return "≈ flat", "trend-flat"
        arrow = "▲" if pct > 0 else "▼"
        cls   = "trend-up" if pct > 0 else "trend-down"
        return f"{arrow} {abs(pct):.1f}%", cls

    try:
        df_prior = _load_df_cached(site_id, hours * 2)
        if not df_prior.empty and "kwh" in df_prior.columns:
            half   = len(df_prior) // 2
            prior_kwh_total = float(df_prior["kwh"].iloc[:half].sum())
            prior_kwh_peak  = float(df_prior["kwh"].iloc[:half].max())
        else:
            prior_kwh_total = today_kwh
            prior_kwh_peak  = peak_kw
    except Exception:
        prior_kwh_total = today_kwh
        prior_kwh_peak  = peak_kw

    prior_co2 = prior_kwh_total * GRID_FACTOR
    kwh_trend,  kwh_cls  = _trend(today_kwh, prior_kwh_total)
    co2_trend,  co2_cls  = _trend(today_co2, prior_co2)
    peak_trend, peak_cls = _trend(peak_kw,   prior_kwh_peak)


    # ── Consumption + Forecast chart ─────────────────────────────────────────
    fig_cons = go.Figure()
    fig_cons.add_trace(go.Scatter(
        x=ts, y=kwh, mode="lines", name="Actual kWh",
        line=dict(color=color, width=1.8),
        fill="tozeroy", fillcolor=_rgba(color, 0.07),
        hovertemplate="%{y:.1f} kWh<extra></extra>",
    ))

    mae_label = ""
    if "forecast" in model_toggles:
        try:
            forecaster = DemandForecaster(site_id=site_id, horizon_hours=24)
            metrics = forecaster.train(df)
            forecast_df = forecaster.predict(df)

            fig_cons.add_trace(go.Scatter(
                x=forecast_df["timestamp"], y=forecast_df["kwh_forecast"],
                mode="lines", name="24h Forecast",
                line=dict(color=COLORS["accent_purple"], width=2, dash="dot"),
                hovertemplate="%{y:.1f} kWh<extra>Forecast</extra>",
            ))
            fig_cons.add_trace(go.Scatter(
                x=pd.concat([forecast_df["timestamp"], forecast_df["timestamp"][::-1]]),
                y=pd.concat([forecast_df["upper_bound"], forecast_df["lower_bound"][::-1]]),
                fill="toself", fillcolor=_rgba(COLORS["accent_purple"], 0.10),
                line=dict(color="rgba(0,0,0,0)"),
                name="95% CI", showlegend=True,
                hoverinfo="skip",
            ))
            mae_label = f"MAE {metrics['mae']:.1f} kWh"
        except Exception:
            pass

    fig_cons.update_layout(**CHART_BASE, yaxis_title="kWh",
                            title="", height=340)

    # ── Anomaly chart — uses cached model (trained once, reused across calls) ──
    anomaly_count = 0
    if "anomaly" in model_toggles:
        from voltedge.models.anomaly.cache import get_anomaly_results
        detector, anom_df = get_anomaly_results(site_id, df, hours=hours)

        if detector is None or anom_df is None:
            # Graceful placeholder — not a blank chart with faint text
            fig_anom = _placeholder_fig(
                icon="⚡",
                title="Insufficient data for anomaly model",
                detail=f"Need at least 48 hourly readings. "
                       f"Currently {len(df)} rows for {site_id} "
                       f"over {hours}h."
            )
        else:
            flagged       = anom_df[anom_df["anomaly_score"] >= threshold]
            anomaly_count = int(anom_df["is_anomaly"].sum())

            fig_anom = go.Figure()
            fig_anom.add_trace(go.Scatter(
                x=anom_df["timestamp"], y=anom_df["anomaly_score"],
                mode="lines", name="Score",
                line=dict(color=COLORS["accent_yellow"], width=1.5),
                fill="tozeroy", fillcolor=_rgba(COLORS["accent_yellow"], 0.12),
                hovertemplate="%{y:.3f}<extra>Anomaly Score</extra>",
            ))
            if not flagged.empty:
                fig_anom.add_trace(go.Scatter(
                    x=flagged["timestamp"], y=flagged["anomaly_score"],
                    mode="markers", name="Flagged",
                    marker=dict(color=COLORS["accent_red"], size=6, symbol="x"),
                    hovertemplate="%{customdata}<extra>Flagged</extra>",
                    customdata=flagged.get("anomaly_label", ["anomaly"] * len(flagged)),
                ))
            fig_anom.add_hline(y=threshold, line_dash="dot",
                                line_color=_rgba(COLORS["accent_red"], 0.5),
                                annotation_text=f"threshold {threshold}",
                                annotation_font_color=COLORS["accent_red"],
                                annotation_font_size=10)
            fig_anom.update_layout(**{
                **CHART_BASE,
                "yaxis": dict(range=[0, 1], gridcolor=COLORS["border"], zeroline=False, title="Score"),
                "height": 340,
            })
    else:
        fig_anom = _placeholder_fig(
            icon="○",
            title="Anomaly detection off",
            detail="Enable the Anomalies layer in AI Layers to see scores."
        )

    # ── Heatmap ───────────────────────────────────────────────────────────────
    df_h = df.copy()
    df_h["hour"] = pd.to_datetime(df_h["timestamp"]).dt.hour
    df_h["dow"]  = pd.to_datetime(df_h["timestamp"]).dt.day_name()
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    pivot = df_h.pivot_table(values="kwh", index="hour", columns="dow", aggfunc="mean")
    pivot = pivot.reindex(columns=[d for d in day_order if d in pivot.columns])

    fig_heat = go.Figure(go.Heatmap(
        z=pivot.values, x=list(pivot.columns), y=list(pivot.index),
        colorscale="Plasma",
        colorbar=dict(title="kWh", tickfont=dict(size=10)),
        hovertemplate="Hour %{y}:00 · %{x}<br>%{z:.1f} kWh<extra></extra>",
    ))
    fig_heat.update_layout(**CHART_BASE, yaxis_title="Hour of Day", height=300)

    # ── Site comparison ───────────────────────────────────────────────────────
    fig_comp = go.Figure()
    # Primary site daily
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    daily_primary = df.groupby("date")["kwh"].sum()
    fig_comp.add_trace(go.Bar(
        x=daily_primary.index, y=daily_primary.values,
        name=site_id.split("-")[0], marker_color=color, opacity=0.85,
        hovertemplate="%{y:,.0f} kWh<extra></extra>",
    ))
    if compare_id and compare_id != "none" and compare_id != site_id:
        df_cmp = _load_df_cached(compare_id, hours)
        if not df_cmp.empty:
            df_cmp["date"] = pd.to_datetime(df_cmp["timestamp"]).dt.date
            daily_cmp = df_cmp.groupby("date")["kwh"].sum()
            fig_comp.add_trace(go.Bar(
                x=daily_cmp.index, y=daily_cmp.values,
                name=compare_id.split("-")[0],
                marker_color=SITE_COLORS.get(compare_id, COLORS["accent_purple"]),
                opacity=0.85,
                hovertemplate="%{y:,.0f} kWh<extra></extra>",
            ))
    fig_comp.update_layout(**CHART_BASE, barmode="group",
                            yaxis_title="Daily kWh", height=300)

    # ── ESG chart ─────────────────────────────────────────────────────────────
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    daily = df.groupby("date")["kwh"].sum().reset_index()
    daily["co2_total"]     = daily["kwh"] * GRID_FACTOR
    daily["co2_renewable"] = daily["co2_total"] * 0.3  # 30% renewable assumption

    fig_esg = go.Figure()
    fig_esg.add_trace(go.Bar(
        x=daily["date"], y=daily["co2_total"],
        name="Total CO₂ (location)", marker_color=COLORS["accent_orange"], opacity=0.7,
    ))
    fig_esg.add_trace(go.Bar(
        x=daily["date"], y=daily["co2_renewable"],
        name="Net CO₂ (market-based)", marker_color=COLORS["accent_green"], opacity=0.9,
    ))
    fig_esg.update_layout(**CHART_BASE, barmode="overlay",
                           yaxis_title="kg CO₂e", height=280)

    # ── Feature importance chart ──────────────────────────────────────────────
    if "forecast" in model_toggles:
        try:
            fi_df = forecaster.feature_importance().head(10)
            fig_fi = go.Figure(go.Bar(
                x=fi_df["importance"], y=fi_df["feature"],
                orientation="h",
                marker=dict(
                    color=fi_df["importance"],
                    colorscale="Blues", showscale=False,
                ),
                hovertemplate="%{y}: %{x:.4f}<extra></extra>",
            ))
            fig_fi.update_layout(**{
                **CHART_BASE,
                "xaxis_title": "Importance",
                "yaxis": dict(autorange="reversed", gridcolor=COLORS["border"]),
                "height": 280,
            })
        except Exception:
            fig_fi = _empty_fig("Train forecast to see feature importance")
    else:
        fig_fi = _empty_fig("Enable Forecast layer")

    # ── Portfolio chart ───────────────────────────────────────────────────────
    fig_port = go.Figure()
    for sid in SITES:
        try:
            pf = _load_df_cached(sid, min(hours, 168))
            if not pf.empty:
                pf["date"] = pd.to_datetime(pf["timestamp"]).dt.date
                daily_pf = pf.groupby("date")["kwh"].sum()
                fig_port.add_trace(go.Scatter(
                    x=daily_pf.index, y=daily_pf.values,
                    mode="lines+markers", name=sid.split("-")[0],
                    line=dict(color=SITE_COLORS.get(sid, "#888"), width=2),
                    marker=dict(size=4),
                    hovertemplate=f"{sid}<br>%{{y:,.0f}} kWh<extra></extra>",
                ))
        except Exception:
            pass
    chart_base_port = {**CHART_BASE, "legend": dict(orientation="h", y=-0.25,
                                                       bgcolor="rgba(0,0,0,0)", font=dict(size=11))}
    fig_port.update_layout(**chart_base_port, yaxis_title="Daily kWh", height=230)

    # ── Final assembly ────────────────────────────────────────────────────────
    updated_str = f"Updated {datetime.now(timezone.utc).strftime('%H:%M')} UTC"
    anom_trend,  anom_cls  = (f"▲ {anomaly_count}", "trend-up") if anomaly_count > 0 else ("✓ clear", "trend-down")

    return (
        f"{today_kwh:,.0f}",
        f"{today_co2:,.1f}",
        str(anomaly_count),
        f"{peak_kw:,.1f}",
        # trend labels
        kwh_trend, co2_trend, anom_trend, peak_trend,
        # trend CSS classes
        kwh_cls, co2_cls, anom_cls, peak_cls,
        fig_cons, fig_anom, fig_heat, fig_comp,
        fig_esg, fig_fi, fig_port,
        updated_str,
        mae_label,
    )


# ── Anomaly drawer ────────────────────────────────────────────────────────────

@callback(
    Output("anomaly-drawer",         "is_open"),
    Output("anomaly-drawer-content", "children"),
    Input("anomaly-chart",  "clickData"),
    State("site-selector",  "value"),
    State("time-window-store", "data"),
    prevent_initial_call=True,
)
def open_anomaly_drawer(click_data, site_id, hours):
    if not click_data:
        return no_update, no_update
    point = click_data["points"][0]
    ts_str = point.get("x", "unknown")
    score  = point.get("y", 0.0)
    label  = point.get("customdata", "anomaly")
    from dash import html
    content = [
        html.P(f"Timestamp: {ts_str}", style={"fontSize": "0.85rem"}),
        html.P(f"Anomaly score: {score:.4f}", style={"fontSize": "0.85rem"}),
        html.P(f"Classification: {label}", style={"fontSize": "0.85rem",
                                                    "color": COLORS["accent_yellow"]}),
        html.Hr(style={"borderColor": COLORS["border"]}),
        html.P("Recommended actions:", style={"fontWeight": "600", "fontSize": "0.85rem"}),
        html.Ul([
            html.Li("Inspect sensor connectivity at this timestamp."),
            html.Li("Compare with neighbouring site baseline."),
            html.Li("Check maintenance log for scheduled activity."),
        ], style={"fontSize": "0.8rem", "color": COLORS["text_muted"]}),
    ]
    return True, content


# ── Sidebar page navigation ───────────────────────────────────────────────────

# ── URL routing ──────────────────────────────────────────────────────────────
# Two-callback pattern: nav buttons write to URL; URL controls what's shown.
# This gives browser back/forward and bookmarkable URLs for free.

_ROUTES = {
    "/":          "portfolio",
    "/portfolio": "portfolio",
    "/esg":       "esg",
    "/ai":        "ai",
}
_NAV_ROUTES = {
    "portfolio": "/portfolio",
    "esg":       "/esg",
    "ai":        "/ai",
}


@callback(
    Output("url", "pathname"),
    Input("navlink-portfolio", "n_clicks"),
    Input("navlink-esg",       "n_clicks"),
    Input("navlink-ai",        "n_clicks"),
    prevent_initial_call=True,
)
def navigate_to_page(_p, _e, _a):
    """Nav button clicks update the browser URL — no page shows/hides here."""
    try:
        triggered = ctx.triggered_id or "navlink-portfolio"
    except Exception:
        triggered = "navlink-portfolio"
    page = triggered.replace("navlink-", "")
    return _NAV_ROUTES.get(page, "/portfolio")


@callback(
    Output("page-portfolio",    "style"),
    Output("page-esg",          "style"),
    Output("page-ai",           "style"),
    Output("navlink-portfolio", "style"),
    Output("navlink-esg",       "style"),
    Output("navlink-ai",        "style"),
    Input("url",                "pathname"),
)
def switch_page(pathname):
    """URL change (incl. page load from bookmark) controls which page is visible."""
    active = _ROUTES.get(pathname or "/", "portfolio")

    page_styles = {
        "portfolio": {} if active == "portfolio" else {"display": "none"},
        "esg":       {} if active == "esg"       else {"display": "none"},
        "ai":        {} if active == "ai"         else {"display": "none"},
    }

    base   = {"fontSize": "0.82rem", "padding": "8px 14px", "borderRadius": "8px",
               "textDecoration": "none", "border": "none"}
    active_style   = {**base, "color": COLORS["accent_teal"],
                      "background": "#E1F5EE", "fontWeight": "500"}
    inactive_style = {**base, "color": COLORS["text_muted"]}

    link_styles = {k: (active_style if k == active else inactive_style)
                   for k in page_styles}

    return (
        page_styles["portfolio"], page_styles["esg"], page_styles["ai"],
        link_styles["portfolio"], link_styles["esg"], link_styles["ai"],
    )


# ── Causal AI insight card ────────────────────────────────────────────────────

@callback(
    Output("causal-insight-card", "children"),
    Output("causal-insight-card", "style"),
    Input("site-selector",     "value"),
    Input("time-window-store", "data"),
    Input("anomaly-threshold", "value"),
    Input("model-toggles",     "value"),
    Input("refresh-interval",  "n_intervals"),
)
def update_causal_insight(site_id, hours, threshold, model_toggles, _interval):
    """
    When the anomaly layer detects a flagged event, run causal root-cause
    analysis and surface the narrative inline above the charts.
    """
    from voltedge.dashboard.layouts import causal_insight_body

    hours = hours or 168
    model_toggles = model_toggles or []
    if "anomaly" not in model_toggles:
        return [], {"display": "none"}

    df = _load_df_cached(site_id, hours)
    if df.empty or len(df) < 48:
        return [], {"display": "none"}

    try:
        from voltedge.models.anomaly.cache import get_anomaly_results
        detector, anom_df = get_anomaly_results(site_id, df, hours=hours)
        if detector is None or anom_df is None:
            return [], {"display": "none"}
        flagged = anom_df[anom_df["anomaly_score"] >= threshold]
        if flagged.empty:
            return [], {"display": "none"}

        # Most recent flagged anomaly
        top = flagged.sort_values("timestamp").iloc[-1]

        from voltedge.causal.graph import CausalGraphBuilder
        from voltedge.causal.root_cause import RootCauseAnalyser

        graph = CausalGraphBuilder(min_samples=48).build(anom_df, site_id)
        analyser = RootCauseAnalyser(graph)
        report = analyser.analyse(
            anom_df, top["timestamp"], anomaly_variable="kwh",
            anomaly_label=str(top.get("anomaly_label", "")),
        )

        recommendation = report.recommendations[0] if report.recommendations else ""
        return causal_insight_body(report.narrative, recommendation), {"display": "block"}
    except Exception:
        return [], {"display": "none"}


# ── Dark / light mode toggle ──────────────────────────────────────────────────
from dash import clientside_callback

# Applies theme to DOM whenever theme-store changes (including on page load
# when localStorage value is restored — this eliminates the flash of light).
clientside_callback(
    """
    function(theme) {
        theme = theme || 'light';
        document.documentElement.setAttribute('data-theme', theme);

        var icon  = document.getElementById('theme-icon');
        var label = document.getElementById('theme-label');
        if (icon)  icon.className  = theme === 'dark' ? 'bi bi-sun-fill' : 'bi bi-moon-fill';
        if (label) label.innerText = theme === 'dark' ? 'Light' : 'Dark';

        var textColor = theme === 'dark' ? '#f1f5ff' : '#111827';
        var bgColor   = theme === 'dark' ? '#1e2d47' : '#ffffff';

        function applyDropdownTheme() {
            var wrappers = document.querySelectorAll(
                '.dash-dropdown-value, .dash-dropdown-value-item, ' +
                '.dash-dropdown-value-count, .dash-dropdown-wrapper'
            );
            for (var i = 0; i < wrappers.length; i++) {
                try {
                    wrappers[i].style.color = textColor;
                    wrappers[i].style.background = bgColor;
                    wrappers[i].style.webkitTextFillColor = textColor;
                } catch(e) {}
            }
            var inner = document.querySelectorAll('.dash-dropdown-wrapper *');
            for (var j = 0; j < inner.length; j++) {
                try {
                    var cn = inner[j].className;
                    // className on SVG elements is SVGAnimatedString — use typeof guard
                    var cnStr = (typeof cn === 'string') ? cn : '';
                    if (cnStr.indexOf('icon') >= 0 || cnStr.indexOf('clear') >= 0 || cnStr.indexOf('arrow') >= 0) continue;
                    inner[j].style.color = textColor;
                    inner[j].style.webkitTextFillColor = textColor;
                } catch(e) {}
            }
        }

        applyDropdownTheme();
        setTimeout(applyDropdownTheme, 200);
        setTimeout(applyDropdownTheme, 600);
        return '';
    }
    """,
    Output("theme-dom-sync", "children"),
    Input("theme-store", "data"),
)

# Toggle button flips the store value; the callback above applies it to the DOM.
clientside_callback(
    """
    function(n, current) {
        if (!n) return current || 'light';
        return (current === 'dark') ? 'light' : 'dark';
    }
    """,
    Output("theme-store", "data"),
    Input("theme-toggle-btn", "n_clicks"),
    State("theme-store", "data"),
    prevent_initial_call=True,
)


# ── KPI drill-down modal ──────────────────────────────────────────────────────

def _kpi_modal_content(card_id: str, site_id: str, hours: int) -> tuple[bool, str, list]:
    """Generate modal content for a clicked KPI card."""
    df = _load_df_cached(site_id, hours)
    if df.empty:
        return True, "No data", [html.P("No data available for this site.")]

    kwh = df["kwh"]
    ts  = df["timestamp"]
    today_kwh = float(kwh.sum())
    peak_kw   = float(kwh.max())
    avg_kw    = float(kwh.mean())
    co2_kg    = today_kwh * GRID_FACTOR

    if "kwh" in card_id:
        title = "Total Consumption — Detail"
        body  = [
            html.P(f"Total: {today_kwh:,.0f} kWh over {hours}h"),
            html.P(f"Average hourly load: {avg_kw:,.1f} kWh/h"),
            html.P(f"Peak: {peak_kw:,.1f} kW"),
            html.P(f"Equivalent cost (@ £0.28/kWh): £{today_kwh * 0.28:,.0f}"),
            html.P(f"Equivalent CO₂: {co2_kg:,.0f} kg "
                   f"(at UK grid factor {GRID_FACTOR} kg/kWh)"),
        ]
    elif "co2" in card_id:
        title = "CO₂ Emissions — Detail"
        body  = [
            html.P(f"Total CO₂: {co2_kg:,.1f} kg ({co2_kg/1000:.3f} tCO₂e)"),
            html.P(f"Carbon intensity: {GRID_FACTOR} kg CO₂/kWh (UK 2023 grid factor)"),
            html.P(f"Scope 2 (location-based): {co2_kg/1000:.4f} tCO₂e"),
            html.P("Scope 2 (market-based): Set VERRA_API_KEY to get REC-adjusted figure."),
            html.P("GRI 302-1 energy total (GJ): "
                   f"{today_kwh * 0.0036:,.2f} GJ"),
        ]
    elif "anom" in card_id:
        title = "Anomalies — Detail"
        try:
            from voltedge.models.anomaly.cache import get_anomaly_results
            _, anom_df = get_anomaly_results(site_id, df, hours=hours)
            if anom_df is None:
                body = [html.P("Anomaly model warming up — try again in a moment.")]
            else:
              flagged = anom_df[anom_df["anomaly_score"] >= 0.75]
            body = [
                html.P(f"Total anomalies detected: {len(flagged)} of {len(df)} readings"),
                html.P(f"Highest score: {anom_df['anomaly_score'].max():.3f}"),
                html.P(f"Anomaly rate: {len(flagged)/len(df)*100:.1f}%"),
            ]
            if not flagged.empty:
                body.append(html.P("Most recent anomalies:"))
                for _, row in flagged.tail(5).iterrows():
                    label = row.get("anomaly_label", "anomaly")
                    body.append(html.Li(
                        f"{str(row.get('timestamp',''))[:16]} — score {row['anomaly_score']:.3f}"
                        f" ({label})"
                    ))
        except Exception:
            body = [html.P("Anomaly model unavailable.")]
    else:
        title = "Peak Demand — Detail"
        body  = [
            html.P(f"Peak demand: {peak_kw:,.1f} kW"),
            html.P(f"Average demand: {avg_kw:,.1f} kW"),
            html.P(f"Load factor: {avg_kw/peak_kw:.1%} "
                   "(higher = more efficient use of capacity)"),
            html.P("Triad risk: peak demand during Nov–Feb 16:00–19:30 "
                   "incurs Transmission Network Use of System (TNUoS) charges."),
        ]
    return True, title, body


for _kpi_id in ("kpi-kwh", "kpi-co2", "kpi-anom", "kpi-peak"):
    @callback(
        Output("kpi-modal",       "is_open",   allow_duplicate=True),
        Output("kpi-modal-title", "children",  allow_duplicate=True),
        Output("kpi-modal-body",  "children",  allow_duplicate=True),
        Input(f"{_kpi_id}-card",  "n_clicks"),
        State("site-selector",    "value"),
        State("time-window-store","data"),
        prevent_initial_call=True,
    )
    def _open_kpi_modal(n, site_id, hours, _kpi_id=_kpi_id):
        if not n:
            return False, "", []
        is_open, title, body = _kpi_modal_content(_kpi_id, site_id or "LONDON-FACTORY-01", hours or 168)
        return is_open, title, body


@callback(
    Output("kpi-modal", "is_open"),
    Input("kpi-modal-close", "n_clicks"),
    prevent_initial_call=True,
)
def close_kpi_modal(_):
    return False


# ── Anomaly threshold value pill ──────────────────────────────────────────────

@callback(
    Output("threshold-value-pill", "children"),
    Input("anomaly-threshold", "value"),
)
def update_threshold_pill(value):
    """Keep our custom pill text in sync with the slider value.
    Visual styling (bg, colour, border) is handled entirely by CSS so dark mode works.
    """
    return f"{(value or 0.75):.2f}"


# ── Kill RC Slider tooltip via DOM (portal evades CSS) ───────────────────────
from dash import clientside_callback as _csc

_csc(
    """
    function(value) {
        // RC Slider renders tooltip in a React portal at document.body level.
        // CSS selectors can't reliably target portals, so we remove via JS.
        function removeTooltips() {
            document.querySelectorAll(
                '.rc-slider-tooltip, .rc-slider-tooltip-content, .rc-slider-tooltip-inner'
            ).forEach(function(el) { el.style.display = 'none'; });
        }
        removeTooltips();
        // Also observe for future tooltip appearances
        var obs = new MutationObserver(removeTooltips);
        obs.observe(document.body, { childList: true, subtree: true });
        return value;
    }
    """,
    Output("threshold-value-pill", "children", allow_duplicate=True),
    Input("anomaly-threshold", "value"),
    prevent_initial_call="initial_duplicate",
)


# ── CSV export ────────────────────────────────────────────────────────────────

@callback(
    Output("download-csv", "data"),
    Input("export-csv-btn", "n_clicks"),
    State("site-selector",     "value"),
    State("time-window-store", "data"),
    prevent_initial_call=True,
)
def export_csv(_, site_id, hours):
    """Download the current site's readings as a CSV file."""
    from dash import dcc as _dcc
    import io

    site_id = site_id or "LONDON-FACTORY-01"
    hours   = hours   or 168

    df = _load_df_cached(site_id, hours)
    if df.empty:
        return no_update

    # Keep the columns that are meaningful to a non-technical user
    export_cols = [c for c in [
        "timestamp", "kwh", "voltage_v", "current_a", "power_factor",
        "temperature_c", "hour", "is_weekend", "is_business_hour",
    ] if c in df.columns]

    export_df = df[export_cols].copy()
    export_df["co2_kg"] = (export_df["kwh"] * GRID_FACTOR).round(3)

    buf = io.StringIO()
    export_df.to_csv(buf, index=False)

    filename = f"voltedge_{site_id}_{hours}h.csv"
    return dict(content=buf.getvalue(), filename=filename)


# ── ESG page — KPIs + carbon intensity + DR savings ──────────────────────────

@callback(
    Output("esg-kpi-scope2",     "children"),
    Output("esg-kpi-intensity",  "children"),
    Output("esg-kpi-gri",        "children"),
    Output("esg-kpi-dr",         "children"),
    Output("esg-kpi-scope2-d",   "children"),
    Output("esg-kpi-intensity-d","children"),
    Output("esg-kpi-gri-d",      "children"),
    Output("esg-kpi-dr-d",       "children"),
    Output("esg-intensity-chart","figure"),
    Output("esg-dr-chart",       "figure"),
    Input("site-selector",       "value"),
    Input("time-window-store",   "data"),
    Input("refresh-interval",    "n_intervals"),
)
def update_esg_page(site_id, hours, _):
    """Populate all ESG page elements from the database."""
    site_id = site_id or "LONDON-FACTORY-01"
    hours   = hours or 168

    df = _load_df_cached(site_id, hours)
    if df.empty:
        empty = _empty_fig("No data")
        return "—", "—", "—", "—", empty, empty

    kwh_total    = float(df["kwh"].sum())
    co2_total_kg = kwh_total * GRID_FACTOR
    co2_total_t  = co2_total_kg / 1000
    intensity    = co2_total_kg / (kwh_total + 1e-9)
    gri_gj       = kwh_total * 0.0036  # kWh → GJ

    # Estimated DR savings: off-peak shifting of peak hours at £0.05/kWh differential
    peak_mask  = df["is_business_hour"] == 1 if "is_business_hour" in df.columns else pd.Series(False, index=df.index)
    peak_kwh   = float(df.loc[peak_mask, "kwh"].sum()) if peak_mask.any() else 0
    dr_savings = round(peak_kwh * 0.032, 2)  # Conservative 3.2p/kWh DR incentive

    # ── Carbon intensity rolling 7-day chart ─────────────────────────────────
    df_daily = (
        df.set_index("timestamp")
          .resample("D")["kwh"]
          .sum()
          .reset_index()
    )
    df_daily["co2_kg"]    = df_daily["kwh"] * GRID_FACTOR
    df_daily["intensity"] = df_daily["co2_kg"] / (df_daily["kwh"] + 1e-9)
    df_daily["intensity_roll7"] = df_daily["intensity"].rolling(7, min_periods=1).mean()

    fig_intensity = go.Figure()
    fig_intensity.add_trace(go.Scatter(
        x=df_daily["timestamp"], y=df_daily["intensity_roll7"],
        mode="lines", name="7-day rolling avg",
        line=dict(color=COLORS["accent_teal"], width=2),
        fill="tozeroy", fillcolor=_rgba(COLORS["accent_teal"], 0.08),
        hovertemplate="%{y:.3f} kg/kWh<extra></extra>",
    ))
    # Grid average reference line
    fig_intensity.add_hline(y=GRID_FACTOR, line_dash="dot",
                             line_color=_rgba(COLORS["accent_orange"], 0.7),
                             annotation_text="UK grid avg",
                             annotation_font_color=COLORS["accent_orange"],
                             annotation_font_size=10)
    fig_intensity.update_layout(**CHART_BASE, yaxis_title="kgCO₂/kWh", height=340)

    # ── DR savings chart ──────────────────────────────────────────────────────
    df_daily["dr_saving_gbp"] = (
        df_daily["kwh"] * 0.032 *
        (df_daily["timestamp"].dt.dayofweek < 5).astype(float)  # weekdays only
    )

    fig_dr = go.Figure()
    fig_dr.add_trace(go.Bar(
        x=df_daily["timestamp"], y=df_daily["dr_saving_gbp"].round(2),
        name="Est. DR saving",
        marker_color=COLORS["accent_green"],
        hovertemplate="£%{y:.2f}<extra>DR saving</extra>",
    ))
    fig_dr.update_layout(**CHART_BASE, yaxis_title="£ / day", height=300)

    return (
        f"{co2_total_t:.2f}",
        f"{intensity:.3f}",
        f"{gri_gj:,.1f}",
        f"£{dr_savings:,.0f}",
        "", "", "", "",   # delta pills (not used on ESG page)
        fig_intensity,
        fig_dr,
    )
