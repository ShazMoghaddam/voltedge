"""
VoltEdge — Dashboard Entry Point

Run locally:
    PYTHONPATH=/path/to/voltedge python3 voltedge/dashboard/app.py

Deploy (Railway / Render):
    See railway.json / render.yaml in the project root.

The seeder and anomaly pre-warmer run in a background thread on first
startup. The dashboard is available immediately; data populates within
~5 seconds on first run, instantly on subsequent runs.
"""
from __future__ import annotations

import os
import threading

import dash
import dash_bootstrap_components as dbc

from voltedge.dashboard.ai_panel import register_ai_callbacks
from voltedge.dashboard.layouts import build_layout
import voltedge.dashboard.callbacks  # noqa: F401

_HERE = os.path.dirname(os.path.abspath(__file__))


def _seed_in_background() -> None:
    """Seed 90-day SQLite database then pre-warm anomaly models.
    Runs in a daemon thread so the dashboard is available immediately."""
    try:
        from voltedge.db.seeder import seed
        seed()
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(f"Seeder error: {exc}")

    try:
        from voltedge.models.anomaly.cache import prewarm_all
        prewarm_all()
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(f"Anomaly pre-warm error: {exc}")


def create_app() -> dash.Dash:
    app = dash.Dash(
        __name__,
        assets_folder=os.path.join(_HERE, "assets"),
        external_stylesheets=[
            dbc.themes.FLATLY,
            dbc.icons.BOOTSTRAP,
        ],
        title="VoltEdge | Energy Intelligence",
        suppress_callback_exceptions=True,
        meta_tags=[{
            "name": "viewport",
            "content": "width=device-width, initial-scale=1",
        }],
    )
    app.layout = build_layout()
    register_ai_callbacks(app)
    return app


if __name__ == "__main__":
    threading.Thread(target=_seed_in_background, daemon=True).start()
    app = create_app()

    # Use PORT env var if set (Railway / Render inject this)
    port = int(os.environ.get("PORT", 8050))
    debug = os.environ.get("VOLTEDGE_DEBUG", "false").lower() == "true"

    app.run(host="0.0.0.0", port=port, debug=debug)
