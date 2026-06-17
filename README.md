# VoltEdge — Enterprise Energy Intelligence Platform

> Real-time energy monitoring, ML anomaly detection, causal AI root-cause analysis, and CSRD-ready ESG reporting — built for multi-site industrial and commercial operations.

**Built by [Shaz Moghaddam](https://shazmoghaddam.github.io) · [GitHub](https://github.com/ShazMoghaddam) · [LinkedIn](https://www.linkedin.com/in/shazmoghaddam/)**

---

## What it does

VoltEdge gives energy managers and sustainability leads a single platform to:

- **Monitor** energy consumption across all sites in real time
- **Detect anomalies** automatically using an isolation-forest model trained on each site's baseline
- **Explain anomalies** using a causal AI graph that identifies root causes with quantified causal strength scores
- **Report** Scope 2 emissions (location and market-based), GRI 302-1, and carbon intensity — CSRD-compliant
- **Ask questions** in plain English via the AI assistant ("What caused the spike on LONDON-FACTORY-01 last Thursday?")
- **Export** consumption and emissions data as CSV for auditors and finance teams

---

## Screenshots

| Portfolio overview | ESG & carbon | AI assistant |
|---|---|---|
| ![Portfolio](docs/screenshots/portfolio.png) | ![ESG](docs/screenshots/esg.png) | ![AI](docs/screenshots/ai.png) |

---

## Tech stack

| Layer | Technology |
|---|---|
| Dashboard | Plotly Dash 4, Bootstrap 5, Bootstrap Icons |
| API | FastAPI, Uvicorn |
| ML — anomaly | Scikit-learn isolation forest |
| ML — forecasting | Gradient boosting (scikit-learn) |
| ML — causal AI | PC algorithm (NetworkX) + root-cause analyser |
| Data persistence | SQLite (demo) / PostgreSQL-ready |
| AI assistant | Anthropic Claude (direct mode — no API server required) |
| Auth | JWT + RBAC |
| Language | Python 3.11 |

---

## Running locally

### Prerequisites

- Python 3.11+
- macOS or Linux (Windows via WSL)

### Setup

```bash
git clone https://github.com/ShazMoghaddam/voltedge.git
cd voltedge
pip install -r requirements.txt
pip install -e .
```

### Start the dashboard

```bash
PYTHONPATH=$(pwd) python3 voltedge/dashboard/app.py
```

Open `http://localhost:8050` in your browser.

The first run seeds a 90-day SQLite database (~34,000 readings across 4 demo sites) and pre-warms the anomaly detection models. This takes approximately 5–10 seconds. Subsequent runs start instantly.

### Enable the AI assistant (optional)

```bash
export ANTHROPIC_API_KEY=sk-ant-your-key-here
PYTHONPATH=$(pwd) python3 voltedge/dashboard/app.py
```

Without the API key the platform works fully — the AI assistant page shows setup instructions instead of a chat interface.

### Run the test suite

```bash
python3 -m pytest tests/ \
  --ignore=tests/test_models/test_lstm \
  --ignore=tests/test_models/test_lstm_forecaster.py \
  -q
```

903 tests, ~65 seconds on Apple Silicon M-series.

---

## Deployment

### Railway (recommended)

1. Fork or push to your GitHub account at [github.com/ShazMoghaddam](https://github.com/ShazMoghaddam)
2. Connect repo to [Railway](https://railway.app)
3. Railway auto-detects `railway.json` and deploys
4. Optionally set `ANTHROPIC_API_KEY` in Railway environment variables

### Render

1. Push to GitHub
2. Create a new Web Service on [Render](https://render.com)
3. Point to repo — Render reads `render.yaml` automatically
4. Set `ANTHROPIC_API_KEY` in environment variables if needed

Both platforms provide a public URL on deploy. Cold start is approximately 30–60 seconds on free tiers.

---

## Project structure

```
voltedge/
├── ai/              — AI assistant (Claude integration, tool executor)
├── api/             — FastAPI REST API
├── causal/          — PC-algorithm causal graph, root-cause analyser, counterfactuals
├── dashboard/       — Plotly Dash UI (layouts, callbacks, assets)
│   └── assets/      — voltedge.css (full design system)
├── db/              — SQLite models, seeder, loader
├── esg/             — ESG metrics, PDF report generator
├── ingestion/       — MQTT connector, simulators
├── models/
│   ├── anomaly/     — Isolation forest + in-memory cache
│   └── forecasting/ — GBM demand forecaster
├── processing/      — Feature engineering transformer
├── rbac/            — Role-based access control, SOC 2 audit trail
└── streaming/       — WebSocket broker, event hub, live simulator
tests/               — 903 tests across all modules
```

---

## Demo sites

The platform ships with 90 days of pre-seeded data for four realistic demo sites:

| Site | Type | Profile |
|---|---|---|
| LONDON-FACTORY-01 | Industrial factory | Strong weekday peaks at 10:00 and 14:00, low weekends |
| DUBAI-OFFICE-01 | Commercial office | Sun–Thu work week, high cooling load |
| ROTTERDAM-WAREHOUSE-01 | Logistics warehouse | Extended operating hours, flat load profile |
| FRANKFURT-DC-01 | Data centre | Flat 24/7 baseload, very low variance |

Each site has 4 deliberate anomaly events seeded across the 90-day window for demonstration purposes.

---

## Licence

MIT — see [LICENSE](LICENSE)

---

## Author

Designed and developed by **Shaz Moghaddam** · London

- 🌐 [shazmoghaddam.github.io](https://shazmoghaddam.github.io)
- 💼 [linkedin.com/in/shazmoghaddam](https://www.linkedin.com/in/shazmoghaddam/)
- 🐙 [github.com/ShazMoghaddam](https://github.com/ShazMoghaddam)
