# VoltEdge — Technical Product Requirements Document

**Version:** 1.0.0 | **Status:** Draft | **Last Updated:** 2026-05-13

---

## 1. Executive Summary

VoltEdge is a cloud-native energy intelligence platform targeting global energy enterprises (Shell, BP, Eni, TotalEnergies). It ingests real-time IoT sensor data from industrial sites, applies ML-driven analytics, and surfaces actionable insights through an interactive operations dashboard. Initial release targets SME pilots on a free-tier cloud footprint, scaling to enterprise contracts via premium APIs and multi-region cloud deployments.

**Business objective:** Enable operators to reduce energy costs by 12–25% and meet Scope 2 emissions targets without capital expenditure on proprietary hardware.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                       DATA SOURCES                          │
│  IoT Sensors (MQTT) · REST APIs · CSV Batch · ERP Feeds    │
└─────────────────────────────┬───────────────────────────────┘
                              │
                   ┌──────────▼──────────┐
                   │   INGESTION LAYER   │
                   │  BaseConnector ABC  │
                   │  SimulatedConnector │
                   │  MQTTConnector*     │
                   └──────────┬──────────┘
                              │ EnergyReading (Pydantic)
                   ┌──────────▼──────────┐
                   │  PROCESSING LAYER   │
                   │  EnergyTransformer  │
                   │  Clean→Features→Lag │
                   └──────────┬──────────┘
             ┌────────────────┼──────────────────┐
             │                │                  │
  ┌──────────▼──────┐ ┌───────▼──────┐ ┌────────▼──────┐
  │  STORAGE LAYER  │ │   ML LAYER   │ │   ESG LAYER   │
  │  LocalStore     │ │ Forecaster   │ │ ESGCalculator │
  │  S3Store*       │ │ AnomalyDet.  │ │ GHG Protocol  │
  │  Parquet/Hive   │ │ CostOptimizer│ │ GRI 302       │
  └─────────────────┘ └───────┬──────┘ └────────┬──────┘
                              └─────────┬────────┘
                                        │
                             ┌──────────▼──────────┐
                             │   DASHBOARD LAYER   │
                             │  Plotly Dash        │
                             │  KPIs · Charts      │
                             │  Anomaly Drawer     │
                             └─────────────────────┘
* planned Phase 2
```

**Layered dependency rule:** Each layer may only import from layers below it. Enforced by module structure and verified in CI.

---

## 3. Module Reference

| Module | Path | Responsibility |
|--------|------|---------------|
| Settings | `config/settings.py` | Pydantic-settings singleton; all env vars |
| Logger | `voltedge/utils/logger.py` | structlog JSON/console |
| BaseConnector | `voltedge/ingestion/base.py` | ABC + EnergyReading validation |
| SimulatedConnector | `voltedge/ingestion/simulators.py` | Deterministic demo data |
| EnergyTransformer | `voltedge/processing/transformer.py` | Clean → 30+ feature columns |
| LocalStore | `voltedge/storage/base.py` | Parquet persistence, Hive layout |
| DemandForecaster | `voltedge/models/forecasting/` | GBM 24h demand forecast |
| AnomalyDetector | `voltedge/models/anomaly/` | Isolation Forest scoring |
| CostOptimizer | `voltedge/models/optimization/` | ToU tariff analysis |
| ModelScheduler | `voltedge/core/scheduler.py` | APScheduler weekly retraining |
| EnergyPipeline | `voltedge/core/pipeline.py` | Ingest → process → store |
| ESGCalculator | `voltedge/esg/metrics.py` | Scope 2, GRI 302-1, CDP |
| Dashboard Layout | `voltedge/dashboard/layouts/` | Reusable Dash components |
| Dashboard Callbacks | `voltedge/dashboard/callbacks/` | All @callback definitions |
| CLI | `voltedge/cli.py` | pipeline/dashboard/esg commands |

---

## 4. Data Schema

### 4.1 EnergyReading (canonical ingestion schema)

```python
class EnergyReading(BaseModel):
    reading_id:    str        # UUID4
    site_id:       str        # "LONDON-FACTORY-01"
    sensor_id:     str        # "LONDON-FACTORY-01-MAIN"
    timestamp:     datetime   # Always UTC
    kwh:           float      # >= 0
    voltage_v:     float | None
    current_a:     float | None
    power_factor:  float | None   # 0–1
    temperature_c: float | None
    metadata:      dict       # Source-specific extras
```

### 4.2 Processed Feature Set (30 columns post-transformer)

| Group | Columns |
|-------|---------|
| Raw | `kwh`, `voltage_v`, `current_a`, `power_factor` |
| Temporal | `hour`, `day_of_week`, `month`, `is_weekend`, `is_business_hour` |
| Cyclic | `hour_sin/cos`, `dow_sin/cos`, `month_sin/cos` |
| Rolling | `kwh_roll_mean/std_{3,6,12,24}h`, `roll_max/min_24h` |
| Lag | `kwh_lag_{1,2,3,6,12,24,48,168}h` |
| Derived | `apparent_power_kva`, `pf_poor`, `kwh_delta`, `kwh_pct_change` |

### 4.3 Storage Layout (Hive-partitioned Parquet)

```
data/
├── raw/site_id=LONDON-FACTORY-01/year=2026/month=05/day=13/a1b2.parquet
└── processed/site_id=LONDON-FACTORY-01/year=2026/month=05/day=13/c3d4.parquet
```

Cloud-portable: `LocalStore` and `S3Store` produce identical partition paths.

---

## 5. ML Model Specifications

### 5.1 DemandForecaster

| Property | Value |
|----------|-------|
| Algorithm | Gradient Boosting Regressor (sklearn) |
| Validation | 5-fold TimeSeriesSplit |
| Key metrics | MAE, RMSE, R², MAPE, CV-MAE mean/std |
| Acceptance | CV-MAE ≤ 15% of mean consumption |
| Persistence | `model.pkl` + `scaler.pkl` + `metadata.json` |
| Retrain cadence | Weekly via ModelScheduler |

### 5.2 AnomalyDetector

| Property | Value |
|----------|-------|
| Algorithm | Isolation Forest (sklearn) |
| Scaler | RobustScaler (outlier-resistant) |
| Contamination | 5% (env-configurable) |
| Labels | `sudden_spike`, `flat_line_sensor`, `poor_power_factor`, `extreme_consumption` |
| Acceptance | Precision ≥ 0.70, Recall ≥ 0.60 on labelled test set |

### 5.3 CostOptimizer

| Property | Value |
|----------|-------|
| Method | Rule-based ToU tariff analysis |
| Tariffs | UK Half-Hourly, EU Industrial, US Commercial |
| Output | `OptimizationReport` with recommendations sorted by annual saving |

---

## 6. ESG Compliance Mapping

| Standard | Metric | VoltEdge Field |
|----------|--------|---------------|
| GHG Protocol Scope 2 | Location-based | `scope_2_location_based_tco2e` |
| GHG Protocol Scope 2 | Market-based (RECs) | `scope_2_market_based_tco2e` |
| GRI 302-1 | Energy consumption (GJ) | `gri_302_1` |
| GRI 302-3 | Energy intensity (kWh/m²) | `energy_intensity_kwh_per_unit` |
| CDP Climate | Annual CO₂ | `total_co2_kg` × annualisation factor |
| ISO 50001 | EnPI | Custom via `ESGCalculator.summarise_sites()` |

---

## 7. Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | `development` | `development` / `staging` / `production` |
| `CLOUD_PROVIDER` | `local` | `local` / `aws` / `azure` |
| `INGESTION_INTERVAL_SECONDS` | `60` | Pipeline poll frequency |
| `FORECAST_HORIZON_HOURS` | `24` | Forecaster lookahead |
| `ANOMALY_CONTAMINATION` | `0.05` | Isolation Forest rate |
| `MODEL_RETRAIN_HOURS` | `168` | Scheduler cadence |
| `CARBON_INTENSITY_KG_PER_KWH` | `0.207` | UK 2023 grid average |
| `LOG_FORMAT` | `console` | `console` (dev) / `json` (prod) |

---

## 8. Deployment Runbook

### Local Development
```bash
git clone https://github.com/your-org/voltedge.git && cd voltedge
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
voltedge pipeline simulate --site-type factory --hours 720
voltedge dashboard   # http://localhost:8050
```

### Docker (staging)
```bash
docker-compose up --build
# Dashboard: localhost:8050 · LocalStack S3: localhost:4566
```

### AWS Free Tier
1. `aws s3 mb s3://voltedge-data-dev`
2. Set `CLOUD_PROVIDER=aws`, `AWS_S3_BUCKET=voltedge-data-dev`
3. Deploy via Elastic Beanstalk (Python platform) or EC2 t2.micro

---

## 9. Architecture Decision Records

### ADR-001: Pydantic v2 settings + boundary validation
**Decision:** `pydantic-settings` for all config; `BaseModel` for `EnergyReading`.  
**Rationale:** Catches misconfigured environments and malformed sensor payloads at the ingestion boundary, not deep in the stack. Pydantic v2 is 5–50× faster than v1 for validation.

### ADR-002: Parquet + Hive partitioning
**Decision:** Snappy-compressed Parquet with `site_id/year/month/day` partitions.  
**Rationale:** 10–20× compression vs CSV. Zero-code migration to AWS Athena or Azure Synapse for SQL querying at scale.

### ADR-003: Gradient Boosting over LSTM (v1.0)
**Decision:** GBM (sklearn) rather than TensorFlow LSTM for initial forecaster.  
**Rationale:** Trains in seconds on free-tier CPU; comparable accuracy on hourly energy data (CV-MAE 8–14%). LSTM is the clear v2.0 upgrade path once GPU resources are available.

### ADR-004: `@property` for computed paths in Settings
**Decision:** Computed filesystem paths are `@property` methods, not Pydantic fields.  
**Rationale:** Pydantic v2 validates fields before `model_post_init`, causing `ValidationError: None is not a valid Path`. Properties bypass this and compute lazily.

### ADR-005: DatetimeIndex for time interpolation
**Decision:** `_clean()` temporarily sets timestamp as index before `interpolate(method="time")`, then resets to `RangeIndex`.  
**Rationale:** pandas 2.x enforces DatetimeIndex for time-weighted interpolation. Two extra index operations per transform call is negligible overhead.

---

## 10. Test Coverage

| Suite | Tests | Status |
|-------|-------|--------|
| Ingestion / Simulator | 5 | ✅ All pass |
| Processing / Transformer | 8 | ✅ All pass |
| ML / DemandForecaster | 5 | ✅ All pass |
| ML / AnomalyDetector | 4 | ✅ All pass |
| ML / CostOptimizer | 12 | ✅ All pass |
| ML / ModelScheduler | 4 | ✅ All pass |
| Dashboard / Layout | 7 | ✅ All pass |
| **Total** | **45** | **45/45 ✅** |

`pytest tests/ -v` from project root.

---

## 11. Roadmap

| Phase | Deliverables |
|-------|-------------|
| **Next (v1.1)** | S3Store + AzureBlobStore · MQTT connector · FastAPI REST layer · JWT auth |
| **v2.0** | LSTM forecaster · ESG PDF export · Slack/PagerDuty alerts · CI/CD pipeline |
| **v3.0 Enterprise** | SAP/Oracle ERP adapters · RBAC · SOC 2 Type II · Multi-region · White-label OEM |

---
*Maintained by VoltEdge Engineering. Raise a PR to update.*
