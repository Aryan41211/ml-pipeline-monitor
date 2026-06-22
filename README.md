# 🚀 ML Pipeline Monitor

**A production-ready MLOps observability and operations platform** built with **Streamlit** (UI) and **FastAPI** (inference API).  
Track experiments, manage model lifecycles, monitor data drift, and correlate pipeline runs with system health — using a backend persistence layer (SQLite by default, PostgreSQL supported).

Enterprise features:
- JWT authentication with refresh tokens
- Versioned REST API (`/v1/*`) with rate limiting
- Model caching for low-latency inference
- Alembic database migrations
- PostgreSQL + connection pooling
- Prometheus + Grafana + Alertmanager monitoring stack
- Structured logging with correlation IDs
- CI/CD with linting, security scanning, and Docker builds
- 180+ unit & integration tests with 80%+ coverage

---

## ✅ Project Overview

Managing ML experiments at scale requires more than notebooks. **ML Pipeline Monitor** provides:

- **Pipeline orchestration** with per-stage progress tracking (enterprise-style UI)
- **Experiment tracking** with persisted metrics & filters
- **Model registry** with lifecycle management:
  - `development → staging → production → archived`
- **Data drift detection**:
  - **Kolmogorov–Smirnov (KS)** test
  - **Population Stability Index (PSI)**
- **System resource monitoring** to understand performance bottlenecks
- **Dataset management & drift references**
- **Production inference API** (FastAPI) with JWT auth, model caching, and Prometheus metrics
- **Celery worker** for scheduled pipeline execution
- **Backup/restore** utilities for PostgreSQL and SQLite

---

## ✨ Features (Quick Table)

| Area | What you get | Where |
|---|---|---|
| Dataset Hub | Dataset preview + feature statistics | `pages/0_Dataset_Management.py` |
| Pipeline Runner | Configure + execute training pipeline | `pages/1_Pipeline_Runner.py` |
| Experiment Tracking | Compare runs + metrics | `pages/2_Experiment_Tracking.py` |
| Model Registry | Promote models across stages | `pages/3_Model_Registry.py` |
| Drift Detection | KS + PSI + drift reports | `pages/4_Data_Drift.py` |
| Data Health | Quality checks & governance context | `pages/5_Data_Health.py` |
| Governance | Audit trail & compliance policy UI | `pages/6_Governance.py` |

> Note: The UI and services are designed to support additional MLOps extensions like scheduling, alerts, and dataset lineage.

---

## 🧱 Architecture Diagram

### High-level flow (Mermaid)
```mermaid
flowchart TB
  UI[Streamlit Pages (pages/)] --> SVC[services/*]
  SVC --> CORE[src/*]
  CORE --> DB[src/database.py]
  DB --> STORE[(SQLite/Postgres)]
  SVC --> API[FastAPI (services/api/*)]
  API --> MODEL[Model Artifacts (joblib)]
```

### Layered architecture
- **UI layer**: `pages/*.py`
- **Service layer**: `services/` orchestration + integration
- **Core layer**: `src/` pipeline/drift/loader/metrics/auth helpers
- **Persistence layer**: `src/database.py` + backend abstraction in `src/db_engine.py`

---

## 🖼️ Screenshots

Add/attach screenshots from your app UI here. Common useful screenshots:
- Dataset Hub preview + missing values panel
- Pipeline Runner live progress timeline
- Experiment Tracking metric comparison chart
- Model Registry production/staging model cards
- Data Drift PSI/KS results

*(If you want, you can paste a screenshot list and I can format it.)*

---

## 🧰 Tech Stack

| Category | Tech |
|---|---|
| UI | Streamlit |
| Inference API | FastAPI + Uvicorn |
| ML | scikit-learn |
| Drift/Stats | SciPy-like statistical tests (KS test), PSI logic |
| Visuals | Plotly |
| Data | pandas + NumPy |
| Persistence | SQLite + PostgreSQL (switchable backend) |
| Artifacts | joblib (models + scalers) |
| Monitoring | Prometheus + Grafana (docker stack) |

---

## 📌 Tech Details (For Developers & Recruiters)

### Persistence & backend abstraction
The repository uses a backend-agnostic persistence layer:
- `src/database.py`: CRUD + domain persistence
- `src/db_engine.py`: backend connector (SQLite / PostgreSQL)
- `src/db_interface.py`: shared interface contracts

### Drift logic
- `src/drift_detector.py`: drift analysis (KS + PSI + per-feature results)
- `services/drift_service.py`: orchestrates drift runs and persists reports + references

### Pipeline logic
- `src/pipeline.py`: stage-by-stage training pipeline
- `services/pipeline_service.py`: caching splits, running pipeline, persisting experiments/models

---

## 📍 Project Structure

```text
ml-pipeline-monitor/
├── app.py                           # Dashboard entry (Streamlit)
├── run_app.py                       # Launcher script (auto port)
├── Dockerfile
├── docker-compose.yml
├── config.yaml
├── requirements.txt
├── src/
│   ├── pipeline.py                 # Core pipeline with stages
│   ├── data_loader.py             # Dataset loading and splitting
│   ├── database.py                # Persistence layer (SQLite/Postgres)
│   ├── db_engine.py               # Backend connectors
│   ├── drift_detector.py          # KS + PSI
│   ├── feature_store.py          # Caching splits
│   └── system_monitor.py         # Host metrics
├── pages/
│   ├── 0_Dataset_Management.py
│   ├── 1_Pipeline_Runner.py
│   ├── 2_Experiment_Tracking.py
│   ├── 3_Model_Registry.py
│   ├── 4_Data_Drift.py
│   ├── 5_Data_Health.py
│   └── 6_Governance.py
├── services/
│   ├── app_service.py            # app-level initialization
│   ├── pipeline_service.py      # orchestrates pipeline runs
│   ├── model_service.py         # loads production model artifacts
│   ├── drift_service.py         # orchestrates drift detection runs
│   └── api/
│       └── main.py              # FastAPI endpoints
└── tests/
    ├── test_pipeline.py
    ├── test_api_service.py
    └── test_database_lineage.py
```

---

## 🧠 Machine Learning Pipeline Flow

```mermaid
flowchart LR
  A[Load dataset & split] --> B[Preprocess (scaling)]
  B --> C[Feature analysis]
  C --> D[Cross-validation]
  D --> E[Train model]
  E --> F[Evaluate]
  F --> G[Persist experiment + artifacts]
  G --> H[Register model stage]
```

Pipeline stages are implemented in `src/pipeline.py`, while orchestration & persistence happen in `services/pipeline_service.py`.

---

## ♻️ MLOps Features

### Experiment lifecycle
- Each pipeline run is persisted as an experiment
- Metrics and model metadata are stored
- Model artifacts are saved to `artifacts/`

### Model lifecycle
- Models are registered automatically
- Stage changes are recorded for governance/audit views

### Drift monitoring
- Drift references and drift reports are stored
- Drift pages show KS/PSI interpretation and drift severity cues

### System health correlation
- Pipeline metrics can be correlated with system load over time

---

## 📡 API Documentation (FastAPI)

### Base URLs
- **V1 API**: `http://localhost:8000/v1`
- **Legacy API**: `http://localhost:8000` (deprecated)
- **Docs**: `http://localhost:8000/v1/docs`
- **Redoc**: `http://localhost:8000/v1/redoc`

### Authentication
The API supports two authentication methods:

1. **JWT Bearer Token** (recommended for v1):
   ```bash
   curl -X POST http://localhost:8000/v1/auth/login \
     -H "Content-Type: application/json" \
     -d '{"username":"admin","password":"password","refresh":true}'
   ```

2. **API Key** (legacy):
   ```bash
   curl -X POST http://localhost:8000/predict \
     -H "X-API-Key: YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"features": {...}}'
   ```

### V1 Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/auth/login` | Get JWT access + refresh tokens |
| `POST` | `/v1/auth/refresh` | Refresh access token |
| `GET` | `/v1/auth/me` | Get current user info |
| `POST` | `/v1/predict` | Predict with production model (rate limited) |

### Health Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Overall health + version |
| `GET` | `/health/live` | Liveness probe |
| `GET` | `/health/ready` | Readiness probe (DB check) |
| `GET` | `/health/detailed` | System metrics + DB status |
| `GET` | `/metrics` | Prometheus metrics |

### Example Prediction (V1)
```bash
# 1. Login
TOKEN=$(curl -s -X POST http://localhost:8000/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"password"}' | jq -r .access_token)

# 2. Predict
curl -X POST http://localhost:8000/v1/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"features":{"sepal length (cm)":5.1,"sepal width (cm)":3.5,"petal length (cm)":1.4,"petal width (cm)":0.2},"dataset":"Iris Species"}'
```

---

## 🏗️ Architecture Diagram

### High-level flow (Mermaid)
```mermaid
flowchart TB
  UI[Streamlit Pages (pages/)] --> SVC[services/*]
  SVC --> CORE[src/*]
  CORE --> DB[src/database.py]
  DB --> STORE[(SQLite/Postgres)]
  SVC --> API[FastAPI (services/api/*)]
  API --> MODEL[Model Artifacts (joblib)]
  API --> METRICS[Prometheus Metrics]
  METRICS --> GRAFANA[Grafana]
  METRICS --> ALERT[Alertmanager]
```

### Layered architecture
- **UI layer**: `pages/*.py`
- **Service layer**: `services/` orchestration + integration
- **Core layer**: `src/` pipeline/drift/loader/metrics/auth helpers
- **Persistence layer**: `src/database.py` + backend abstraction in `src/db_engine.py`
- **API layer**: `services/api/app.py` (FastAPI with JWT, rate limiting, model caching)
- **Monitoring layer**: Prometheus + Grafana + Alertmanager

---

## 📚 Dataset Management

### What the Dataset Hub does
- Lets users browse supported datasets
- Shows:
  - feature statistics
  - missing values
  - train/test split details
  - basic class/target distribution summaries

Datasets are sourced from scikit-learn or generated synthetically, so the project runs without external data downloads.

---

## 🔍 Monitoring & Drift Detection

The drift analysis compares:
- a **reference distribution** window
- against a **current distribution** window

### PSI interpretation table
| PSI | Meaning |
|---|---|
| < 0.10 | Stable |
| 0.10 – 0.25 | Moderate drift — monitor |
| > 0.25 | Significant drift — retrain recommended |

Drift artifacts:
- drift reports
- drift references per dataset

---

## 📦 Model Registry

Models move through stages:
- `development` (default)
- `staging` (pre-production)
- `production` (serving)
- `archived` (retired)

The **Governance** page visualizes the audit trail and stage change history.

---

## 🤖 Automated Retraining

Retraining is supported as a conceptual flow through:
- drift detection → recommendation/alerts → pipeline execution
- model performance checks → promotion/archiving
- Celery worker (`services/worker.py`) for scheduled execution

> Scheduling + fully automated retraining via cron is designed to integrate cleanly with the existing services and persistence layer.

---

## 🔒 Security

- **JWT authentication** with HS256 signing and refresh tokens
- **Rate limiting** on API endpoints (default: 60 req/min)
- **bcrypt** password hashing for credentials
- **Secrets management** via environment variables or `SecretsManager` (env → files → `.secrets.json`)
- **Structured logging** with correlation/request IDs for tracing
- **Input validation** at every boundary (UI, API, persistence)
- **Session security** with configurable timeout and max login attempts

---

## 📊 Monitoring & Observability

The platform includes a full observability stack:

| Component | Image | Port | Purpose |
|---|---|---|---|
| **Prometheus** | `prom/prometheus:v2.54.1` | 9090 | Metrics collection |
| **Grafana** | `grafana/grafana:11.1.0` | 3000 | Visualization dashboards |
| **Alertmanager** | `prom/alertmanager:v0.27.0` | 9093 | Alert routing |
| **Flower** | `mher/flower:2.0.1` | 5555 | Celery task monitoring |

### Prometheus Metrics
- Pipeline runs, stage durations, and status counters
- API request rates, latency histograms, and error counts
- Prediction latency and throughput
- Drift detections, scores, and feature counts
- System CPU, memory, disk, and temperature
- Model registry events (registrations, promotions)
- Dataset validation counts and row/column metrics

### Grafana Dashboards
Pre-provisioned dashboards for:
- System health overview
- Drift metrics tracking
- API performance monitoring
- Pipeline execution metrics

---

## 🔄 CI/CD

GitHub Actions workflows (`.github/workflows/ci.yml`):

| Job | Purpose |
|---|---|
| **lint** | Ruff + Black + isort checks |
| **security** | Bandit + Safety dependency scanning |
| **test** | Pytest on Python 3.10/3.11/3.12 with coverage |
| **docker-build** | Multi-stage Docker builds for app, API, worker |
| **e2e** | Playwright E2E tests |

Pull requests require:
- ✅ Lint pass
- ✅ Tests pass with >= 80% coverage
- ✅ Docker build succeeds
- ✅ Security scan clear

---

## 🗄️ Database

### Migrations
Alembic is configured for schema versioning:
```bash
alembic upgrade head
alembic revision --autogenerate -m "description"
```

Initial migration creates:
- `experiments`
- `models`
- `model_stage_events`
- `drift_reports`
- `drift_references`
- `predictions`
- `predictions_log`

### Backup/Restore
```bash
# SQLite backup
python -m scripts.backup backup sqlite .pipeline_monitor.db

# PostgreSQL backup
python -m scripts.backup backup postgres mydb --dsn "postgresql://user:pass@localhost/db"

# Restore
python -m scripts.backup restore sqlite backups/pipeline_monitor_20240101_120000.db
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose (for full stack)

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run Streamlit UI
streamlit run app.py

# Run FastAPI (separate terminal)
uvicorn services.api.main:app --reload --port 8000

# Run tests
pytest -q
```

### Docker Compose (Full Stack)
```bash
# Development
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Production
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# With PostgreSQL
docker-compose --profile postgres up -d
```

---

## 🧪 Testing

```bash
# Unit + integration tests
pytest tests/ --ignore=tests/load --ignore=tests/e2e

# With coverage report
pytest tests/ --ignore=tests/load --ignore=tests/e2e --cov=src --cov=services --cov-report=html

# E2E tests (requires Playwright)
playwright install chromium
pytest tests/e2e/ --base-url=http://localhost:8501

# Load tests (requires Locust)
locust -f tests/load/test_api_load.py --host=http://localhost:8000
```

**Current coverage**: 80%+ across 180+ tests

---

## 🔮 Roadmap

- [ ] Dataset lineage & versioning UI
- [ ] Workspace-based isolation and multi-team support
- [ ] DB-backed user/role management for Streamlit
- [ ] Slack + email alert channels
- [ ] Pipeline scheduling engine with cron parsing
- [ ] Model serving with A/B testing
- [ ] Feature store integration
- [ ] Kubernetes Helm charts

---

## 🤝 Contributing Guide

1. Fork the repository
2. Create a feature branch
   
```bash
   git checkout -b feat/your-feature
   
```
3. Run tests locally
   
```bash
   pytest -q
   
```
4. Ensure formatting and linting (if configured)
5. Open a Pull Request

---

## 🧾 License

MIT License. See [LICENSE](LICENSE) for details.
