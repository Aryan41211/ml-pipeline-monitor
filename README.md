# 🚀 ML Pipeline Monitor

**A beginner-friendly + production-minded MLOps observability platform** built with **Streamlit** (UI) and **FastAPI** (inference API).  
Track experiments, manage model lifecycles, monitor data drift, and correlate pipeline runs with system health — using a backend persistence layer (SQLite by default, PostgreSQL supported).

![MLOps Observability](https://img.shields.io/badge/MLOps-Observability-024ad8?style=flat&logo=mlflow)

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
- **Production inference API** (FastAPI) and model artifact loading

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

### Base URL
- `http://localhost:8000`

### Endpoints
- **`GET /health`** — service health check
- **`POST /predict`** — predict using the latest production model

### Example (curl)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"dataset":"Iris Species","features":{"sepal length (cm)":5.1,"sepal width (cm)":3.5,"petal length (cm)":1.4,"petal width (cm)":0.2}}'
```

Swagger docs:
- `http://localhost:8000/docs`

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

> Scheduling + fully automated retraining via cron is designed to integrate cleanly with the existing services and persistence layer.

---

## 🚧 Future Improvements

- Add **dataset lineage & versioning** UI
- Add **workspace-based isolation** and real **team workspaces**
- Add **DB-backed user/role management** (currently env-based auth for Streamlit)
- Add **Slack + email alert channels** (webhook + sink abstraction)
- Add **pipeline scheduling engine** with cron parsing and run history

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
