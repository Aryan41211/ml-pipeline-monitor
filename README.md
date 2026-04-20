# ML Pipeline Monitor

A production-ready MLOps observability platform that gives data science teams real-time visibility into model training pipelines, experiment outcomes, and data distribution health — all through a professional Streamlit interface backed by a local SQLite experiment store.

---

## Overview

Managing ML experiments at scale requires more than a notebook.  This platform provides:

- **End-to-end pipeline orchestration** with per-stage progress tracking and structured logging
- **Experiment tracking** with metric persistence, filtering, and visual comparison
- **Model registry** with lifecycle stage management (development → staging → production)
- **Data drift detection** using the Kolmogorov-Smirnov test and Population Stability Index (PSI)
- **System resource monitoring** to correlate pipeline latency with host pressure

---

## Screenshots

| Page | Description |
|---|---|
| Overview | At-a-glance dashboard with KPIs, recent runs, and system health |
| Pipeline Runner | Configure dataset, algorithm, and hyperparameters; monitor live progress |
| Experiment Tracking | Compare runs across metrics with interactive Plotly charts |
| Model Registry | Manage model lifecycle stages; benchmark performance |
| Data Drift | KS test + PSI analysis with distribution overlay visualisations |

---

## Technology Stack

| Layer | Technology |
|---|---|
| UI framework | [Streamlit](https://streamlit.io) |
| Inference API | [FastAPI](https://fastapi.tiangolo.com), [Uvicorn](https://www.uvicorn.org) |
| ML framework | [scikit-learn](https://scikit-learn.org) |
| Gradient boosting | [XGBoost](https://xgboost.readthedocs.io) |
| Statistical tests | [SciPy](https://scipy.org) |
| Visualisation | [Plotly](https://plotly.com/python/) |
| Data manipulation | [pandas](https://pandas.pydata.org), [NumPy](https://numpy.org) |
| Persistence | SQLite and PostgreSQL (switchable backend layer) |
| System monitoring | [psutil](https://psutil.readthedocs.io) |
| Experiment registry (optional) | [MLflow](https://mlflow.org) |
| Config | [PyYAML](https://pyyaml.org) |
| Serialisation | [joblib](https://joblib.readthedocs.io) |

---

## Supported Algorithms

**Classification**
- Random Forest
- XGBoost
- Gradient Boosting
- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree

**Regression**
- Random Forest Regressor
- XGBoost Regressor
- Gradient Boosting Regressor
- Ridge Regression
- Support Vector Regressor (SVR)
- Decision Tree Regressor

---

## Supported Datasets

- Breast Cancer Wisconsin (classification)
- Wine Recognition (classification)
- Iris Species (classification)
- Handwritten Digits (classification)
- Synthetic Classification (2,000 samples, 20 features)
- Synthetic Regression (2,000 samples, 20 features)

All datasets are sourced from scikit-learn's built-in dataset library — no external downloads required.

---

## Project Structure

```
ml-pipeline-monitor/
├── app.py                          # Home dashboard (Streamlit app file)
├── run_app.py                      # Recommended launcher (auto-detects free port)
├── Dockerfile                      # Container image for Streamlit + API
├── docker-compose.yml              # App, API, optional postgres profile
├── requirements.txt
├── setup.py
├── config.yaml                     # Application configuration
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions test + coverage gate
├── src/
│   ├── pipeline.py                 # Core ML pipeline with stage orchestration
│   ├── data_loader.py              # Dataset loading and splitting utilities
│   ├── database.py                 # Persistence layer (backend-aware)
│   ├── db_interface.py             # DB connection/backend interface contracts
│   ├── db_engine.py                # Backend implementations (sqlite + postgres)
│   ├── drift_detector.py           # KS test + PSI drift detection
│   └── system_monitor.py          # Host resource metrics (psutil)
├── pages/
│   ├── 1_Pipeline_Runner.py        # Interactive pipeline execution
│   ├── 2_Experiment_Tracking.py   # Experiment comparison and filtering
│   ├── 3_Model_Registry.py        # Model lifecycle management
│   └── 4_Data_Drift.py            # Distribution drift analysis
├── services/
│   ├── app_service.py              # App-level facade
│   ├── pipeline_service.py         # Pipeline orchestration + automation
│   ├── model_service.py            # Registry + inference model loading
│   ├── drift_service.py            # Drift run/persistence logic
│   └── api/
│       └── main.py                 # FastAPI inference service (/predict)
└── tests/
  ├── test_pipeline.py            # pipeline + drift + loader tests
  ├── test_api_service.py         # FastAPI endpoint tests
  └── test_database_lineage.py    # lineage + stage governance tests
```

---

## Architecture

This project follows a layered architecture:

- UI layer: Streamlit pages in `pages/`
- Service layer: `services/` orchestrates use-cases and decouples page logic
- Core layer: `src/` modules for pipeline, drift, loader, auth, and logging
- Persistence layer: `src/database.py` over backend abstraction in `src/db_engine.py` and contracts in `src/db_interface.py`

Data flow:

`Streamlit UI -> services -> core modules -> DB`

Inference flow:

`Client -> FastAPI /predict -> services/model_service.py -> production model artifacts`

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- pip

### Installation

```bash
git clone https://github.com/manpatell/ml-pipeline-monitor.git
cd ml-pipeline-monitor

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Running the application (recommended)

Use the launcher script to avoid port conflicts. It finds the first available port starting at 8501, starts Streamlit, and prints the final URL.

```bash
python run_app.py
```

### Running the application (manual)

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

### Running the FastAPI inference service

```bash
uvicorn services.api.main:app --host 0.0.0.0 --port 8000
```

API docs and endpoints:

- Swagger UI: `http://localhost:8000/docs`
- Health: `GET /health`
- Prediction: `POST /predict`

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"dataset":"Iris Species","features":{"sepal length (cm)":5.1,"sepal width (cm)":3.5,"petal length (cm)":1.4,"petal width (cm)":0.2}}'
```

### Running tests

```bash
pytest -q
```

Coverage is enforced through `pytest.ini` (includes `src/` and `services/`).
If total coverage drops below the configured minimum, pytest exits with a failure code.

---

## CI (GitHub Actions)

CI runs on every push and pull request via `.github/workflows/ci.yml`.

It performs:

- checkout repository
- setup Python 3.10 and 3.11
- install dependencies from `requirements.txt`
- run `pytest -q` (tests + coverage gate from `pytest.ini`)
- upload HTML coverage report artifact (`htmlcov`)

---

## Docker

### Build and run Streamlit app

```bash
docker build -t ml-pipeline-monitor .
docker run --rm -p 8501:8501 ml-pipeline-monitor
```

### Docker Compose (app + api)

```bash
docker compose up --build
```

### Optional Postgres profile (upgrade readiness)

```bash
docker compose --profile postgres up --build
```

Note: runtime backend remains SQLite by default; postgres profile is included for migration planning.

---

## Configuration

Pipeline defaults and UI settings are controlled via `config.yaml`:

```yaml
pipeline:
  random_seed: 42
  test_size: 0.20
  cv_folds: 5

monitoring:
  drift_significance_level: 0.05
  psi_moderate_threshold: 0.10
  psi_significant_threshold: 0.25

storage:
  backend: "sqlite"        # or "postgres"
  db_path: ".pipeline_monitor.db"
  postgres_dsn: ""
```

The database path can be overridden with the `PIPELINE_DB` environment variable:

```bash
PIPELINE_DB=/data/experiments.db streamlit run app.py
```

For PostgreSQL backend, set `storage.backend: "postgres"` and provide DSN:

```bash
PIPELINE_DB_DSN=postgresql://user:pass@localhost:5432/ml_monitor
```

### Authentication via Environment Variables

Authentication is environment-only (no default credentials in `config.yaml`).

Set either a single-user credential pair:

```bash
AUTH_USERNAME=admin
AUTH_PASSWORD=secure_pass
AUTH_ROLE=admin   # optional: viewer | operator | admin
```

Or set a JSON user map for multi-user access:

```bash
AUTH_USERS_JSON='{"viewer":{"password":"view_pass","role":"viewer"},"operator":{"password":"op_pass","role":"operator"},"admin":{"password":"admin_pass","role":"admin"}}'
```

Optional auth toggle:

```bash
MLMONITOR_AUTH_ENABLED=true
```

### Alerting

This project supports two alert surfaces:

- Console/file alert logs via structured logger
- Simulated email alerts written to a local sink file (`logs/alerts_email_simulated.log` by default)

Configure sink path in `config.yaml`:

```yaml
alerting:
  email_simulation_file: "logs/alerts_email_simulated.log"
```

Drift alerts and pipeline failures trigger both console and simulated email alerts.

### MLflow Integration (Optional)

MLflow logging is integrated and controlled by config:

```yaml
mlflow:
  enabled: true
  tracking_uri: "file:./mlruns"
  experiment: "ml-pipeline-monitor"
```

When enabled, pipeline service logs parameters, metrics, model artifact, and model object to MLflow.

---

## Pipeline Stages

Each training run proceeds through seven sequential stages:

1. **Data Validation** — schema checks, missing value counts, class balance
2. **Preprocessing** — StandardScaler fit on training split, applied to test
3. **Feature Analysis** — correlation matrix, high-correlation pair count, variance check
4. **Cross-Validation** — stratified K-fold CV with F1 (classification) or R² (regression)
5. **Training** — fit estimator on full training set with timing
6. **Evaluation** — accuracy, precision, recall, F1, ROC-AUC / RMSE, MAE, R² on holdout
7. **Feature Importance** — tree-based importances or coefficient magnitudes

---

## Drift Detection

The drift analysis page compares a reference window against a current window using:

- **Kolmogorov-Smirnov test**: non-parametric two-sample test for distribution equality
- **Population Stability Index (PSI)**: measures how much the distribution of a variable has shifted

| PSI | Interpretation |
|---|---|
| < 0.10 | Stable — no action required |
| 0.10 – 0.25 | Moderate shift — monitor closely |
| > 0.25 | Significant shift — consider retraining |

---

## License

MIT License.  See [LICENSE](LICENSE) for details.
