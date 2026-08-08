# ML Pipeline Monitor

Enterprise-grade MLOps platform for monitoring, tracking, and governing machine learning pipelines.

## Features

- **Dataset Management**: Centralized dataset hub with versioning and lineage tracking
- **Pipeline Orchestration**: Visual workflow execution with hyperparameter optimization
- **Experiment Tracking**: Comprehensive experiment logging and metric comparison
- **Model Registry**: Versioned model inventory with stage-based lifecycle management
- **Data Drift Detection**: Statistical monitoring using PSI and Kolmogorov-Smirnov tests
- **System Health**: Real-time infrastructure telemetry and audit logging
- **Governance**: Compliance tracking, audit trails, and policy enforcement

## Architecture

```
ML-pipeline-monitor/
├── app.py                    # Streamlit entry point
├── run_app.py                # Auto port discovery launcher
│
├── src/
│   └── ml_pipeline_monitor/  # Python package
│       ├── api/               # FastAPI inference API
│       │   ├── main.py        # Routes, middleware, auth
│       │   └── __main__.py    # uvicorn launcher
│       │
│       ├── core/              # Cross-cutting concerns
│       │   ├── config_loader.py
│       │   ├── logger.py
│       │   ├── metrics.py
│       │   ├── auth.py
│       │   ├── jwt_auth.py
│       │   ├── secrets.py
│       │   ├── alerts.py
│       │   └── system_monitor.py
│       │
│       ├── database/          # Persistence layer
│       │   ├── connection.py  # SQLite + PostgreSQL backends
│       │   ├── schema.py      # Table DDL + initialization
│       │   ├── experiments.py # Experiment CRUD
│       │   ├── models.py      # Model registry CRUD
│       │   ├── drift.py       # Drift report CRUD
│       │   ├── predictions.py # Prediction history
│       │   ├── governance.py  # Teams, users, workspaces
│       │   └── lineage.py     # Dataset lineage
│       │
│       ├── ml/                # ML / data-science logic
│       │   ├── pipeline.py    # Stage-by-stage ML pipeline
│       │   ├── data_loader.py # Dataset loading
│       │   ├── drift_detector.py
│       │   ├── feature_store.py
│       │   ├── model_cache.py
│       │   └── data_validation.py
│       │
│       ├── services/          # Business logic layer
│       │   ├── app_service.py
│       │   ├── pipeline_service.py
│       │   ├── model_service.py
│       │   ├── drift_service.py
│       │   ├── data_health_service.py
│       │   ├── telemetry_service.py
│       │   └── worker.py      # Celery background worker
│       │
│       └── utils/
│           └── ui_theme.py    # Enterprise design system
│
├── pages/                     # Streamlit UI pages
│   ├── 0_Dataset_Management.py
│   ├── 1_Pipeline_Runner.py
│   ├── 2_Experiment_Tracking.py
│   ├── 3_Model_Registry.py
│   ├── 4_Data_Drift.py
│   ├── 5_Data_Health.py
│   └── 6_Governance.py
│
├── tests/                     # Test suite
│   ├── unit/                  # 13 unit test files
│   ├── integration/           # 11 integration test files
│   ├── e2e/                   # 8 Playwright E2E tests
│   └── load/                  # Load tests
│
├── alembic/                   # Database migrations
│   └── versions/
│
├── config/
│   ├── config.yaml            # Development configuration
│   └── config.prod.yaml       # Production overrides
│
├── deployment/                # Infrastructure configs
│   ├── prometheus/
│   ├── grafana/
│   ├── alertmanager/
│   └── nginx/
│
├── scripts/
│   ├── database/              # init-db.sql, backup.py
│   └── development/           # update_imports.py
│
├── data/                      # Dataset storage
├── artifacts/                 # Generated model artifacts
├── docs/
└── .github/workflows/
    ├── ci.yml
    └── e2e.yml
```

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL (optional, SQLite works out of the box)
- Docker & Docker Compose (for monitoring stack)

### Installation

```bash
# Clone repository
git clone https://github.com/Aryan41211/ml-pipeline-monitor.git
cd ml-pipeline-monitor

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Initialize database
python -c "from ml_pipeline_monitor.database import initialize_db; initialize_db()"
```

### Running the Application

```bash
# Start Streamlit UI
streamlit run app.py

# Start FastAPI server (in separate terminal)
uvicorn ml_pipeline_monitor.api.main:app --reload --port 8000
```

### Docker Deployment

```bash
# Full stack with monitoring
docker-compose up -d

# Development mode
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# With PostgreSQL
docker-compose --profile postgres up -d
```

## Configuration

Edit `config/config.yaml` to customize:

```yaml
pipeline:
  test_size: 0.20
  random_seed: 42
  cv_folds: 5

storage:
  backend: sqlite  # or postgres
  db_path: .pipeline_monitor.db
  artifacts_root: artifacts

monitoring:
  drift_significance_level: 0.05
  psi_moderate_threshold: 0.10
  psi_significant_threshold: 0.25
```

## Testing

```bash
# Run unit + integration tests
pytest tests/ --ignore=tests/load --ignore=tests/e2e

# Run with coverage
pytest tests/ --cov=ml_pipeline_monitor --cov-report=html

# Run E2E tests (requires Playwright)
playwright install
pytest tests/e2e/

# Run load tests
pytest tests/load/ -v
```

## Tech Stack

**Backend:**
- FastAPI - REST API with JWT auth
- SQLAlchemy-compatible backends - SQLite + PostgreSQL
- scikit-learn, XGBoost - ML frameworks
- Prometheus - Metrics collection

**Frontend:**
- Streamlit - Web UI
- Plotly - Interactive charts
- Custom HP Design System - Enterprise UI components

**Infrastructure:**
- Docker - Multi-stage containerization
- PostgreSQL - Production database
- Redis - Caching & Celery broker
- Celery - Background tasks
- Prometheus + Grafana - Monitoring & alerting

## Documentation

- [Deployment Guide](docs/DEPLOYMENT.md)
- [Operations Manual](docs/OPERATIONS.md)
- [CLAUDE.md](CLAUDE.md) - Development guidelines and architecture decisions

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request
