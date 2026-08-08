# CLAUDE.md — ML Pipeline Monitor (Permanent AI Instructions)

Use this file as the **source of truth** for AI coding assistant behavior when modifying this repository.

## 1) Project Overview & Goals

ML Pipeline Monitor is a production-oriented MLOps observability and operations platform. It provides:
- Streamlit dashboards for pipeline runs, experiment tracking, model registry, data drift, data health, and governance.
- Service-layer orchestration (Streamlit UI -> services -> core logic -> persistence).
- FastAPI inference API with rate limiting, Prometheus metrics, and JWT/auth.
- Celery background worker for scheduled/automated jobs.
- Prometheus + Grafana + Alertmanager monitoring stack.
- SQLite or PostgreSQL backend with connection pooling.
- E2E test automation via Playwright.

Primary goals:
- Reliability: deterministic orchestration and consistent persistence.
- Traceability: stage-level progress, lineage, and structured logging.
- Maintainability: clear modular boundaries and assistant-friendly conventions.

## 2) Architecture (UI -> Services -> Core -> Persistence)

```
Streamlit UI  ->  services/  ->  core/  +  ml/  ->  database/
    |                                |                   |
  app.py                      config, auth,          schema, models,
  pages/*.py                  logger, metrics        experiments, etc.
```

All business logic lives in `src/ml_pipeline_monitor/`. Streamlit pages and API routes are thin controllers that delegate to services.

## 3) Complete File Structure & Responsibilities

```
ML-pipeline-monitor/
├── app.py                           # Main Streamlit entry point ("Executive Command Center")
├── run_app.py                       # Launcher script (auto port discovery)
│
├── src/
│   └── ml_pipeline_monitor/         # <<< THE PYTHON PACKAGE (all source code)
│       ├── __init__.py
│       │
│       ├── api/                     # FastAPI inference API
│       │   ├── __init__.py
│       │   ├── __main__.py          # uvicorn launcher
│       │   └── main.py              # FastAPI app, routes, middleware, auth
│       │
│       ├── core/                    # Cross-cutting concerns
│       │   ├── __init__.py
│       │   ├── alerts.py            # Console + simulated email alerts
│       │   ├── auth.py              # Streamlit auth helpers (bcrypt)
│       │   ├── config_loader.py     # YAML config loading + secrets injection
│       │   ├── jwt_auth.py          # JWT token creation/verification
│       │   ├── logger.py            # Structured logging (JSON + console)
│       │   ├── metrics.py           # Prometheus metrics (30+ metrics)
│       │   ├── secrets.py           # SecretsManager (env -> Docker/K8s -> .secrets.json)
│       │   └── system_monitor.py    # psutil CPU/Memory/Disk metrics
│       │
│       ├── database/                # Persistence layer
│       │   ├── __init__.py          # Public API exports
│       │   ├── connection.py        # SQLite + PostgreSQL backends with pooling
│       │   ├── drift.py             # Drift report CRUD
│       │   ├── experiments.py       # Experiment CRUD
│       │   ├── governance.py        # Teams, users, workspaces, alerts, schedules
│       │   ├── interfaces.py        # DatabaseBackend/DatabaseConnection protocols
│       │   ├── lineage.py           # Dataset versions, schema changes, lineage edges
│       │   ├── models.py            # Model registry CRUD + stage events
│       │   ├── predictions.py       # Prediction history CRUD
│       │   └── schema.py            # Table DDL + initialization
│       │
│       ├── ml/                      # ML / data-science logic
│       │   ├── __init__.py
│       │   ├── data_loader.py       # Dataset loading and splitting (sklearn + synthetic)
│       │   ├── data_validation.py   # Quality score, anomalies, validation
│       │   ├── drift_detector.py    # KS test + PSI drift detection
│       │   ├── feature_store.py     # Caching splits with hashlib keys
│       │   ├── mlflow_tracker.py    # Optional MLflow integration
│       │   ├── model_cache.py       # In-memory model artifact caching
│       │   └── pipeline.py          # Stage-by-stage ML pipeline (7 stages)
│       │
│       ├── services/                # Application/use-case orchestration
│       │   ├── __init__.py
│       │   ├── app_service.py       # Dashboard snapshot, app initialization
│       │   ├── data_health_service.py  # Quality checks, schema comparison
│       │   ├── drift_service.py     # Orchestrates drift detection runs
│       │   ├── model_service.py     # Model registry + inference coordination
│       │   ├── pipeline_service.py  # Orchestrates pipeline runs
│       │   ├── telemetry_service.py # User action tracking
│       │   └── worker.py            # Celery background worker
│       │
│       └── utils/                   # UI / presentation helpers
│           ├── __init__.py
│           └── ui_theme.py          # Enterprise design system (HP-inspired)
│
├── pages/                           # Streamlit pages (auto-discovered by numbering)
│   ├── 0_Dataset_Management.py
│   ├── 1_Pipeline_Runner.py
│   ├── 2_Experiment_Tracking.py
│   ├── 3_Model_Registry.py
│   ├── 4_Data_Drift.py
│   ├── 5_Data_Health.py
│   └── 6_Governance.py
│
├── tests/
│   ├── conftest.py
│   ├── unit/                        # 13 unit test files
│   ├── integration/                 # 11 integration test files
│   ├── load/                        # Load tests
│   └── e2e/                         # 8 Playwright E2E tests
│
├── alembic/                         # Database migrations
│   ├── env.py
│   ├── script.py.mako
│   └── versions/
│
├── config/
│   ├── config.yaml                  # Development configuration
│   └── config.prod.yaml             # Production overrides
│
├── deployment/
│   ├── prometheus/                  # prometheus.yml
│   ├── grafana/                     # provisioning + dashboards
│   ├── alertmanager/                # alertmanager.yml
│   └── nginx/                       # nginx.conf + conf.d/
│
├── scripts/
│   ├── database/                    # init-db.sql, backup.py
│   └── development/                 # update_imports.py
│
├── data/                            # Dataset storage
│   ├── raw/
│   ├── processed/
│   └── sample/
│
├── artifacts/                       # Generated model/scaler artifacts
│   ├── models/
│   ├── scalers/
│   └── feature_store/
│
├── docs/
│   ├── DEPLOYMENT.md
│   └── OPERATIONS.md
│
├── .github/workflows/
│   ├── ci.yml                       # Lint, test, security, Docker build, E2E
│   └── e2e.yml                      # Dedicated E2E test pipeline
│
├── Dockerfile                       # Multi-stage build (base, deps, dev, prod, api, worker)
├── docker-compose.yml               # Production stack with profiles
├── docker-compose.dev.yml           # Development overrides
├── docker-compose.prod.yml          # Production overrides
├── alembic.ini                      # Alembic configuration
├── setup.py                         # Package setup (pip install -e .)
├── requirements.txt                 # Runtime dependencies
├── requirements-dev.txt             # Development dependencies
├── pytest.ini                       # Test config
├── .coveragerc                      # Coverage config
├── playwright.config.ts             # E2E test config
├── .gitignore
├── CLAUDE.md                        # This file
├── README.md
└── LICENSE
```

## 4) Coding Standards (Non-Negotiable)

- **Type hints**: Use explicit type hints for public functions/classes.
- **Docstrings**: Every module-level public function/class must have a docstring.
- **Modular design**: small, composable functions; single responsibility.
- **Avoid God classes**: no large all-in-one classes/modules.
- **Avoid duplicate code**: factor shared behavior into reusable helpers.
- **Correctness over cleverness**: prefer explicit, readable implementations.

## 5) Import Convention

All imports use the `ml_pipeline_monitor` package namespace:

```python
# Good
from ml_pipeline_monitor.services.drift_service import run_drift_detection
from ml_pipeline_monitor.database import initialize_db
from ml_pipeline_monitor.core.config_loader import load_config

# Bad (fragile relative imports)
from ..services.drift_service import run_drift_detection
```

## 6) Database Rules

- Use the existing persistence abstractions (do not bypass `database/connection.py`).
- Prefer parameterized queries / safe patterns used by the project.
- Ensure schema/lineage operations remain consistent with existing lineage tracking.
- Keep migrations/DB init consistent with current backend setup.

SQLite backend (`SQLiteBackend`):
- Connection pooling via `queue.Queue`
- WAL journal mode, foreign keys enabled
- `_PooledConnection` wrapper returns connections to pool on `close()`

PostgreSQL backend (`PostgresBackend`):
- Uses `psycopg-pool` (`ConnectionPool`)
- `PostgresConnectionAdapter` normalizes `?` -> `%s` and `datetime(...)` calls
- DSN from `PIPELINE_DB_DSN` env var or `storage.postgres_dsn` config

## 7) API Design Standards (FastAPI)

All endpoints in `src/ml_pipeline_monitor/api/main.py`:
- `GET /health` -- DB connectivity check
- `GET /health/live` -- liveness probe
- `GET /health/ready` -- readiness probe (DB check)
- `GET /health/detailed` -- system metrics + DB status
- `GET /metrics` -- Prometheus metrics endpoint
- `POST /v1/auth/login` -- JWT login
- `POST /v1/auth/refresh` -- JWT refresh
- `GET /v1/auth/me` -- Current user info
- `POST /v1/predict` -- Predict using latest production model (JWT required, rate limited)
- `POST /predict` -- Legacy predict (deprecated, API key required)

Design rules:
- Validate request bodies with Pydantic (`PredictRequest`).
- Return consistent response shapes; avoid leaking internal exceptions.
- Keep endpoints thin: route to `services/`.
- Middleware logs all requests with correlation/request IDs.
- Rate limiting via `slowapi` (default 60/min, configurable).
- Global exception handlers for: `Exception`, `RequestValidationError`, `RateLimitExceeded`.
- Graceful shutdown with signal handlers (SIGTERM, SIGINT).

## 8) Streamlit Page Conventions (`pages/*.py`)

Page functions should primarily:
- define widgets,
- call corresponding services,
- render results.

Do not implement business logic directly in pages.
Keep UI state localized; do not rely on implicit global mutation.
Any expensive computation should be delegated to services/core and cached if applicable.

### Pages Index
| File | Purpose |
|---|---|
| `pages/0_Dataset_Management.py` | Dataset preview + feature statistics |
| `pages/1_Pipeline_Runner.py` | Configure + execute training pipeline |
| `pages/2_Experiment_Tracking.py` | Compare runs + filter metrics |
| `pages/3_Model_Registry.py` | Promote models across stages |
| `pages/4_Data_Drift.py` | KS + PSI + drift reports + references |
| `pages/5_Data_Health.py` | Data quality checks + outlier summaries |
| `pages/6_Governance.py` | Audit trail + compliance policy UI |

## 9) Logging Conventions

- Use the repository logger abstraction (`ml_pipeline_monitor.core.logger`).
- Structured logging with JSON file output + colored console output.
- Context propagation via `contextvars`: `correlation_id`, `request_id`, `operation_context`, `actor_context`, `service_context`.
- Use `LogContext` context manager for setting contexts.
- Log with structured context where possible (IDs: run_id, experiment_id, dataset name, stage).
- Avoid logging secrets or raw credentials.
- Use log levels consistently: debug/info/warning/error.
- `ErrorCategory` enum for categorizing exceptions.

## 10) Testing Requirements

- Add/extend **unit tests** under `tests/unit/` for core and services.
- Add/extend **integration tests** under `tests/integration/` for service-layer flows.
- Add/extend **e2e tests** under `tests/e2e/` using Playwright when user-facing flows change.
- Maintain coverage expectations enforced by `pytest.ini` (80% minimum).
- Tests must be deterministic; avoid time-based flakiness.
- E2E tests use `pytest-playwright` with `playwright.config.ts`.

## 11) Git Commit Format

Use Conventional-like prefixes exactly as below:
- `feat:` new feature
- `fix:` bug fix
- `refactor:` structural improvement without behavior change
- `docs:` documentation-only changes

## 12) Dependency Policy

- Prefer existing libraries already used in `requirements.txt`.
- Avoid unnecessary new packages.
- Do not add heavy dependencies unless clearly required.

## 13) Performance Rules

- Cache expensive operations (use Streamlit `@st.cache_data` or existing caching in services/core).
- Feature store (`ml_pipeline_monitor.ml.feature_store`) caches processed dataset splits with hashlib-derived keys.
- Avoid repeated disk I/O (read once, reuse; memoize in-process when appropriate).
- Minimize unnecessary DB roundtrips.
- Database connection pooling: SQLite (5-connection pool), PostgreSQL (configurable via `connection_pool`).
- Streamlit page caching with TTL where applicable.

## 14) Security Rules

- **bcrypt only** for password hashing and verification.
- Secrets come from **environment variables** or `SecretsManager` (priority: env vars -> Docker/K8s secrets -> `.secrets.json`).
- Input validation everywhere:
  - validate request payloads (Pydantic in API),
  - validate UI form inputs,
  - validate any persistence-layer inputs.
- Rate limiting on API (`slowapi`, default 60/min).
- API key authentication (`X-API-Key` header) on legacy `/predict`.
- JWT authentication on `/v1/*` endpoints.
- Session timeout, max login attempts, lockout configured in `config.yaml`/`config.prod.yaml`.

## 15) Configuration System

| File | Purpose |
|---|---|
| `config/config.yaml` | Development/default configuration |
| `config/config.prod.yaml` | Production overrides |
| `src/ml_pipeline_monitor/core/config_loader.py` | Loads YAML, deep-merges with `DEFAULT_CONFIG`, injects secrets |
| `src/ml_pipeline_monitor/core/secrets.py` | `SecretsManager` with env -> files -> `.secrets.json` priority |

Key config sections:
- `pipeline`: random_seed, test_size, cv_folds
- `datasets`: display names, task types, sources
- `monitoring`: thresholds, automated retraining settings
- `storage`: backend selection, db_path, artifacts_root, connection_pool
- `auth`: enabled, session timeout, max login attempts
- `logging`: level, file path, rotation settings
- `alerting`: email simulation file, SMTP settings, Slack webhook
- `mlflow`: enabled, tracking URI, experiment name
- `ui`: max_experiments_displayed

Environment variables:
- `CONFIG_PATH` -- override config file path
- `PIPELINE_DB_DSN` -- PostgreSQL DSN
- `PIPELINE_DB` -- SQLite path override
- `MLMONITOR_AUTH_ENABLED`, `AUTH_USERNAME`, `AUTH_PASSWORD`, `AUTH_ROLE`, `AUTH_USERS_JSON`
- `MLMONITOR_API_KEY`, `MLMONITOR_RATE_LIMIT`
- `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`
- `SLACK_WEBHOOK`, `MLFLOW_TRACKING_URI`, `JWT_SECRET`, `JWT_ALGORITHM`

## 16) Monitoring & Observability Stack

### Prometheus (metrics collection)
- Scrapes: API (`/metrics`, 10s), Streamlit app (`/metrics`, 30s), self-monitoring
- Config: `deployment/prometheus/prometheus.yml`

### Grafana (visualization)
- Provisioned datasource: Prometheus
- Dashboards: `deployment/grafana/dashboards/`

### Prometheus Metrics (`ml_pipeline_monitor.core.metrics`)
- Pipeline: `ml_pipeline_runs_total`, `ml_pipeline_duration_seconds`, `ml_pipeline_stage_duration_seconds`
- API: `ml_api_requests_total`, `ml_api_request_duration_seconds`, `ml_api_errors_total`
- Predictions: `ml_predictions_total`, `ml_prediction_latency_seconds`
- Drift: `ml_drift_detections_total`, `ml_drift_score`, `ml_drift_features_count`
- System: CPU%, memory%, disk%, temperature (host + process)

## 17) Docker & Deployment

### Multi-stage Dockerfile
1. `base` -- Python 3.12-slim, non-root user (UID/GID 1000), system deps
2. `dependencies` -- pip install requirements
3. `development` -- dev deps, Playwright browsers, hot reload
4. `production` -- minimal image, only production code, healthcheck
5. `api` -- extends production, runs `uvicorn ml_pipeline_monitor.api.__main__:app`
6. `worker` -- extends production, runs `python -m ml_pipeline_monitor.services.worker`

### docker-compose.yml Services
| Service | Target | Port | Profile |
|---|---|---|---|
| app | production | 8501 | default |
| api | api | 8000 | default |
| worker | worker | -- | default |
| flower | mher/flower:2.0.1 | 5555 | monitoring |
| postgres | postgres:16-alpine | 5432 | postgres |
| redis | redis:7-alpine | 6379 | default |
| prometheus | prom/prometheus:v2.54.1 | 9090 | default |
| grafana | grafana/grafana:11.1.0 | 3000 | default |
| alertmanager | prom/alertmanager:v0.27.0 | 9093 | monitoring |
| nginx | nginx:alpine | 80/443 | production |

Start commands:
- Development: `docker-compose -f docker-compose.yml -f docker-compose.dev.yml up`
- Production: `docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d`
- With Postgres: `docker-compose --profile postgres up -d`

## 18) CLI Entry Points

```bash
mlmonitor-api        # Launch FastAPI server (from setup.py entry_points)
mlmonitor hash-password <password>  # Generate bcrypt hash (if cli.py exists)
```

## 19) ML Pipeline Stages

`ml_pipeline_monitor.ml.pipeline` implements `MLPipeline` with these stages:
1. Data Validation
2. Preprocessing (StandardScaler)
3. Feature Analysis
4. Cross-Validation (KFold / StratifiedKFold)
5. Training (Random Forest, XGBoost, Gradient Boosting, Logistic Regression, SVM, Decision Tree + regressors)
6. Evaluation (accuracy, precision, recall, F1, confusion matrix, ROC-AUC, MSE, MAE, R-squared)
7. Feature Importance Extraction

## 20) Model Lifecycle

Stages: `development` -> `staging` -> `production` -> `archived`
Governance page shows audit trail and stage change history.
`model_stage_events` table tracks all promotions/demotions.

## 21) Drift Detection

`ml_pipeline_monitor.ml.drift_detector`:
- KS test (scipy.stats) per feature
- PSI (Population Stability Index)
- Severity classification: none / moderate / significant
- Configurable thresholds in `config.yaml`

## 22) Data Quality

`ml_pipeline_monitor.ml.data_validation`:
- `ValidationResult` dataclass with quality_score (0-100), status, report, recommendations
- Missing values, duplicates, outlier detection (IQR + Z-score)
- `DataQualityFailed` exception stops training when score below threshold

`ml_pipeline_monitor.services.data_health_service`:
- `missing_value_report()`, `class_imbalance_report()`, feature analysis, shape validation
