# CLAUDE.md — ML Pipeline Monitor (Permanent AI Instructions)

Use this file as the **source of truth** for AI coding assistant behavior when modifying this repository.

## 1) Project Overview & Goals

ML Pipeline Monitor is a production-oriented MLOps observability and operations platform. It provides:
- Streamlit dashboards for pipeline runs, experiment tracking, model registry, data drift, data health, and governance.
- Service-layer orchestration (Streamlit UI → services → core logic → persistence).
- FastAPI inference API with rate limiting, Prometheus metrics, and JWT/auth.
- Celery background worker for scheduled/automated jobs.
- Prometheus + Grafana + Alertmanager monitoring stack.
- PostgreSQL or SQLite backend with connection pooling.
- E2E test automation via Playwright.

Primary goals:
- Reliability: deterministic orchestration and consistent persistence.
- Traceability: stage-level progress, lineage, and structured logging.
- Maintainability: clear modular boundaries and assistant-friendly conventions.

## 2) Existing Architecture (UI → Services → Core → Persistence)

- **UI (Streamlit)**: `app.py` (dashboard entry) + `pages/*.py` (7 feature pages)
- **Services**: `services/*.py` (use-cases and orchestration) + `services/api/*.py` (FastAPI)
- **Core**: `src/*.py` (domain logic: pipeline, drift, loaders, auth, UI theme, metrics, logger, etc.)
- **Persistence**: `src/database.py`, `src/db_engine.py`, `src/db_interface.py`

Data flow guideline:
`Streamlit UI -> services -> src core -> persistence/db layer`

## 3) Complete File Structure & Responsibilities

```
├── app.py                           # Main Streamlit dashboard ("Executive Command Center")
├── run_app.py                       # Launcher script (auto port discovery)
├── fibonacci.py                     # Side helper file
├── Dockerfile                       # Multi-stage build (base, deps, dev, prod, api, worker)
├── docker-compose.yml               # Production stack with profiles
├── docker-compose.dev.yml           # Development overrides
├── docker-compose.prod.yml          # Production overrides
├── config.yaml                      # Runtime configuration
├── config.prod.yaml                 # Production-specific configuration
├── requirements.txt                 # Runtime dependencies
├── requirements-dev.txt             # Development dependencies
├── setup.py                         # Package setup with CLI entry points
├── pytest.ini                       # Test config (50% coverage minimum)
├── playwright.config.ts             # E2E test configuration
├── .github/workflows/ci.yml         # CI pipeline
├── .github/workflows/e2e.yml        # E2E test pipeline
│
├── pages/
│   ├── 0_Dataset_Management.py      # Dataset hub + feature statistics
│   ├── 1_Pipeline_Runner.py         # Configure + execute training pipeline
│   ├── 1_Dashboard.py               # Alternative dashboard page (exists alongside app.py)
│   ├── 2_Experiment_Tracking.py    # Compare runs + metrics
│   ├── 3_Model_Registry.py          # Promote models across stages
│   ├── 4_Data_Drift.py              # KS + PSI + drift reports
│   ├── 5_Data_Health.py             # Quality checks + governance context
│   └── 6_Governance.py              # Audit trail + compliance policy UI
│
├── services/
│   ├── __init__.py
│   ├── app_service.py               # Dashboard snapshot, app initialization
│   ├── pipeline_service.py          # Orchestrates pipeline runs
│   ├── model_service.py             # Model registry + inference coordination
│   ├── drift_service.py             # Orchestrates drift detection runs
│   ├── data_health_service.py       # Quality checks, schema comparison, outlier summaries
│   ├── telemetry_service.py         # User action tracking
│   └── api/
│       ├── __init__.py
│       ├── main.py                  # FastAPI launcher
│       └── app.py                   # FastAPI application (health, predict, metrics, etc.)
│
├── src/
│   ├── __init__.py
│   ├── pipeline.py                  # Stage-by-stage ML pipeline (7 stages)
│   ├── data_loader.py               # Dataset loading and splitting (sklearn + synthetic)
│   ├── database.py                  # Persistence layer (1512 lines, SQLite/Postgres)
│   ├── db_engine.py                 # Backend abstraction (SQLite + PostgreSQL)
│   ├── db_interface.py              # Database interface contracts (Protocol)
│   ├── drift_detector.py            # Statistical drift detection (KS + PSI)
│   ├── feature_store.py             # Caching splits with hashlib keys
│   ├── system_monitor.py            # psutil-based CPU/Memory/Disk metrics
│   ├── auth.py                      # Authentication + role helpers (bcrypt)
│   ├── secrets.py                   # Secrets management (env, Docker/K8s secrets, local)
│   ├── config_loader.py             # YAML config loading + secrets injection
│   ├── logger.py                    # Structured logging (JSON + console formatters)
│   ├── metrics.py                   # Prometheus metrics (30+ metrics)
│   ├── alerts.py                    # Console + simulated email alerts
│   ├── mlflow_tracker.py            # Optional MLflow integration
│   ├── ui_theme.py                  # Enterprise design system (HP-inspired)
│   ├── data_validation.py           # Data quality validation (quality score, anomalies)
│   ├── components.py                # Reusable UI components
│   ├── design_system.py             # Centralized design tokens (colors, typography, spacing)
│   └── cli.py                       # CLI entry point (hash-password, launch)
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_pipeline.py
│   ├── test_api_service.py
│   ├── test_app_service.py
│   ├── test_alerts.py
│   ├── test_data_validation.py
│   ├── test_data_health_service.py
│   ├── test_database_lineage.py
│   ├── test_drift_service.py
│   ├── test_model_service.py
│   └── e2e/
│       ├── conftest.py
│       ├── test_auth.py
│       ├── test_dataset_management.py
│       ├── test_system_health.py
│       ├── test_data_drift.py
│       ├── test_model_registry.py
│       ├── test_experiment_tracking.py
│       └── test_pipeline_runner.py
│
├── monitoring/
│   ├── prometheus/
│   │   └── prometheus.yml            # Scrape configs for API, app, prometheus self
│   ├── grafana/
│   │   ├── provisioning/
│   │   │   ├── datasources/
│   │   │   │   └── datasource.yml    # Prometheus datasource provisioning
│   │   │   └── dashboards/
│   │   │       └── dashboards.yml    # Dashboard provider config
│   │   └── dashboards/
│   │       ├── system-health.json
│   │       ├── drift-metrics.json
│   │       ├── api-metrics.json
│   │       └── pipeline-metrics.json
│   └── alertmanager/
│       └── alertmanager.yml          # Alertmanager configuration
│
├── scripts/
│   └── init-db.sql                   # PostgreSQL initialization script
│
├── artifacts/
│   ├── models/                       # Saved model artifacts (joblib)
│   ├── scalers/                      # Saved scaler artifacts (joblib)
│   └── feature_store/                # Cached dataset splits
│
├── datasets/                         # Dataset storage
├── logs/                             # Application logs
└── .pipeline_monitor.db             # Default SQLite database
```

## 4) Coding Standards (Non-Negotiable)

- **Type hints**: Use explicit type hints for public functions/classes.
- **Docstrings**: Every module-level public function/class must have a docstring.
- **Modular design**: small, composable functions; single responsibility.
- **Avoid God classes**: no large all-in-one classes/modules.
- **Avoid duplicate code**: factor shared behavior into reusable helpers.
- **Correctness over cleverness**: prefer explicit, readable implementations.

## 5) Database Rules

- Use the existing persistence abstractions (do not bypass `db_interface`/`db_engine`).
- Prefer parameterized queries / safe ORM patterns used by the project.
- Ensure schema/lineage operations remain consistent with existing lineage tracking.
- Keep migrations/DB init consistent with current backend setup.

SQLite backend (`SQLiteBackend`):
- Connection pooling via `queue.Queue`
- WAL journal mode, foreign keys enabled
- `_PooledConnection` wrapper returns connections to pool on `close()`

PostgreSQL backend (`PostgresBackend`):
- Uses `psycopg-pool` (`ConnectionPool`)
- `PostgresConnectionAdapter` normalizes `?` → `%s` and `datetime(...)` calls
- DSN from `PIPELINE_DB_DSN` env var or `storage.postgres_dsn` config

## 6) API Design Standards (FastAPI)

Endpoints in `services/api/app.py`:
- `GET /health` — DB connectivity check
- `GET /health/live` — liveness probe
- `GET /health/ready` — readiness probe (DB check)
- `GET /metrics` — Prometheus metrics endpoint
- `GET /health/detailed` — system metrics + DB status
- `POST /predict` — predict using latest production model (API key required, rate limited)

Design rules:
- Follow established patterns in `services/api/app.py`.
- Validate request bodies with Pydantic (`PredictRequest`).
- Return consistent response shapes; avoid leaking internal exceptions.
- Keep endpoints thin: route to `services/`.
- Middleware logs all requests with correlation/request IDs.
- Rate limiting via `slowapi` (default 60/min, configurable).
- Global exception handlers for: `Exception`, `RequestValidationError`, `RateLimitExceeded`.
- Graceful shutdown with signal handlers (SIGTERM, SIGINT).

## 7) Streamlit Page Conventions (`pages/*.py`)

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
| `pages/1_Dashboard.py` | Alternative dashboard view |
| `pages/2_Experiment_Tracking.py` | Compare runs + filter metrics |
| `pages/3_Model_Registry.py` | Promote models across stages |
| `pages/4_Data_Drift.py` | KS + PSI + drift reports + references |
| `pages/5_Data_Health.py` | Data quality checks + outlier summaries |
| `pages/6_Governance.py` | Audit trail + compliance policy UI |

## 8) Logging Conventions

- Use the repository logger abstraction (`src/logger.py`).
- Structured logging with JSON file output + colored console output.
- Context propagation via `contextvars`: `correlation_id`, `request_id`, `operation_context`, `actor_context`, `service_context`.
- Use `LogContext` context manager for setting contexts.
- Log with structured context where possible (IDs: run_id, experiment_id, dataset name, stage).
- Avoid logging secrets or raw credentials.
- Use log levels consistently: debug/info/warning/error.
- `ErrorCategory` enum for categorizing exceptions: validation, configuration, database, model, pipeline, drift, authentication, authorization, external_service, internal, unknown.

## 9) Testing Requirements

- Add/extend **unit tests** under `tests/` for core and services.
- Add/extend **e2e tests** under `tests/e2e/` using Playwright when user-facing flows change.
- Maintain coverage expectations enforced by `pytest.ini` (50% minimum, term-missing + html reports).
- Tests must be deterministic; avoid time-based flakiness.
- E2E tests use `pytest-playwright` with `pytest-base-url`.

## 10) Git Commit Format

Use Conventional-like prefixes exactly as below:
- `feat:` new feature
- `fix:` bug fix
- `refactor:` structural improvement without behavior change
- `docs:` documentation-only changes

## 11) Dependency Policy

- Prefer existing libraries already used in `requirements.txt`.
- Avoid unnecessary new packages.
- Do not add heavy dependencies unless clearly required.

Key runtime dependencies:
- streamlit==1.56.0, fastapi==0.136.0, uvicorn==0.44.0
- scikit-learn==1.8.0, xgboost==3.2.0
- pandas>=2.0.0,<3.0.0, numpy>=1.26.0
- plotly==6.7.0, altair==6.0.0
- sqlite (stdlib) + psycopg-pool==3.3.1 (PostgreSQL)
- prometheus-client==0.21.1, slowapi==0.1.9
- bcrypt==5.0.0, pydantic==2.13.2
- psutil==7.2.2, joblib==1.5.3
- pytest==9.0.3, pytest-playwright==0.8.0
- playwright==1.60.0

Key dev dependencies:
- ruff==0.6.9, black==24.10.0, isort==5.13.2, mypy==1.13.0
- alembic==1.13.1, celery==5.4.0, redis==5.1.0, flower==2.0.1
- bandit==1.7.9, safety==3.3.1

## 12) Performance Rules

- Cache expensive operations (use Streamlit `@st.cache_data` or existing caching in services/core).
- Feature store (`src/feature_store.py`) caches processed dataset splits with hashlib-derived keys.
- Avoid repeated disk I/O (read once, reuse; memoize in-process when appropriate).
- Minimize unnecessary DB roundtrips.
- Database connection pooling: SQLite (5-connection pool), PostgreSQL (configurable via `connection_pool`).
- Streamlit page caching with TTL where applicable.

## 13) Security Rules

- **bcrypt only** for password hashing and verification.
- Secrets come from **environment variables** or `SecretsManager` (priority: env vars → Docker/K8s secrets → `.secrets.json`).
- Input validation everywhere:
  - validate request payloads (Pydantic in API),
  - validate UI form inputs,
  - validate any persistence-layer inputs.
- Rate limiting on API (`slowapi`, default 60/min).
- API key authentication (`X-API-Key` header) on `/predict`.
- Session timeout, max login attempts, lockout configured in `config.yaml`/`config.prod.yaml`.
- Production security headers in `config.prod.yaml` (X-Frame-Options, CSP, etc.).

## 14) Configuration System

| File | Purpose |
|---|---|
| `config.yaml` | Development/default configuration |
| `config.prod.yaml` | Production overrides (Postgres, JWT, security, worker, etc.) |
| `src/config_loader.py` | Loads YAML, deep-merges with `DEFAULT_CONFIG`, injects secrets |
| `src/secrets.py` | `SecretsManager` with env → files → `.secrets.json` priority |

Key config sections:
- `pipeline`: random_seed, test_size, cv_folds, n_jobs
- `datasets`: display names, task types, sources
- `monitoring`: thresholds, automated retraining settings
- `storage`: backend selection, db_path, artifacts_root, connection_pool
- `auth`: enabled, session timeout, max login attempts
- `logging`: level, file path, rotation settings, format (json/console)
- `alerting`: email simulation file, SMTP settings, Slack webhook
- `mlflow`: enabled, tracking URI, experiment name
- `ui`: primary color, max experiments, chart height
- `api` (prod): rate limit, CORS, JWT settings
- `worker` (prod): concurrency, prefetch, max tasks
- `security` (prod): bcrypt rounds, password policy, CORS, secure headers

Environment variables:
- `CONFIG_PATH` — override config file path
- `PIPELINE_DB_DSN` — PostgreSQL DSN
- `MLMONITOR_AUTH_ENABLED`, `AUTH_USERNAME`, `AUTH_PASSWORD`, `AUTH_ROLE`, `AUTH_USERS_JSON`
- `MLMONITOR_API_KEY`, `MLMONITOR_RATE_LIMIT`
- `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`
- `SLACK_WEBHOOK`, `MLFLOW_TRACKING_URI`, `JWT_SECRET`, `JWT_ALGORITHM`

## 15) Monitoring & Observability Stack

### Prometheus (metrics collection)
- Image: `prom/prometheus:v2.54.1`
- Port: 9090
- Scrapes: API (`/metrics`, 10s), Streamlit app (`/metrics`, 30s), self-monitoring
- Retention: 30d / 10GB

### Grafana (visualization)
- Image: `grafana/grafana:11.1.0`
- Port: 3000
- Provisioned datasource: Prometheus
- Dashboards: system-health, drift-metrics, api-metrics, pipeline-metrics

### Alertmanager (alert routing)
- Image: `prom/alertmanager:v0.27.0`
- Port: 9093
- Profile: `monitoring`

### Prometheus Metrics (`src/metrics.py`)
- Pipeline: `ml_pipeline_runs_total`, `ml_pipeline_duration_seconds`, `ml_pipeline_stage_duration_seconds`
- API: `ml_api_requests_total`, `ml_api_request_duration_seconds`, `ml_api_errors_total`
- Predictions: `ml_predictions_total`, `ml_prediction_latency_seconds`
- Drift: `ml_drift_detections_total`, `ml_drift_score`, `ml_drift_features_count`, `ml_drift_analysis_duration_seconds`
- System: CPU%, memory%, disk%, temperature (host + process)
- Registry: `ml_experiments_total`, `ml_models_registered_total`, `ml_model_promotions_total`
- Data Health: `ml_dataset_validations_total`, `ml_dataset_rows`, `ml_dataset_columns`

### System Monitoring (`src/system_monitor.py`)
- `get_system_metrics()`: CPU, memory, disk, temperature
- `get_process_metrics()`: PID, RSS, VMS, threads, status

## 16) Docker & Deployment

### Multi-stage Dockerfile
1. `base` — Python 3.10-slim, non-root user (UID/GID 1000), system deps
2. `dependencies` — pip install requirements
3. `development` — dev deps, Playwright browsers, hot reload (`streamlit run --server.runOnSave=true`)
4. `production` — minimal image, only production code, healthcheck, EXPOSE 8501+8000
5. `api` — extends production, runs `uvicorn services.api.main:app`
6. `worker` — extends production, runs `python -m services.worker`

### docker-compose.yml Services
| Service | Image/Target | Port | Profile |
|---|---|---|---|
| app | production target | 8501 | default |
| api | api target | 8000 | default |
| worker | worker target | — | default |
| flower | `mher/flower:2.0.1` | 5555 | monitoring |
| postgres | `postgres:16-alpine` | 5432 | postgres |
| redis | `redis:7-alpine` | 6379 | default |
| prometheus | `prom/prometheus:v2.54.1` | 9090 | default |
| grafana | `grafana/grafana:11.1.0` | 3000 | default |
| alertmanager | `prom/alertmanager:v0.27.0` | 9093 | monitoring |
| nginx | `nginx:alpine` | 80/443 | production |

Start commands:
- Development: `docker-compose -f docker-compose.yml -f docker-compose.dev.yml up`
- Production: `docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d`
- With Postgres: `docker-compose --profile postgres up -d`

## 17) CLI Entry Points

```bash
mlmonitor            # Launch Streamlit app
mlmonitor-api        # Launch FastAPI server
mlmonitor hash-password <password>  # Generate bcrypt hash
```

## 18) ML Pipeline Stages

`src/pipeline.py` implements `MLPipeline` with these stages:
1. Data Validation
2. Preprocessing (StandardScaler)
3. Feature Analysis
4. Cross-Validation (KFold / StratifiedKFold)
5. Training (Random Forest, XGBoost, Gradient Boosting, Logistic Regression, SVM, Decision Tree + regressors)
6. Evaluation (accuracy, precision, recall, F1, confusion matrix, ROC-AUC, MSE, MAE, R²)
7. Feature Importance Extraction

Supported algorithms:
- Classification: Random Forest, XGBoost, Gradient Boosting, Logistic Regression, SVM, Decision Tree
- Regression: Random Forest, XGBoost, Gradient Boosting, Ridge, SVR, Decision Tree

## 19) Model Lifecycle

Stages: `development` → `staging` → `production` → `archived`
Governance page shows audit trail and stage change history.
`model_stage_events` table tracks all promotions/demotions.

## 20) Drift Detection

`src/drift_detector.py`:
- KS test (scipy.stats) per feature
- PSI (Population Stability Index)
- Severity classification: none / moderate / significant
- Configurable thresholds in `config.yaml`

## 21) Data Quality

`src/data_validation.py`:
- `ValidationResult` dataclass with quality_score (0-100), status, report, recommendations, fail_reasons
- Missing values, duplicates, outlier detection (IQR + Z-score)
- `DataQualityFailed` exception stops training when score below threshold

`services/data_health_service.py`:
- `missing_value_report()`, `class_imbalance_report()`, feature analysis, shape validation

## 22) "Automatic Compliance" Expectation

All future modifications must follow this document's rules automatically:
- If you must deviate, document the reason in a `docs:` commit and update the relevant section.
- Do not introduce silent behavior changes without tests.
- Every new module-level public function/class needs a docstring.
- Type hints for public functions/classes.
- Input validation on every boundary (UI, API, persistence).
