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
ml-pipeline-monitor/
├── src/
│   ├── database/           # Modular persistence layer
│   │   ├── schema.py      # Table definitions & migrations
│   │   ├── experiments.py # Experiment CRUD
│   │   ├── models.py      # Model registry CRUD
│   │   ├── drift.py       # Drift reports CRUD
│   │   ├── predictions.py # Prediction history
│   │   ├── governance.py  # Teams, users, workspaces
│   │   └── lineage.py     # Dataset lineage
│   ├── services/          # Business logic layer
│   │   ├── app_service.py
│   │   ├── pipeline_service.py
│   │   ├── model_service.py
│   │   ├── drift_service.py
│   │   ├── data_health_service.py
│   │   └── telemetry_service.py
│   ├── ui_theme.py        # Enterprise design system
│   ├── auth.py            # Authentication & authorization
│   ├── pipeline.py        # ML pipeline orchestration
│   ├── data_loader.py     # Dataset loading utilities
│   ├── drift_detector.py  # Statistical drift detection
│   └── ...
├── pages/                 # Streamlit UI pages
├── services/api/          # FastAPI inference server
├── tests/                 # Test suite
├── docs/                  # Documentation
└── monitoring/            # Prometheus, Grafana configs
```

## Quick Start

### Prerequisites

- Python 3.10+
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

# Initialize database
python -c "from src.database import initialize_db; initialize_db()"
```

### Running the Application

```bash
# Start Streamlit UI
streamlit run app.py

# Start FastAPI server (in separate terminal)
uvicorn services.api.main:app --reload --port 8000
```

### Docker Deployment

```bash
# Full stack with monitoring
docker-compose up -d

# Development mode
docker-compose -f docker-compose.dev.yml up
```

## Configuration

Edit `config.yaml` to customize:

```yaml
database:
  backend: sqlite  # or postgres
  path: data/ml_monitor.db

pipeline:
  test_size: 0.20
  random_seed: 42
  cv_folds: 5

monitoring:
  prometheus_port: 9090
  grafana_port: 3000
```

## Testing

```bash
# Run unit tests
pytest tests/ --ignore=tests/load --ignore=tests/e2e

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run E2E tests (requires Playwright)
playwright install
pytest tests/e2e/
```

## Documentation

- [Architecture Guide](docs/architecture.md)
- [Deployment Guide](docs/deployment.md)
- [API Reference](docs/api_reference.md)
- [Operations Manual](docs/operations.md)
- [Troubleshooting](docs/troubleshooting.md)

## Tech Stack

**Backend:**
- FastAPI - REST API
- SQLAlchemy - Database ORM
- scikit-learn, XGBoost - ML frameworks
- Prometheus - Metrics collection

**Frontend:**
- Streamlit - Web UI
- Plotly - Interactive charts
- Custom HP Design System - Enterprise UI components

**Infrastructure:**
- Docker - Containerization
- PostgreSQL - Production database
- Redis - Caching & job queue
- Celery - Background tasks
- Prometheus + Grafana - Monitoring

## Development

See [CLAUDE.md](CLAUDE.md) for development guidelines and architecture decisions.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## Support

For issues and questions, please open a GitHub issue.