# RUN_LOCAL.md — Local Development Commands

## Prerequisites
- Python 3.10+
- pip

## 0) Create virtual environment
```bash
python -m venv .venv
.
# Windows (cmd):
.venv\Scripts\activate
```

## 1) Environment variables (.env)
This repository expects a local `.env` file (see `.env.example` for safe defaults). Create it as:
```bash
copy .env.example .env
```

Key vars used by the app/API:
- `PIPELINE_DB` (optional, defaults to a local SQLite file)
- `MLMONITOR_AUTH_ENABLED` (optional, defaults to enabled)
- `AUTH_USERNAME`, `AUTH_PASSWORD`, `AUTH_ROLE` (optional; used by Streamlit auth)
- `MLMONITOR_RATE_LIMIT` (optional; default `60/minute`)
- `MLMONITOR_API_KEY` (optional; legacy API auth)
- `JWT_EXPIRATION_MINUTES` (optional; default `60`)

## 2) Install dependencies
```bash
pip install -r requirements.txt
```

If you want dev/test tooling:
```bash
pip install -r requirements-dev.txt
```

## 3) Streamlit frontend
### Command
```bash
streamlit run app.py --server.port 8501 --server.address 127.0.0.1
```

Alternative (auto port selection):
```bash
python run_app.py
```

Open:
- http://127.0.0.1:8501

## 4) FastAPI backend (inference API)
### Command
```bash
uvicorn services.api.main:app --reload --port 8000 --host 0.0.0.0
```

Swagger docs:
- http://localhost:8000/v1/docs

## 5) Run tests
```bash
pytest -q
```

## 6) Docker (if available)
If you use the provided compose files:
```bash
docker-compose up -d
```

For dev stack (if defined in compose.dev.yml):
```bash
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

## Troubleshooting
### A) `ModuleNotFoundError` / imports failing
- Ensure you installed requirements into the active venv.
- Ensure you run from repository root.

### B) DB errors
- Confirm `PIPELINE_DB` points to a writable location.
- Default is a local SQLite DB file under repository root.

### C) Auth issues in Streamlit
- Set `AUTH_USERNAME` and `AUTH_PASSWORD` in `.env`.
- Or set `MLMONITOR_AUTH_ENABLED=false` to disable login (only for local dev).

