"""Runtime configuration loader for ML Pipeline Monitor."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

from src.secrets import get_secrets_manager


ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config.yaml"


DEFAULT_CONFIG: Dict[str, Any] = {
    "pipeline": {
        "random_seed": 42,
        "test_size": 0.20,
        "cv_folds": 5,
    },
    "monitoring": {
        "drift_significance_level": 0.05,
        "psi_moderate_threshold": 0.10,
        "psi_significant_threshold": 0.25,
        "drift_feature_ratio_threshold": 0.20,
        "automated_retraining": {
            "enabled": True,
            "drift_threshold": 0.20,
            "min_performance_improvement": 0.02,
            "cooldown_seconds": 300,
        },
    },
    "storage": {
        "backend": "sqlite",
        "db_path": ".pipeline_monitor.db",
        "artifacts_root": "artifacts",
    },
    "auth": {
        "enabled": True,
    },
    "logging": {
        "level": "INFO",
        "file": "logs/app.log",
        "max_bytes": 5000000,
        "backup_count": 3,
    },
    "mlflow": {
        "enabled": False,
        "tracking_uri": "",
        "experiment": "ml-pipeline-monitor",
    },
    "datasets": {},
    "ui": {
        "max_experiments_displayed": 200,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@lru_cache(maxsize=1)
def load_config() -> Dict[str, Any]:
    """Load and cache YAML config with sane defaults and secrets injection."""
    if not CONFIG_PATH.exists():
        return DEFAULT_CONFIG

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    if not isinstance(raw, dict):
        return DEFAULT_CONFIG

    config = _deep_merge(DEFAULT_CONFIG, raw)
    
    # Inject secrets for sensitive configuration values
    secrets = get_secrets_manager()
    
    # Database secrets
    if "storage" in config:
        storage = config["storage"]
        if storage.get("backend") == "postgres":
            # Override postgres DSN from secrets if available
            dsn = secrets.get("postgres_dsn") or secrets.get("database_url")
            if dsn:
                storage["postgres_dsn"] = dsn
            # Individual components
            for key in ["db_host", "db_port", "db_name", "db_user", "db_password"]:
                val = secrets.get(key)
                if val:
                    storage[key] = val
    
    # API keys
    if "mlflow" in config:
        mlflow = config["mlflow"]
        tracking_uri = secrets.get("mlflow_tracking_uri")
        if tracking_uri:
            mlflow["tracking_uri"] = tracking_uri
    
    # Auth secrets (already handled via env in auth.py, but can be centralized)
    # Alerting secrets
    if "alerting" in config:
        alerting = config["alerting"]
        for key in ["smtp_host", "smtp_port", "smtp_user", "smtp_password", "slack_webhook"]:
            val = secrets.get(key)
            if val:
                alerting[key] = val
    
    return config


def get_artifact_dirs() -> Dict[str, Path]:
    """Return canonical artifact directories and ensure they exist."""
    cfg = load_config()
    root = ROOT_DIR / cfg.get("storage", {}).get("artifacts_root", "artifacts")
    models = root / "models"
    scalers = root / "scalers"

    models.mkdir(parents=True, exist_ok=True)
    scalers.mkdir(parents=True, exist_ok=True)

    return {"root": root, "models": models, "scalers": scalers}
