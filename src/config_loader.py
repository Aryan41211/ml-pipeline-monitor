"""Runtime configuration loader for ML Pipeline Monitor."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml


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
    },
    "storage": {
        "backend": "sqlite",
        "db_path": ".pipeline_monitor.db",
        "artifacts_root": "artifacts",
    },
    "auth": {
        "username": "admin",
        "password": "admin123",
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
    """Load and cache YAML config with sane defaults."""
    if not CONFIG_PATH.exists():
        return DEFAULT_CONFIG

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    if not isinstance(raw, dict):
        return DEFAULT_CONFIG

    return _deep_merge(DEFAULT_CONFIG, raw)


def get_artifact_dirs() -> Dict[str, Path]:
    """Return canonical artifact directories and ensure they exist."""
    cfg = load_config()
    root = ROOT_DIR / cfg.get("storage", {}).get("artifacts_root", "artifacts")
    models = root / "models"
    scalers = root / "scalers"

    models.mkdir(parents=True, exist_ok=True)
    scalers.mkdir(parents=True, exist_ok=True)

    return {"root": root, "models": models, "scalers": scalers}
