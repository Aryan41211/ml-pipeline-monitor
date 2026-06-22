"""In-memory model cache for fast inference in the API layer."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib

from src.config_loader import get_artifact_dirs
from src.logger import get_app_logger

LOGGER = get_app_logger("model_cache")

_lock = threading.Lock()
_cache: Dict[str, Tuple[Any, Any, str]] = {}


def _resolve_artifact(run_id: str) -> Tuple[Optional[Path], Optional[Path]]:
    dirs = get_artifact_dirs()
    model_path = dirs["models"] / f"{run_id}_model.joblib"
    scaler_path = dirs["scalers"] / f"{run_id}_scaler.joblib"
    if not model_path.exists():
        return None, None
    return model_path, scaler_path if scaler_path.exists() else None


def get_model(run_id: str) -> Optional[Tuple[Any, Any, str]]:
    key = run_id.strip()
    if not key:
        return None
    with _lock:
        entry = _cache.get(key)
        if entry is not None:
            LOGGER.debug("Model cache hit", extra={"run_id": key})
            return entry
    model_path, scaler_path = _resolve_artifact(key)
    if model_path is None:
        return None
    try:
        model = joblib.load(str(model_path))
        scaler = joblib.load(str(scaler_path)) if scaler_path else None
        entry = (model, scaler, str(model_path))
        with _lock:
            _cache[key] = entry
        LOGGER.debug("Model cached", extra={"run_id": key})
        return entry
    except Exception as exc:
        LOGGER.error("Failed to load model artifact", extra={"run_id": key, "error": str(exc)})
        return None


def get_latest_production_model(dataset: Optional[str] = None) -> Optional[Tuple[Any, Any, str]]:
    from src.database import get_latest_production_model
    record = get_latest_production_model(dataset=dataset)
    if record is None:
        return None
    artifact_path = record.get("artifact_path", "")
    run_id = Path(artifact_path).stem.replace("_model", "") if artifact_path else record.get("run_id", "")
    return get_model(run_id) if run_id else None


def invalidate(run_id: str) -> None:
    with _lock:
        _cache.pop(run_id, None)
    LOGGER.debug("Model cache invalidated", extra={"run_id": run_id})


def clear_cache() -> None:
    with _lock:
        _cache.clear()
    LOGGER.info("Model cache cleared")
