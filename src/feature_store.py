"""Local feature store for reusing processed dataset splits across runs."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

import joblib

from src.config_loader import ROOT_DIR, load_config


def _feature_store_root() -> Path:
    cfg = load_config().get("storage", {})
    root = ROOT_DIR / cfg.get("artifacts_root", "artifacts") / "feature_store"
    root.mkdir(parents=True, exist_ok=True)
    return root


def make_feature_key(dataset_key: str, test_size: float, random_state: int) -> str:
    raw = f"{dataset_key}|{round(float(test_size), 4)}|{int(random_state)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def load_cached_splits(feature_key: str) -> Optional[Dict[str, Any]]:
    path = _feature_store_root() / f"{feature_key}.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def save_cached_splits(feature_key: str, payload: Dict[str, Any]) -> str:
    path = _feature_store_root() / f"{feature_key}.joblib"
    joblib.dump(payload, path)
    return str(path)
