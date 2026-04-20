"""Model service for registry operations and online inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd

from src.database import (
    get_latest_production_model,
    get_model_stage_events,
    get_model_lineage,
    get_models,
    get_recent_production_models,
    update_model_stage,
)


def list_models(limit: int = 100, dataset: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return model registry records with optional dataset filter."""
    records = get_models(limit=limit)
    if dataset:
        return [row for row in records if row.get("dataset") == dataset]
    return records


def list_lineage(limit: int = 200, dataset: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return model lineage records."""
    return get_model_lineage(limit=limit, dataset=dataset)


def set_model_stage(model_id: str, stage: str) -> None:
    """Promote/demote model lifecycle stage."""
    update_model_stage(model_id=model_id, stage=stage)


def get_stage_timeline(model_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Return stage change history for a model, newest first."""
    return get_model_stage_events(model_id=model_id, limit=limit)


def get_rollback_hint(dataset: str) -> Dict[str, Any]:
    """Return current and previous production model hints for rollback UX."""
    recent = get_recent_production_models(dataset=dataset, limit=2)
    current = recent[0] if len(recent) >= 1 else None
    previous = recent[1] if len(recent) >= 2 else None
    return {
        "current_production": current,
        "previous_production": previous,
    }


def revert_to_previous_production(dataset: str) -> Dict[str, Any]:
    """Promote the previous production model back to production for a dataset."""
    hint = get_rollback_hint(dataset)
    previous = hint.get("previous_production")
    if not previous:
        raise ValueError("No previous production model available to revert.")

    model_id = str(previous.get("model_id"))
    update_model_stage(model_id=model_id, stage="production")
    return previous


def _derive_scaler_path(model_path: Path) -> Path:
    """Infer scaler artifact path from stored model artifact path convention."""
    scaler_name = model_path.name.replace("_model.joblib", "_scaler.joblib")
    return model_path.parent.parent / "scalers" / scaler_name


def load_production_artifacts(
    dataset: Optional[str] = None,
) -> Tuple[Any, Optional[Any], Dict[str, Any]]:
    """Load latest production model (+ optional scaler) and its metadata."""
    model_meta = get_latest_production_model(dataset=dataset)
    if model_meta is None:
        ds_msg = f" for dataset '{dataset}'" if dataset else ""
        raise ValueError(f"No production model found{ds_msg}")

    artifact_path = model_meta.get("artifact_path")
    if not artifact_path:
        raise ValueError("Production model is missing artifact_path")

    model_path = Path(str(artifact_path))
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}")

    model = joblib.load(model_path)

    scaler = None
    scaler_path = _derive_scaler_path(model_path)
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)

    return model, scaler, model_meta


def _to_dataframe(payload: Any) -> pd.DataFrame:
    """Normalize prediction input payload into a DataFrame."""
    if isinstance(payload, dict):
        return pd.DataFrame([payload])

    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return pd.DataFrame(payload)

    if isinstance(payload, list) and payload and isinstance(payload[0], (int, float)):
        return pd.DataFrame([payload])

    if isinstance(payload, list) and payload and isinstance(payload[0], list):
        return pd.DataFrame(payload)

    raise ValueError(
        "Invalid input payload. Use a feature dictionary, list of feature dictionaries, "
        "a single feature vector, or a list of vectors."
    )


def _align_features(model: Any, frame: pd.DataFrame) -> pd.DataFrame:
    """Align request payload columns with model training schema when available."""
    expected = getattr(model, "feature_names_in_", None)
    if expected is None:
        return frame

    expected_cols = [str(col) for col in list(expected)]
    missing = [col for col in expected_cols if col not in frame.columns]
    if missing:
        raise ValueError(
            "Missing required features for production model: " + ", ".join(missing)
        )

    aligned = frame.reindex(columns=expected_cols)
    return aligned


def predict_from_payload(payload: Any, dataset: Optional[str] = None) -> Dict[str, Any]:
    """Run prediction against latest production model."""
    model, scaler, model_meta = load_production_artifacts(dataset=dataset)
    X = _align_features(model, _to_dataframe(payload))

    X_infer = scaler.transform(X) if scaler is not None else X
    preds = model.predict(X_infer)

    response: Dict[str, Any] = {
        "model_id": model_meta.get("model_id"),
        "dataset": model_meta.get("dataset"),
        "version": model_meta.get("version"),
        "stage": model_meta.get("stage"),
        "predictions": preds.tolist(),
    }

    if hasattr(model, "predict_proba"):
        try:
            response["probabilities"] = model.predict_proba(X_infer).tolist()
        except Exception:
            pass

    return response
