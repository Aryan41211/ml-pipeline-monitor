"""Optional MLflow tracking integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.config_loader import ROOT_DIR, load_config
from src.logger import get_app_logger

LOGGER = get_app_logger("mlflow")


def _is_enabled() -> bool:
    return bool(load_config().get("mlflow", {}).get("enabled", False))


def _log_model_to_mlflow(model: Any, artifact_path: str = "model") -> None:
    """Log model using appropriate MLflow flavor based on model type."""
    model_class_name = model.__class__.__name__
    
    if "XGB" in model_class_name or "XGBoost" in model_class_name:
        import mlflow.xgboost
        mlflow.xgboost.log_model(model, artifact_path=artifact_path)
    elif "LGBM" in model_class_name:
        import mlflow.lightgbm
        mlflow.lightgbm.log_model(model, artifact_path=artifact_path)
    else:
        import mlflow.sklearn
        mlflow.sklearn.log_model(model, artifact_path=artifact_path)


def log_pipeline_run(
    *,
    run_name: str,
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    artifact_path: str,
    model: Any,
) -> None:
    """Log run metadata and model to MLflow when enabled."""
    if not _is_enabled():
        return

    try:
        import mlflow

        cfg = load_config().get("mlflow", {})
        tracking_uri = str(cfg.get("tracking_uri", f"file:{(ROOT_DIR / 'mlruns').as_posix()}"))
        experiment_name = str(cfg.get("experiment", "ml-pipeline-monitor"))

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({k: v for k, v in params.items() if isinstance(v, (str, int, float, bool))})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
            if artifact_path:
                art = Path(artifact_path)
                if art.exists():
                    mlflow.log_artifact(str(art), artifact_path="artifacts")
            _log_model_to_mlflow(model, artifact_path="model")
    except Exception as exc:
        LOGGER.warning("MLflow logging skipped: %s", exc)
