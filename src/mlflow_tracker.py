"""Optional MLflow tracking integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.config_loader import ROOT_DIR, load_config
from src.logger import get_app_logger

LOGGER = get_app_logger("mlflow")


def _is_enabled() -> bool:
    return bool(load_config().get("mlflow", {}).get("enabled", False))


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
        import mlflow.sklearn

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
            mlflow.sklearn.log_model(model, artifact_path="model")
    except Exception as exc:
        LOGGER.warning("MLflow logging skipped: %s", exc)
