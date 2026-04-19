"""Pipeline service for manual and scheduled execution workflows."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional

import joblib

from src.alerts import emit_console_alert
from src.config_loader import get_artifact_dirs, load_config
from src.data_loader import load_dataset
from src.database import get_experiments, save_experiment, save_model
from src.feature_store import load_cached_splits, make_feature_key, save_cached_splits
from src.logger import get_app_logger
from src.mlflow_tracker import log_pipeline_run
from src.pipeline import MLPipeline, PipelineResult


ProgressCallback = Callable[[str, float, str], None]
LOGGER = get_app_logger("pipeline_service")


def list_experiments(limit: int = 200) -> list[Dict[str, Any]]:
    """Return persisted experiments for analytics views."""
    return get_experiments(limit=limit)


def _persist_artifacts(run_id: str, model: object, scaler: object) -> Dict[str, str]:
    """Persist model and scaler using canonical artifact layout."""
    dirs = get_artifact_dirs()
    model_path = dirs["models"] / f"{run_id}_model.joblib"
    scaler_path = dirs["scalers"] / f"{run_id}_scaler.joblib"

    joblib.dump(model, model_path)
    if scaler is not None:
        joblib.dump(scaler, scaler_path)

    return {
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
    }


def run_pipeline_and_persist(
    *,
    dataset_label: str,
    dataset_key: str,
    model_type: str,
    task: str,
    params: Dict[str, Any],
    test_size: float,
    cv_folds: int,
    random_state: int,
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """Execute pipeline run end-to-end and persist experiment/model artifacts."""
    app_cfg = load_config()
    pipeline_cfg = app_cfg.get("pipeline", {})
    feature_key = make_feature_key(dataset_key, test_size, random_state)
    ds = load_cached_splits(feature_key)
    if ds is None:
        ds = load_dataset(
            dataset_key,
            test_size=float(test_size),
            random_state=int(random_state),
        )
        save_cached_splits(feature_key, ds)
        LOGGER.info("Feature store miss: cached new splits for key=%s", feature_key)
    else:
        LOGGER.info("Feature store hit: reused splits for key=%s", feature_key)

    try:
        pipeline = MLPipeline(
            dataset_name=dataset_label,
            model_type=model_type,
            task=task,
            params=params,
            cv_folds=int(cv_folds),
            random_state=int(random_state),
            n_jobs=int(pipeline_cfg.get("n_jobs", -1)),
            progress_callback=progress_callback,
        )

        result: PipelineResult = pipeline.run(
            ds["X_train"],
            ds["X_test"],
            ds["y_train"],
            ds["y_test"],
        )
    except Exception as exc:
        emit_console_alert("critical", f"Pipeline failure for dataset={dataset_label}: {exc}")
        LOGGER.exception("Pipeline run failed")
        raise

    save_experiment(
        run_id=result.run_id,
        name=f"{dataset_label} / {model_type}",
        dataset=dataset_label,
        model_type=model_type,
        task=task,
        params=params,
        metrics=result.metrics,
        duration=result.duration,
    )

    artifact_paths = _persist_artifacts(result.run_id, result.model, result.scaler)

    model_record = save_model(
        model_id=result.run_id,
        run_id=result.run_id,
        name=model_type,
        dataset=dataset_label,
        model_type=model_type,
        task=task,
        metrics=result.metrics,
        artifact_path=artifact_paths["model_path"],
        params=params,
        experiment_id=result.run_id,
    )

    LOGGER.info(
        "Pipeline run completed run_id=%s dataset=%s model=%s version=%s",
        result.run_id,
        dataset_label,
        model_type,
        model_record.get("version"),
    )

    log_pipeline_run(
        run_name=f"{dataset_label}-{model_type}-{result.run_id}",
        params=params,
        metrics=result.metrics,
        artifact_path=artifact_paths["model_path"],
        model=result.model,
    )

    return {
        "result": result,
        "dataset": ds,
        "model_record": model_record,
        "artifacts": artifact_paths,
    }


def compute_next_run_ts(interval_minutes: int) -> datetime:
    """Return next scheduled run timestamp from now."""
    return datetime.utcnow() + timedelta(minutes=max(1, int(interval_minutes)))


def should_trigger_scheduled_run(
    enabled: bool,
    next_run_at: Optional[datetime],
    now: Optional[datetime] = None,
) -> bool:
    """Determine if the simulated cron run should trigger now."""
    if not enabled or next_run_at is None:
        return False

    now = now or datetime.utcnow()
    return now >= next_run_at
