"""Background worker for scheduled pipeline execution and automated retraining."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from src.config_loader import load_config
from src.logger import get_app_logger
from services.pipeline_service import run_pipeline_and_persist

LOGGER = get_app_logger("worker")


def _execute_scheduled_task(task_config: Dict[str, Any]) -> None:
    task_type = task_config.get("type", "pipeline_run")
    LOGGER.info("Executing scheduled task: %s", task_type)

    if task_type == "pipeline_run":
        params = task_config.get("params", {})
        dataset = params.get("dataset", "iris")
        model_type = params.get("model_type", "Random Forest")
        task = params.get("task", "classification")
        test_size = float(params.get("test_size", 0.2))
        cv_folds = int(params.get("cv_folds", 5))
        random_seed = int(params.get("random_seed", 42))
        run_pipeline_and_persist(
            dataset_label=dataset,
            dataset_key=dataset,
            model_type=model_type,
            task=task,
            test_size=test_size,
            cv_folds=cv_folds,
            random_state=random_seed,
        )
    else:
        LOGGER.warning("Unknown task type: %s", task_type)


def run_worker_loop(concurrency: int = 4, poll_interval: float = 5.0) -> None:
    cfg = load_config().get("worker", {})
    concurrency = int(cfg.get("concurrency", concurrency))
    poll_interval = float(cfg.get("poll_interval", poll_interval))

    LOGGER.info("Worker starting with concurrency=%d, poll_interval=%.1fs", concurrency, poll_interval)

    running = True
    while running:
        try:
            _run_once(poll_interval=poll_interval)
        except KeyboardInterrupt:
            LOGGER.info("Worker received shutdown signal")
            running = False
        except Exception as exc:
            LOGGER.exception("Worker loop error: %s", exc)
            time.sleep(poll_interval)

    LOGGER.info("Worker shut down cleanly")


def _run_once(poll_interval: float = 5.0) -> None:
    time.sleep(poll_interval)


if __name__ == "__main__":
    run_worker_loop()
