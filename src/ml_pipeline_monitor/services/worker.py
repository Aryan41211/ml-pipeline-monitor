"""Background worker for scheduled pipeline execution and automated retraining.

Polls the governance ``schedules`` table and executes any enabled schedule
whose ``next_run_at`` timestamp is due, recording each execution in the
``schedule_runs`` history table.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Set

from ml_pipeline_monitor.core.config_loader import load_config
from ml_pipeline_monitor.core.logger import get_app_logger
from ml_pipeline_monitor.database import (
    initialize_governance_registry,
    list_schedules,
    record_schedule_run,
    update_schedule,
)
from ml_pipeline_monitor.services.pipeline_service import run_pipeline_and_persist

LOGGER = get_app_logger("worker")

_FIELD_SPECS = (
    ("minute", 0, 59),
    ("hour", 0, 23),
    ("day-of-month", 1, 31),
    ("month", 1, 12),
    ("day-of-week", 0, 7),
)


def _parse_cron_field(field: str, lo: int, hi: int) -> Set[int]:
    """Parse a single cron field into the set of allowed values.

    Supports ``*``, ``*/n``, ``n``, ``n-m``, ``n-m/s`` and comma lists.
    """
    allowed: Set[int] = set()

    def _expand(token: str) -> None:
        token = token.strip()
        if token == "*":
            allowed.update(range(lo, hi + 1))
            return
        if "/" in token:
            step = int(token.rsplit("/", 1)[1])
            base = token.rsplit("/", 1)[0]
            if base == "*":
                allowed.update(range(lo, hi + 1, step))
            else:
                for value in _expand_range(base, step):
                    allowed.add(value)
            return
        allowed.update(_expand_range(token, 1))

    def _expand_range(token: str, step: int) -> Set[int]:
        values: Set[int] = set()
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start, end = int(start_s), int(end_s)
            values.update(range(start, end + 1, step))
        else:
            values.add(int(token))
        return values

    for part in field.split(","):
        _expand(part)
    return {v for v in allowed if lo <= v <= hi}


def _next_run_from_cron(cron_expression: str, from_dt: datetime) -> datetime:
    """Compute the next trigger datetime (UTC) matching a 5-field cron expression."""
    fields = cron_expression.split()
    if len(fields) != 5:
        raise ValueError(f"Invalid cron expression: {cron_expression!r}")

    minutes, hours, days_of_month, months, weekdays = (
        _parse_cron_field(fields[0], *_FIELD_SPECS[0][1:]),
        _parse_cron_field(fields[1], *_FIELD_SPECS[1][1:]),
        _parse_cron_field(fields[2], *_FIELD_SPECS[2][1:]),
        _parse_cron_field(fields[3], *_FIELD_SPECS[3][1:]),
        _parse_cron_field(fields[4], *_FIELD_SPECS[4][1:]),
    )
    # Cron day-of-week (0=Sunday) -> Python weekday (0=Monday)
    weekday_py = {(value + 6) % 7 for value in weekdays}

    candidate = (from_dt + timedelta(minutes=1)).replace(second=0, microsecond=0)
    horizon = from_dt + timedelta(days=365 * 4)
    while candidate <= horizon:
        if (
            candidate.minute in minutes
            and candidate.hour in hours
            and candidate.day in days_of_month
            and candidate.month in months
            and candidate.weekday() in weekday_py
        ):
            return candidate
        candidate += timedelta(minutes=1)

    raise ValueError(f"Could not determine next run within horizon for {cron_expression!r}")


def _parse_dt(value: Any) -> datetime | None:
    """Parse a stored ISO timestamp into an aware datetime (or None)."""
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _build_task_config(schedule: Dict[str, Any]) -> Dict[str, Any]:
    """Translate a schedule row into the worker task configuration."""
    schedule_type = str(schedule.get("schedule_type", "pipeline_run"))
    return {
        "type": schedule_type,
        "params": {
            "dataset": schedule.get("pipeline_dataset") or "iris",
            "model_type": schedule.get("pipeline_model_type") or "Random Forest",
        },
    }


def _claim_due_schedules(now: datetime | None = None) -> List[Dict[str, Any]]:
    """Return enabled schedules due now, advancing their next_run_at to avoid double-fire."""
    now = now or datetime.now(timezone.utc)
    due: List[Dict[str, Any]] = []
    for schedule in list_schedules(limit=1000):
        if not schedule.get("enabled"):
            continue
        next_run_raw = schedule.get("next_run_at")
        if next_run_raw:
            next_run = _parse_dt(next_run_raw)
            if next_run and next_run > now:
                continue
        try:
            next_ts = _next_run_from_cron(
                str(schedule.get("cron_expression", "* * * * *")), now
            )
        except ValueError:
            LOGGER.warning(
                "Skipping schedule %s: invalid cron %r",
                schedule.get("schedule_name"),
                schedule.get("cron_expression"),
            )
            continue
        update_schedule(
            schedule_id=int(schedule["id"]),
            last_run_at=now.isoformat(),
            next_run_at=next_ts.isoformat(),
        )
        due.append(schedule)
    return due


def _run_schedule(schedule: Dict[str, Any]) -> None:
    """Execute a single due schedule and record its outcome."""
    schedule_id = int(schedule["id"])
    schedule_name = schedule.get("schedule_name", schedule_id)
    LOGGER.info(
        "Running schedule %s (%s)", schedule_name, schedule.get("schedule_type")
    )
    try:
        _execute_scheduled_task(_build_task_config(schedule))
        record_schedule_run(schedule_id=schedule_id, status="success")
        LOGGER.info("Schedule %s completed successfully", schedule_name)
    except Exception as exc:
        LOGGER.exception("Schedule %s failed: %s", schedule_name, exc)
        record_schedule_run(schedule_id=schedule_id, status="failed", error=str(exc))


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
            params=params.get("hyperparameters", {}),
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

    initialize_governance_registry()

    LOGGER.info("Worker starting with concurrency=%d, poll_interval=%.1fs", concurrency, poll_interval)

    running = True
    while running:
        try:
            _run_once(poll_interval=poll_interval, concurrency=concurrency)
        except KeyboardInterrupt:
            LOGGER.info("Worker received shutdown signal")
            running = False
        except Exception as exc:
            LOGGER.exception("Worker loop error: %s", exc)
            time.sleep(poll_interval)

    LOGGER.info("Worker shut down cleanly")


def _run_once(poll_interval: float = 5.0, concurrency: int = 4) -> None:
    due = _claim_due_schedules()
    if due:
        LOGGER.info("Worker found %d due schedule(s)", len(due))
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
            futures = [executor.submit(_run_schedule, schedule) for schedule in due]
            for future in futures:
                try:
                    future.result()
                except Exception as exc:
                    LOGGER.exception("Schedule task error: %s", exc)
    time.sleep(poll_interval)


if __name__ == "__main__":
    run_worker_loop()
