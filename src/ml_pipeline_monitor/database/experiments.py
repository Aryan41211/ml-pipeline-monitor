"""
Experiment tracking CRUD operations.

Handles saving and retrieving ML pipeline experiments, including metrics,
parameters, and execution metadata.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ml_pipeline_monitor.database.schema import _get_connection


def save_experiment(
    run_id: str,
    name: str,
    dataset: str,
    model_type: str,
    task: str,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    duration: float,
    tags: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a completed experiment run."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO experiments
                (run_id, name, dataset, model_type, task, status, started_at,
                 completed_at, duration_seconds, params, metrics, tags)
            VALUES (?, ?, ?, ?, ?, 'completed', ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                name=excluded.name,
                dataset=excluded.dataset,
                model_type=excluded.model_type,
                task=excluded.task,
                status='completed',
                started_at=excluded.started_at,
                completed_at=excluded.completed_at,
                duration_seconds=excluded.duration_seconds,
                params=excluded.params,
                metrics=excluded.metrics,
                tags=excluded.tags
            """,
            (
                run_id,
                name,
                dataset,
                model_type,
                task,
                now,
                now,
                duration,
                json.dumps(params),
                json.dumps(metrics),
                json.dumps(tags or {}),
            ),
        )


def get_experiments(limit: int = 200) -> List[Dict[str, Any]]:
    """Retrieve recent experiments, ordered by creation date."""
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM experiments ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_experiment_by_run_id(run_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a single experiment by its run ID."""
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM experiments WHERE run_id = ?", (run_id,)
        ).fetchone()
    return dict(row) if row else None