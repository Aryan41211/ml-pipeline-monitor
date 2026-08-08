"""
Model registry CRUD operations.

Handles model metadata, stage transitions, lineage tracking,
and production model queries.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from ml_pipeline_monitor.database.schema import _get_connection


def save_model(
    model_id: str,
    run_id: str,
    name: str,
    dataset: str,
    model_type: str,
    task: str,
    metrics: Dict[str, float],
    artifact_path: str,
    params: Optional[Dict[str, Any]] = None,
    experiment_id: Optional[str] = None,
    parent_model_id: Optional[str] = None,
    version: Optional[int] = None,
    confusion_matrix: Optional[np.ndarray] = None,
    feature_importances: Optional[Any] = None,
) -> Dict[str, Any]:
    """Register a new model or update existing model metadata."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _get_connection() as conn:
        prev_model = conn.execute(
            """
            SELECT model_id, version
            FROM models
            WHERE dataset = ?
            ORDER BY version DESC, datetime(created_at) DESC
            LIMIT 1
            """,
            (dataset,),
        ).fetchone()

        if version is None:
            version = (int(prev_model["version"]) + 1) if prev_model else 1

        if parent_model_id is None and prev_model is not None:
            parent_model_id = str(prev_model["model_id"])

        effective_experiment_id = experiment_id or run_id

        confusion_json: Optional[str] = None
        if confusion_matrix is not None:
            confusion_json = json.dumps(np.asarray(confusion_matrix).tolist())

        feature_importances_json: Optional[str] = None
        if feature_importances is not None:
            try:
                if hasattr(feature_importances, "to_dict"):
                    feature_importances_json = json.dumps(
                        {str(k): float(v) for k, v in feature_importances.to_dict().items()}
                    )
                else:
                    feature_importances_json = json.dumps(feature_importances)
            except Exception:
                feature_importances_json = json.dumps(str(feature_importances))

        conn.execute(
            """
            INSERT INTO models
                (model_id, run_id, name, version, dataset, dataset_name, model_type, task,
                 metrics, params, confusion_matrix, feature_importances,
                 experiment_id, parent_model_id, artifact_path, stage, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                    COALESCE((SELECT stage FROM models WHERE model_id = ?), 'development'), ?)
            ON CONFLICT(model_id) DO UPDATE SET
                run_id=excluded.run_id,
                name=excluded.name,
                version=excluded.version,
                dataset=excluded.dataset,
                dataset_name=excluded.dataset_name,
                model_type=excluded.model_type,
                task=excluded.task,
                metrics=excluded.metrics,
                params=excluded.params,
                confusion_matrix=excluded.confusion_matrix,
                feature_importances=excluded.feature_importances,
                experiment_id=excluded.experiment_id,
                parent_model_id=excluded.parent_model_id,
                artifact_path=excluded.artifact_path,
                created_at=excluded.created_at
            """,
            (
                model_id,
                run_id,
                name,
                version,
                dataset,
                dataset,
                model_type,
                task,
                json.dumps(metrics),
                json.dumps(params or {}),
                confusion_json,
                feature_importances_json,
                effective_experiment_id,
                parent_model_id,
                artifact_path,
                model_id,
                now,
            ),
        )

        existing_stage_events = conn.execute(
            "SELECT COUNT(1) AS cnt FROM model_stage_events WHERE model_id = ?",
            (model_id,),
        ).fetchone()
        if int(existing_stage_events["cnt"]) == 0:
            conn.execute(
                """
                INSERT INTO model_stage_events (model_id, dataset, from_stage, to_stage, changed_at, note)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (model_id, dataset, None, "development", now, "initial registration"),
            )

    return {
        "model_id": model_id,
        "version": version,
        "artifact_path": artifact_path,
        "created_at": now,
        "experiment_id": effective_experiment_id,
        "parent_model_id": parent_model_id,
    }


def get_models(limit: int = 100) -> List[Dict[str, Any]]:
    """Retrieve all models, ordered by creation date."""
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM models ORDER BY created_at DESC, version DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_latest_production_model(dataset: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Get the latest production model, optionally filtered by dataset."""
    with _get_connection() as conn:
        if dataset:
            row = conn.execute(
                """
                SELECT * FROM models
                WHERE stage = 'production' AND dataset = ?
                ORDER BY created_at DESC, version DESC
                LIMIT 1
                """,
                (dataset,),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT * FROM models
                WHERE stage = 'production'
                ORDER BY created_at DESC, version DESC
                LIMIT 1
                """
            ).fetchone()
    return dict(row) if row else None


def get_recent_production_models(dataset: str, limit: int = 2) -> List[Dict[str, Any]]:
    """Get recent production models for rollback scenarios."""
    with _get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM models
            WHERE stage = 'production' AND dataset = ?
            ORDER BY created_at DESC, version DESC
            LIMIT ?
            """,
            (dataset, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_model_stage_events(model_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Get stage change history for a model."""
    with _get_connection() as conn:
        rows = conn.execute(
            """
            SELECT model_id, dataset, from_stage, to_stage, changed_at, note
            FROM model_stage_events
            WHERE model_id = ?
            ORDER BY changed_at DESC
            LIMIT ?
            """,
            (model_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_model_lineage(limit: int = 200, dataset: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get model lineage with optional dataset filter."""
    with _get_connection() as conn:
        if dataset:
            rows = conn.execute(
                """
                SELECT
                    model_id,
                    run_id,
                    dataset,
                    dataset_name,
                    model_type,
                    task,
                    stage,
                    version,
                    artifact_path,
                    params,
                    metrics,
                    experiment_id,
                    parent_model_id,
                    created_at
                FROM models
                WHERE dataset = ?
                ORDER BY created_at DESC, version DESC
                LIMIT ?
                """,
                (dataset, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT
                    model_id,
                    run_id,
                    dataset,
                    dataset_name,
                    model_type,
                    task,
                    stage,
                    version,
                    artifact_path,
                    params,
                    metrics,
                    experiment_id,
                    parent_model_id,
                    created_at
                FROM models
                ORDER BY created_at DESC, version DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
    return [dict(r) for r in rows]


def update_model_stage(model_id: str, stage: str) -> None:
    """Update model stage and record the transition."""
    valid = {"development", "staging", "production", "archived"}
    if stage not in valid:
        raise ValueError(f"Stage must be one of {valid}")

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT model_id, dataset, stage FROM models WHERE model_id = ?",
            (model_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown model_id: {model_id}")

        current_stage = str(row["stage"])
        dataset = str(row["dataset"])

        if current_stage == stage:
            return

        if stage == "production":
            to_demote = conn.execute(
                """
                SELECT model_id, stage
                FROM models
                WHERE dataset = ? AND stage = 'production' AND model_id != ?
                """,
                (dataset, model_id),
            ).fetchall()

            conn.execute(
                """
                UPDATE models
                SET stage = 'staging'
                WHERE dataset = ? AND stage = 'production' AND model_id != ?
                """,
                (dataset, model_id),
            )

            for demoted in to_demote:
                conn.execute(
                    """
                    INSERT INTO model_stage_events (model_id, dataset, from_stage, to_stage, changed_at, note)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(demoted["model_id"]),
                        dataset,
                        str(demoted["stage"]),
                        "staging",
                        now,
                        f"auto-demoted while promoting {model_id}",
                    ),
                )

            try:
                from ml_pipeline_monitor.ml.data_loader import load_dataset
                from ml_pipeline_monitor.core.config_loader import load_config
                pipeline_cfg = load_config().get("pipeline", {})
                ds = load_dataset(
                    dataset,
                    test_size=float(pipeline_cfg.get("test_size", 0.20)),
                    random_state=int(pipeline_cfg.get("random_seed", 42)),
                )
                conn.execute(
                    """
                    INSERT INTO drift_references (dataset, feature_names, reference_data)
                    VALUES (?, ?, ?)
                    ON CONFLICT(dataset) DO UPDATE SET
                        feature_names=excluded.feature_names,
                        reference_data=excluded.reference_data
                    """,
                    (dataset, json.dumps(ds["feature_names"]), json.dumps(ds["X_train"].values.tolist())),
                )
            except Exception:
                pass

        conn.execute(
            "UPDATE models SET stage = ? WHERE model_id = ?",
            (stage, model_id),
        )
        conn.execute(
            """
            INSERT INTO model_stage_events (model_id, dataset, from_stage, to_stage, changed_at, note)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (model_id, dataset, current_stage, stage, now, "manual stage update"),
        )