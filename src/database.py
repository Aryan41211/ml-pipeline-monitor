"""SQLite persistence layer for experiments, models, and drift reports."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional

from src.db_engine import get_backend


def _json_default(value: Any) -> Any:
    """Convert numpy/pandas scalar-like values to JSON-native types."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")
    if isinstance(value, set):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _to_json(payload: Any) -> str:
    return json.dumps(payload, default=_json_default)


def _connect() -> sqlite3.Connection:
    return get_backend().connect()


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def initialize_db() -> None:
    """Create tables, migrate schema, and create useful indexes."""
    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id           TEXT    UNIQUE NOT NULL,
                name             TEXT    NOT NULL,
                dataset          TEXT    NOT NULL,
                model_type       TEXT    NOT NULL,
                task             TEXT    NOT NULL DEFAULT 'classification',
                status           TEXT    NOT NULL DEFAULT 'pending',
                started_at       TEXT,
                completed_at     TEXT,
                duration_seconds REAL,
                params           TEXT,
                metrics          TEXT,
                tags             TEXT,
                created_at       TEXT    DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS models (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id        TEXT    UNIQUE NOT NULL,
                run_id          TEXT    NOT NULL,
                name            TEXT    NOT NULL,
                version         INTEGER NOT NULL DEFAULT 1,
                dataset         TEXT    NOT NULL,
                dataset_name    TEXT,
                model_type      TEXT    NOT NULL,
                task            TEXT    NOT NULL DEFAULT 'classification',
                metrics         TEXT,
                artifact_path   TEXT,
                stage           TEXT    DEFAULT 'development',
                created_at      TEXT    DEFAULT CURRENT_TIMESTAMP,
                registered_at   TEXT    DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES experiments (run_id)
            );

            CREATE TABLE IF NOT EXISTS drift_reports (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                report_id        TEXT    UNIQUE NOT NULL,
                dataset          TEXT    NOT NULL,
                reference_size   INTEGER,
                current_size     INTEGER,
                drift_detected   INTEGER NOT NULL DEFAULT 0,
                drift_score      REAL,
                features_drifted INTEGER DEFAULT 0,
                feature_results  TEXT,
                created_at       TEXT    DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS model_lineage (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id          TEXT    NOT NULL,
                experiment_id     TEXT    NOT NULL,
                dataset           TEXT    NOT NULL,
                params            TEXT,
                parent_model_id   TEXT,
                created_at        TEXT    DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (model_id) REFERENCES models(model_id),
                FOREIGN KEY (experiment_id) REFERENCES experiments(run_id)
            );

            CREATE TABLE IF NOT EXISTS model_stage_events (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id          TEXT    NOT NULL,
                dataset           TEXT    NOT NULL,
                from_stage        TEXT,
                to_stage          TEXT    NOT NULL,
                reason            TEXT,
                created_at        TEXT    DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (model_id) REFERENCES models(model_id)
            );

            CREATE INDEX IF NOT EXISTS idx_experiments_dataset_created
                ON experiments(dataset, created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_models_dataset_stage
                ON models(dataset, stage);

            CREATE INDEX IF NOT EXISTS idx_models_dataset_version
                ON models(dataset, version DESC);

            CREATE INDEX IF NOT EXISTS idx_drift_reports_dataset_created
                ON drift_reports(dataset, created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_model_lineage_dataset_created
                ON model_lineage(dataset, created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_model_lineage_model
                ON model_lineage(model_id);

            CREATE INDEX IF NOT EXISTS idx_stage_events_model_created
                ON model_stage_events(model_id, created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_stage_events_dataset_created
                ON model_stage_events(dataset, created_at DESC);
            """
        )

        model_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(models)").fetchall()
        }

        if "dataset_name" not in model_columns:
            conn.execute("ALTER TABLE models ADD COLUMN dataset_name TEXT")

        if "created_at" not in model_columns:
            conn.execute("ALTER TABLE models ADD COLUMN created_at TEXT DEFAULT CURRENT_TIMESTAMP")

        conn.execute(
            "UPDATE models SET dataset_name = COALESCE(dataset_name, dataset)"
        )
        conn.execute(
            "UPDATE models SET created_at = COALESCE(created_at, registered_at, CURRENT_TIMESTAMP)"
        )


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def save_experiment(
    run_id: str,
    name: str,
    dataset: str,
    model_type: str,
    task: str,
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    duration: float,
    tags: Optional[Dict[str, Any]] = None,
) -> None:
    now = datetime.utcnow().isoformat(timespec="seconds")
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO experiments
                (run_id, name, dataset, model_type, task, status, started_at,
                 completed_at, duration_seconds, params, metrics, tags)
            VALUES (?, ?, ?, ?, ?, 'completed', ?, ?, ?, ?, ?, ?)
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
                _to_json(params),
                _to_json(metrics),
                _to_json(tags or {}),
            ),
        )


def get_experiments(limit: int = 200) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM experiments ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_experiment_by_run_id(run_id: str) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM experiments WHERE run_id = ?", (run_id,)
        ).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

def save_model(
    model_id: str,
    run_id: str,
    name: str,
    dataset: str,
    model_type: str,
    task: str,
    metrics: Dict[str, Any],
    artifact_path: str,
    params: Optional[Dict[str, Any]] = None,
    experiment_id: Optional[str] = None,
    version: Optional[int] = None,
) -> Dict[str, Any]:
    now = datetime.utcnow().isoformat(timespec="seconds")
    with get_connection() as conn:
        if version is None:
            row = conn.execute(
                "SELECT COALESCE(MAX(version), 0) AS max_version FROM models WHERE dataset = ?",
                (dataset,),
            ).fetchone()
            version = int(row["max_version"]) + 1

        conn.execute(
            """
            INSERT OR REPLACE INTO models
                (model_id, run_id, name, version, dataset, dataset_name, model_type, task,
                 metrics, artifact_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                _to_json(metrics),
                artifact_path,
                now,
            ),
        )

        prev = conn.execute(
            """
            SELECT model_id
            FROM models
            WHERE dataset = ? AND version < ?
            ORDER BY version DESC
            LIMIT 1
            """,
            (dataset, version),
        ).fetchone()

        save_model_lineage(
            model_id=model_id,
            experiment_id=experiment_id or run_id,
            dataset=dataset,
            params=params or {},
            parent_model_id=prev["model_id"] if prev else None,
            conn=conn,
        )

        conn.execute(
            """
            INSERT INTO model_stage_events (model_id, dataset, from_stage, to_stage, reason)
            VALUES (?, ?, ?, ?, ?)
            """,
            (model_id, dataset, None, "development", "initial_registration"),
        )

    return {
        "model_id": model_id,
        "version": version,
        "artifact_path": artifact_path,
        "created_at": now,
    }


def get_models(limit: int = 100) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM models ORDER BY registered_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_latest_production_model(dataset: Optional[str] = None) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        if dataset:
            row = conn.execute(
                """
                SELECT *
                FROM models
                WHERE stage = 'production' AND dataset = ?
                ORDER BY created_at DESC, version DESC
                LIMIT 1
                """,
                (dataset,),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT *
                FROM models
                WHERE stage = 'production'
                ORDER BY created_at DESC, version DESC
                LIMIT 1
                """
            ).fetchone()

    return dict(row) if row else None


def update_model_stage(model_id: str, stage: str) -> None:
    valid = {"development", "staging", "production", "archived"}
    if stage not in valid:
        raise ValueError(f"Stage must be one of {valid}")

    with get_connection() as conn:
        row = conn.execute(
            "SELECT model_id, dataset, stage FROM models WHERE model_id = ?",
            (model_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown model_id: {model_id}")

        old_stage = row["stage"] or "development"
        if old_stage == stage:
            return

        if stage == "production":
            to_demote = conn.execute(
                """
                SELECT model_id
                FROM models
                WHERE dataset = ? AND stage = 'production' AND model_id != ?
                """,
                (row["dataset"], model_id),
            ).fetchall()

            conn.execute(
                """
                UPDATE models
                SET stage = 'staging'
                WHERE dataset = ? AND stage = 'production' AND model_id != ?
                """,
                (row["dataset"], model_id),
            )

            for demoted in to_demote:
                conn.execute(
                    """
                    INSERT INTO model_stage_events (model_id, dataset, from_stage, to_stage, reason)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (demoted["model_id"], row["dataset"], "production", "staging", "auto_demote_on_replacement"),
                )

        conn.execute(
            "UPDATE models SET stage = ? WHERE model_id = ?",
            (stage, model_id),
        )

        conn.execute(
            """
            INSERT INTO model_stage_events (model_id, dataset, from_stage, to_stage, reason)
            VALUES (?, ?, ?, ?, ?)
            """,
            (model_id, row["dataset"], old_stage, stage, "manual_update"),
        )


def get_model_stage_events(model_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT model_id, dataset, from_stage, to_stage, reason, created_at
            FROM model_stage_events
            WHERE model_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (model_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_recent_production_models(dataset: str, limit: int = 2) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT m.model_id, m.name, m.version, m.dataset, m.stage, m.registered_at
            FROM model_stage_events e
            JOIN models m ON m.model_id = e.model_id
            WHERE e.dataset = ? AND e.to_stage = 'production'
            ORDER BY e.created_at DESC
            LIMIT ?
            """,
            (dataset, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def save_model_lineage(
    model_id: str,
    experiment_id: str,
    dataset: str,
    params: Dict[str, Any],
    parent_model_id: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    payload = _to_json(params or {})

    if conn is not None:
        conn.execute(
            """
            INSERT INTO model_lineage
                (model_id, experiment_id, dataset, params, parent_model_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (model_id, experiment_id, dataset, payload, parent_model_id),
        )
        return

    with get_connection() as local_conn:
        local_conn.execute(
            """
            INSERT INTO model_lineage
                (model_id, experiment_id, dataset, params, parent_model_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (model_id, experiment_id, dataset, payload, parent_model_id),
        )


def get_model_lineage(
    limit: int = 200,
    dataset: Optional[str] = None,
) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        if dataset:
            rows = conn.execute(
                """
                SELECT ml.*, m.version, m.stage
                FROM model_lineage ml
                LEFT JOIN models m ON m.model_id = ml.model_id
                WHERE ml.dataset = ?
                ORDER BY ml.created_at DESC
                LIMIT ?
                """,
                (dataset, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT ml.*, m.version, m.stage
                FROM model_lineage ml
                LEFT JOIN models m ON m.model_id = ml.model_id
                ORDER BY ml.created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Drift Reports
# ---------------------------------------------------------------------------

def save_drift_report(
    report_id: str,
    dataset: str,
    reference_size: int,
    current_size: int,
    drift_detected: bool,
    drift_score: float,
    features_drifted: int,
    feature_results: Dict[str, Any],
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO drift_reports
                (report_id, dataset, reference_size, current_size,
                 drift_detected, drift_score, features_drifted, feature_results)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report_id,
                dataset,
                reference_size,
                current_size,
                int(drift_detected),
                drift_score,
                features_drifted,
                _to_json(feature_results),
            ),
        )


def get_drift_reports(limit: int = 50) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM drift_reports ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
