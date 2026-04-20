"""Persistence layer for experiments, models, and drift reports."""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional

from src.db_engine import get_backend


def _backend_name() -> str:
    return str(get_backend().name)


def _connect():
    return get_backend().connect()


@contextmanager
def get_connection() -> Iterator[Any]:
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
    backend = _backend_name()

    sqlite_schema = """
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
                params          TEXT,
                experiment_id   TEXT,
                parent_model_id TEXT,
                artifact_path   TEXT,
                stage           TEXT    DEFAULT 'development',
                created_at      TEXT    DEFAULT CURRENT_TIMESTAMP,
                registered_at   TEXT    DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES experiments (run_id)
            );

            CREATE TABLE IF NOT EXISTS model_stage_events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id    TEXT NOT NULL,
                dataset     TEXT NOT NULL,
                from_stage  TEXT,
                to_stage    TEXT NOT NULL,
                changed_at  TEXT DEFAULT CURRENT_TIMESTAMP,
                note        TEXT,
                FOREIGN KEY (model_id) REFERENCES models (model_id)
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

            CREATE INDEX IF NOT EXISTS idx_experiments_dataset_created
                ON experiments(dataset, created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_models_dataset_stage
                ON models(dataset, stage);

            CREATE INDEX IF NOT EXISTS idx_models_dataset_version
                ON models(dataset, version DESC);

            CREATE INDEX IF NOT EXISTS idx_models_created_at
                ON models(created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_stage_events_model_changed
                ON model_stage_events(model_id, changed_at DESC);

            CREATE INDEX IF NOT EXISTS idx_stage_events_dataset_changed
                ON model_stage_events(dataset, changed_at DESC);

            CREATE INDEX IF NOT EXISTS idx_drift_reports_dataset_created
                ON drift_reports(dataset, created_at DESC);
    """

    postgres_schema = """
            CREATE TABLE IF NOT EXISTS experiments (
                id               BIGSERIAL PRIMARY KEY,
                run_id           TEXT    UNIQUE NOT NULL,
                name             TEXT    NOT NULL,
                dataset          TEXT    NOT NULL,
                model_type       TEXT    NOT NULL,
                task             TEXT    NOT NULL DEFAULT 'classification',
                status           TEXT    NOT NULL DEFAULT 'pending',
                started_at       TEXT,
                completed_at     TEXT,
                duration_seconds DOUBLE PRECISION,
                params           TEXT,
                metrics          TEXT,
                tags             TEXT,
                created_at       TEXT    DEFAULT CURRENT_TIMESTAMP::text
            );

            CREATE TABLE IF NOT EXISTS models (
                id              BIGSERIAL PRIMARY KEY,
                model_id        TEXT    UNIQUE NOT NULL,
                run_id          TEXT    NOT NULL,
                name            TEXT    NOT NULL,
                version         INTEGER NOT NULL DEFAULT 1,
                dataset         TEXT    NOT NULL,
                dataset_name    TEXT,
                model_type      TEXT    NOT NULL,
                task            TEXT    NOT NULL DEFAULT 'classification',
                metrics         TEXT,
                params          TEXT,
                experiment_id   TEXT,
                parent_model_id TEXT,
                artifact_path   TEXT,
                stage           TEXT    DEFAULT 'development',
                created_at      TEXT    DEFAULT CURRENT_TIMESTAMP::text,
                registered_at   TEXT    DEFAULT CURRENT_TIMESTAMP::text,
                FOREIGN KEY (run_id) REFERENCES experiments (run_id)
            );

            CREATE TABLE IF NOT EXISTS model_stage_events (
                id          BIGSERIAL PRIMARY KEY,
                model_id    TEXT NOT NULL,
                dataset     TEXT NOT NULL,
                from_stage  TEXT,
                to_stage    TEXT NOT NULL,
                changed_at  TEXT DEFAULT CURRENT_TIMESTAMP::text,
                note        TEXT,
                FOREIGN KEY (model_id) REFERENCES models (model_id)
            );

            CREATE TABLE IF NOT EXISTS drift_reports (
                id               BIGSERIAL PRIMARY KEY,
                report_id        TEXT    UNIQUE NOT NULL,
                dataset          TEXT    NOT NULL,
                reference_size   INTEGER,
                current_size     INTEGER,
                drift_detected   INTEGER NOT NULL DEFAULT 0,
                drift_score      DOUBLE PRECISION,
                features_drifted INTEGER DEFAULT 0,
                feature_results  TEXT,
                created_at       TEXT    DEFAULT CURRENT_TIMESTAMP::text
            );

            CREATE INDEX IF NOT EXISTS idx_experiments_dataset_created
                ON experiments(dataset, created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_models_dataset_stage
                ON models(dataset, stage);

            CREATE INDEX IF NOT EXISTS idx_models_dataset_version
                ON models(dataset, version DESC);

            CREATE INDEX IF NOT EXISTS idx_models_created_at
                ON models(created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_stage_events_model_changed
                ON model_stage_events(model_id, changed_at DESC);

            CREATE INDEX IF NOT EXISTS idx_stage_events_dataset_changed
                ON model_stage_events(dataset, changed_at DESC);

            CREATE INDEX IF NOT EXISTS idx_drift_reports_dataset_created
                ON drift_reports(dataset, created_at DESC);
    """

    with get_connection() as conn:
        conn.executescript(postgres_schema if backend == "postgres" else sqlite_schema)

        if backend == "postgres":
            model_columns = {
                row["column_name"]
                for row in conn.execute(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = ?
                    """,
                    ("models",),
                ).fetchall()
            }
        else:
            model_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(models)").fetchall()
            }

        if "dataset_name" not in model_columns:
            conn.execute("ALTER TABLE models ADD COLUMN dataset_name TEXT")

        if "created_at" not in model_columns:
            conn.execute("ALTER TABLE models ADD COLUMN created_at TEXT DEFAULT CURRENT_TIMESTAMP")

        if "params" not in model_columns:
            conn.execute("ALTER TABLE models ADD COLUMN params TEXT")

        if "experiment_id" not in model_columns:
            conn.execute("ALTER TABLE models ADD COLUMN experiment_id TEXT")

        if "parent_model_id" not in model_columns:
            conn.execute("ALTER TABLE models ADD COLUMN parent_model_id TEXT")

        conn.execute(
            "UPDATE models SET dataset_name = COALESCE(dataset_name, dataset)"
        )
        conn.execute(
            "UPDATE models SET created_at = COALESCE(created_at, registered_at, CURRENT_TIMESTAMP)"
        )
        conn.execute(
            "UPDATE models SET experiment_id = COALESCE(experiment_id, run_id)"
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
    metrics: Dict[str, float],
    duration: float,
    tags: Optional[Dict[str, Any]] = None,
) -> None:
    now = datetime.utcnow().isoformat(timespec="seconds")
    with get_connection() as conn:
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
    metrics: Dict[str, float],
    artifact_path: str,
    params: Optional[Dict[str, Any]] = None,
    experiment_id: Optional[str] = None,
    parent_model_id: Optional[str] = None,
    version: Optional[int] = None,
) -> Dict[str, Any]:
    now = datetime.utcnow().isoformat(timespec="seconds")
    with get_connection() as conn:
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

        conn.execute(
            """
            INSERT INTO models
                (model_id, run_id, name, version, dataset, dataset_name, model_type, task,
                 metrics, params, experiment_id, parent_model_id, artifact_path, stage, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT stage FROM models WHERE model_id = ?), 'development'), ?)
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
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM models ORDER BY created_at DESC, version DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_latest_production_model(dataset: Optional[str] = None) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
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
    with get_connection() as conn:
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
    with get_connection() as conn:
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
    with get_connection() as conn:
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
    valid = {"development", "staging", "production", "archived"}
    if stage not in valid:
        raise ValueError(f"Stage must be one of {valid}")

    now = datetime.utcnow().isoformat(timespec="seconds")
    with get_connection() as conn:
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
            INSERT INTO drift_reports
                (report_id, dataset, reference_size, current_size,
                 drift_detected, drift_score, features_drifted, feature_results)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(report_id) DO UPDATE SET
                dataset=excluded.dataset,
                reference_size=excluded.reference_size,
                current_size=excluded.current_size,
                drift_detected=excluded.drift_detected,
                drift_score=excluded.drift_score,
                features_drifted=excluded.features_drifted,
                feature_results=excluded.feature_results
            """,
            (
                report_id,
                dataset,
                reference_size,
                current_size,
                int(drift_detected),
                drift_score,
                features_drifted,
                json.dumps(feature_results),
            ),
        )


def get_drift_reports(limit: int = 50) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM drift_reports ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
