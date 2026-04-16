"""
Persistence layer backed by SQLite.

Stores experiment runs, model registry entries, and drift reports
without requiring an external tracking server.
"""
import json
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

DB_PATH = os.getenv("PIPELINE_DB", ".pipeline_monitor.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def initialize_db() -> None:
    """Create all tables if they do not already exist."""
    with _connect() as conn:
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
                model_type      TEXT    NOT NULL,
                task            TEXT    NOT NULL DEFAULT 'classification',
                metrics         TEXT,
                artifact_path   TEXT,
                stage           TEXT    DEFAULT 'development',
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
            """
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
    with _connect() as conn:
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
                json.dumps(params),
                json.dumps(metrics),
                json.dumps(tags or {}),
            ),
        )


def get_experiments(limit: int = 200) -> List[Dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM experiments ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_experiment_by_run_id(run_id: str) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
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
    version: int = 1,
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO models
                (model_id, run_id, name, version, dataset, model_type, task,
                 metrics, artifact_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_id,
                run_id,
                name,
                version,
                dataset,
                model_type,
                task,
                json.dumps(metrics),
                artifact_path,
            ),
        )


def get_models(limit: int = 100) -> List[Dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM models ORDER BY registered_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def update_model_stage(model_id: str, stage: str) -> None:
    valid = {"development", "staging", "production", "archived"}
    if stage not in valid:
        raise ValueError(f"Stage must be one of {valid}")
    with _connect() as conn:
        conn.execute(
            "UPDATE models SET stage = ? WHERE model_id = ?",
            (stage, model_id),
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
    with _connect() as conn:
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
                json.dumps(feature_results),
            ),
        )


def get_drift_reports(limit: int = 50) -> List[Dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM drift_reports ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
