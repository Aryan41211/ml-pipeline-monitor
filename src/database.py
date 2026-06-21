"""Persistence layer for experiments, models, and drift reports."""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, UTC
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

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
                confusion_matrix TEXT,
                feature_importances TEXT,
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

            CREATE TABLE IF NOT EXISTS drift_references (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset          TEXT    NOT NULL,
                feature_names    TEXT    NOT NULL,
                reference_data   TEXT    NOT NULL,
                created_at       TEXT    DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(dataset)
            );

            CREATE INDEX IF NOT EXISTS idx_experiments_dataset_created
                ON experiments(dataset, created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_models_dataset_stage
                ON models(dataset, stage);

            CREATE INDEX IF NOT EXISTS idx_models_dataset_version
                ON models(dataset, version DESC);

            CREATE INDEX IF NOT EXISTS idx_models_created_at
                ON models(created_at DESC);

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
                confusion_matrix TEXT,
                feature_importances TEXT,
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

            CREATE TABLE IF NOT EXISTS drift_references (
                id               BIGSERIAL PRIMARY KEY,
                dataset          TEXT    NOT NULL UNIQUE,
                feature_names    TEXT    NOT NULL,
                reference_data   TEXT    NOT NULL,
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

            CREATE INDEX IF NOT EXISTS idx_drift_reports_dataset_created
                ON drift_reports(dataset, created_at DESC);
    """

    with get_connection() as conn:
        conn.executescript(postgres_schema if backend == "postgres" else sqlite_schema)

        def ensure_column_exists(table: str, column: str, definition: str) -> None:
            if backend == "postgres":
                existing_columns = {
                    row["column_name"]
                    for row in conn.execute(
                        """
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_schema = 'public' AND table_name = ?
                        """,
                        (table,),
                    ).fetchall()
                }
            else:
                existing_columns = {
                    row["name"]
                    for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
                }

            if column not in existing_columns:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

        ensure_column_exists("models", "dataset_name", "TEXT")
        ensure_column_exists("models", "created_at", "TEXT DEFAULT CURRENT_TIMESTAMP")
        ensure_column_exists("models", "params", "TEXT")
        ensure_column_exists("models", "confusion_matrix", "TEXT")
        ensure_column_exists("models", "feature_importances", "TEXT")
        ensure_column_exists("models", "experiment_id", "TEXT")
        ensure_column_exists("models", "parent_model_id", "TEXT")

        ensure_column_exists("model_stage_events", "dataset", "TEXT")
        ensure_column_exists("model_stage_events", "changed_at", "TIMESTAMP")
        ensure_column_exists("model_stage_events", "note", "TEXT")

        conn.execute(
            "UPDATE models SET dataset_name = COALESCE(dataset_name, dataset)"
        )
        conn.execute(
            "UPDATE models SET created_at = COALESCE(created_at, registered_at, CURRENT_TIMESTAMP)"
        )
        conn.execute(
            "UPDATE models SET experiment_id = COALESCE(experiment_id, run_id)"
        )
        conn.execute(
            "UPDATE model_stage_events SET dataset = COALESCE(dataset, '')"
        )
        conn.execute(
            "UPDATE model_stage_events SET changed_at = COALESCE(changed_at, CURRENT_TIMESTAMP)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_stage_events_model_changed ON model_stage_events(model_id, changed_at DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_stage_events_dataset_changed ON model_stage_events(dataset, changed_at DESC)"
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
    now = datetime.now(UTC).isoformat(timespec="seconds")
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
    confusion_matrix: Optional[np.ndarray] = None,
    feature_importances: Optional[Any] = None,
) -> Dict[str, Any]:
    now = datetime.now(UTC).isoformat(timespec="seconds")
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

        confusion_json: Optional[str] = None
        if confusion_matrix is not None:
            confusion_json = json.dumps(np.asarray(confusion_matrix).tolist())

        feature_importances_json: Optional[str] = None
        if feature_importances is not None:
            # Accept pd.Series or array-like
            try:
                if hasattr(feature_importances, "to_dict"):
                    feature_importances_json = json.dumps(
                        {str(k): float(v) for k, v in feature_importances.to_dict().items()}
                    )
                else:
                    feature_importances_json = json.dumps(feature_importances)
            except Exception:
                # Last resort: store string representation
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

    now = datetime.now(UTC).isoformat(timespec="seconds")
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

            # Store reference distribution from training data for drift detection
            try:
                from src.data_loader import load_dataset
                from src.config_loader import load_config
                pipeline_cfg = load_config().get("pipeline", {})
                ds = load_dataset(
                    dataset,
                    test_size=float(pipeline_cfg.get("test_size", 0.20)),
                    random_state=int(pipeline_cfg.get("random_seed", 42)),
                )
                import json
                import numpy as np
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
            except Exception as exc:
                # Log but don't fail promotion if reference storage fails
                import logging
                logging.getLogger(__name__).warning("Failed to store drift reference for %s: %s", dataset, exc)

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


def save_drift_reference(dataset: str, feature_names: List[str], reference_data: np.ndarray) -> None:
    """Store reference distribution for a dataset."""
    import json
    import numpy as np
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO drift_references (dataset, feature_names, reference_data)
            VALUES (?, ?, ?)
            ON CONFLICT(dataset) DO UPDATE SET
                feature_names=excluded.feature_names,
                reference_data=excluded.reference_data
            """,
            (dataset, json.dumps(feature_names), json.dumps(reference_data.tolist())),
        )


def get_drift_reference(dataset: str) -> Optional[Dict[str, Any]]:
    """Retrieve stored reference distribution for a dataset."""
    import json
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM drift_references WHERE dataset = ?",
            (dataset,),
        ).fetchone()
    if row:
        d = dict(row)
        d["feature_names"] = json.loads(d["feature_names"])
        d["reference_data"] = np.array(json.loads(d["reference_data"]))
        return d
    return None


# ---------------------------------------------------------------------------
# Prediction History (Batch/Online Inference Logging)
# ---------------------------------------------------------------------------

def initialize_prediction_registry() -> None:
    """Create tables for prediction request/history and latency tracking."""
    backend = _backend_name()

    prediction_registry_sqlite = """
        CREATE TABLE IF NOT EXISTS prediction_requests (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id         TEXT UNIQUE,
            correlation_id    TEXT,
            model_id           TEXT NOT NULL,
            dataset            TEXT,
            input_type         TEXT NOT NULL,
            input_hash         TEXT,
            num_predictions    INTEGER NOT NULL,
            status             TEXT NOT NULL,
            duration_ms        REAL,
            error              TEXT,
            created_at         TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id        TEXT NOT NULL,
            row_index         INTEGER NOT NULL,
            prediction        TEXT NOT NULL,
            probability       TEXT,
            created_at        TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (request_id) REFERENCES prediction_requests(request_id)
        );

        CREATE INDEX IF NOT EXISTS idx_prediction_requests_created
            ON prediction_requests(created_at DESC);

        CREATE INDEX IF NOT EXISTS idx_predictions_request_row
            ON predictions(request_id, row_index);
    """

    prediction_registry_postgres = """
        CREATE TABLE IF NOT EXISTS prediction_requests (
            id                 BIGSERIAL PRIMARY KEY,
            request_id         TEXT UNIQUE,
            correlation_id    TEXT,
            model_id           TEXT NOT NULL,
            dataset            TEXT,
            input_type         TEXT NOT NULL,
            input_hash         TEXT,
            num_predictions    INTEGER NOT NULL,
            status             TEXT NOT NULL,
            duration_ms        DOUBLE PRECISION,
            error              TEXT,
            created_at         TEXT DEFAULT CURRENT_TIMESTAMP::text
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id                 BIGSERIAL PRIMARY KEY,
            request_id        TEXT NOT NULL,
            row_index         INTEGER NOT NULL,
            prediction        TEXT NOT NULL,
            probability       TEXT,
            created_at        TEXT DEFAULT CURRENT_TIMESTAMP::text,
            FOREIGN KEY (request_id) REFERENCES prediction_requests(request_id)
        );

        CREATE INDEX IF NOT EXISTS idx_prediction_requests_created
            ON prediction_requests(created_at DESC);

        CREATE INDEX IF NOT EXISTS idx_predictions_request_row
            ON predictions(request_id, row_index);
    """

    with get_connection() as conn:
        conn.executescript(
            prediction_registry_postgres if backend == "postgres" else prediction_registry_sqlite
        )


def save_prediction_request(
    *,
    request_id: str,
    correlation_id: str | None,
    model_id: str,
    dataset: str | None,
    input_type: str,
    input_hash: str | None,
    num_predictions: int,
    status: str,
    duration_ms: float | None,
    error: str | None,
) -> None:
    """Persist prediction request metadata."""
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO prediction_requests
                (request_id, correlation_id, model_id, dataset, input_type, input_hash,
                 num_predictions, status, duration_ms, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(request_id) DO UPDATE SET
                correlation_id=excluded.correlation_id,
                model_id=excluded.model_id,
                dataset=excluded.dataset,
                input_type=excluded.input_type,
                input_hash=excluded.input_hash,
                num_predictions=excluded.num_predictions,
                status=excluded.status,
                duration_ms=excluded.duration_ms,
                error=excluded.error
            """,
            (
                request_id,
                correlation_id,
                model_id,
                dataset,
                input_type,
                input_hash,
                int(num_predictions),
                status,
                duration_ms,
                error,
            ),
        )


def save_predictions_for_request(
    *,
    request_id: str,
    predictions: list[Any],
    probabilities: list[Any] | None = None,
) -> None:
    """Persist per-row predictions for a request."""
    probabilities = probabilities or [None] * len(predictions)
    with get_connection() as conn:
        for idx, pred in enumerate(predictions):
            conn.execute(
                """
                INSERT INTO predictions (request_id, row_index, prediction, probability)
                VALUES (?, ?, ?, ?)
                """,
                (
                    request_id,
                    int(idx),
                    str(pred),
                    None if probabilities[idx] is None else str(probabilities[idx]),
                ),
            )


def get_prediction_history(limit: int = 50) -> list[dict[str, Any]]:
    """Fetch prediction request history (newest first)."""
    if limit <= 0 or limit > 1000:
        limit = 50
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM prediction_requests
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_prediction_history_by_request_id(request_id: str) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM prediction_requests WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        if not row:
            return None
        req = dict(row)

        preds = conn.execute(
            """
            SELECT row_index, prediction, probability
            FROM predictions
            WHERE request_id = ?
            ORDER BY row_index ASC
            """,
            (request_id,),
        ).fetchall()
        req["predictions"] = [
            {
                "row_index": int(r["row_index"]),
                "prediction": r["prediction"],
                "probability": r["probability"],
            }
            for r in preds
        ]
    return req


# ---------------------------------------------------------------------------
# Governance: Users, Teams, Workspaces, Alerts, Scheduling
# ---------------------------------------------------------------------------

def initialize_governance_registry() -> None:
    """Create governance tables (users/teams/workspaces, alert history, schedules)."""
    backend = _backend_name()

    governance_registry_sqlite = """
        CREATE TABLE IF NOT EXISTS teams (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            team_name    TEXT NOT NULL UNIQUE,
            created_at    TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS users (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role          TEXT NOT NULL,
            team_id       INTEGER,
            created_at    TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (team_id) REFERENCES teams(id)
        );

        CREATE TABLE IF NOT EXISTS workspaces (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_name TEXT NOT NULL UNIQUE,
            team_id         INTEGER NOT NULL,
            created_at     TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (team_id) REFERENCES teams(id)
        );

        CREATE TABLE IF NOT EXISTS workspace_members (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_id    INTEGER NOT NULL,
            user_id         INTEGER NOT NULL,
            role_override   TEXT,
            created_at      TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (workspace_id) REFERENCES workspaces(id),
            FOREIGN KEY (user_id) REFERENCES users(id),
            UNIQUE(workspace_id, user_id)
        );

        CREATE TABLE IF NOT EXISTS user_activity_logs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         INTEGER NOT NULL,
            workspace_id   INTEGER,
            action          TEXT NOT NULL,
            metadata_json   TEXT NOT NULL,
            created_at      TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (workspace_id) REFERENCES workspaces(id)
        );

        CREATE TABLE IF NOT EXISTS alert_events (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_id    INTEGER,
            alert_type      TEXT NOT NULL,
            severity        TEXT NOT NULL,
            message         TEXT NOT NULL,
            metadata_json  TEXT NOT NULL,
            created_at      TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (workspace_id) REFERENCES workspaces(id)
        );

        CREATE TABLE IF NOT EXISTS alert_channels (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_id  INTEGER NOT NULL,
            channel_type  TEXT NOT NULL,  -- email|slack
            enabled        INTEGER NOT NULL DEFAULT 1,
            target         TEXT NOT NULL, -- email address or slack webhook url
            created_at     TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (workspace_id) REFERENCES workspaces(id),
            UNIQUE(workspace_id, channel_type, target)
        );

        CREATE TABLE IF NOT EXISTS schedules (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_id     INTEGER NOT NULL,
            schedule_name    TEXT NOT NULL,
            schedule_type    TEXT NOT NULL, -- training|drift_scan|retraining
            cron_expression  TEXT NOT NULL,
            timezone         TEXT NOT NULL DEFAULT 'UTC',
            enabled          INTEGER NOT NULL DEFAULT 1,
            next_run_at      TEXT,
            last_run_at      TEXT,
            pipeline_dataset TEXT,
            pipeline_model_type TEXT,
            created_at       TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at       TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (workspace_id) REFERENCES workspaces(id)
        );

        CREATE TABLE IF NOT EXISTS schedule_runs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            schedule_id    INTEGER NOT NULL,
            status         TEXT NOT NULL, -- queued|running|success|failed
            started_at     TEXT DEFAULT CURRENT_TIMESTAMP,
            completed_at   TEXT,
            error           TEXT,
            metadata_json  TEXT NOT NULL,
            FOREIGN KEY (schedule_id) REFERENCES schedules(id),
            UNIQUE(schedule_id, started_at)
        );

        CREATE INDEX IF NOT EXISTS idx_alert_events_workspace_created
            ON alert_events(workspace_id, created_at DESC);

        CREATE INDEX IF NOT EXISTS idx_schedule_next_run
            ON schedules(next_run_at, enabled);
    """

    governance_registry_postgres = """
        CREATE TABLE IF NOT EXISTS teams (
            id            BIGSERIAL PRIMARY KEY,
            team_name    TEXT NOT NULL UNIQUE,
            created_at    TEXT DEFAULT CURRENT_TIMESTAMP::text
        );

        CREATE TABLE IF NOT EXISTS users (
            id             BIGSERIAL PRIMARY KEY,
            username      TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role          TEXT NOT NULL,
            team_id       BIGINT,
            created_at    TEXT DEFAULT CURRENT_TIMESTAMP::text,
            FOREIGN KEY (team_id) REFERENCES teams(id)
        );

        CREATE TABLE IF NOT EXISTS workspaces (
            id               BIGSERIAL PRIMARY KEY,
            workspace_name  TEXT NOT NULL UNIQUE,
            team_id          BIGINT NOT NULL,
            created_at      TEXT DEFAULT CURRENT_TIMESTAMP::text,
            FOREIGN KEY (team_id) REFERENCES teams(id)
        );

        CREATE TABLE IF NOT EXISTS workspace_members (
            id             BIGSERIAL PRIMARY KEY,
            workspace_id  BIGINT NOT NULL,
            user_id       BIGINT NOT NULL,
            role_override TEXT,
            created_at    TEXT DEFAULT CURRENT_TIMESTAMP::text,
            FOREIGN KEY (workspace_id) REFERENCES workspaces(id),
            FOREIGN KEY (user_id) REFERENCES users(id),
            UNIQUE(workspace_id, user_id)
        );

        CREATE TABLE IF NOT EXISTS user_activity_logs (
            id              BIGSERIAL PRIMARY KEY,
            user_id         BIGINT NOT NULL,
            workspace_id   BIGINT,
            action          TEXT NOT NULL,
            metadata_json  TEXT NOT NULL,
            created_at      TEXT DEFAULT CURRENT_TIMESTAMP::text,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (workspace_id) REFERENCES workspaces(id)
        );

        CREATE TABLE IF NOT EXISTS alert_events (
            id               BIGSERIAL PRIMARY KEY,
            workspace_id    BIGINT,
            alert_type      TEXT NOT NULL,
            severity        TEXT NOT NULL,
            message         TEXT NOT NULL,
            metadata_json  TEXT NOT NULL,
            created_at      TEXT DEFAULT CURRENT_TIMESTAMP::text,
            FOREIGN KEY (workspace_id) REFERENCES workspaces(id)
        );

        CREATE TABLE IF NOT EXISTS alert_channels (
            id             BIGSERIAL PRIMARY KEY,
            workspace_id  BIGINT NOT NULL,
            channel_type  TEXT NOT NULL,  -- email|slack
            enabled        INTEGER NOT NULL DEFAULT 1,
            target         TEXT NOT NULL, -- email address or slack webhook url
            created_at     TEXT DEFAULT CURRENT_TIMESTAMP::text,
            FOREIGN KEY (workspace_id) REFERENCES workspaces(id),
            UNIQUE(workspace_id, channel_type, target)
        );

        CREATE TABLE IF NOT EXISTS schedules (
            id                BIGSERIAL PRIMARY KEY,
            workspace_id     BIGINT NOT NULL,
            schedule_name    TEXT NOT NULL,
            schedule_type    TEXT NOT NULL, -- training|drift_scan|retraining
            cron_expression  TEXT NOT NULL,
            timezone         TEXT NOT NULL DEFAULT 'UTC',
            enabled          INTEGER NOT NULL DEFAULT 1,
            next_run_at      TEXT,
            last_run_at      TEXT,
            pipeline_dataset TEXT,
            pipeline_model_type TEXT,
            created_at       TEXT DEFAULT CURRENT_TIMESTAMP::text,
            updated_at       TEXT DEFAULT CURRENT_TIMESTAMP::text,
            FOREIGN KEY (workspace_id) REFERENCES workspaces(id)
        );

        CREATE TABLE IF NOT EXISTS schedule_runs (
            id              BIGSERIAL PRIMARY KEY,
            schedule_id    BIGINT NOT NULL,
            status         TEXT NOT NULL, -- queued|running|success|failed
            started_at     TEXT DEFAULT CURRENT_TIMESTAMP::text,
            completed_at   TEXT,
            error           TEXT,
            metadata_json  TEXT NOT NULL,
            FOREIGN KEY (schedule_id) REFERENCES schedules(id),
            UNIQUE(schedule_id, started_at)
        );

        CREATE INDEX IF NOT EXISTS idx_alert_events_workspace_created
            ON alert_events(workspace_id, created_at DESC);

        CREATE INDEX IF NOT EXISTS idx_schedule_next_run
            ON schedules(next_run_at, enabled);
    """

    with get_connection() as conn:
        conn.executescript(
            governance_registry_postgres if backend == "postgres" else governance_registry_sqlite
        )


# -------------------- Minimal CRUD helpers --------------------

def create_team(team_name: str) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO teams (team_name) VALUES (?) ON CONFLICT(team_name) DO NOTHING RETURNING id",
            (team_name,),
        )
        row = cur.fetchone()
        if row:
            return int(row["id"])
        existing = conn.execute("SELECT id FROM teams WHERE team_name = ?", (team_name,)).fetchone()
        return int(existing["id"])


def create_user(*, username: str, password_hash: str, role: str, team_id: int) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO users (username, password_hash, role, team_id)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(username) DO UPDATE SET
                password_hash=excluded.password_hash,
                role=excluded.role,
                team_id=excluded.team_id
            RETURNING id
            """,
            (username, password_hash, role, team_id),
        )
        row = cur.fetchone()
        return int(row["id"])


def create_workspace(*, workspace_name: str, team_id: int) -> int:
    with get_connection() as conn:
        row = conn.execute(
            """
            INSERT INTO workspaces (workspace_name, team_id)
            VALUES (?, ?)
            ON CONFLICT(workspace_name) DO UPDATE SET team_id=excluded.team_id
            RETURNING id
            """,
            (workspace_name, team_id),
        ).fetchone()
        return int(row["id"])


def log_user_activity(*, user_id: int, workspace_id: int | None, action: str, metadata: dict[str, Any] | None = None) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO user_activity_logs (user_id, workspace_id, action, metadata_json)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, workspace_id, action, json.dumps(metadata or {})),
        )


def save_alert_event(
    *,
    workspace_id: int | None,
    alert_type: str,
    severity: str,
    message: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO alert_events (workspace_id, alert_type, severity, message, metadata_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (workspace_id, alert_type, severity, message, json.dumps(metadata or {})),
        )


def list_alert_events(*, workspace_id: int | None = None, limit: int = 50) -> list[dict[str, Any]]:
    if limit <= 0 or limit > 1000:
        limit = 50
    with get_connection() as conn:
        if workspace_id is None:
            rows = conn.execute(
                """
                SELECT * FROM alert_events ORDER BY created_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM alert_events
                WHERE workspace_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (workspace_id, limit),
            ).fetchall()
    return [dict(r) for r in rows]


def create_schedule(
    *,
    workspace_id: int,
    schedule_name: str,
    schedule_type: str,
    cron_expression: str,
    timezone: str = "UTC",
    enabled: bool = True,
    next_run_at: str | None = None,
    pipeline_dataset: str | None = None,
    pipeline_model_type: str | None = None,
) -> int:
    with get_connection() as conn:
        row = conn.execute(
            """
            INSERT INTO schedules
                (workspace_id, schedule_name, schedule_type, cron_expression, timezone, enabled, next_run_at, pipeline_dataset, pipeline_model_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING id
            """,
            (
                workspace_id,
                schedule_name,
                schedule_type,
                cron_expression,
                timezone,
                1 if enabled else 0,
                next_run_at,
                pipeline_dataset,
                pipeline_model_type,
            ),
        ).fetchone()
        return int(row["id"])


def list_schedules(*, workspace_id: int | None = None, limit: int = 200) -> list[dict[str, Any]]:
    if limit <= 0 or limit > 1000:
        limit = 200
    with get_connection() as conn:
        if workspace_id is None:
            rows = conn.execute(
                "SELECT * FROM schedules ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM schedules WHERE workspace_id = ? ORDER BY created_at DESC LIMIT ?",
                (workspace_id, limit),
            ).fetchall()
    return [dict(r) for r in rows]


def record_schedule_run(
    *,
    schedule_id: int,
    status: str,
    error: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO schedule_runs (schedule_id, status, error, metadata_json)
            VALUES (?, ?, ?, ?)
            """,
            (schedule_id, status, error, json.dumps(metadata or {})),
        )


# ---------------------------------------------------------------------------
# Dataset Registry (Dataset Versioning + Data Lineage)
# ---------------------------------------------------------------------------

def initialize_dataset_registry() -> None:
    """
    Create dataset registry tables without modifying the existing initialize_db()
    schema blocks (safer migrations for this codebase).
    """
    backend = _backend_name()

    dataset_registry_sqlite = """
        CREATE TABLE IF NOT EXISTS datasets (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id    TEXT UNIQUE NOT NULL,
            dataset_name  TEXT NOT NULL,
            created_at    TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS dataset_versions (
            id                       INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id               TEXT NOT NULL,
            version                  INTEGER NOT NULL,
            hash                     TEXT NOT NULL,
            row_count                INTEGER NOT NULL,
            column_count             INTEGER NOT NULL,
            missing_values_summary  TEXT NOT NULL,
            created_at               TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id),
            UNIQUE(dataset_id, version),
            UNIQUE(dataset_id, hash)
        );

        CREATE TABLE IF NOT EXISTS dataset_schema_snapshots (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_version_id  INTEGER NOT NULL,
            column_name          TEXT NOT NULL,
            dtype                 TEXT NOT NULL,
            FOREIGN KEY (dataset_version_id) REFERENCES dataset_versions(id),
            UNIQUE(dataset_version_id, column_name)
        );

        CREATE TABLE IF NOT EXISTS dataset_schema_changes (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id      TEXT NOT NULL,
            from_version    INTEGER NOT NULL,
            to_version      INTEGER NOT NULL,
            added_columns   TEXT NOT NULL,
            removed_columns TEXT NOT NULL,
            dtype_changes   TEXT NOT NULL,
            detected_at     TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id),
            UNIQUE(dataset_id, from_version, to_version)
        );

        CREATE TABLE IF NOT EXISTS dataset_lineage_edges (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            edge_type        TEXT NOT NULL,
            from_dataset_id  TEXT,
            from_version     INTEGER,
            to_dataset_id    TEXT,
            to_version       INTEGER,
            from_run_id      TEXT,
            to_run_id        TEXT,
            to_model_id      TEXT,
            from_model_id    TEXT,
            created_at       TEXT DEFAULT CURRENT_TIMESTAMP,
            note             TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_dataset_versions_dataset_created
            ON dataset_versions(dataset_id, created_at DESC);

        CREATE INDEX IF NOT EXISTS idx_dataset_schema_changes_dataset
            ON dataset_schema_changes(dataset_id, detected_at DESC);

        CREATE INDEX IF NOT EXISTS idx_dataset_lineage_edges_type_created
            ON dataset_lineage_edges(edge_type, created_at DESC);
    """

    dataset_registry_postgres = """
        CREATE TABLE IF NOT EXISTS datasets (
            id            BIGSERIAL PRIMARY KEY,
            dataset_id    TEXT UNIQUE NOT NULL,
            dataset_name  TEXT NOT NULL,
            created_at    TEXT DEFAULT CURRENT_TIMESTAMP::text
        );

        CREATE TABLE IF NOT EXISTS dataset_versions (
            id                       BIGSERIAL PRIMARY KEY,
            dataset_id               TEXT NOT NULL,
            version                  INTEGER NOT NULL,
            hash                     TEXT NOT NULL,
            row_count                INTEGER NOT NULL,
            column_count             INTEGER NOT NULL,
            missing_values_summary  TEXT NOT NULL,
            created_at               TEXT DEFAULT CURRENT_TIMESTAMP::text,
            FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id),
            UNIQUE(dataset_id, version),
            UNIQUE(dataset_id, hash)
        );

        CREATE TABLE IF NOT EXISTS dataset_schema_snapshots (
            id                    BIGSERIAL PRIMARY KEY,
            dataset_version_id  BIGINT NOT NULL,
            column_name          TEXT NOT NULL,
            dtype                 TEXT NOT NULL,
            FOREIGN KEY (dataset_version_id) REFERENCES dataset_versions(id),
            UNIQUE(dataset_version_id, column_name)
        );

        CREATE TABLE IF NOT EXISTS dataset_schema_changes (
            id               BIGSERIAL PRIMARY KEY,
            dataset_id      TEXT NOT NULL,
            from_version    INTEGER NOT NULL,
            to_version      INTEGER NOT NULL,
            added_columns   TEXT NOT NULL,
            removed_columns TEXT NOT NULL,
            dtype_changes   TEXT NOT NULL,
            detected_at     TEXT DEFAULT CURRENT_TIMESTAMP::text,
            FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id),
            UNIQUE(dataset_id, from_version, to_version)
        );

        CREATE TABLE IF NOT EXISTS dataset_lineage_edges (
            id                BIGSERIAL PRIMARY KEY,
            edge_type        TEXT NOT NULL,
            from_dataset_id  TEXT,
            from_version     INTEGER,
            to_dataset_id    TEXT,
            to_version       INTEGER,
            from_run_id      TEXT,
            to_run_id        TEXT,
            to_model_id      TEXT,
            from_model_id    TEXT,
            created_at       TEXT DEFAULT CURRENT_TIMESTAMP::text,
            note             TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_dataset_versions_dataset_created
            ON dataset_versions(dataset_id, created_at DESC);

        CREATE INDEX IF NOT EXISTS idx_dataset_schema_changes_dataset
            ON dataset_schema_changes(dataset_id, detected_at DESC);

        CREATE INDEX IF NOT EXISTS idx_dataset_lineage_edges_type_created
            ON dataset_lineage_edges(edge_type, created_at DESC);
    """

    with get_connection() as conn:
        conn.executescript(dataset_registry_postgres if backend == "postgres" else dataset_registry_sqlite)
