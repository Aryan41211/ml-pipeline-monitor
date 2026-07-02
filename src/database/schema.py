"""
Database schema management and migrations.

Handles table creation, schema initialization, and migrations for both
SQLite and PostgreSQL backends.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from src.db_engine import get_backend


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

    with _get_connection() as conn:
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

    with _get_connection() as conn:
        conn.executescript(dataset_registry_postgres if backend == "postgres" else dataset_registry_sqlite)


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

    with _get_connection() as conn:
        conn.executescript(
            prediction_registry_postgres if backend == "postgres" else prediction_registry_sqlite
        )


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
            created_at       TEXT DEFAULT CURRENT_TIMESTAMP::text,
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

    with _get_connection() as conn:
        conn.executescript(governance_registry_postgres if backend == "postgres" else governance_registry_sqlite)


def _backend_name() -> str:
    return str(get_backend().name)


@contextmanager
def _get_connection() -> Iterator[Any]:
    """Context manager for database connections with automatic commit/rollback."""
    conn = get_backend().connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
