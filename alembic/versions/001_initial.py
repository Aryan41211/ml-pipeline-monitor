"""Initial schema: create all core tables.

Revision ID: 001_initial
Revises:
Create Date: 2026-06-23
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
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
        )
    """)

    op.execute("""
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
        )
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS model_stage_events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id    TEXT NOT NULL,
            dataset     TEXT NOT NULL,
            from_stage  TEXT,
            to_stage    TEXT NOT NULL,
            changed_at  TEXT DEFAULT CURRENT_TIMESTAMP,
            note        TEXT,
            FOREIGN KEY (model_id) REFERENCES models (model_id)
        )
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS drift_reports (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id        TEXT    UNIQUE NOT NULL,
            dataset          TEXT    NOT NULL,
            reference_size   INTEGER,
            current_size     INTEGER,
            drift_detected   INTEGER NOT NULL DEFAULT 0,
            drift_score      REAL,
            features_drifted INTEGER DEFAULT 0,
            severity         TEXT,
            features         TEXT,
            summary          TEXT,
            created_at       TEXT    DEFAULT CURRENT_TIMESTAMP
        )
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS drift_references (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset     TEXT    NOT NULL,
            reference_data TEXT NOT NULL,
            feature_stats  TEXT,
            created_at     TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at     TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id  TEXT    NOT NULL,
            model_id    TEXT,
            dataset     TEXT,
            input_data  TEXT,
            predictions TEXT,
            status      TEXT    NOT NULL DEFAULT 'success',
            latency_ms  REAL,
            created_at  TEXT    DEFAULT CURRENT_TIMESTAMP
        )
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS predictions_log (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id   TEXT    NOT NULL,
            model_id     TEXT,
            dataset      TEXT,
            payload      TEXT,
            predictions  TEXT,
            latency_ms   REAL,
            error        TEXT,
            created_at   TEXT    DEFAULT CURRENT_TIMESTAMP
        )
    """)

    op.execute("CREATE INDEX IF NOT EXISTS idx_experiments_dataset ON experiments(dataset)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_models_stage ON models(stage)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_models_dataset ON models(dataset)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_drift_reports_dataset ON drift_reports(dataset)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_drift_reports_dataset")
    op.execute("DROP INDEX IF EXISTS idx_models_dataset")
    op.execute("DROP INDEX IF EXISTS idx_models_stage")
    op.execute("DROP INDEX IF EXISTS idx_experiments_status")
    op.execute("DROP INDEX IF EXISTS idx_experiments_dataset")
    op.execute("DROP TABLE IF EXISTS predictions_log")
    op.execute("DROP TABLE IF EXISTS predictions")
    op.execute("DROP TABLE IF EXISTS drift_references")
    op.execute("DROP TABLE IF EXISTS drift_reports")
    op.execute("DROP TABLE IF EXISTS model_stage_events")
    op.execute("DROP TABLE IF EXISTS models")
    op.execute("DROP TABLE IF EXISTS experiments")
