"""Fix schema discrepancies between alembic and schema.py.

The initial migration created drift_reports with severity/features/summary columns
but the code uses feature_results. drift_references had feature_stats/updated_at
but code uses feature_names. predictions table was old schema; code uses
prediction_requests + predictions (row-level).

Revision ID: 002_fix_schema
Revises: 001_initial
Create Date: 2026-08-08
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "002_fix_schema"
down_revision: Union[str, None] = "001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _column_exists(table: str, column: str, bind) -> bool:
    """Check if a column exists in a table (SQLite)."""
    dialect = bind.dialect.name
    if dialect == "sqlite":
        rows = bind.execute(sa.text(f"PRAGMA table_info({table})")).fetchall()
        return any(row[1] == column for row in rows)
    else:
        rows = bind.execute(
            sa.text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name = :tbl"
            ),
            {"tbl": table},
        ).fetchall()
        return any(row[0] == column for row in rows)


def upgrade() -> None:
    bind = op.get_bind()

    # --- Fix drift_reports ---
    # Alembic created: severity, features, summary
    # Code expects: feature_results
    if _column_exists("drift_reports", "feature_results", bind):
        pass  # already correct
    elif _column_exists("drift_reports", "severity", bind):
        # Drop old columns and add correct one
        # SQLite doesn't support DROP COLUMN before 3.35, so recreate table
        dialect = bind.dialect.name
        if dialect == "sqlite":
            op.execute("""
                CREATE TABLE IF NOT EXISTS drift_reports_new (
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
                )
            """)
            op.execute("""
                INSERT INTO drift_reports_new
                    (id, report_id, dataset, reference_size, current_size,
                     drift_detected, drift_score, features_drifted, created_at)
                SELECT id, report_id, dataset, reference_size, current_size,
                       drift_detected, drift_score, features_drifted, created_at
                FROM drift_reports
            """)
            op.execute("DROP TABLE drift_reports")
            op.execute("ALTER TABLE drift_reports_new RENAME TO drift_reports")
        else:
            op.execute("ALTER TABLE drift_reports DROP COLUMN IF EXISTS severity")
            op.execute("ALTER TABLE drift_reports DROP COLUMN IF EXISTS features")
            op.execute("ALTER TABLE drift_reports DROP COLUMN IF EXISTS summary")
            op.execute("ALTER TABLE drift_reports ADD COLUMN feature_results TEXT")

    # --- Fix drift_references ---
    # Alembic created: feature_stats, updated_at
    # Code expects: feature_names with UNIQUE(dataset)
    if _column_exists("drift_references", "feature_names", bind):
        pass  # already correct
    elif _column_exists("drift_references", "feature_stats", bind):
        dialect = bind.dialect.name
        if dialect == "sqlite":
            op.execute("""
                CREATE TABLE IF NOT EXISTS drift_references_new (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset          TEXT    NOT NULL UNIQUE,
                    feature_names    TEXT    NOT NULL,
                    reference_data   TEXT    NOT NULL,
                    created_at       TEXT    DEFAULT CURRENT_TIMESTAMP
                )
            """)
            op.execute("""
                INSERT INTO drift_references_new
                    (id, dataset, reference_data, created_at)
                SELECT id, dataset, reference_data, created_at
                FROM drift_references
            """)
            op.execute("DROP TABLE drift_references")
            op.execute("ALTER TABLE drift_references_new RENAME TO drift_references")
        else:
            op.execute("ALTER TABLE drift_references DROP COLUMN IF EXISTS feature_stats")
            op.execute("ALTER TABLE drift_references DROP COLUMN IF EXISTS updated_at")
            op.execute("ALTER TABLE drift_references ADD COLUMN feature_names TEXT NOT NULL DEFAULT '[]'")

    # --- Fix predictions ---
    # Alembic created: old predictions + predictions_log
    # Code expects: prediction_requests + predictions (row-level)
    # Check if prediction_requests exists
    tables = bind.execute(sa.text(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )).fetchall() if bind.dialect.name == "sqlite" else bind.execute(sa.text(
        "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
    )).fetchall()
    table_names = {row[0] for row in tables}

    if "prediction_requests" not in table_names:
        # Drop old tables
        for tbl in ["predictions_log", "predictions"]:
            if tbl in table_names:
                op.execute(f"DROP TABLE {tbl}")

        dialect = bind.dialect.name
        if dialect == "sqlite":
            op.execute("""
                CREATE TABLE IF NOT EXISTS prediction_requests (
                    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id         TEXT UNIQUE,
                    correlation_id     TEXT,
                    model_id           TEXT NOT NULL,
                    dataset            TEXT,
                    input_type         TEXT NOT NULL,
                    input_hash         TEXT,
                    num_predictions    INTEGER NOT NULL,
                    status             TEXT NOT NULL,
                    duration_ms        REAL,
                    error              TEXT,
                    created_at         TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            op.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id        TEXT NOT NULL,
                    row_index         INTEGER NOT NULL,
                    prediction        TEXT NOT NULL,
                    probability       TEXT,
                    created_at        TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (request_id) REFERENCES prediction_requests(request_id)
                )
            """)
        else:
            op.execute("""
                CREATE TABLE IF NOT EXISTS prediction_requests (
                    id                 BIGSERIAL PRIMARY KEY,
                    request_id         TEXT UNIQUE,
                    correlation_id     TEXT,
                    model_id           TEXT NOT NULL,
                    dataset            TEXT,
                    input_type         TEXT NOT NULL,
                    input_hash         TEXT,
                    num_predictions    INTEGER NOT NULL,
                    status             TEXT NOT NULL,
                    duration_ms        DOUBLE PRECISION,
                    error              TEXT,
                    created_at         TEXT DEFAULT CURRENT_TIMESTAMP::text
                )
            """)
            op.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id                 BIGSERIAL PRIMARY KEY,
                    request_id        TEXT NOT NULL,
                    row_index         INTEGER NOT NULL,
                    prediction        TEXT NOT NULL,
                    probability       TEXT,
                    created_at        TEXT DEFAULT CURRENT_TIMESTAMP::text,
                    FOREIGN KEY (request_id) REFERENCES prediction_requests(request_id)
                )
            """)

        op.execute("CREATE INDEX IF NOT EXISTS idx_prediction_requests_created ON prediction_requests(created_at DESC)")
        op.execute("CREATE INDEX IF NOT EXISTS idx_predictions_request_row ON predictions(request_id, row_index)")

    # --- Add missing indexes from schema.py ---
    op.execute("CREATE INDEX IF NOT EXISTS idx_experiments_dataset_created ON experiments(dataset, created_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_models_dataset_stage ON models(dataset, stage)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_models_dataset_version ON models(dataset, version DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_models_created_at ON models(created_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_drift_reports_dataset_created ON drift_reports(dataset, created_at DESC)")


def downgrade() -> None:
    # This migration is not safely reversible without data loss
    pass
