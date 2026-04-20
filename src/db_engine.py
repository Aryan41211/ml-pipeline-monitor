"""Database backend abstraction with a SQLite implementation."""

from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path

from src.db_interface import DatabaseBackend, DatabaseConnection

from src.config_loader import ROOT_DIR, load_config


class PostgresConnectionAdapter:
    """Compatibility adapter to keep sqlite-like calls in persistence layer."""

    def __init__(self, connection) -> None:
        self._connection = connection

    @staticmethod
    def _normalize_query(query: str) -> str:
        normalized = query.replace("?", "%s")
        normalized = re.sub(r"datetime\(([^)]+)\)", r"\1", normalized)
        return normalized

    def execute(self, query: str, params=None):
        return self._connection.execute(self._normalize_query(query), params or ())

    def executescript(self, script: str) -> None:
        statements = [stmt.strip() for stmt in script.split(";") if stmt.strip()]
        with self._connection.cursor() as cur:
            for statement in statements:
                cur.execute(self._normalize_query(statement))

    def commit(self) -> None:
        self._connection.commit()

    def rollback(self) -> None:
        self._connection.rollback()

    def close(self) -> None:
        self._connection.close()


class SQLiteBackend:
    """SQLite backend implementation used by default."""

    name = "sqlite"

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    def connect(self) -> DatabaseConnection:
        path = Path(self.db_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn


class PostgresBackend:
    """PostgreSQL backend implementation via psycopg."""

    name = "postgres"

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    def connect(self) -> DatabaseConnection:
        try:
            from psycopg import connect
            from psycopg.rows import dict_row
        except Exception as exc:
            raise RuntimeError(
                "PostgreSQL backend requires psycopg. Install with 'pip install psycopg[binary]'."
            ) from exc

        conn = connect(self.dsn, row_factory=dict_row)
        return PostgresConnectionAdapter(conn)


def resolve_sqlite_db_path() -> str:
    """Resolve SQLite path from env/config with sane defaults."""
    env_db = os.getenv("PIPELINE_DB")
    if env_db:
        return env_db

    cfg_db = load_config().get("storage", {}).get("db_path", ".pipeline_monitor.db")
    return str((ROOT_DIR / cfg_db).resolve())


def get_backend() -> DatabaseBackend:
    """Return configured DB backend instance.

    The project is currently SQLite-only at runtime, but this function provides
    a single extension point to support PostgreSQL later.
    """
    storage_cfg = load_config().get("storage", {})
    backend = str(storage_cfg.get("backend", "sqlite")).strip().lower()

    if backend == "sqlite":
        return SQLiteBackend(resolve_sqlite_db_path())

    if backend == "postgres":
        dsn = os.getenv("PIPELINE_DB_DSN") or str(storage_cfg.get("postgres_dsn", "")).strip()
        if not dsn:
            raise ValueError(
                "PostgreSQL backend selected but no DSN configured. "
                "Set PIPELINE_DB_DSN or storage.postgres_dsn in config.yaml."
            )
        return PostgresBackend(dsn)

    raise ValueError(
        f"Unsupported database backend '{backend}'. Supported backends: 'sqlite', 'postgres'."
    )
