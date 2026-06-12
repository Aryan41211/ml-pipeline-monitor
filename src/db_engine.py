"""Database backend abstraction with a SQLite implementation."""

from __future__ import annotations

import os
import queue
import re
import sqlite3
import threading
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
    """SQLite backend implementation used by default with connection pooling."""

    name = "sqlite"

    def __init__(self, db_path: str, pool_size: int = 5) -> None:
        self.db_path = db_path
        self.pool_size = pool_size
        self._pool: queue.Queue[sqlite3.Connection] = queue.Queue(maxsize=pool_size)
        self._lock = threading.Lock()
        self._initialized = False

    def _init_pool(self) -> None:
        """Initialize the connection pool."""
        with self._lock:
            if self._initialized:
                return
            path = Path(self.db_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            for _ in range(self.pool_size):
                conn = sqlite3.connect(str(path), check_same_thread=False)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA foreign_keys=ON")
                self._pool.put(conn)
            self._initialized = True

    def connect(self) -> DatabaseConnection:
        self._init_pool()
        try:
            conn = self._pool.get_nowait()
        except queue.Empty:
            # Pool exhausted, create a temporary connection
            path = Path(self.db_path)
            conn = sqlite3.connect(str(path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
        return _PooledConnection(conn, self._pool)

    def close_all(self) -> None:
        """Close all connections in the pool."""
        with self._lock:
            while not self._pool.empty():
                try:
                    conn = self._pool.get_nowait()
                    conn.close()
                except queue.Empty:
                    break


class _PooledConnection:
    """Wrapper that returns connection to pool on close."""

    def __init__(self, conn: sqlite3.Connection, pool: queue.Queue) -> None:
        self._conn = conn
        self._pool = pool
        self._closed = False

    def __getattr__(self, name: str):
        return getattr(self._conn, name)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                self._pool.put_nowait(self._conn)
            except queue.Full:
                self._conn.close()

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()


class PostgresBackend:
    """PostgreSQL backend implementation via psycopg with connection pooling."""

    name = "postgres"

    def __init__(self, dsn: str, pool_size: int = 5) -> None:
        self.dsn = dsn
        self.pool_size = pool_size
        self._pool = None
        self._init_pool()

    def _init_pool(self) -> None:
        try:
            from psycopg_pool import ConnectionPool
            from psycopg.rows import dict_row
        except Exception as exc:
            raise RuntimeError(
                "PostgreSQL backend requires psycopg-pool. Install with 'pip install psycopg-pool'."
            ) from exc

        self._pool = ConnectionPool(
            self.dsn,
            min_size=1,
            max_size=self.pool_size,
            kwargs={"row_factory": dict_row},
        )

    def connect(self) -> DatabaseConnection:
        if self._pool is None:
            self._init_pool()
        conn = self._pool.getconn()
        return PostgresConnectionAdapter(conn)

    def close_all(self) -> None:
        """Close all connections in the pool."""
        if self._pool is not None:
            self._pool.closeall()
            self._pool = None


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
