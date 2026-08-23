"""Database backend abstraction with a SQLite implementation."""

from __future__ import annotations

import os
import queue
import re
import sqlite3
import threading
from pathlib import Path

from ml_pipeline_monitor.database.interfaces import DatabaseBackend, DatabaseConnection

from ml_pipeline_monitor.core.config_loader import ROOT_DIR, load_config


class PostgresConnectionAdapter:
    """Compatibility adapter to keep sqlite-like calls in persistence layer.

    ``close()`` returns the connection to the owning pool rather than tearing it
    down, so a pool actually recycles connections instead of draining itself.
    """

    def __init__(self, connection, pool=None) -> None:
        self._connection = connection
        self._pool = pool
        self._closed = False

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
        if self._closed:
            return
        self._closed = True
        if self._pool is not None:
            self._pool.putconn(self._connection)
        else:
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

    def __init__(self, dsn: str, pool_size: int = 5, min_size: int = 1) -> None:
        self.dsn = dsn
        self.pool_size = max(1, int(pool_size))
        self.min_size = max(1, min(int(min_size), self.pool_size))
        self._pool = None
        self._init_pool()

    def _init_pool(self) -> None:
        try:
            from psycopg.rows import dict_row
            from psycopg_pool import ConnectionPool
        except Exception as exc:
            raise RuntimeError(
                "PostgreSQL backend requires psycopg-pool. Install with 'pip install psycopg-pool'."
            ) from exc

        self._pool = ConnectionPool(
            self.dsn,
            min_size=self.min_size,
            max_size=self.pool_size,
            kwargs={"row_factory": dict_row},
        )

    def connect(self) -> DatabaseConnection:
        if self._pool is None:
            self._init_pool()
        conn = self._pool.getconn()
        return PostgresConnectionAdapter(conn, self._pool)

    def close_all(self) -> None:
        """Close all connections in the pool."""
        if self._pool is not None:
            self._pool.close()
            self._pool = None


def resolve_sqlite_db_path() -> str:
    """Resolve SQLite path from env/config with sane defaults."""
    env_db = os.getenv("PIPELINE_DB")
    if env_db:
        return env_db

    cfg_db = load_config().get("storage", {}).get("db_path", ".pipeline_monitor.db")
    return str((ROOT_DIR / cfg_db).resolve())


_backend_lock = threading.Lock()
_backend_cache: tuple[tuple[str, str], DatabaseBackend] | None = None


def _resolve_backend_key() -> tuple[str, str]:
    """Return the (backend_name, target) pair identifying the configured backend."""
    storage_cfg = load_config().get("storage", {})
    backend = str(storage_cfg.get("backend", "sqlite")).strip().lower()

    if backend == "sqlite":
        return backend, resolve_sqlite_db_path()

    if backend == "postgres":
        dsn = os.getenv("PIPELINE_DB_DSN") or str(storage_cfg.get("postgres_dsn", "")).strip()
        if not dsn:
            raise ValueError(
                "PostgreSQL backend selected but no DSN configured. "
                "Set PIPELINE_DB_DSN or storage.postgres_dsn in config.yaml."
            )
        return backend, dsn

    raise ValueError(
        f"Unsupported database backend '{backend}'. Supported backends: 'sqlite', 'postgres'."
    )


def _build_backend(key: tuple[str, str]) -> DatabaseBackend:
    backend, target = key
    pool_cfg = load_config().get("storage", {}).get("connection_pool", {}) or {}

    if backend == "sqlite":
        return SQLiteBackend(target, pool_size=int(pool_cfg.get("max_size", 5)))

    return PostgresBackend(
        target,
        pool_size=int(pool_cfg.get("max_size", 5)),
        min_size=int(pool_cfg.get("min_size", 1)),
    )


def get_backend() -> DatabaseBackend:
    """Return the process-wide DB backend for the current configuration.

    The backend owns a connection pool, so it must be built once and reused. It
    is cached against the resolved (backend, target) pair: when the target
    changes -- a different ``PIPELINE_DB`` in a test, say -- the previous
    backend is closed and a new one takes its place.
    """
    global _backend_cache

    key = _resolve_backend_key()

    cached = _backend_cache
    if cached is not None and cached[0] == key:
        return cached[1]

    with _backend_lock:
        cached = _backend_cache
        if cached is not None and cached[0] == key:
            return cached[1]

        if cached is not None:
            close_all = getattr(cached[1], "close_all", None)
            if callable(close_all):
                try:
                    close_all()
                except Exception:  # pragma: no cover - best-effort teardown
                    pass

        backend_instance = _build_backend(key)
        _backend_cache = (key, backend_instance)
        return backend_instance


def reset_backend() -> None:
    """Close and forget the cached backend (used by tests and on shutdown)."""
    global _backend_cache
    with _backend_lock:
        cached = _backend_cache
        _backend_cache = None
    if cached is None:
        return
    close_all = getattr(cached[1], "close_all", None)
    if callable(close_all):
        try:
            close_all()
        except Exception:  # pragma: no cover - best-effort teardown
            pass
