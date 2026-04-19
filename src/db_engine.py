"""Database backend abstraction with a SQLite implementation."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Protocol

from src.config_loader import ROOT_DIR, load_config


class DatabaseBackend(Protocol):
    """Protocol for pluggable database backends."""

    def connect(self) -> sqlite3.Connection:
        """Return an open DB-API connection."""


class SQLiteBackend:
    """SQLite backend implementation used by default."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    def connect(self) -> sqlite3.Connection:
        path = Path(self.db_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn


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

    if backend != "sqlite":
        raise ValueError(
            f"Unsupported database backend '{backend}'. Only 'sqlite' is currently implemented."
        )

    return SQLiteBackend(resolve_sqlite_db_path())
