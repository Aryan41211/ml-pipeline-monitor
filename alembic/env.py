"""Alembic environment configuration for ML Pipeline Monitor."""

from __future__ import annotations

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config_loader import load_config

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

config.set_main_option(
    "sqlalchemy.url",
    os.getenv("ALEMBIC_DB_DSN", "sqlite:///.pipeline_monitor.db"),
)


def _get_connectable():
    cfg = load_config().get("storage", {})
    backend = cfg.get("backend", "sqlite")
    if backend == "postgres":
        dsn = os.getenv("PIPELINE_DB_DSN") or cfg.get("postgres_dsn", "")
        if not dsn:
            raise RuntimeError("PostgreSQL backend selected but no DSN configured for Alembic")
        return create_engine(dsn)
    db_path = str((Path(__file__).resolve().parent.parent / cfg.get("db_path", ".pipeline_monitor.db")).resolve())
    return create_engine(f"sqlite:///{db_path}")


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, literal_binds=True, dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = _get_connectable()
    with connectable.connect() as connection:
        context.configure(connection=connection)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
