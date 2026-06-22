"""Database backup and restore utilities for ML Pipeline Monitor."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, UTC
from pathlib import Path
from typing import Optional


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def backup_sqlite(db_path: str, output_dir: str = "backups") -> str:
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"SQLite database not found: {db_path}")
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    backup_path = out_dir / f"pipeline_monitor_{_timestamp()}.db"
    
    import sqlite3
    src = sqlite3.connect(str(path))
    dst = sqlite3.connect(str(backup_path))
    src.backup(dst)
    dst.close()
    src.close()
    
    print(f"SQLite backup created: {backup_path}")
    return str(backup_path)


def backup_postgres(
    dsn: str,
    output_dir: str = "backups",
    db_name: Optional[str] = None,
) -> str:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    backup_path = out_dir / f"pipeline_monitor_{_timestamp()}.sql"
    
    env = os.environ.copy()
    if "PGPASSWORD" not in env:
        import re
        match = re.search(r"password=([^\\s]+)", dsn, re.IGNORECASE)
        if match:
            env["PGPASSWORD"] = match.group(1)
    
    cmd = [
        "pg_dump",
        "--format=plain",
        "--no-owner",
        "--no-privileges",
        "-f", str(backup_path),
        dsn if not db_name else f"-d {dsn} -d {db_name}",
    ]
    # Remove the malformed args above and use proper pg_dump syntax
    cmd = [
        "pg_dump",
        "--format=plain",
        "--no-owner",
        "--no-privileges",
        "-f", str(backup_path),
        dsn,
    ]
    
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"pg_dump failed: {result.stderr}")
    
    print(f"PostgreSQL backup created: {backup_path}")
    return str(backup_path)


def restore_sqlite(backup_path: str, target_path: str = ".pipeline_monitor.db") -> None:
    src = Path(backup_path)
    if not src.exists():
        raise FileNotFoundError(f"Backup file not found: {backup_path}")
    
    import sqlite3
    src_conn = sqlite3.connect(str(src))
    dst_conn = sqlite3.connect(str(target_path))
    src_conn.backup(dst_conn)
    dst_conn.close()
    src_conn.close()
    print(f"Restored to: {target_path}")


def restore_postgres(backup_path: str, dsn: str) -> None:
    path = Path(backup_path)
    if not path.exists():
        raise FileNotFoundError(f"Backup file not found: {backup_path}")
    
    env = os.environ.copy()
    import re
    match = re.search(r"password=([^\\s]+)", dsn, re.IGNORECASE)
    if match:
        env["PGPASSWORD"] = match.group(1)
    
    cmd = ["psql", "-f", str(path), dsn]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"psql restore failed: {result.stderr}")
    print(f"PostgreSQL restored from: {backup_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/backup.py backup|restore <backend> [path] [dsn]")
        sys.exit(1)
    
    action = sys.argv[1]
    backend = sys.argv[2].lower()
    
    if action == "backup":
        if backend == "sqlite":
            backup_sqlite(sys.argv[3] if len(sys.argv) > 3 else ".pipeline_monitor.db")
        elif backend == "postgres":
            backup_postgres(
                sys.argv[4] if len(sys.argv) > 4 else os.getenv("PIPELINE_DB_DSN", ""),
                db_name=sys.argv[3] if len(sys.argv) > 3 else None,
            )
    elif action == "restore":
        if backend == "sqlite":
            restore_sqlite(sys.argv[3])
        elif backend == "postgres":
            restore_postgres(sys.argv[3], sys.argv[4])
