"""
Governance CRUD operations.

Handles teams, users, workspaces, alert events, schedules,
and user activity logging for multi-tenant governance.
"""

from __future__ import annotations

import json
from typing import Any

from ml_pipeline_monitor.database.schema import _get_connection


def create_team(team_name: str) -> int:
    """Create a new team or return existing team ID."""
    with _get_connection() as conn:
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
    """Create a new user or update existing user credentials."""
    with _get_connection() as conn:
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
    """Create a new workspace or return existing workspace ID."""
    with _get_connection() as conn:
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
    """Log user activity for audit trail."""
    with _get_connection() as conn:
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
    """Save an alert event for monitoring and notification."""
    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO alert_events (workspace_id, alert_type, severity, message, metadata_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (workspace_id, alert_type, severity, message, json.dumps(metadata or {})),
        )


def list_alert_events(*, workspace_id: int | None = None, limit: int = 50) -> list[dict[str, Any]]:
    """Retrieve alert events, optionally filtered by workspace."""
    if limit <= 0 or limit > 1000:
        limit = 50
    with _get_connection() as conn:
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
    """Create a new scheduled task."""
    with _get_connection() as conn:
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
    """Retrieve scheduled tasks, optionally filtered by workspace."""
    if limit <= 0 or limit > 1000:
        limit = 200
    with _get_connection() as conn:
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
    """Record a schedule execution run."""
    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO schedule_runs (schedule_id, status, error, metadata_json)
            VALUES (?, ?, ?, ?)
            """,
            (schedule_id, status, error, json.dumps(metadata or {})),
        )


def update_schedule(
    *,
    schedule_id: int,
    enabled: bool | None = None,
    next_run_at: str | None = None,
    last_run_at: str | None = None,
) -> None:
    """Update schedule bookkeeping fields (enabled, next/last run timestamps)."""
    updates: list[str] = []
    params: list[Any] = []
    if enabled is not None:
        updates.append("enabled = ?")
        params.append(1 if enabled else 0)
    if next_run_at is not None:
        updates.append("next_run_at = ?")
        params.append(next_run_at)
    if last_run_at is not None:
        updates.append("last_run_at = ?")
        params.append(last_run_at)
    if not updates:
        return
    params.append(schedule_id)
    with _get_connection() as conn:
        conn.execute(
            f"UPDATE schedules SET {', '.join(updates)}, updated_at=CURRENT_TIMESTAMP WHERE id = ?",
            params,
        )