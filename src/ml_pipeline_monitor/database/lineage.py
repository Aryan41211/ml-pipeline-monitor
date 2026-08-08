"""
Dataset lineage tracking operations.

Handles dataset versioning, schema snapshots, schema changes,
and lineage edges for tracking data provenance.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ml_pipeline_monitor.database.schema import _get_connection


def create_dataset(dataset_id: str, dataset_name: str) -> int:
    """Create a new dataset entry."""
    with _get_connection() as conn:
        row = conn.execute(
            """
            INSERT INTO datasets (dataset_id, dataset_name)
            VALUES (?, ?)
            ON CONFLICT(dataset_id) DO UPDATE SET dataset_name=excluded.dataset_name
            RETURNING id
            """,
            (dataset_id, dataset_name),
        ).fetchone()
        return int(row["id"])


def create_dataset_version(
    dataset_id: str,
    version: int,
    hash: str,
    row_count: int,
    column_count: int,
    missing_values_summary: str,
) -> int:
    """Create a new dataset version."""
    with _get_connection() as conn:
        row = conn.execute(
            """
            INSERT INTO dataset_versions (dataset_id, version, hash, row_count, column_count, missing_values_summary)
            VALUES (?, ?, ?, ?, ?, ?)
            RETURNING id
            """,
            (dataset_id, version, hash, row_count, column_count, missing_values_summary),
        ).fetchone()
        return int(row["id"])


def save_schema_snapshot(dataset_version_id: int, column_name: str, dtype: str) -> None:
    """Save a column schema snapshot for a dataset version."""
    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO dataset_schema_snapshots (dataset_version_id, column_name, dtype)
            VALUES (?, ?, ?)
            ON CONFLICT(dataset_version_id, column_name) DO UPDATE SET dtype=excluded.dtype
            """,
            (dataset_version_id, column_name, dtype),
        )


def save_schema_change(
    dataset_id: str,
    from_version: int,
    to_version: int,
    added_columns: List[str],
    removed_columns: List[str],
    dtype_changes: Dict[str, str],
) -> None:
    """Record a schema change between dataset versions."""
    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO dataset_schema_changes (dataset_id, from_version, to_version, added_columns, removed_columns, dtype_changes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                dataset_id,
                from_version,
                to_version,
                json.dumps(added_columns),
                json.dumps(removed_columns),
                json.dumps(dtype_changes),
            ),
        )


def create_lineage_edge(
    edge_type: str,
    from_dataset_id: Optional[str] = None,
    from_version: Optional[int] = None,
    to_dataset_id: Optional[str] = None,
    to_version: Optional[int] = None,
    from_run_id: Optional[str] = None,
    to_run_id: Optional[str] = None,
    to_model_id: Optional[str] = None,
    from_model_id: Optional[str] = None,
    note: Optional[str] = None,
) -> int:
    """Create a lineage edge between data assets."""
    with _get_connection() as conn:
        row = conn.execute(
            """
            INSERT INTO dataset_lineage_edges
                (edge_type, from_dataset_id, from_version, to_dataset_id, to_version,
                 from_run_id, to_run_id, to_model_id, from_model_id, note)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING id
            """,
            (
                edge_type,
                from_dataset_id,
                from_version,
                to_dataset_id,
                to_version,
                from_run_id,
                to_run_id,
                to_model_id,
                from_model_id,
                note,
            ),
        ).fetchone()
        return int(row["id"])


def get_dataset_versions(dataset_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Get version history for a dataset."""
    with _get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM dataset_versions
            WHERE dataset_id = ?
            ORDER BY version DESC
            LIMIT ?
            """,
            (dataset_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_schema_changes(dataset_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Get schema change history for a dataset."""
    with _get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM dataset_schema_changes
            WHERE dataset_id = ?
            ORDER BY detected_at DESC
            LIMIT ?
            """,
            (dataset_id, limit),
        ).fetchall()
    results = []
    for r in rows:
        d = dict(r)
        d["added_columns"] = json.loads(d["added_columns"])
        d["removed_columns"] = json.loads(d["removed_columns"])
        d["dtype_changes"] = json.loads(d["dtype_changes"])
        results.append(d)
    return results


def get_lineage_edges(edge_type: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    """Get lineage edges, optionally filtered by type."""
    with _get_connection() as conn:
        if edge_type:
            rows = conn.execute(
                """
                SELECT * FROM dataset_lineage_edges
                WHERE edge_type = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (edge_type, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM dataset_lineage_edges
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
    return [dict(r) for r in rows]