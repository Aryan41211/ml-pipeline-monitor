"""Governance service: alert history and dataset lineage for the UI.

These records are written by the alerting path (`core.alerts`) and the pipeline
service. This module is the read side, so pages never touch the database layer
directly.
"""

from __future__ import annotations

import json
from typing import Any

from ml_pipeline_monitor.database import (
    get_dataset_versions,
    get_lineage_edges,
    get_schema_changes,
    list_alert_events,
)


def _coerce_metadata(raw: Any) -> dict[str, Any]:
    """Parse a metadata_json column into a dict, tolerating bad rows."""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def list_alerts(limit: int = 100) -> list[dict[str, Any]]:
    """Return recent alert events, newest first."""
    if limit <= 0 or limit > 1000:
        raise ValueError("limit must be between 1 and 1000")

    rows = list_alert_events(limit=limit)
    for row in rows:
        row["metadata"] = _coerce_metadata(row.get("metadata_json"))
    return rows


def alert_summary(alerts: list[dict[str, Any]]) -> dict[str, int]:
    """Count alerts by severity."""
    summary = {"critical": 0, "warning": 0, "info": 0}
    for alert in alerts:
        severity = str(alert.get("severity", "info")).lower()
        summary[severity] = summary.get(severity, 0) + 1
    return summary


def list_lineage_edges(limit: int = 200) -> list[dict[str, Any]]:
    """Return dataset -> model lineage edges, newest first."""
    if limit <= 0 or limit > 1000:
        raise ValueError("limit must be between 1 and 1000")
    return get_lineage_edges(limit=limit)


def list_dataset_versions(dataset_id: str, limit: int = 50) -> list[dict[str, Any]]:
    """Return the recorded version history for one dataset."""
    if not dataset_id or not dataset_id.strip():
        raise ValueError("dataset_id is required")
    if limit <= 0 or limit > 500:
        raise ValueError("limit must be between 1 and 500")
    return get_dataset_versions(dataset_id.strip(), limit=limit)


def list_schema_changes(dataset_id: str, limit: int = 50) -> list[dict[str, Any]]:
    """Return recorded schema changes for one dataset."""
    if not dataset_id or not dataset_id.strip():
        raise ValueError("dataset_id is required")
    if limit <= 0 or limit > 500:
        raise ValueError("limit must be between 1 and 500")
    return get_schema_changes(dataset_id.strip(), limit=limit)
