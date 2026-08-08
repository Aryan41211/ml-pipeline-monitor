"""
Drift detection CRUD operations.

Handles saving and retrieving drift reports and reference distributions
for data drift monitoring.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np

from ml_pipeline_monitor.database.schema import _get_connection


def save_drift_report(
    report_id: str,
    dataset: str,
    reference_size: int,
    current_size: int,
    drift_detected: bool,
    drift_score: float,
    features_drifted: int,
    feature_results: Dict[str, Any],
) -> None:
    """Save a drift detection report."""
    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO drift_reports
                (report_id, dataset, reference_size, current_size,
                 drift_detected, drift_score, features_drifted, feature_results)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(report_id) DO UPDATE SET
                dataset=excluded.dataset,
                reference_size=excluded.reference_size,
                current_size=excluded.current_size,
                drift_detected=excluded.drift_detected,
                drift_score=excluded.drift_score,
                features_drifted=excluded.features_drifted,
                feature_results=excluded.feature_results
            """,
            (
                report_id,
                dataset,
                reference_size,
                current_size,
                int(drift_detected),
                drift_score,
                features_drifted,
                json.dumps(feature_results),
            ),
        )


def get_drift_reports(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve recent drift reports, ordered by creation date."""
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM drift_reports ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def save_drift_reference(dataset: str, feature_names: List[str], reference_data: np.ndarray) -> None:
    """Store reference distribution for a dataset."""
    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO drift_references (dataset, feature_names, reference_data)
            VALUES (?, ?, ?)
            ON CONFLICT(dataset) DO UPDATE SET
                feature_names=excluded.feature_names,
                reference_data=excluded.reference_data
            """,
            (dataset, json.dumps(feature_names), json.dumps(reference_data.tolist())),
        )


def get_drift_reference(dataset: str) -> Optional[Dict[str, Any]]:
    """Retrieve stored reference distribution for a dataset."""
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM drift_references WHERE dataset = ?",
            (dataset,),
        ).fetchone()
    if row:
        d = dict(row)
        d["feature_names"] = json.loads(d["feature_names"])
        d["reference_data"] = np.array(json.loads(d["reference_data"]))
        return d
    return None