"""
Prediction history CRUD operations.

Handles prediction request logging, individual prediction storage,
and prediction history retrieval for the inference API.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ml_pipeline_monitor.database.schema import _get_connection


def save_prediction_request(
    *,
    request_id: str,
    correlation_id: str | None,
    model_id: str,
    dataset: str | None,
    input_type: str,
    input_hash: str | None,
    num_predictions: int,
    status: str,
    duration_ms: float | None,
    error: str | None,
) -> None:
    """Persist prediction request metadata."""
    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO prediction_requests
                (request_id, correlation_id, model_id, dataset, input_type, input_hash,
                 num_predictions, status, duration_ms, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(request_id) DO UPDATE SET
                correlation_id=excluded.correlation_id,
                model_id=excluded.model_id,
                dataset=excluded.dataset,
                input_type=excluded.input_type,
                input_hash=excluded.input_hash,
                num_predictions=excluded.num_predictions,
                status=excluded.status,
                duration_ms=excluded.duration_ms,
                error=excluded.error
            """,
            (
                request_id,
                correlation_id,
                model_id,
                dataset,
                input_type,
                input_hash,
                int(num_predictions),
                status,
                duration_ms,
                error,
            ),
        )


def save_predictions_for_request(
    *,
    request_id: str,
    predictions: list[Any],
    probabilities: list[Any] | None = None,
) -> None:
    """Persist per-row predictions for a request."""
    probabilities = probabilities or [None] * len(predictions)
    with _get_connection() as conn:
        for idx, pred in enumerate(predictions):
            conn.execute(
                """
                INSERT INTO predictions (request_id, row_index, prediction, probability)
                VALUES (?, ?, ?, ?)
                """,
                (
                    request_id,
                    int(idx),
                    str(pred),
                    None if probabilities[idx] is None else str(probabilities[idx]),
                ),
            )


def get_prediction_history(limit: int = 50) -> list[dict[str, Any]]:
    """Fetch prediction request history (newest first)."""
    if limit <= 0 or limit > 1000:
        limit = 50
    with _get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM prediction_requests
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_prediction_history_by_request_id(request_id: str) -> dict[str, Any] | None:
    """Fetch a single prediction request with its individual predictions."""
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM prediction_requests WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        if not row:
            return None
        req = dict(row)

        preds = conn.execute(
            """
            SELECT row_index, prediction, probability
            FROM predictions
            WHERE request_id = ?
            ORDER BY row_index ASC
            """,
            (request_id,),
        ).fetchall()
        req["predictions"] = [
            {
                "row_index": int(r["row_index"]),
                "prediction": r["prediction"],
                "probability": r["probability"],
            }
            for r in preds
        ]
    return req