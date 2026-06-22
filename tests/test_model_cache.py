"""Test suite for model cache module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.model_cache import (
    clear_cache,
    get_latest_production_model,
    get_model,
    invalidate,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_cache()
    yield
    clear_cache()


def _fake_record(run_id: str = "test-run-123"):
    return {
        "run_id": run_id,
        "model_id": "model-001",
        "artifact_path": f"artifacts/models/{run_id}_model.joblib",
        "stage": "production",
    }


@patch("src.model_cache._resolve_artifact")
@patch("joblib.load")
def test_get_model_cache_miss_and_load(mock_load, mock_resolve):
    mock_resolve.return_value = ("path/to/model.joblib", "path/to/scaler.joblib")
    mock_load.side_effect = lambda p: {"model": p, "scaler": p + "_scaler"}

    result = get_model("abc123")
    assert result is not None
    model, scaler, path = result
    assert isinstance(path, str)
    assert result is get_model("abc123")  # cache hit


@patch("src.model_cache._resolve_artifact")
def test_get_model_returns_none_when_missing(mock_resolve):
    mock_resolve.return_value = (None, None)
    assert get_model("missing") is None


@patch("src.model_cache._resolve_artifact")
@patch("joblib.load")
def test_invalidate_removes_entry(mock_load, mock_resolve):
    mock_resolve.return_value = ("path.joblib", None)
    mock_load.return_value = "model_obj"
    assert get_model("run-x") is not None
    invalidate("run-x")
    assert get_model("run-x") is not None  # reload after invalidation


@patch("src.database.get_latest_production_model")
@patch("src.model_cache.get_model")
def test_get_latest_production_model(mock_get_model, mock_db_get):
    mock_db_get.return_value = _fake_record()
    mock_get_model.return_value = ("model", "scaler", "path")
    result = get_latest_production_model()
    assert result is not None
