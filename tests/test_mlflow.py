"""Tests for mlflow_tracker module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.mlflow_tracker import _is_enabled, log_pipeline_run


def test_is_enabled_true():
    with patch("src.mlflow_tracker.load_config", return_value={"mlflow": {"enabled": True}}):
        assert _is_enabled() is True


def test_is_enabled_false():
    with patch("src.mlflow_tracker.load_config", return_value={"mlflow": {"enabled": False}}):
        assert _is_enabled() is False


def test_is_enabled_missing():
    with patch("src.mlflow_tracker.load_config", return_value={}):
        assert _is_enabled() is False


def test_log_pipeline_run_disabled():
    with patch("src.mlflow_tracker._is_enabled", return_value=False):
        with patch("src.mlflow_tracker.load_config", return_value={"mlflow": {}}):
            log_pipeline_run(run_name="test", params={}, metrics={}, artifact_path="", model=MagicMock())


def test_log_pipeline_run_enabled():
    mock_mlflow = MagicMock()
    with patch("src.mlflow_tracker._is_enabled", return_value=True):
        with patch("src.mlflow_tracker.load_config", return_value={"mlflow": {"tracking_uri": "file:///tmp/mlruns", "experiment": "test-exp"}}):
            with patch.dict("sys.modules", {"mlflow": mock_mlflow, "mlflow.xgboost": MagicMock(), "mlflow.lightgbm": MagicMock(), "mlflow.sklearn": MagicMock()}):
                mock_model = MagicMock()
                mock_model.__class__.__name__ = "RandomForestClassifier"
                log_pipeline_run(run_name="test-run", params={"n_estimators": 100}, metrics={"accuracy": 0.95}, artifact_path="", model=mock_model)


def test_log_pipeline_run_exception():
    with patch("src.mlflow_tracker._is_enabled", return_value=True):
        with patch("src.mlflow_tracker.load_config", side_effect=Exception("config error")):
            with patch("src.mlflow_tracker.LOGGER") as mock_logger:
                log_pipeline_run(run_name="test", params={}, metrics={}, artifact_path="", model=MagicMock())
                assert mock_logger.warning.called
