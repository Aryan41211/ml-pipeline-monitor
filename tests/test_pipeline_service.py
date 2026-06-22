"""Tests for pipeline service layer."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from services.pipeline_service import (
    _validate_pipeline_inputs,
    get_dataset_options,
    get_dataset_preview,
    get_pipeline_defaults,
    get_task_and_model_options,
    list_experiments,
)


def test_get_pipeline_defaults():
    with patch("services.pipeline_service.load_config", return_value={"pipeline": {"random_seed": 42, "test_size": 0.2, "cv_folds": 5, "n_jobs": -1}}):
        defaults = get_pipeline_defaults()
    assert defaults["random_seed"] == 42
    assert defaults["test_size"] == 0.2
    assert defaults["cv_folds"] == 5


def test_get_dataset_options():
    with patch("services.pipeline_service.DATASET_OPTIONS", {"Iris Species": "iris"}):
        opts = get_dataset_options()
    assert "Iris Species" in opts
    assert opts["Iris Species"] == "iris"


def test_get_task_and_model_options_classification():
    cfg = {
        "datasets": {
            "iris": {"task": "classification"},
        }
    }
    with patch("services.pipeline_service.load_config", return_value=cfg):
        result = get_task_and_model_options("iris")
    assert result["task"] == "classification"
    assert "Random Forest" in result["model_options"]


def test_get_task_and_model_options_regression():
    cfg = {
        "datasets": {
            "synthetic_reg": {"task": "regression"},
        }
    }
    with patch("services.pipeline_service.load_config", return_value=cfg):
        result = get_task_and_model_options("synthetic_reg")
    assert result["task"] == "regression"
    assert "Random Forest" in result["model_options"]


def test_get_task_and_model_options_unknown_dataset():
    cfg = {"datasets": {}}
    with patch("services.pipeline_service.load_config", return_value=cfg):
        result = get_task_and_model_options("unknown")
    assert result["task"] == "regression"


def test_get_dataset_preview():
    with patch("services.pipeline_service.load_dataset", return_value={"X_train": "x_train", "X_test": "x_test", "y_train": "y_train", "y_test": "y_test", "feature_names": ["f1", "f2"]}):
        with patch("services.pipeline_service.get_feature_statistics", return_value={"mean": [1.0, 2.0]}):
            result = get_dataset_preview("iris", test_size=0.2, random_state=42)
    assert "dataset" in result
    assert "feature_stats" in result


def test_validate_pipeline_inputs_success():
    _validate_pipeline_inputs("iris", "iris", "Random Forest", "classification", 0.2, 5, 42)


def test_validate_pipeline_inputs_missing_label():
    with pytest.raises(ValueError, match="dataset_label is required"):
        _validate_pipeline_inputs("", "iris", "Random Forest", "classification", 0.2, 5, 42)


def test_validate_pipeline_inputs_invalid_model():
    with pytest.raises(ValueError, match="Unsupported model_type"):
        _validate_pipeline_inputs("iris", "iris", "", "classification", 0.2, 5, 42)


def test_validate_pipeline_inputs_bad_test_size():
    with pytest.raises(ValueError, match="test_size must be between"):
        _validate_pipeline_inputs("iris", "iris", "Random Forest", "classification", -0.1, 5, 42)


def test_list_experiments():
    with patch("services.pipeline_service.get_experiments", return_value=[{"run_id": "abc"}]) as m:
        result = list_experiments(limit=10)
    m.assert_called_once_with(limit=10)
    assert len(result) == 1
