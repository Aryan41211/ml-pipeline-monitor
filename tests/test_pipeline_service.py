"""Tests for pipeline_service run_pipeline_and_persist."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from services.pipeline_service import (
    _validate_pipeline_inputs,
    get_pipeline_defaults,
    run_pipeline_and_persist,
)


def _fake_pipeline_result():
    return SimpleNamespace(
        run_id="run-123",
        model=MagicMock(),
        scaler=MagicMock(),
        metrics={"accuracy": 0.95},
        confusion_matrix=[[1, 0], [0, 1]],
        confusion_mat=[[1, 0], [0, 1]],
        feature_importances={"f1": 0.5},
        duration=2.5,
        predictions=[0, 1],
    )


def test_run_pipeline_and_persist_success():
    with patch("services.pipeline_service.load_config", return_value={"pipeline": {"n_jobs": 1}}):
        with patch("services.pipeline_service.make_feature_key", return_value="key-123"):
            with patch("services.pipeline_service.load_cached_splits", return_value=None):
                with patch("services.pipeline_service.load_dataset", return_value={"X_train": [[1]], "X_test": [[1]], "y_train": [0], "y_test": [0]}):
                    with patch("services.pipeline_service.save_cached_splits"):
                        with patch("services.pipeline_service.MLPipeline") as mock_pipe:
                            mock_pipe.return_value.run.return_value = _fake_pipeline_result()
                            with patch("services.pipeline_service.save_experiment"):
                                with patch("services.pipeline_service.save_model"):
                                    with patch("services.pipeline_service.record_pipeline_run"):
                                        with patch("services.pipeline_service.emit_console_alert"):
                                            with patch("services.pipeline_service.emit_email_alert"):
                                                with patch("services.pipeline_service._persist_artifacts"):
                                                    result = run_pipeline_and_persist(
                                                        dataset_label="iris",
                                                        dataset_key="iris",
                                                        model_type="Random Forest",
                                                        task="classification",
                                                        params={"n_estimators": 10},
                                                        test_size=0.2,
                                                        cv_folds=5,
                                                        random_state=42,
                                                    )
    assert result["result"].run_id == "run-123"
    assert result["dataset"] == {"X_train": [[1]], "X_test": [[1]], "y_train": [0], "y_test": [0]}


def test_run_pipeline_and_persist_failure():
    with patch("services.pipeline_service.load_config", return_value={"pipeline": {"n_jobs": 1}}):
        with patch("services.pipeline_service.make_feature_key", return_value="key-123"):
            with patch("services.pipeline_service.load_cached_splits", return_value=None):
                with patch("services.pipeline_service.load_dataset", return_value={"X_train": [[1]], "X_test": [[1]], "y_train": [0], "y_test": [0]}):
                    with patch("services.pipeline_service.save_cached_splits"):
                        with patch("services.pipeline_service.MLPipeline") as mock_pipe:
                            mock_pipe.return_value.run.side_effect = RuntimeError("boom")
                            with patch("services.pipeline_service.record_pipeline_run"):
                                with patch("services.pipeline_service.emit_console_alert"):
                                    with patch("services.pipeline_service.emit_email_alert"):
                                        with patch("services.pipeline_service._persist_artifacts"):
                                            with pytest.raises(RuntimeError, match="boom"):
                                                run_pipeline_and_persist(
                                                    dataset_label="iris",
                                                    dataset_key="iris",
                                                    model_type="Random Forest",
                                                    task="classification",
                                                    params={},
                                                    test_size=0.2,
                                                    cv_folds=5,
                                                    random_state=42,
                                                )


def test_get_pipeline_defaults():
    with patch("services.pipeline_service.load_config", return_value={"pipeline": {"random_seed": 42, "test_size": 0.2, "cv_folds": 5, "n_jobs": -1}}):
        defaults = get_pipeline_defaults()
    assert defaults["random_seed"] == 42
    assert defaults["test_size"] == 0.2
    assert defaults["cv_folds"] == 5
    assert defaults["n_jobs"] == -1


def test_validate_pipeline_inputs_bad_random_state_type():
    with pytest.raises((ValueError, TypeError)):
        _validate_pipeline_inputs("iris", "iris", "Random Forest", "classification", 0.2, 5, None)


def test_validate_pipeline_inputs_bad_dataset_label():
    with pytest.raises(ValueError, match="dataset_label is required"):
        _validate_pipeline_inputs("", "iris", "Random Forest", "classification", 0.2, 5, 42)
