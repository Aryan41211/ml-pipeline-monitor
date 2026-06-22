"""Tests for Prometheus metrics module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.metrics import (
    dataset_columns,
    dataset_rows,
    dataset_validations_total,
    drift_detections_total,
    drift_features_count,
    drift_score,
    experiments_total,
    models_registered_total,
    model_promotions_total,
    pipeline_runs_total,
    pipeline_stage_duration_seconds,
    predictions_total,
    record_api_request,
    record_api_error,
    record_dataset_validation,
    record_drift_detection,
    record_experiment,
    record_model_promotion,
    record_model_registration,
    record_pipeline_run,
    record_pipeline_stage,
    record_prediction,
    system_cpu_percent,
    system_disk_percent,
    system_memory_percent,
    update_system_metrics,
)


def test_record_pipeline_run():
    record_pipeline_run(status="completed", dataset="iris", model_type="RF", duration_seconds=10.0)
    assert pipeline_runs_total.labels(status="completed", dataset="iris", model_type="RF")._value.get() > 0


def test_record_pipeline_stage():
    record_pipeline_stage(stage="training", dataset="iris", model_type="RF", duration_seconds=5.0)
    assert pipeline_stage_duration_seconds.labels(stage="training", dataset="iris", model_type="RF")._sum.get() >= 5.0


def test_record_api_request():
    record_api_request(method="GET", endpoint="/health", status_code="200", duration_seconds=0.1)
    assert True  # Should not raise


def test_record_api_error():
    record_api_error(method="POST", endpoint="/predict", error_type="http_500")
    assert True


def test_record_prediction():
    record_prediction(model_id="m1", dataset="iris", status="success", latency_seconds=0.05)
    assert predictions_total.labels(model_id="m1", dataset="iris", status="success")._value.get() > 0


def test_record_drift_detection():
    record_drift_detection(
        dataset="iris",
        severity="moderate",
        drift_detected=True,
        drift_score_value=0.15,
        features_drifted=3,
        duration_seconds=1.5,
    )
    assert drift_detections_total.labels(dataset="iris", severity="moderate", drift_detected="true")._value.get() > 0
    assert drift_score.labels(dataset="iris")._value.get() == 0.15
    assert drift_features_count.labels(dataset="iris")._value.get() == 3


def test_record_experiment():
    record_experiment(status="completed", dataset="iris", model_type="RF")
    assert experiments_total.labels(status="completed", dataset="iris", model_type="RF")._value.get() > 0


def test_record_model_registration():
    record_model_registration(dataset="iris", model_type="RF", stage="production")
    assert models_registered_total.labels(dataset="iris", model_type="RF", stage="production")._value.get() > 0


def test_record_model_promotion():
    record_model_promotion(
        dataset="iris", model_type="RF", from_stage="staging", to_stage="production", status="success"
    )
    assert model_promotions_total.labels(dataset="iris", model_type="RF", from_stage="staging", to_stage="production", status="success")._value.get() > 0


def test_record_dataset_validation():
    record_dataset_validation(dataset="iris", status="pass", rows=150, columns=5)
    assert dataset_validations_total.labels(dataset="iris", status="pass")._value.get() > 0
    assert dataset_rows.labels(dataset="iris")._value.get() == 150
    assert dataset_columns.labels(dataset="iris")._value.get() == 5


def test_update_system_metrics():
    with patch("psutil.cpu_percent", return_value=25.0):
        with patch("psutil.sensors_temperatures", return_value={}, create=True):
            with patch("psutil.virtual_memory") as mock_mem:
                mock_mem.return_value = MagicMock(total=16*1024**3, used=8*1024**3, available=8*1024**3, percent=50.0)
                with patch("psutil.disk_usage") as mock_disk:
                    mock_disk.return_value = MagicMock(total=500*1024**3, used=200*1024**3, free=300*1024**3, percent=40.0)
                    mock_proc = MagicMock()
                    mock_proc.cpu_percent.return_value = 5.0
                    mock_proc.memory_info.return_value = MagicMock(rss=1024*1024*100, vms=1024*1024*200)
                    mock_proc.num_threads.return_value = 4
                    mock_proc.status.return_value = "running"
                    with patch("psutil.Process", return_value=mock_proc):
                        with patch("psutil.pids", return_value=[1, 2, 3]):
                            update_system_metrics()
    assert system_cpu_percent._value.get() == 25.0
    assert system_memory_percent._value.get() == 50.0
    assert system_disk_percent._value.get() == 40.0
