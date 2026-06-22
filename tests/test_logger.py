"""Tests for structured logging module."""

from __future__ import annotations

import logging
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from src.logger import (
    LogContext,
    LogLevel,
    clear_actor_context,
    clear_correlation_id,
    clear_operation_context,
    clear_request_id,
    clear_service_context,
    get_actor_context,
    get_app_logger,
    get_correlation_id,
    get_operation_context,
    get_request_id,
    get_service_context,
    log_dataset_upload,
    log_dataset_validation,
    log_drift_detection,
    log_experiment_creation,
    log_governance_action,
    log_model_promotion,
    log_model_registration,
    log_pipeline_run,
    log_prediction_request,
    log_user_action,
    set_actor_context,
    set_correlation_id,
    set_operation_context,
    set_request_id,
    set_service_context,
)


def test_get_app_logger_returns_logger():
    logger = get_app_logger("test_module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_module"


def test_get_set_clear_correlation_id():
    clear_correlation_id()
    assert get_correlation_id() is not None
    cid = "test-cid-123"
    set_correlation_id(cid)
    assert get_correlation_id() == cid
    clear_correlation_id()
    assert get_correlation_id() is not None


def test_get_set_clear_request_id():
    clear_request_id()
    assert get_request_id() is not None
    rid = "test-rid-456"
    set_request_id(rid)
    assert get_request_id() == rid
    clear_request_id()


def test_get_set_clear_operation_context():
    clear_operation_context()
    assert get_operation_context() is None
    set_operation_context("pipeline_run")
    assert get_operation_context() == "pipeline_run"
    clear_operation_context()
    assert get_operation_context() is None


def test_get_set_clear_actor_context():
    clear_actor_context()
    assert get_actor_context() is None
    set_actor_context("admin")
    assert get_actor_context() == "admin"
    clear_actor_context()


def test_get_set_clear_service_context():
    clear_service_context()
    assert get_service_context() is None
    set_service_context("streamlit")
    assert get_service_context() == "streamlit"
    clear_service_context()


def test_log_context_manager():
    clear_correlation_id()
    clear_request_id()
    with LogContext(operation="test_op", actor="tester", service="test_service") as ctx:
        assert get_operation_context() == "test_op"
        assert get_actor_context() == "tester"
        assert get_service_context() == "test_service"
    assert get_operation_context() is None
    assert get_actor_context() is None
    assert get_service_context() is None


def test_log_user_action(caplog):
    logger = get_app_logger("user_actions")
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    try:
        log_user_action("click", page="dashboard", metadata={"button": "sync"})
    finally:
        logger.removeHandler(handler)
    output = stream.getvalue()
    assert "click" in output or "dashboard" in output


def test_log_pipeline_run():
    with patch("src.logger.get_app_logger") as mock_get:
        mock_logger = MagicMock()
        mock_get.return_value = mock_logger
        log_pipeline_run(
            run_id="run-1",
            dataset="iris",
            model_type="Random Forest",
            status="completed",
            duration_seconds=12.5,
            metrics={"accuracy": 0.95},
        )
    assert mock_logger.log.called


def test_log_experiment_creation():
    with patch("src.logger.get_app_logger") as mock_get:
        mock_logger = MagicMock()
        mock_get.return_value = mock_logger
        log_experiment_creation(
            experiment_id="exp-1",
            exp_name="test",
            dataset="iris",
            model_type="RF",
            status="success",
        )
    assert mock_logger.log.called


def test_log_model_registration():
    with patch("src.logger.get_app_logger") as mock_get:
        mock_logger = MagicMock()
        mock_get.return_value = mock_logger
        log_model_registration(
            model_id="m1",
            model_name="RF-v1",
            dataset="iris",
            version=1,
            stage="production",
            status="success",
        )
    assert mock_logger.log.called


def test_log_drift_detection():
    with patch("src.logger.get_app_logger") as mock_get:
        mock_logger = MagicMock()
        mock_get.return_value = mock_logger
        log_drift_detection(
            dataset="iris",
            reference_window="w1",
            current_window="w2",
            status="success",
            drift_detected=True,
            drift_score=0.3,
        )
    assert mock_logger.log.called


def test_log_prediction_request():
    with patch("src.logger.get_app_logger") as mock_get:
        mock_logger = MagicMock()
        mock_get.return_value = mock_logger
        log_prediction_request(
            model_id="m1",
            dataset="iris",
            status="success",
            num_predictions=10,
        )
    assert mock_logger.log.called


def test_log_model_promotion():
    with patch("src.logger.get_app_logger") as mock_get:
        mock_logger = MagicMock()
        mock_get.return_value = mock_logger
        log_model_promotion(
            model_id="m1",
            model_name="RF-v1",
            dataset="iris",
            from_stage="staging",
            to_stage="production",
            status="success",
        )
    assert mock_logger.log.called


def test_log_governance_action():
    with patch("src.logger.get_app_logger") as mock_get:
        mock_logger = MagicMock()
        mock_get.return_value = mock_logger
        log_governance_action(
            action="approve",
            entity_type="model",
            entity_id="m1",
            status="success",
        )
    assert mock_logger.log.called


def test_log_dataset_validation():
    with patch("src.logger.get_app_logger") as mock_get:
        mock_logger = MagicMock()
        mock_get.return_value = mock_logger
        log_dataset_validation(
            dataset_name="iris",
            status="success",
            rows=150,
            columns=5,
        )
    assert mock_logger.log.called
