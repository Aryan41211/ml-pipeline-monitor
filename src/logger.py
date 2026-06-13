"""Application logging utilities with structured file output."""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

from src.config_loader import ROOT_DIR, load_config

# Context variable for correlation ID propagation across async boundaries
correlation_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("correlation_id", default=None)

# Context variable for operation context (e.g., "pipeline_run", "drift_detection")
operation_context_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("operation_context", default=None)

# Context variable for user/actor context
actor_context_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("actor_context", default=None)


class LogLevel:
    """Standard log levels with numeric values."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class ErrorCategory:
    """Error categorization for structured error handling."""
    VALIDATION = "validation_error"
    CONFIGURATION = "configuration_error"
    DATABASE = "database_error"
    MODEL = "model_error"
    PIPELINE = "pipeline_error"
    DRIFT = "drift_error"
    AUTHENTICATION = "authentication_error"
    AUTHORIZATION = "authorization_error"
    EXTERNAL_SERVICE = "external_service_error"
    INTERNAL = "internal_error"
    UNKNOWN = "unknown_error"


def get_correlation_id() -> str:
    """Get current correlation ID or generate a new one."""
    cid = correlation_id_var.get()
    if cid is None:
        cid = str(uuid.uuid4())[:8]
        correlation_id_var.set(cid)
    return cid


def set_correlation_id(cid: str) -> None:
    """Set correlation ID for current context."""
    correlation_id_var.set(cid)


def clear_correlation_id() -> None:
    """Clear correlation ID from current context."""
    correlation_id_var.set(None)


def get_operation_context() -> Optional[str]:
    """Get current operation context."""
    return operation_context_var.get()


def set_operation_context(ctx: str) -> None:
    """Set operation context for current context."""
    operation_context_var.set(ctx)


def clear_operation_context() -> None:
    """Clear operation context."""
    operation_context_var.set(None)


def get_actor_context() -> Optional[str]:
    """Get current actor/user context."""
    return actor_context_var.get()


def set_actor_context(actor: str) -> None:
    """Set actor/user context for current context."""
    actor_context_var.set(actor)


def clear_actor_context() -> None:
    """Clear actor context."""
    actor_context_var.set(None)


class JsonFormatter(logging.Formatter):
    """Enhanced JSON formatter for structured logs with error categorization."""

    def format(self, record: logging.LogRecord) -> str:
        # Use datetime for microsecond precision since time.strftime doesn't support %f
        from datetime import datetime
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        payload: Dict[str, Any] = {
            "timestamp": timestamp,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", None) or get_correlation_id(),
            "operation": getattr(record, "operation", None) or get_operation_context(),
            "actor": getattr(record, "actor", None) or get_actor_context(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
            payload["error_category"] = getattr(record, "error_category", None) or ErrorCategory.INTERNAL
        # Include any extra fields from the log record
        # Skip reserved LogRecord attributes to avoid KeyError
        reserved = {
            "name", "msg", "args", "created", "filename", "funcName", "levelname", "levelno", "lineno",
            "module", "msecs", "message", "pathname", "process", "processName", "relativeCreated",
            "thread", "threadName", "exc_info", "exc_text", "stack_info", "correlation_id",
            "operation", "actor", "error_category"
        }
        for key, value in record.__dict__.items():
            if key not in reserved:
                payload[key] = value
        return json.dumps(payload, ensure_ascii=True)


class ConsoleFormatter(logging.Formatter):
    """Human-readable console formatter with color support."""

    COLORS = {
        "DEBUG": "\033[36m",      # Cyan
        "INFO": "\033[32m",       # Green
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "CRITICAL": "\033[35m",   # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        reset = self.RESET
        correlation_id = getattr(record, "correlation_id", None) or get_correlation_id()
        operation = getattr(record, "operation", None) or get_operation_context()
        actor = getattr(record, "actor", None) or get_actor_context()

        base = f"{color}{self.formatTime(record, '%H:%M:%S')} | {record.levelname:8s} | {record.name}{reset}"
        context_parts = []
        if correlation_id:
            context_parts.append(f"cid={correlation_id}")
        if operation:
            context_parts.append(f"op={operation}")
        if actor:
            context_parts.append(f"actor={actor}")
        context_str = f" [{', '.join(context_parts)}]" if context_parts else ""

        message = record.getMessage()
        if record.exc_info:
            message += f"\n{self.formatException(record.exc_info)}"

        return f"{base}{context_str} | {message}"


def _resolve_log_path() -> Path:
    cfg = load_config().get("logging", {})
    relative = cfg.get("file", "logs/app.log")
    path = (ROOT_DIR / relative).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_app_logger(name: str = "ml_monitor") -> logging.Logger:
    """Return configured logger instance."""
    cfg = load_config().get("logging", {})
    level_name = str(cfg.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)

    file_handler = RotatingFileHandler(
        _resolve_log_path(),
        maxBytes=int(cfg.get("max_bytes", 5_000_000)),
        backupCount=int(cfg.get("backup_count", 3)),
        encoding="utf-8",
    )
    file_handler.setFormatter(JsonFormatter())

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ConsoleFormatter())

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False
    return logger


def log_user_action(action: str, *, page: str, metadata: Dict[str, Any] | None = None) -> None:
    """Emit structured user action logs for UI observability."""
    logger = get_app_logger("user_actions")
    payload = {
        "event": "user_action",
        "action": str(action),
        "page": str(page),
        "metadata": metadata or {},
    }
    logger.info(json.dumps(payload, ensure_ascii=True))


# ============================================================
# Structured logging helpers for major operations
# ============================================================

def log_dataset_upload(
    dataset_name: str,
    *,
    status: str,
    rows: int,
    columns: int,
    file_size_bytes: Optional[int] = None,
    error: Optional[str] = None,
    correlation_id: Optional[str] = None,
    actor: Optional[str] = None,
) -> None:
    """Log dataset upload operation."""
    logger = get_app_logger("dataset_upload")
    extra = {
        "event": "dataset_upload",
        "dataset": dataset_name,
        "status": status,
        "rows": rows,
        "columns": columns,
        "file_size_bytes": file_size_bytes,
    }
    if error:
        extra["error"] = error
        extra["error_category"] = ErrorCategory.VALIDATION if status == "failed" else None

    _log_with_context(
        logger,
        level=logging.INFO if status == "success" else logging.ERROR,
        message=f"Dataset upload {status}: {dataset_name} ({rows} rows, {columns} cols)",
        extra=extra,
        correlation_id=correlation_id,
        operation="dataset_upload",
        actor=actor,
    )


def log_pipeline_run(
    run_id: str,
    dataset: str,
    model_type: str,
    *,
    status: str,
    duration_seconds: Optional[float] = None,
    metrics: Optional[Dict[str, float]] = None,
    error: Optional[str] = None,
    stage: Optional[str] = None,
    correlation_id: Optional[str] = None,
    actor: Optional[str] = None,
) -> None:
    """Log pipeline run operation."""
    logger = get_app_logger("pipeline_run")
    extra = {
        "event": "pipeline_run",
        "run_id": run_id,
        "dataset": dataset,
        "model_type": model_type,
        "status": status,
        "duration_seconds": duration_seconds,
        "metrics": metrics or {},
    }
    if error:
        extra["error"] = error
        extra["error_category"] = ErrorCategory.PIPELINE
    if stage:
        extra["stage"] = stage

    _log_with_context(
        logger,
        level=logging.INFO if status in ("started", "completed", "stage_completed") else logging.ERROR,
        message=f"Pipeline {status}: run_id={run_id} dataset={dataset} model={model_type}" + (f" stage={stage}" if stage else ""),
        extra=extra,
        correlation_id=correlation_id,
        operation="pipeline_run",
        actor=actor,
    )


def log_experiment_creation(
    experiment_id: str,
    exp_name: str,
    dataset: str,
    model_type: str,
    *,
    status: str,
    metrics: Optional[Dict[str, float]] = None,
    duration_seconds: Optional[float] = None,
    error: Optional[str] = None,
    correlation_id: Optional[str] = None,
    actor: Optional[str] = None,
) -> None:
    """Log experiment creation operation."""
    logger = get_app_logger("experiment")
    extra = {
        "event": "experiment_creation",
        "experiment_id": experiment_id,
        "exp_name": exp_name,
        "dataset": dataset,
        "model_type": model_type,
        "status": status,
        "metrics": metrics or {},
        "duration_seconds": duration_seconds,
    }
    if error:
        extra["error"] = error
        extra["error_category"] = ErrorCategory.DATABASE

    _log_with_context(
        logger,
        level=logging.INFO if status == "success" else logging.ERROR,
        message=f"Experiment {status}: {exp_name} (id={experiment_id}) dataset={dataset} model={model_type}",
        extra=extra,
        correlation_id=correlation_id,
        operation="experiment_creation",
        actor=actor,
    )


def log_model_registration(
    model_id: str,
    model_name: str,
    dataset: str,
    version: int,
    stage: str,
    *,
    status: str,
    metrics: Optional[Dict[str, float]] = None,
    error: Optional[str] = None,
    correlation_id: Optional[str] = None,
    actor: Optional[str] = None,
) -> None:
    """Log model registration operation."""
    logger = get_app_logger("model_registry")
    extra = {
        "event": "model_registration",
        "model_id": model_id,
        "model_name": model_name,
        "dataset": dataset,
        "version": version,
        "stage": stage,
        "status": status,
        "metrics": metrics or {},
    }
    if error:
        extra["error"] = error
        extra["error_category"] = ErrorCategory.MODEL

    _log_with_context(
        logger,
        level=logging.INFO if status == "success" else logging.ERROR,
        message=f"Model registration {status}: {model_name} v{version} ({stage}) dataset={dataset}",
        extra=extra,
        correlation_id=correlation_id,
        operation="model_registration",
        actor=actor,
    )


def log_drift_detection(
    dataset: str,
    reference_window: str,
    current_window: str,
    *,
    status: str,
    drift_detected: bool,
    drift_score: Optional[float] = None,
    features_drifted: Optional[int] = None,
    error: Optional[str] = None,
    correlation_id: Optional[str] = None,
    actor: Optional[str] = None,
) -> None:
    """Log drift detection operation."""
    logger = get_app_logger("drift_detection")
    extra = {
        "event": "drift_detection",
        "dataset": dataset,
        "reference_window": reference_window,
        "current_window": current_window,
        "status": status,
        "drift_detected": drift_detected,
        "drift_score": drift_score,
        "features_drifted": features_drifted,
    }
    if error:
        extra["error"] = error
        extra["error_category"] = ErrorCategory.DRIFT

    _log_with_context(
        logger,
        level=logging.INFO if status == "success" else logging.ERROR,
        message=f"Drift detection {status}: dataset={dataset} drift={drift_detected} score={drift_score}",
        extra=extra,
        correlation_id=correlation_id,
        operation="drift_detection",
        actor=actor,
    )


def log_prediction_request(
    model_id: str,
    dataset: str,
    *,
    status: str,
    num_predictions: int,
    duration_ms: Optional[float] = None,
    error: Optional[str] = None,
    correlation_id: Optional[str] = None,
    actor: Optional[str] = None,
) -> None:
    """Log prediction request operation."""
    logger = get_app_logger("prediction")
    extra = {
        "event": "prediction_request",
        "model_id": model_id,
        "dataset": dataset,
        "status": status,
        "num_predictions": num_predictions,
        "duration_ms": duration_ms,
    }
    if error:
        extra["error"] = error
        extra["error_category"] = ErrorCategory.MODEL

    _log_with_context(
        logger,
        level=logging.INFO if status == "success" else logging.ERROR,
        message=f"Prediction {status}: model={model_id} dataset={dataset} count={num_predictions} duration_ms={duration_ms}",
        extra=extra,
        correlation_id=correlation_id,
        operation="prediction",
        actor=actor,
    )


def _log_with_context(
    logger: logging.Logger,
    level: int,
    message: str,
    extra: Dict[str, Any],
    correlation_id: Optional[str] = None,
    operation: Optional[str] = None,
    actor: Optional[str] = None,
) -> None:
    """Internal helper to log with context variables set."""
    # Temporarily set context vars for this log call
    cid_token = correlation_id_var.set(correlation_id or get_correlation_id())
    op_token = operation_context_var.set(operation) if operation else None
    actor_token = actor_context_var.set(actor) if actor else None

    try:
        # Add context to extra for formatter
        log_extra = dict(extra)
        log_extra["correlation_id"] = correlation_id or get_correlation_id()
        log_extra["operation"] = operation
        log_extra["actor"] = actor
        logger.log(level, message, extra=log_extra)
    finally:
        correlation_id_var.reset(cid_token)
        if op_token:
            operation_context_var.reset(op_token)
        if actor_token:
            actor_context_var.reset(actor_token)


class LogContext:
    """Context manager for setting correlation ID, operation, and actor context."""

    def __init__(
        self,
        correlation_id: Optional[str] = None,
        operation: Optional[str] = None,
        actor: Optional[str] = None,
    ):
        self.correlation_id = correlation_id or str(uuid.uuid4())[:8]
        self.operation = operation
        self.actor = actor
        self._cid_token = None
        self._op_token = None
        self._actor_token = None

    def __enter__(self) -> "LogContext":
        self._cid_token = correlation_id_var.set(self.correlation_id)
        if self.operation:
            self._op_token = operation_context_var.set(self.operation)
        if self.actor:
            self._actor_token = actor_context_var.set(self.actor)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        correlation_id_var.reset(self._cid_token)
        if self._op_token:
            operation_context_var.reset(self._op_token)
        if self._actor_token:
            actor_context_var.reset(self._actor_token)

    def get_logger(self, name: str = "ml_monitor") -> logging.Logger:
        """Get logger within this context."""
        return get_app_logger(name)


def get_error_category(error: Exception) -> str:
    """Categorize an exception into an error category."""
    error_type = type(error).__name__
    error_msg = str(error).lower()

    if isinstance(error, (ValueError, KeyError, TypeError)):
        return ErrorCategory.VALIDATION
    if isinstance(error, (FileNotFoundError, PermissionError)):
        return ErrorCategory.CONFIGURATION
    if "database" in error_msg or "sql" in error_msg or "connection" in error_msg:
        return ErrorCategory.DATABASE
    if "model" in error_msg or "artifact" in error_msg or "joblib" in error_msg:
        return ErrorCategory.MODEL
    if "pipeline" in error_msg or "stage" in error_msg:
        return ErrorCategory.PIPELINE
    if "drift" in error_msg or "psi" in error_msg or "kolmogorov" in error_msg:
        return ErrorCategory.DRIFT
    if "auth" in error_msg or "login" in error_msg or "credential" in error_msg:
        return ErrorCategory.AUTHENTICATION
    if "unauthorized" in error_msg or "forbidden" in error_msg or "permission" in error_msg:
        return ErrorCategory.AUTHORIZATION
    if "mlflow" in error_msg or "external" in error_msg:
        return ErrorCategory.EXTERNAL_SERVICE

    return ErrorCategory.INTERNAL