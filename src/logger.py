"""Application logging utilities with structured file output."""

from __future__ import annotations

import contextvars
import json
import logging
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

from src.config_loader import ROOT_DIR, load_config

# Context variable for correlation ID propagation across async boundaries
correlation_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("correlation_id", default=None)


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


class JsonFormatter(logging.Formatter):
    """Simple JSON formatter for structured logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", None) or get_correlation_id(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # Include any extra fields from the log record
        for key, value in record.__dict__.items():
            if key not in {"name", "msg", "args", "created", "filename", "funcName", "levelname", "levelno", "lineno", "module", "msecs", "message", "name", "pathname", "process", "processName", "relativeCreated", "thread", "threadName", "exc_info", "exc_text", "stack_info", "correlation_id"}:
                payload[key] = value
        return json.dumps(payload, ensure_ascii=True)


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

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(JsonFormatter())

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
