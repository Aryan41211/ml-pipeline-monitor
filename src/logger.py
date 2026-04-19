"""Application logging utilities with structured file output."""

from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict

from src.config_loader import ROOT_DIR, load_config


class JsonFormatter(logging.Formatter):
    """Simple JSON formatter for structured logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
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
