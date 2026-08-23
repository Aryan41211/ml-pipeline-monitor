"""Alert helpers for console, file, and persisted alert-event surfaces."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ml_pipeline_monitor.core.config_loader import ROOT_DIR, load_config
from ml_pipeline_monitor.core.logger import get_app_logger

LOGGER = get_app_logger("alerts")

_SEVERITY_ALIASES = {
    "critical": "critical",
    "error": "critical",
    "warning": "warning",
    "warn": "warning",
}


def _normalize_severity(severity: str) -> str:
    """Map the severity spellings used across the codebase onto a fixed set."""
    return _SEVERITY_ALIASES.get(str(severity).strip().lower(), str(severity).strip().lower() or "info")


def _persist_alert_event(
    *,
    severity: str,
    alert_type: str,
    message: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Record the alert in the ``alert_events`` table for the governance view.

    Imported lazily and failure-tolerant: alerting is a notification path and
    must never take down the operation that raised the alert.
    """
    try:
        from ml_pipeline_monitor.database import save_alert_event

        save_alert_event(
            workspace_id=None,
            alert_type=alert_type,
            severity=severity,
            message=message,
            metadata=metadata or {},
        )
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.debug("Could not persist alert event: %s", exc)


def emit_console_alert(severity: str, message: str) -> dict[str, str]:
    """Emit a console/file alert event and return the normalized payload."""
    sev = _normalize_severity(severity)
    if sev == "critical":
        LOGGER.error("ALERT: %s", message)
    elif sev == "warning":
        LOGGER.warning("ALERT: %s", message)
    else:
        LOGGER.info("ALERT: %s", message)

    _persist_alert_event(severity=sev, alert_type="console", message=message)
    return {"severity": sev, "message": message}


def _resolve_alert_sink() -> Path:
    cfg = load_config().get("alerting", {})
    relative = str(cfg.get("email_simulation_file", "logs/alerts_email_simulated.log"))
    path = (ROOT_DIR / relative).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def emit_email_alert(
    severity: str,
    subject: str,
    message: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Simulate an email alert by appending a JSON record to the local sink file."""
    sev = _normalize_severity(severity)
    payload = {
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
        "severity": sev,
        "subject": str(subject),
        "message": str(message),
        "metadata": metadata or {},
    }

    sink = _resolve_alert_sink()
    try:
        with sink.open("a", encoding="utf-8") as fh:
            # One JSON object per line, so the sink is machine-readable. It used
            # to hold Python repr(), which no JSON parser can read back.
            fh.write(json.dumps(payload, default=str) + "\n")
    except OSError as exc:
        LOGGER.warning("Could not write alert sink %s: %s", sink, exc)

    _persist_alert_event(
        severity=sev,
        alert_type="email",
        message=f"{subject}: {message}",
        metadata=metadata or {},
    )

    LOGGER.info("SIMULATED_EMAIL_ALERT: %s", json.dumps(payload, default=str))
    return {"severity": sev, "subject": str(subject), "sink": str(sink)}
