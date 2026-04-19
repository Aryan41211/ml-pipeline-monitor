"""Alert helpers for console and Streamlit UI surfaces."""

from __future__ import annotations

from typing import Dict

from src.logger import get_app_logger

LOGGER = get_app_logger("alerts")


def emit_console_alert(severity: str, message: str) -> Dict[str, str]:
    """Emit a console/file alert event and return normalized payload."""
    sev = severity.strip().lower()
    if sev in {"critical", "error"}:
        LOGGER.error("ALERT: %s", message)
    elif sev in {"warning", "warn"}:
        LOGGER.warning("ALERT: %s", message)
    else:
        LOGGER.info("ALERT: %s", message)

    return {"severity": sev, "message": message}
