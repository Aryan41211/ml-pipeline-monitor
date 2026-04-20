"""Alert helpers for console and Streamlit UI surfaces."""

from __future__ import annotations

from datetime import datetime, UTC
from pathlib import Path
from typing import Any
from typing import Dict

from src.config_loader import ROOT_DIR, load_config
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
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, str]:
    """Simulate email alerts by appending a structured record to a local sink file."""
    sev = severity.strip().lower()
    now = datetime.now(UTC).isoformat(timespec="seconds")
    payload = {
        "time": now,
        "severity": sev,
        "subject": str(subject),
        "message": str(message),
        "metadata": metadata or {},
    }

    sink = _resolve_alert_sink()
    with sink.open("a", encoding="utf-8") as fh:
        fh.write(f"{payload}\n")

    LOGGER.info("SIMULATED_EMAIL_ALERT: %s", payload)
    return {"severity": sev, "subject": str(subject), "sink": str(sink)}
