"""Service wrapper for structured telemetry and user interaction events."""

from __future__ import annotations

from typing import Any, Dict

from src.logger import log_user_action


def track_user_action(
    action: str,
    page: str | None = None,
    metadata: Dict[str, Any] | None = None,
) -> None:
    """Record a structured user event without exposing logging internals to UI."""
    resolved_page = page or "unknown"
    log_user_action(action=action, page=resolved_page, metadata=metadata)
