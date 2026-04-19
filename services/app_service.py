"""Application-level service facade for dashboard pages.

Enforces UI -> services -> core -> database layering by centralizing
initialization and read operations used by Streamlit pages.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.config_loader import load_config
from src.database import get_experiments, get_models, initialize_db
from src.system_monitor import get_system_metrics


def initialize_application() -> None:
    """Initialize persistent schema and runtime prerequisites."""
    initialize_db()


def list_experiments(limit: int = 200) -> List[Dict[str, Any]]:
    return get_experiments(limit=limit)


def list_models(limit: int = 100) -> List[Dict[str, Any]]:
    return get_models(limit=limit)


def get_dashboard_snapshot(limit: int = 200) -> Dict[str, Any]:
    """Return dashboard data in one service call for robust page loading."""
    return {
        "experiments": list_experiments(limit=limit),
        "models": list_models(limit=100),
        "system": get_system_metrics(),
    }


def get_ui_settings() -> Dict[str, Any]:
    """Return UI settings from config via service layer."""
    return load_config().get("ui", {})
