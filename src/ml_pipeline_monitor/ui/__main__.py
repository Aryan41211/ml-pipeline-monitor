"""Launcher for the Streamlit dashboard.

Starts the Prometheus exporter before handing control to Streamlit's CLI.

Streamlit only executes ``app.py`` when a browser session connects, so calling
``start_metrics_server()`` from inside the script leaves the exporter down --
and the Prometheus target with it -- until somebody opens the dashboard. In a
monitored deployment that reads as an outage.

Streamlit runs the script in a thread of this same process, so starting the
exporter here shares one registry with everything the app records.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _app_path() -> str:
    """Locate app.py relative to the installed package or the repo root."""
    override = os.getenv("MLMONITOR_APP_PATH")
    if override:
        return override

    # src/ml_pipeline_monitor/ui/__main__.py -> repo root is four levels up.
    candidate = Path(__file__).resolve().parents[3] / "app.py"
    if candidate.is_file():
        return str(candidate)

    cwd_candidate = Path.cwd() / "app.py"
    if cwd_candidate.is_file():
        return str(cwd_candidate)

    raise FileNotFoundError("Could not locate app.py; set MLMONITOR_APP_PATH.")


def run() -> None:
    """Start the metrics exporter, then run Streamlit in this process."""
    from ml_pipeline_monitor.core.logger import get_app_logger
    from ml_pipeline_monitor.core.metrics import start_metrics_server

    logger = get_app_logger("ui")

    try:
        start_metrics_server()
    except OSError as exc:
        # A busy port must not stop the dashboard from serving.
        logger.warning("Metrics exporter did not start: %s", exc)

    from streamlit.web import cli as stcli

    argv = sys.argv[1:]
    sys.argv = [
        "streamlit",
        "run",
        _app_path(),
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
        *argv,
    ]
    sys.exit(stcli.main())


if __name__ == "__main__":
    run()
