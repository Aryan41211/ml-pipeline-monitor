"""Launcher for the FastAPI inference API."""

from __future__ import annotations

import uvicorn

from ml_pipeline_monitor.api.main import app


def run() -> None:
    """Run local dev server."""
    uvicorn.run("ml_pipeline_monitor.api.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()
