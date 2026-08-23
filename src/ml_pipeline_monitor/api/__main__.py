"""Launcher for the FastAPI inference API."""

from __future__ import annotations

import os

import uvicorn


def run() -> None:
    """Run the API server.

    Binds all interfaces so the container is reachable on the compose network;
    override with MLMONITOR_API_HOST when running directly on a host.
    """
    host = os.getenv("MLMONITOR_API_HOST", "0.0.0.0")  # nosec B104 - containerised bind
    port = int(os.getenv("MLMONITOR_API_PORT", "8000"))
    uvicorn.run("ml_pipeline_monitor.api.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run()
