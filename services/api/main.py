"""Launcher for the FastAPI inference API."""

from __future__ import annotations

import uvicorn

from services.api.app import app
from services.model_service import predict_from_payload


def run() -> None:
    """Run local dev server."""
    uvicorn.run("services.api.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()
