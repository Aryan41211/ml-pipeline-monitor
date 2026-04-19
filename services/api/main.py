"""FastAPI inference API for latest production model predictions."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from services.model_service import predict_from_payload
from src.database import initialize_db


class PredictRequest(BaseModel):
    """Prediction request payload."""

    features: Union[
        Dict[str, float],
        List[Dict[str, float]],
        List[float],
        List[List[float]],
    ] = Field(..., description="Feature payload for one or many predictions")
    dataset: Optional[str] = Field(
        default=None,
        description="Optional dataset name to target production model selection",
    )


app = FastAPI(title="ML Pipeline Monitor Inference API", version="1.0.0")


@app.on_event("startup")
def _startup() -> None:
    initialize_db()


@app.get("/health")
def health() -> Dict[str, str]:
    """Basic health endpoint."""
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest) -> Dict[str, Any]:
    """Predict using latest production model."""
    try:
        return predict_from_payload(payload=request.features, dataset=request.dataset)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


def run() -> None:
    """Run local dev server."""
    uvicorn.run("services.api.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()
