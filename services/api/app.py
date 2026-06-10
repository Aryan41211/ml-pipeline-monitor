"""FastAPI inference API for latest production model predictions."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from services.model_service import predict_from_payload
from src.database import initialize_db


API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    api_key_env = os.getenv("MLMONITOR_API_KEY", "")
    if not api_key_env:
        raise HTTPException(status_code=500, detail="API key not configured on server")
    if not api_key or api_key != api_key_env:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


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


@asynccontextmanager
async def _lifespan(_: FastAPI):
    """Initialize persistence before serving requests."""
    initialize_db()
    yield


app = FastAPI(
    title="ML Pipeline Monitor Inference API",
    version="1.0.0",
    lifespan=_lifespan,
)


@app.get("/health")
def health() -> Dict[str, str]:
    """Basic health endpoint."""
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest, api_key: str = Depends(verify_api_key)) -> Dict[str, Any]:
    """Predict using latest production model artifact from registry."""
    try:
        # Backward compatibility: allow legacy monkeypatches on services.api.main.
        try:
            from services.api import main as main_module

            predict_fn = getattr(main_module, "predict_from_payload", predict_from_payload)
        except Exception:
            predict_fn = predict_from_payload

        return predict_fn(payload=request.features, dataset=request.dataset)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
