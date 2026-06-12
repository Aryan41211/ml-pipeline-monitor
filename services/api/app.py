"""FastAPI inference API for latest production model predictions."""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from services.model_service import predict_from_payload
from services.telemetry_service import track_user_action
from src.database import initialize_db
from src.logger import get_app_logger


LOGGER = get_app_logger("api")
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Rate limiter: 60 requests per minute per IP by default
RATE_LIMIT = os.getenv("MLMONITOR_RATE_LIMIT", "60/minute")
limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT])


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

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing."""
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    LOGGER.info(
        "api_request",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(duration * 1000, 2),
        },
    )
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    LOGGER.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


@app.get("/health")
def health() -> Dict[str, str]:
    """Basic health endpoint."""
    return {"status": "ok"}


@app.get("/health/detailed")
def detailed_health() -> Dict[str, Any]:
    """Detailed health endpoint with system metrics."""
    from src.system_monitor import get_system_metrics, get_process_metrics
    
    sys_metrics = get_system_metrics()
    proc_metrics = get_process_metrics()
    
    return {
        "status": "ok",
        "system": sys_metrics,
        "process": proc_metrics,
    }


@app.post("/predict")
@limiter.limit(RATE_LIMIT)
def predict(request: Request, request_body: PredictRequest, api_key: str = Depends(verify_api_key)) -> Dict[str, Any]:
    """Predict using latest production model artifact from registry."""
    start = time.time()
    
    try:
        # Backward compatibility: allow legacy monkeypatches on services.api.main.
        try:
            from services.api import main as main_module

            predict_fn = getattr(main_module, "predict_from_payload", predict_from_payload)
        except Exception:
            predict_fn = predict_from_payload

        result = predict_fn(payload=request_body.features, dataset=request_body.dataset)
        
        duration = time.time() - start
        track_user_action(page="api", action="prediction", metadata={"duration_ms": round(duration * 1000, 2)})
        
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
