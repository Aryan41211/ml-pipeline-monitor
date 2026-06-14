"""FastAPI inference API for latest production model predictions."""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ValidationError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

import signal
import sys

from services.model_service import predict_from_payload
from services.telemetry_service import track_user_action
from src.database import initialize_db
from src.db_engine import get_backend
from src.logger import get_app_logger, get_correlation_id, set_correlation_id, get_request_id, set_request_id, set_service_context, get_error_category, ErrorCategory


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
    """Initialize persistence before serving requests, with graceful shutdown."""
    initialize_db()
    
    # Setup signal handlers for graceful shutdown
    shutdown_event = asyncio.Event()
    
    def _signal_handler(signum: int, frame: Any) -> None:
        LOGGER.info("Received signal %s, initiating graceful shutdown", signum)
        shutdown_event.set()
    
    # Register handlers (SIGTERM for container shutdown, SIGINT for Ctrl+C)
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _signal_handler)
        except (OSError, ValueError):
            # Not available on all platforms (e.g., Windows)
            pass
    
    # Run shutdown watcher in background
    async def _shutdown_watcher():
        await shutdown_event.wait()
        LOGGER.info("Shutdown signal received, closing database connections")
        try:
            backend = get_backend()
            if hasattr(backend, "close_all"):
                backend.close_all()
        except Exception as e:
            LOGGER.warning("Error during shutdown cleanup: %s", e)
        sys.exit(0)
    
    import asyncio
    watcher_task = asyncio.create_task(_shutdown_watcher())
    
    try:
        yield
    finally:
        watcher_task.cancel()
        try:
            await watcher_task
        except asyncio.CancelledError:
            pass
        # Final cleanup
        try:
            backend = get_backend()
            if hasattr(backend, "close_all"):
                backend.close_all()
        except Exception as e:
            LOGGER.warning("Error during final cleanup: %s", e)


app = FastAPI(
    title="ML Pipeline Monitor Inference API",
    version="1.0.0",
    lifespan=_lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing, correlation ID, and request ID."""
    # Get or generate correlation ID
    correlation_id = request.headers.get("X-Correlation-ID") or get_correlation_id()
    set_correlation_id(correlation_id)

    # Generate or get request ID
    request_id = request.headers.get("X-Request-ID") or get_request_id()
    set_request_id(request_id)

    # Set service context
    set_service_context("api")

    start = time.time()
    response = await call_next(request)
    duration = time.time() - start

    # Add correlation ID and request ID to response headers
    response.headers["X-Correlation-ID"] = correlation_id
    response.headers["X-Request-ID"] = request_id

    LOGGER.info(
        "api_request",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(duration * 1000, 2),
            "correlation_id": correlation_id,
            "request_id": request_id,
            "service": "api",
        },
    )
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    error_category = get_error_category(exc)
    correlation_id = get_correlation_id()
    request_id = get_request_id()

    LOGGER.exception(
        "Unhandled exception: %s",
        exc,
        extra={
            "error_category": error_category,
            "correlation_id": correlation_id,
            "request_id": request_id,
        },
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error_category": error_category,
            "correlation_id": correlation_id,
            "request_id": request_id,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors with detailed messages."""
    correlation_id = get_correlation_id()
    request_id = get_request_id()

    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(x) for x in error["loc"]),
            "message": error["msg"],
            "type": error["type"],
        })

    LOGGER.warning(
        "Request validation failed",
        extra={
            "error_category": ErrorCategory.VALIDATION,
            "validation_errors": errors,
            "correlation_id": correlation_id,
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
        },
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Request validation failed",
            "error_category": ErrorCategory.VALIDATION,
            "errors": errors,
            "correlation_id": correlation_id,
            "request_id": request_id,
        },
    )


@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    """Handle rate limit exceeded errors."""
    correlation_id = get_correlation_id()
    request_id = get_request_id()

    LOGGER.warning(
        "Rate limit exceeded",
        extra={
            "error_category": "rate_limit_exceeded",
            "correlation_id": correlation_id,
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "client_ip": get_remote_address(request),
        },
    )

    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "detail": "Rate limit exceeded",
            "error_category": "rate_limit_exceeded",
            "retry_after": exc.retry_after,
            "correlation_id": correlation_id,
            "request_id": request_id,
        },
    )


@app.get("/health")
def health() -> Dict[str, Any]:
    """Basic health endpoint with database connectivity check."""
    from src.db_engine import get_backend

    db_status = "ok"
    try:
        backend = get_backend()
        conn = backend.connect()
        conn.execute("SELECT 1")
        conn.close()
    except Exception as e:
        db_status = f"error: {e}"

    return {
        "status": "ok" if db_status == "ok" else "degraded",
        "database": db_status,
    }


@app.get("/health/live")
def health_live() -> Dict[str, str]:
    """Liveness probe - indicates if the process is running.

    Used by Kubernetes/container orchestrators to determine if the container
    should be restarted. Returns 200 if the process is alive.
    """
    return {"status": "alive"}


@app.get("/health/ready")
def health_ready() -> Dict[str, Any]:
    """Readiness probe - indicates if the service can handle requests.

    Used by Kubernetes/container orchestrators to determine if traffic
    should be routed to this instance. Checks database connectivity.
    """
    from src.db_engine import get_backend

    db_status = "ok"
    try:
        backend = get_backend()
        conn = backend.connect()
        conn.execute("SELECT 1")
        conn.close()
    except Exception as e:
        db_status = f"error: {e}"

    ready = db_status == "ok"
    status_code = 200 if ready else 503

    return {
        "status": "ready" if ready else "not_ready",
        "database": db_status,
    }


@app.get("/health/detailed")
def detailed_health() -> Dict[str, Any]:
    """Detailed health endpoint with system metrics and database check."""
    from src.system_monitor import get_system_metrics, get_process_metrics
    from src.db_engine import get_backend

    sys_metrics = get_system_metrics()
    proc_metrics = get_process_metrics()

    db_status = "ok"
    db_info = {}
    try:
        backend = get_backend()
        conn = backend.connect()
        conn.execute("SELECT 1")
        conn.close()
        db_info = {"backend": backend.name, "connected": True}
    except Exception as e:
        db_status = f"error: {e}"
        db_info = {"connected": False, "error": str(e)}

    return {
        "status": "ok" if db_status == "ok" else "degraded",
        "system": sys_metrics,
        "process": proc_metrics,
        "database": db_info,
    }


@app.post("/predict")
@limiter.limit(RATE_LIMIT)
def predict(request: Request, request_body: PredictRequest, api_key: str = Depends(verify_api_key)) -> Dict[str, Any]:
    """Predict using latest production model artifact from registry."""
    correlation_id = get_correlation_id()
    request_id = get_request_id()
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

        # Log successful prediction
        log_prediction_request(
            model_id=result.get("model_id", "unknown"),
            dataset=request_body.dataset or "unknown",
            status="success",
            num_predictions=result.get("predictions_count", 1),
            duration_ms=round(duration * 1000, 2),
            correlation_id=correlation_id,
            request_id=request_id,
            service="api",
        )

        return result
    except FileNotFoundError as exc:
        error_category = get_error_category(exc)
        log_prediction_request(
            model_id="unknown",
            dataset=request_body.dataset or "unknown",
            status="failed",
            num_predictions=0,
            error=str(exc),
            correlation_id=correlation_id,
            request_id=request_id,
            service="api",
        )
        raise HTTPException(
            status_code=500,
            detail={"message": str(exc), "error_category": error_category, "correlation_id": correlation_id, "request_id": request_id},
        ) from exc
    except ValueError as exc:
        error_category = get_error_category(exc)
        log_prediction_request(
            model_id="unknown",
            dataset=request_body.dataset or "unknown",
            status="failed",
            num_predictions=0,
            error=str(exc),
            correlation_id=correlation_id,
            request_id=request_id,
            service="api",
        )
        raise HTTPException(
            status_code=400,
            detail={"message": str(exc), "error_category": error_category, "correlation_id": correlation_id, "request_id": request_id},
        ) from exc
    except Exception as exc:
        error_category = get_error_category(exc)
        log_prediction_request(
            model_id="unknown",
            dataset=request_body.dataset or "unknown",
            status="failed",
            num_predictions=0,
            error=str(exc),
            correlation_id=correlation_id,
            request_id=request_id,
            service="api",
        )
        LOGGER.exception(
            "Prediction failed",
            extra={
                "error_category": error_category,
                "correlation_id": correlation_id,
                "request_id": request_id,
            },
        )
        raise HTTPException(
            status_code=500,
            detail={"message": f"Prediction failed: {exc}", "error_category": error_category, "correlation_id": correlation_id, "request_id": request_id},
        ) from exc
