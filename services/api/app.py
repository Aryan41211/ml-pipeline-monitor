"""FastAPI inference API with JWT auth, versioning, and model caching."""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

from fastapi import Depends, FastAPI, HTTPException, Request, Response, Security, status
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ValidationError

try:
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
except ModuleNotFoundError:  # pragma: no cover
    generate_latest = None
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

import signal
import sys

from services.model_service import predict_from_payload
from services.telemetry_service import track_user_action
from src.database import initialize_db
from src.db_engine import get_backend
from src.jwt_auth import (
    TokenPayload,
    create_access_token,
    create_refresh_token,
    decode_token,
    verify_token,
)
from src.logger import (
    get_app_logger,
    get_correlation_id,
    set_correlation_id,
    get_request_id,
    set_request_id,
    set_service_context,
    get_error_category,
    ErrorCategory,
)
from src.metrics import (
    registry,
    record_api_request,
    record_api_error,
    record_prediction,
    update_system_metrics,
)
from src.model_cache import get_latest_production_model

LOGGER = get_app_logger("api")

# ---------------------------------------------------------------------------
# Auth schemes
# ---------------------------------------------------------------------------
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
JWT_SCHEME = HTTPBearer(auto_error=False)

RATE_LIMIT = os.getenv("MLMONITOR_RATE_LIMIT", "60/minute")
limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT])

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")
    refresh: bool = Field(False, description="Return refresh token as well")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: Optional[str] = None
    role: str


class RefreshRequest(BaseModel):
    refresh_token: str = Field(..., description="Valid refresh token")


class PredictRequest(BaseModel):
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


# ---------------------------------------------------------------------------
# Auth dependencies
# ---------------------------------------------------------------------------

async def _get_api_key(api_key: str = Security(API_KEY_HEADER)) -> Optional[str]:
    if api_key:
        return api_key
    return None


async def _get_jwt_token(credentials: Optional[HTTPAuthorizationCredentials] = Security(JWT_SCHEME)) -> Optional[str]:
    if credentials and credentials.scheme.lower() == "bearer":
        return credentials.credentials
    return None


async def _authenticate(
    api_key: Optional[str] = Depends(_get_api_key),
    jwt_token: Optional[str] = Depends(_get_jwt_token),
) -> TokenPayload:
    api_key_env = os.getenv("MLMONITOR_API_KEY", "")
    if api_key_env and api_key and api_key == api_key_env:
        return TokenPayload(sub="api_key", role="admin", exp=0, iat=0, jti="api-key")

    if jwt_token:
        try:
            payload = verify_token(jwt_token)
            return payload
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"message": "Invalid or expired token", "error": str(exc)},
                headers={"WWW-Authenticate": "Bearer"},
            ) from exc

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing authentication credentials. Provide X-API-Key or Bearer JWT.",
        headers={"WWW-Authenticate": "Bearer"},
    )


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_prediction_request(
    *,
    model_id: str,
    dataset: str,
    status: str,
    num_predictions: int,
    duration_ms: float | int | None = None,
    error: str | None = None,
    correlation_id: str | None = None,
    request_id: str | None = None,
    service: str = "api",
) -> None:
    extra = {
        "model_id": model_id,
        "dataset": dataset,
        "status": status,
        "num_predictions": num_predictions,
        "duration_ms": duration_ms,
        "error": error,
        "correlation_id": correlation_id,
        "request_id": request_id,
        "service": service,
    }
    extra = {k: v for k, v in extra.items() if v is not None}
    if status == "success":
        LOGGER.info("prediction_request", extra=extra)
    else:
        LOGGER.warning("prediction_request_failed", extra=extra)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(_: FastAPI):
    initialize_db()
    shutdown_event = asyncio.Event()

    def _signal_handler(signum: int, frame: Any) -> None:
        LOGGER.info("Received signal %s, initiating graceful shutdown", signum)
        shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _signal_handler)
        except (OSError, ValueError):
            pass

    async def _shutdown_watcher():
        await shutdown_event.wait()
        LOGGER.info("Shutdown signal received, closing database connections")
        try:
            backend = get_backend()
            if hasattr(backend, "close_all"):
                backend.close_all()
        except Exception as exc:
            LOGGER.warning("Error during shutdown cleanup: %s", exc)
        sys.exit(0)

    watcher_task = asyncio.create_task(_shutdown_watcher())
    try:
        yield
    finally:
        watcher_task.cancel()
        try:
            await watcher_task
        except asyncio.CancelledError:
            pass
        try:
            backend = get_backend()
            if hasattr(backend, "close_all"):
                backend.close_all()
        except Exception as exc:
            LOGGER.warning("Error during final cleanup: %s", exc)


app = FastAPI(
    title="ML Pipeline Monitor Inference API",
    description="Production inference API for ML model registry with JWT authentication, model caching, and Prometheus metrics.",
    version="1.0.0",
    lifespan=_lifespan,
    docs_url="/v1/docs",
    redoc_url="/v1/redoc",
    openapi_url="/v1/openapi.json",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

import asyncio

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    correlation_id = request.headers.get("X-Correlation-ID") or get_correlation_id()
    set_correlation_id(correlation_id)
    request_id = request.headers.get("X-Request-ID") or get_request_id()
    set_request_id(request_id)
    set_service_context("api")

    start = time.time()
    response = await call_next(request)
    duration = time.time() - start

    response.headers["X-Correlation-ID"] = correlation_id
    response.headers["X-Request-ID"] = request_id

    record_api_request(
        method=request.method,
        endpoint=request.url.path,
        status_code=response.status_code,
        duration_seconds=duration,
    )

    if response.status_code >= 400:
        record_api_error(
            method=request.method,
            endpoint=request.url.path,
            error_type=f"http_{response.status_code}",
        )

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


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
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


# ---------------------------------------------------------------------------
# Health endpoints (unversioned, always available)
# ---------------------------------------------------------------------------

def _db_status() -> tuple[str, str]:
    db_status = "ok"
    try:
        backend = get_backend()
        conn = backend.connect()
        conn.execute("SELECT 1")
        conn.close()
    except Exception as exc:
        db_status = f"error: {exc}"
    return "ok" if db_status == "ok" else "degraded", db_status


@app.get("/health")
def health() -> Dict[str, Any]:
    status, db_status = _db_status()
    return {
        "status": status,
        "database": db_status,
        "version": app.version,
    }


@app.get("/health/live")
def health_live() -> Dict[str, str]:
    return {"status": "alive"}


@app.get("/health/ready")
def health_ready() -> Dict[str, Any]:
    status, db_status = _db_status()
    ready = status == "ok"
    return {
        "status": "ready" if ready else "not_ready",
        "database": db_status,
    }


@app.get("/health/detailed")
def health_detailed() -> Dict[str, Any]:
    from src.system_monitor import get_system_metrics, get_process_metrics
    status, db_status = _db_status()
    return {
        "status": status,
        "system": get_system_metrics(),
        "process": get_process_metrics(),
        "database": {"backend": get_backend().name, "status": db_status},
    }


@app.get("/metrics")
def metrics() -> Response:
    update_system_metrics()
    if generate_latest is None:
        return Response(content="", media_type=CONTENT_TYPE_LATEST)
    return Response(content=generate_latest(registry), media_type=CONTENT_TYPE_LATEST)


# ---------------------------------------------------------------------------
# V1 Auth endpoints
# ---------------------------------------------------------------------------

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: Optional[str] = None
    role: str


@app.post("/v1/auth/login", response_model=LoginResponse)
async def login(body: LoginRequest):
    from src.auth import _check_login, _resolve_user, _credentials
    ok, err = _check_login(body.username, body.password)
    if not ok:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=err)
    user = _resolve_user(body.username)
    creds = _credentials()
    role = creds.get(user, {}).get("role", "viewer")
    access = create_access_token(sub=user, role=role)
    refresh = create_refresh_token(sub=user, role=role) if body.refresh else None
    return LoginResponse(
        access_token=access,
        expires_in=_get_expiration_minutes() * 60,
        refresh_token=refresh,
        role=role,
    )


@app.post("/v1/auth/refresh")
async def refresh(body: RefreshRequest):
    try:
        payload = verify_token(body.refresh_token)
        if not payload.refresh:
            raise ValueError("Not a refresh token")
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
    access = create_access_token(sub=payload.sub, role=payload.role)
    return {"access_token": access, "token_type": "bearer", "expires_in": _get_expiration_minutes() * 60}


@app.get("/v1/auth/me", dependencies=[Depends(_authenticate)])
async def me(token: TokenPayload = Depends(_authenticate)):
    return {"sub": token.sub, "role": token.role, "refresh": token.refresh}


# ---------------------------------------------------------------------------
# V1 Prediction endpoint
# ---------------------------------------------------------------------------

@app.post("/v1/predict")
@limiter.limit(RATE_LIMIT)
async def predict_v1(
    request: Request,
    body: PredictRequest,
    token: TokenPayload = Depends(_authenticate),
) -> Dict[str, Any]:
    correlation_id = get_correlation_id()
    request_id = get_request_id()
    start = time.time()

    cached = get_latest_production_model(body.dataset)
    if cached is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No production model available. Train and promote a model first.",
        )

    model, scaler, _ = cached

    try:
        result = predict_from_payload(
            payload=body.features,
            dataset=body.dataset,
            model=model,
            scaler=scaler,
        )
        duration = time.time() - start
        track_user_action(page="api", action="prediction", metadata={"duration_ms": round(duration * 1000, 2)})

        record_prediction(
            model_id=result.get("model_id", "unknown"),
            dataset=body.dataset or "unknown",
            status="success",
            latency_seconds=duration,
        )

        log_prediction_request(
            model_id=result.get("model_id", "unknown"),
            dataset=body.dataset or "unknown",
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
            dataset=body.dataset or "unknown",
            status="failed",
            num_predictions=0,
            error=str(exc),
            correlation_id=correlation_id,
            request_id=request_id,
            service="api",
        )
        record_prediction(model_id="unknown", dataset=body.dataset or "unknown", status="failed", latency_seconds=time.time() - start)
        raise HTTPException(
            status_code=500,
            detail={"message": str(exc), "error_category": error_category, "correlation_id": correlation_id, "request_id": request_id},
        ) from exc
    except ValueError as exc:
        error_category = get_error_category(exc)
        log_prediction_request(
            model_id="unknown",
            dataset=body.dataset or "unknown",
            status="failed",
            num_predictions=0,
            error=str(exc),
            correlation_id=correlation_id,
            request_id=request_id,
            service="api",
        )
        record_prediction(model_id="unknown", dataset=body.dataset or "unknown", status="failed", latency_seconds=time.time() - start)
        raise HTTPException(
            status_code=400,
            detail={"message": str(exc), "error_category": error_category, "correlation_id": correlation_id, "request_id": request_id},
        ) from exc
    except Exception as exc:
        error_category = get_error_category(exc)
        log_prediction_request(
            model_id="unknown",
            dataset=body.dataset or "unknown",
            status="failed",
            num_predictions=0,
            error=str(exc),
            correlation_id=correlation_id,
            request_id=request_id,
            service="api",
        )
        record_prediction(model_id="unknown", dataset=body.dataset or "unknown", status="failed", latency_seconds=time.time() - start)
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


# ---------------------------------------------------------------------------
# Backward-compatible legacy endpoints (deprecated)
# ---------------------------------------------------------------------------

@app.post("/predict", deprecated=True)
@limiter.limit(RATE_LIMIT)
def predict_legacy(request: Request, request_body: PredictRequest, api_key: str = Depends(_get_api_key)) -> Dict[str, Any]:
    if not api_key:
        raise HTTPException(status_code=401, detail="Legacy /predict requires X-API-Key. Use /v1/predict with JWT instead.")
    correlation_id = get_correlation_id()
    request_id = get_request_id()
    start = time.time()

    try:
        import services.model_service as _model_service
        result = _model_service.predict_from_payload(payload=request_body.features, dataset=request_body.dataset)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"message": str(exc)}) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail={"message": str(exc)}) from exc

    duration = time.time() - start
    record_prediction(
        model_id=result.get("model_id", "unknown"),
        dataset=request_body.dataset or "unknown",
        status="success",
        latency_seconds=duration,
    )
    return result


def _get_expiration_minutes() -> int:
    try:
        return int(os.getenv("JWT_EXPIRATION_MINUTES", 60))
    except Exception:
        return 60
