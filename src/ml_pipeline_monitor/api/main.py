"""FastAPI inference API with JWT auth, versioning, and model caching."""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response, Security, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

try:
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
except ModuleNotFoundError:  # pragma: no cover
    generate_latest = None
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"

import hashlib
import json
import secrets as _secrets

from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from ml_pipeline_monitor.core.jwt_auth import (
    TokenPayload,
    create_access_token,
    create_refresh_token,
    verify_token,
)
from ml_pipeline_monitor.core.logger import (
    ErrorCategory,
    get_app_logger,
    get_correlation_id,
    get_error_category,
    get_request_id,
    set_correlation_id,
    set_request_id,
    set_service_context,
)
from ml_pipeline_monitor.core.metrics import (
    record_api_error,
    record_api_request,
    record_prediction,
    registry,
    update_system_metrics,
)
from ml_pipeline_monitor.database import (
    initialize_db,
    initialize_prediction_registry,
    save_prediction_request,
    save_predictions_for_request,
)
from ml_pipeline_monitor.database.connection import get_backend, reset_backend
from ml_pipeline_monitor.ml.model_cache import get_latest_production_model
from ml_pipeline_monitor.services.model_service import predict_from_payload
from ml_pipeline_monitor.services.telemetry_service import track_user_action

LOGGER = get_app_logger("api")

# ---------------------------------------------------------------------------
# Auth schemes
# ---------------------------------------------------------------------------
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
JWT_SCHEME = HTTPBearer(auto_error=False)

def _rate_limit(env_var: str, config_key: str, fallback: str) -> str:
    """Resolve a rate limit from env, then the api config block, then a default."""
    from_env = os.getenv(env_var, "").strip()
    if from_env:
        return from_env
    try:
        from ml_pipeline_monitor.core.config_loader import load_config

        configured = str(load_config().get("api", {}).get(config_key, "")).strip()
    except Exception:
        configured = ""
    return configured or fallback


RATE_LIMIT = _rate_limit("MLMONITOR_RATE_LIMIT", "rate_limit", "60/minute")
# Auth endpoints get their own, much tighter bucket: they are the brute-force
# surface, and slowapi's default_limits do not apply without SlowAPIMiddleware.
AUTH_RATE_LIMIT = _rate_limit("MLMONITOR_AUTH_RATE_LIMIT", "auth_rate_limit", "10/minute")
limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT])

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")
    refresh: bool = Field(False, description="Return refresh token as well")


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: str | None = None
    role: str


class RefreshRequest(BaseModel):
    refresh_token: str = Field(..., description="Valid refresh token")


class PredictRequest(BaseModel):
    features: dict[str, float] | list[dict[str, float]] | list[float] | list[list[float]] = Field(..., description="Feature payload for one or many predictions")
    dataset: str | None = Field(
        default=None,
        description="Optional dataset name to target production model selection",
    )


# ---------------------------------------------------------------------------
# Auth dependencies
# ---------------------------------------------------------------------------

async def _get_api_key(api_key: str = Security(API_KEY_HEADER)) -> str | None:
    if api_key:
        return api_key
    return None


async def _get_jwt_token(credentials: HTTPAuthorizationCredentials | None = Security(JWT_SCHEME)) -> str | None:
    if credentials and credentials.scheme.lower() == "bearer":
        return credentials.credentials
    return None


def _api_key_is_valid(api_key: str | None) -> bool:
    """Constant-time check of a presented API key against the configured one.

    Returns False when no key is configured, so an unset ``MLMONITOR_API_KEY``
    can never be satisfied by an arbitrary header value.
    """
    configured = os.getenv("MLMONITOR_API_KEY", "")
    if not configured or not api_key:
        return False
    return _secrets.compare_digest(str(api_key), configured)


async def _authenticate(
    api_key: str | None = Depends(_get_api_key),
    jwt_token: str | None = Depends(_get_jwt_token),
) -> TokenPayload:
    if _api_key_is_valid(api_key):
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


def _input_type(payload: Any) -> str:
    """Classify a prediction payload shape for the history record."""
    if isinstance(payload, dict):
        return "feature_map"
    if isinstance(payload, list) and payload:
        if isinstance(payload[0], dict):
            return "feature_map_batch"
        if isinstance(payload[0], list):
            return "vector_batch"
        return "vector"
    return "unknown"


def _persist_prediction_history(
    *,
    request_id: str,
    correlation_id: str | None,
    model_id: str,
    dataset: str | None,
    payload: Any,
    result: dict[str, Any] | None,
    status_label: str,
    duration_ms: float,
    error: str | None = None,
) -> None:
    """Record a prediction request (and its rows) in the history tables.

    Persistence must never take down a served prediction, so every failure here
    is logged and swallowed.
    """
    try:
        predictions = list((result or {}).get("predictions") or [])
        payload_digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:32]

        save_prediction_request(
            request_id=request_id,
            correlation_id=correlation_id,
            model_id=model_id,
            dataset=dataset,
            input_type=_input_type(payload),
            input_hash=payload_digest,
            num_predictions=len(predictions),
            status=status_label,
            duration_ms=round(duration_ms, 3),
            error=error,
        )
        if predictions:
            save_predictions_for_request(
                request_id=request_id,
                predictions=predictions,
                probabilities=(result or {}).get("probabilities"),
            )
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Failed to persist prediction history: %s", exc)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(_: FastAPI):
    """Initialize the schema on startup and release the pool on shutdown.

    Signal handling is deliberately left to uvicorn: installing our own SIGTERM
    handler here replaces uvicorn's, which is what makes a container hang until
    SIGKILL instead of draining in-flight requests. Uvicorn instead runs this
    context manager's teardown as part of its own graceful shutdown.
    """
    initialize_db()
    initialize_prediction_registry()
    try:
        yield
    finally:
        LOGGER.info("Shutting down: closing database connections")
        try:
            reset_backend()
        except Exception as exc:
            LOGGER.warning("Error during shutdown cleanup: %s", exc)


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


def _cors_origins() -> list[str]:
    """Allowed CORS origins from MLMONITOR_CORS_ORIGINS or api.cors_origins config."""
    raw = os.getenv("MLMONITOR_CORS_ORIGINS", "")
    if raw.strip():
        return [origin.strip() for origin in raw.split(",") if origin.strip()]
    from ml_pipeline_monitor.core.config_loader import load_config

    configured = load_config().get("api", {}).get("cors_origins", []) or []
    return [str(origin) for origin in configured if str(origin).strip()]


_ALLOWED_ORIGINS = _cors_origins()
if _ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Correlation-ID", "X-Request-ID"],
    )

SECURITY_HEADERS = {
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Cache-Control": "no-store",
}

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
    try:
        response = await call_next(request)
    except Exception:
        # Still record the failure: an unhandled error is exactly the request we
        # most want to see on the dashboard.
        duration = time.time() - start
        record_api_request(
            method=request.method,
            endpoint=request.url.path,
            status_code=500,
            duration_seconds=duration,
        )
        record_api_error(method=request.method, endpoint=request.url.path, error_type="unhandled_exception")
        raise
    duration = time.time() - start

    response.headers["X-Correlation-ID"] = correlation_id
    response.headers["X-Request-ID"] = request_id
    for header, value in SECURITY_HEADERS.items():
        response.headers.setdefault(header, value)

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

    # slowapi exposes the breached limit, not a retry_after attribute; reading
    # one turned every 429 into a 500.
    limit = getattr(exc, "limit", None)
    retry_after = getattr(getattr(limit, "limit", None), "GRANULARITY", None)
    retry_after_seconds = getattr(retry_after, "seconds", None)

    headers = {}
    if retry_after_seconds:
        headers["Retry-After"] = str(int(retry_after_seconds))

    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "detail": "Rate limit exceeded",
            "error_category": "rate_limit_exceeded",
            "limit": str(getattr(exc, "detail", "")) or None,
            "retry_after": int(retry_after_seconds) if retry_after_seconds else None,
            "correlation_id": correlation_id,
            "request_id": request_id,
        },
        headers=headers,
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
def health(response: Response) -> dict[str, Any]:
    health_status, db_status = _db_status()
    if health_status != "ok":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": health_status,
        "database": db_status,
        "version": app.version,
    }


@app.get("/health/live")
def health_live() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/health/ready")
def health_ready(response: Response) -> dict[str, Any]:
    """Readiness probe.

    Returns 503 when the database is unreachable so orchestrators actually take
    the instance out of rotation -- a 200 with ``"not_ready"`` in the body reads
    as healthy to every load balancer and to Docker's own healthcheck.
    """
    health_status, db_status = _db_status()
    ready = health_status == "ok"
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": "ready" if ready else "not_ready",
        "database": db_status,
    }


@app.get("/health/detailed")
def health_detailed(response: Response) -> dict[str, Any]:
    from ml_pipeline_monitor.core.system_monitor import get_process_metrics, get_system_metrics

    health_status, db_status = _db_status()
    if health_status != "ok":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": health_status,
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

@app.post("/v1/auth/login", response_model=LoginResponse)
@limiter.limit(AUTH_RATE_LIMIT)
async def login(request: Request, body: LoginRequest):
    from ml_pipeline_monitor.core.auth import _check_login, _credentials, _resolve_user
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
@limiter.limit(AUTH_RATE_LIMIT)
async def refresh(request: Request, body: RefreshRequest):
    try:
        payload = verify_token(body.refresh_token)
        if not payload.refresh:
            raise ValueError("Not a refresh token")
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
    access = create_access_token(sub=payload.sub, role=payload.role)
    return {"access_token": access, "token_type": "bearer", "expires_in": _get_expiration_minutes() * 60}


@app.get("/v1/auth/me")
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
) -> dict[str, Any]:
    correlation_id = get_correlation_id()
    request_id = get_request_id()
    start = time.time()

    try:
        cached = get_latest_production_model(body.dataset)
    except Exception as exc:
        LOGGER.warning("Failed to fetch production model: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to access model registry. Ensure the database is initialized.",
        ) from exc

    if cached is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No production model available. Train and promote a model first.",
        )

    model, scaler, meta = cached

    try:
        result = predict_from_payload(
            payload=body.features,
            dataset=body.dataset,
            model=model,
            scaler=scaler,
            model_meta=meta,
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
            num_predictions=len(result.get("predictions") or []),
            duration_ms=round(duration * 1000, 2),
            correlation_id=correlation_id,
            request_id=request_id,
            service="api",
        )
        _persist_prediction_history(
            request_id=request_id,
            correlation_id=correlation_id,
            model_id=str(result.get("model_id") or "unknown"),
            dataset=body.dataset,
            payload=body.features,
            result=result,
            status_label="success",
            duration_ms=duration * 1000,
        )
        return result
    except Exception as exc:
        duration = time.time() - start
        # ValueError means the caller sent something unusable; everything else is ours.
        http_status = 400 if isinstance(exc, ValueError) else 500
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
        record_prediction(
            model_id="unknown",
            dataset=body.dataset or "unknown",
            status="failed",
            latency_seconds=duration,
        )
        _persist_prediction_history(
            request_id=request_id,
            correlation_id=correlation_id,
            model_id="unknown",
            dataset=body.dataset,
            payload=body.features,
            result=None,
            status_label="failed",
            duration_ms=duration * 1000,
            error=str(exc),
        )
        if http_status == 500:
            LOGGER.exception(
                "Prediction failed",
                extra={
                    "error_category": error_category,
                    "correlation_id": correlation_id,
                    "request_id": request_id,
                },
            )
        raise HTTPException(
            status_code=http_status,
            detail={
                "message": str(exc) if http_status == 400 else f"Prediction failed: {exc}",
                "error_category": error_category,
                "correlation_id": correlation_id,
                "request_id": request_id,
            },
        ) from exc


# ---------------------------------------------------------------------------
# Backward-compatible legacy endpoints (deprecated)
# ---------------------------------------------------------------------------

@app.post("/predict", deprecated=True)
@limiter.limit(RATE_LIMIT)
def predict_legacy(request: Request, request_body: PredictRequest, api_key: str = Depends(_get_api_key)) -> dict[str, Any]:
    if not _api_key_is_valid(api_key):
        raise HTTPException(
            status_code=401,
            detail="Legacy /predict requires a valid X-API-Key. Use /v1/predict with JWT instead.",
        )
    correlation_id = get_correlation_id()
    request_id = get_request_id()
    start = time.time()

    try:
        import ml_pipeline_monitor.services.model_service as _model_service
        result = _model_service.predict_from_payload(payload=request_body.features, dataset=request_body.dataset)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"message": str(exc)}) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail={"message": str(exc)}) from exc
    except Exception as exc:
        LOGGER.warning("Legacy predict failed: %s", exc)
        raise HTTPException(status_code=500, detail={"message": f"Prediction failed: {exc}"}) from exc

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


# ---------------------------------------------------------------------------
# Alertmanager Webhook Receiver
# ---------------------------------------------------------------------------

@app.post("/v1/alerts/webhook")
async def alertmanager_webhook(request: Request) -> dict[str, Any]:
    """Receive alertmanager webhook notifications and log them."""
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON body") from exc

    if not isinstance(body, dict):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Expected a JSON object")

    alerts = body.get("alerts", []) or []
    group_status = body.get("status", "unknown")

    for alert in alerts:
        alert_name = alert.get("labels", {}).get("alertname", "unknown")
        severity = alert.get("labels", {}).get("severity", "unknown")
        summary = alert.get("annotations", {}).get("summary", "")
        description = alert.get("annotations", {}).get("description", "")
        alert_status = alert.get("status", group_status)

        if alert_status == "resolved":
            LOGGER.info(
                "Alert resolved: %s (severity=%s) - %s",
                alert_name, severity, summary,
            )
        else:
            LOGGER.warning(
                "Alert firing: %s (severity=%s) - %s | %s",
                alert_name, severity, summary, description,
            )

    return {"status": "ok", "alerts_received": len(alerts)}
