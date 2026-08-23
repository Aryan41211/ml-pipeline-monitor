# Multi-stage Dockerfile for ML Pipeline Monitor
# Production-ready with security hardening, non-root user, and optimized layers

# =============================================================================
# Stage 1: Runtime base - only what is needed to RUN the app
# =============================================================================
FROM python:3.12-slim AS base

# Security: Create non-root user
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} appuser && \
    useradd -m -u ${UID} -g ${GID} -s /bin/bash appuser

# Runtime-only system packages. Compilers and headers deliberately live in the
# builder stage below: shipping build-essential in the final image adds ~250MB
# and a large amount of attack surface for no runtime benefit.
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        libpq5 \
        curl \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /app

# =============================================================================
# Stage 2: Builder - compiles wheels, never shipped
# =============================================================================
FROM base AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-dev.txt ./

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Stage 3: Development image with hot reload
# =============================================================================
FROM builder AS development

RUN pip install --no-cache-dir -r requirements-dev.txt

# Install Playwright browsers for e2e tests
RUN playwright install --with-deps chromium

COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8501 8000 8502

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["python", "-m", "ml_pipeline_monitor.ui", "--server.port=8501", "--server.address=0.0.0.0", "--server.runOnSave=true"]

# =============================================================================
# Stage 4: Production image (Streamlit UI) - minimal and secure
# =============================================================================
FROM base AS production

# Only the installed packages come across, not the toolchain that built them.
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser pages/ ./pages/
COPY --chown=appuser:appuser app.py ./
COPY --chown=appuser:appuser config/ ./config/
COPY --chown=appuser:appuser run_app.py ./
COPY --chown=appuser:appuser LICENSE ./

RUN mkdir -p /app/artifacts/models /app/artifacts/scalers /app/logs /app/data /app/artifacts/feature_store && \
    chown -R appuser:appuser /app

USER appuser

# 8501 = Streamlit UI, 8502 = Prometheus exporter started by app.py
EXPOSE 8501 8502

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Launched through the package so the Prometheus exporter starts with the
# container rather than on the first browser session.
CMD ["python", "-m", "ml_pipeline_monitor.ui", "--server.port=8501", "--server.address=0.0.0.0"]

# =============================================================================
# Stage 5: API-only production image
# =============================================================================
FROM production AS api

EXPOSE 8000

# The inherited healthcheck probed Streamlit on 8501, which this image does not
# serve, so the container would have reported unhealthy forever if compose did
# not happen to override it. Probe the API's own readiness endpoint instead.
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health/ready || exit 1

CMD ["uvicorn", "ml_pipeline_monitor.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# =============================================================================
# Stage 6: Worker image (for background jobs)
# =============================================================================
FROM production AS worker

# The worker serves no HTTP port, so liveness is "PID 1 is still our loop".
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import os, sys; sys.exit(0 if os.path.exists('/proc/1') else 1)" || exit 1

CMD ["python", "-m", "ml_pipeline_monitor.services.worker"]
