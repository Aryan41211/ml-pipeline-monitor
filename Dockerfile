# Multi-stage Dockerfile for ML Pipeline Monitor
# Production-ready with security hardening, non-root user, and optimized layers

# =============================================================================
# Stage 1: Base dependencies and system packages
# =============================================================================
FROM python:3.10-slim AS base

# Security: Create non-root user
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} appuser && \
    useradd -m -u ${UID} -g ${GID} -s /bin/bash appuser

# Install system dependencies with security updates
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    ca-certificates \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set Python environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# =============================================================================
# Stage 2: Python dependencies (cached layer)
# =============================================================================
FROM base AS dependencies

# Copy requirements first for better layer caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Stage 3: Development image with hot reload
# =============================================================================
FROM dependencies AS development

# Install development dependencies
RUN pip install --no-cache-dir -r requirements-dev.txt

# Install Playwright browsers for e2e tests
RUN playwright install --with-deps chromium

# Copy source code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Expose ports
EXPOSE 8501 8000

# Development command with hot reload
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.runOnSave=true"]

# =============================================================================
# Stage 4: Production image - minimal and secure
# =============================================================================
FROM base AS production

# Copy only production dependencies
COPY --from=dependencies /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser services/ ./services/
COPY --chown=appuser:appuser pages/ ./pages/
COPY --chown=appuser:appuser app.py ./
COPY --chown=appuser:appuser config.yaml ./
COPY --chown=appuser:appuser config.prod.yaml ./
COPY --chown=appuser:appuser run_app.py ./
COPY --chown=appuser:appuser LICENSE ./

# Create necessary directories with proper permissions
RUN mkdir -p /app/artifacts/models /app/artifacts/scalers /app/logs /app/data /app/artifacts/feature_store && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Expose ports
EXPOSE 8501 8000

# Default production command
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]

# =============================================================================
# Stage 5: API-only production image (for separate API service)
# =============================================================================
FROM production AS api

# Override command for API service
CMD ["uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# =============================================================================
# Stage 6: Worker image (for background jobs)
# =============================================================================
FROM production AS worker

# Override command for background worker
CMD ["python", "-m", "services.worker"]