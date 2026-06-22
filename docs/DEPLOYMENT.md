# Production Deployment Guide

## Overview

This guide covers deploying ML Pipeline Monitor to a production Kubernetes or Docker environment.

## Prerequisites

- Docker 24+ and Docker Compose 2.20+
- Kubernetes 1.27+ (optional, for K8s deployment)
- PostgreSQL 16+ (for production data)
- A domain name with TLS certificate (or use Let's Encrypt)

## Environment Variables

### Required
| Variable | Description |
|---|---|
| `CONFIG_PATH` | Path to config file (default: `config.prod.yaml`) |
| `PIPELINE_DB_DSN` | PostgreSQL DSN (required for Postgres backend) |

### Optional
| Variable | Description |
|---|---|
| `JWT_SECRET` | JWT signing secret (generate with `openssl rand -hex 32`) |
| `JWT_ALGORITHM` | JWT algorithm (default: `HS256`) |
| `JWT_EXPIRATION_MINUTES` | Access token TTL (default: `60`) |
| `JWT_REFRESH_EXPIRATION_DAYS` | Refresh token TTL (default: `7`) |
| `MLMONITOR_API_KEY` | Legacy API key for `/predict` endpoint |
| `MLMONITOR_RATE_LIMIT` | Rate limit string (default: `60/minute`) |
| `SMTP_HOST` | SMTP server for email alerts |
| `SMTP_PORT` | SMTP port |
| `SLACK_WEBHOOK` | Slack webhook URL for alerts |
| `MLFLOW_TRACKING_URI` | MLflow tracking server URI |

## Docker Compose Deployment

### 1. Configure Environment
```bash
# Copy production config
cp config.prod.yaml .env

# Set secrets
export JWT_SECRET=$(openssl rand -hex 32)
export PIPELINE_DB_DSN="postgresql://mlmonitor:securepass@postgres:5432/mlmonitor"
export MLMONITOR_API_KEY=$(openssl rand -hex 32)
```

### 2. Start Services
```bash
# With PostgreSQL
docker-compose --profile postgres up -d

# With monitoring stack
docker-compose --profile monitoring up -d

# Full production stack
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### 3. Run Migrations
```bash
docker-compose exec app alembic upgrade head
```

### 4. Verify
```bash
# Health check
curl https://your-domain.com/health

# API health
curl https://your-domain.com/v1/health

# Metrics
curl https://your-domain.com/metrics
```

## Kubernetes Deployment

### Namespace
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ml-pipeline-monitor
```

### Secrets
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: mlmonitor-secrets
  namespace: ml-pipeline-monitor
type: Opaque
stringData:
  jwt-secret: <generated-secret>
  api-key: <generated-api-key>
  db-dsn: "postgresql://..."
```

### Key Resources
- `Deployment` for app (Streamlit)
- `Deployment` for API (FastAPI)
- `Deployment` for worker (Celery)
- `Service` for each component
- `Ingress` with TLS
- `StatefulSet` for PostgreSQL (or use managed DB)
- `PersistentVolumeClaim` for artifacts and logs

## Database Setup

### PostgreSQL
```sql
-- Run scripts/init-db.sql
CREATE DATABASE mlmonitor;
\c mlmonitor
\i scripts/init-db.sql
```

### Configure Connection Pooling
```yaml
# config.prod.yaml
storage:
  backend: postgres
  postgres_dsn: "${PIPELINE_DB_DSN}"
  connection_pool:
    min_size: 5
    max_size: 20
```

## SSL/TLS

### Using Let's Encrypt
```yaml
# In docker-compose.prod.yml nginx section
certbot:
  image: certbot/certbot
  volumes:
    - ./certs:/etc/letsencrypt
  command: certonly --standalone -d your-domain.com
```

## Backup Strategy

### Automated Daily Backups
```bash
# Add to crontab
0 2 * * * cd /opt/ml-pipeline-monitor && python -m scripts.backup backup postgres mlmonitor --dsn "$PIPELINE_DB_DSN" --output-dir /backups
```

### Retention Policy
- Daily backups: 7 days
- Weekly backups: 4 weeks
- Monthly backups: 12 months

## Health Checks

All services expose health endpoints:
- **App**: `GET /health` (Streamlit)
- **API**: `GET /health/live`, `GET /health/ready`
- **Worker**: Celery Flower at `:5555`

## Scaling

| Component | Horizontal Scaling | Notes |
|---|---|---|
| App (Streamlit) | Limited | Use session affinity |
| API (FastAPI) | Yes | Behind load balancer |
| Worker (Celery) | Yes | Add more worker replicas |
| PostgreSQL | Read replicas | Use managed service recommended |

## Rollback Procedure

```bash
# 1. Identify previous version
docker images | grep ml-pipeline-monitor

# 2. Rollback deployment
docker-compose up -d --force-recreate app

# 3. Run migrations (if needed)
docker-compose exec app alembic downgrade -1
```

## Disaster Recovery

1. **Database**: Restore from latest backup
2. **Artifacts**: Persisted on PVC or S3-compatible storage
3. **Configuration**: Git-versioned, re-apply from repo
4. **Secrets**: Re-inject from secrets manager

## Monitoring on Call

- Prometheus: http://prometheus:9090
- Grafana: http://grafana:3000 (admin / configured password)
- Alertmanager: http://alertmanager:9093
- Flower: http://flower:5555

Alert channels:
- Email (SMTP configured in alertmanager.yml)
- Slack (#ml-pipeline-monitor channel)
- Webhook to app (`http://app:8501/alert`)
