# Production Deployment Guide

## Overview

This guide covers deploying ML Pipeline Monitor to a production Kubernetes or Docker environment.

## Prerequisites

- Docker 24+ and Docker Compose 2.20+
- Kubernetes 1.27+ (optional, for K8s deployment)
- PostgreSQL 16+ (for production data)
- A domain name with TLS certificate (or use Let's Encrypt)

> **Note on migrations.** The application creates and migrates its own schema
> at startup: both the API and the worker call `initialize_db()`, which issues
> idempotent `CREATE TABLE IF NOT EXISTS` / `ensure_column_exists` statements.
> The `alembic/` directory is present but **nothing invokes it at runtime**, and
> its two revisions are not the source of truth. Do not run `alembic upgrade
> head` as a deploy step -- it is not required, and it has not been kept in sync
> with `database/schema.py`.

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
docker compose up -d   # Postgres is a default service, not a profile

# With monitoring stack
docker compose -f docker-compose.yml -f docker-compose.flower.yml up -d   # optional Flower

# Full production stack
docker compose -f docker compose.yml -f docker compose.prod.yml up -d
```

### 3. Run Migrations
```bash
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
- `Deployment` for the polling worker
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
# In docker compose.prod.yml nginx section
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
- **Worker**: container healthcheck only (the polling worker exposes no HTTP port). Flower at `:5555` is for a future Celery-backed worker and shows nothing today.

## Scaling

| Component | Horizontal Scaling | Notes |
|---|---|---|
| App (Streamlit) | Limited | Use session affinity |
| API (FastAPI) | Yes | Behind load balancer |
| Worker | Yes | Schedules are claimed atomically, so replicas will not run the same job twice |
| PostgreSQL | Read replicas | Use managed service recommended |

## Rollback Procedure

```bash
# 1. Identify previous version
docker images | grep ml-pipeline-monitor

# 2. Rollback deployment
docker compose up -d --force-recreate app

# 3. Run migrations (if needed)
# (schema is managed by initialize_db(); restore from backup to roll back)
```

## Disaster Recovery

1. **Database**: Restore from latest backup
2. **Artifacts**: Persisted on PVC or S3-compatible storage
3. **Configuration**: Git-versioned, re-apply from repo
4. **Secrets**: Re-inject from secrets manager

## TLS / HTTPS

The stack ships plain HTTP and switches to HTTPS in one scripted step, because
the ordering is a chicken-and-egg: nginx refuses to start with an
`ssl_certificate` that does not exist yet, and Let's Encrypt cannot validate
the domain unless nginx is already answering on port 80.

### Prerequisites

1. A public DNS A/AAAA record for your domain pointing at this host.
2. Inbound port 80 reachable from the internet (the ACME HTTP-01 challenge).
3. `DOMAIN` and `CERTBOT_EMAIL` set in `.env`.

### Issue the certificate

```bash
# Rehearse first: production issuance is rate limited to 5 failures per
# hostname per hour, and a misconfigured attempt burns that budget.
echo "CERTBOT_STAGING=true" >> .env
./scripts/deployment/enable-tls.sh

# Happy with the result? Switch to a real certificate.
sed -i 's/CERTBOT_STAGING=true/CERTBOT_STAGING=false/' .env
./scripts/deployment/enable-tls.sh
```

The script starts nginx on :80, requests the certificate through the ACME
webroot, renders `tls-available/ml-monitor-tls.conf.template` with your domain
into `conf.d/ml-monitor.conf`, validates it with `nginx -t`, and reloads. If
validation fails it restores the HTTP-only config, so a bad render cannot take
the site down.

### Renewal

Automatic. The `certbot` service runs `certbot renew` every 12 hours (a no-op
until the certificate is within 30 days of expiry) and nginx reloads every 6
hours to pick up a renewed certificate. Port 80 keeps serving
`/.well-known/acme-challenge/` after the switch to HTTPS specifically so
renewals keep working.

Check status:

```bash
docker compose -f docker-compose.prod.yml logs certbot | tail -20
docker compose -f docker-compose.prod.yml run --rm certbot certificates
```

### Security headers and HSTS

`deployment/nginx/conf.d/security-headers.inc` holds the shared header set.
**nginx inherits `add_header` from an outer level only if the current level
declares none of its own**, so any server or location that adds a header must
re-include that file or it silently drops the rest. This is why the TLS
template includes it three times.

HSTS starts at `max-age=86400` (1 day). Raise it in the template only once you
have seen a renewal succeed: a long max-age plus a broken certificate locks
users out of the site, and browsers honour it even after you revert.

### Certificates are not in git

`deployment/nginx/letsencrypt/` is gitignored. Back it up separately: losing it
means re-issuing, which is subject to the same rate limits.


## Monitoring on Call

- Prometheus: http://prometheus:9090
- Grafana: http://grafana:3000 (admin / configured password)
- Alertmanager: http://alertmanager:9093
- Flower: http://flower:5555

Alert channels:
- Email (SMTP configured in alertmanager.yml)
- Slack (#ml-pipeline-monitor channel)
- Webhook to app (`http://app:8501/alert`)
