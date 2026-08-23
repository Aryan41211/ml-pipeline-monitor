# Pre-Launch Checklist

Everything below is verified in the repository unless marked otherwise.

## Verified in this repo

- [x] Lint gate green: `ruff`, `black --check`, `isort --check-only` all pass
- [x] 254 unit + integration tests pass; coverage 82% (gate is 80%)
- [x] 27 Playwright E2E tests pass against a real browser
- [x] `bandit -r src --severity-level medium` exits 0
- [x] `pip-audit --requirement requirements.txt --strict` reports no known vulnerabilities
- [x] Prometheus alert rules present and structurally validated; every metric
      they reference is actually recorded by `core/metrics.py`
- [x] All YAML (compose, prometheus, alertmanager, config) parses

## Requires a Docker daemon (NOT yet run — Docker was unavailable in this environment)

These are the remaining gates before a production deploy. Run them on a host
with Docker running:

- [ ] `docker compose -f docker-compose.yml config --quiet`
- [ ] `docker compose -f docker-compose.prod.yml config --quiet`
- [ ] `promtool check rules deployment/prometheus/rules/ml-pipeline-monitor.yml`
      (the CI `config` job does this; it validates PromQL, which the structural
      check in Python cannot)
- [ ] `amtool check-config deployment/alertmanager/alertmanager.yml`
- [ ] `docker compose build` for the `production`, `api`, and `worker` targets
      (the Dockerfile was restructured into a builder stage; the build itself
      has not been executed)
- [ ] `docker compose up -d` and confirm every service reports healthy
- [ ] Confirm app / api / worker all see the **same** Postgres database
- [ ] Confirm `/health/ready` returns 503 when Postgres is stopped
- [ ] Confirm a fired alert reaches `POST /v1/alerts/webhook`

## Before going live

- [ ] Create `.env` from `.env.example` and fill every value marked REQUIRED.
      Keep comments on their own lines.
- [ ] Set `AUTH_PASSWORD` to a **bcrypt hash**, not a plaintext password:
      `python -c "import bcrypt;print(bcrypt.hashpw(b'PASSWORD', bcrypt.gensalt(12)).decode())"`
- [ ] Rotate any secret that has ever been committed or shared.
- [ ] Terminate TLS. `deployment/nginx/conf.d/ml-monitor.conf` listens on :80
      only; the HTTPS server block and the HSTS header in `nginx.conf` are
      commented out pending a certificate.
- [ ] Set `api.cors_origins` (or `MLMONITOR_CORS_ORIGINS`) only if a browser
      front-end on another origin calls the API. Leave empty otherwise.
- [ ] Decide on `MLMONITOR_API_KEY`. Leaving it blank disables the deprecated
      `/predict` endpoint entirely, which is the safer default.
- [ ] Configure a real Alertmanager receiver. It currently posts only to the
      API's webhook, which logs the alert; no email or Slack is delivered.
- [ ] Back up the Postgres volume (`pgdata`) and `./artifacts`.

## Known limitations (deliberate, documented)

- The background worker is a polling loop, not Celery. Flower is included
  behind the `monitoring` profile but has no tasks to show.
- `_claim_due_schedules()` is not atomic, so run exactly one worker replica
  until schedule claiming uses a database-level lock.
- Email alerting is simulated to a local JSON-lines file; SMTP is configured
  but not wired to a sender.
- The Governance "Apply Thresholds" button scopes to the session only; edit
  `config/config.prod.yaml` to persist policy thresholds.
