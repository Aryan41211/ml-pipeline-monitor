# Pre-Launch Checklist

## Verified in this repository

- [x] Lint gate green: `ruff`, `black --check`, `isort --check-only` all pass
- [x] 254 unit + integration tests pass; coverage 81% (gate is 80%)
- [x] 27 Playwright E2E tests pass against a real browser
- [x] `bandit -r src --severity-level medium` exits 0
- [x] `pip-audit --requirement requirements.txt --strict` finds no known vulnerabilities

## Verified against a live Docker daemon

- [x] `docker compose config` clean for base, prod, dev overlay, and flower overlay
- [x] `promtool check rules` — 14 rules valid
- [x] `promtool check config` — prometheus.yml valid
- [x] `amtool check-config` — alertmanager.yml valid (it was rejected before; see below)
- [x] `docker compose build` succeeds for `production`, `api`, and `worker`
- [x] Images shrank 2.42GB → 1.98GB after moving the toolchain to a builder stage
- [x] `docker compose up -d` — all 9 services report healthy
- [x] app, api and worker share **one** Postgres: the worker writes a row and both
      the api and the app read it back
- [x] All three Prometheus targets report `up`, with no browser session open
- [x] Prometheus is wired to Alertmanager, and a test alert is delivered to the
      API's `/v1/alerts/webhook`
- [x] `/health/ready` returns 503 with Postgres stopped, and recovers to 200
- [x] End to end: train → promote → authenticated `/v1/predict` returns a real
      prediction with probabilities
- [x] Prediction history persists to the database
- [x] Full stack restart (`down` then `up -d`) returns every service to healthy,
      and data written before the restart is still present
- [x] `.streamlit/config.toml` is present in the built image and applied (the
      developer toolbar is hidden and widgets use the project palette)

## Before going live

- [ ] Create `.env` from `.env.example` and fill every value marked REQUIRED.
      Keep comments on their own lines.
- [ ] Set `AUTH_PASSWORD` to a **bcrypt hash**, not a plaintext password:
      `python -c "import bcrypt;print(bcrypt.hashpw(b'PASSWORD', bcrypt.gensalt(12)).decode())"`
- [ ] Rotate every secret that has been committed, shared, or used in testing.
      The `.env` in this working tree holds development values.
- [ ] Terminate TLS: set `DOMAIN` and `CERTBOT_EMAIL` in `.env`, point DNS at
      the host, then run `./scripts/deployment/enable-tls.sh`. Rehearse with
      `CERTBOT_STAGING=true` first. See docs/DEPLOYMENT.md.
      The config, certbot service, redirect and renewal loop are in place and
      validated; only the real certificate is missing, since issuance needs a
      public domain.
- [ ] Set `api.cors_origins` (or `MLMONITOR_CORS_ORIGINS`) only if a browser
      front-end on another origin calls the API. Leave it empty otherwise.
- [ ] Decide on `MLMONITOR_API_KEY`. Leaving it blank disables the deprecated
      `/predict` endpoint entirely, which is the safer default.
- [ ] Point Alertmanager at a real receiver. It currently posts only to the
      API's webhook, which logs the alert; no email or Slack is delivered.
- [ ] Back up the `pgdata` volume and `./artifacts`.
- [ ] Review the Grafana dashboards against the metrics that are actually
      recorded; they were authored before several metrics were wired up.

## Known limitations (deliberate, documented)

- The background worker is a polling loop, not Celery. Flower ships as an
  opt-in overlay (`docker-compose.flower.yml`) for a future Celery worker and
  has no tasks to display today.
- `_claim_due_schedules()` is not atomic, so run exactly one worker replica
  until schedule claiming uses a database-level lock.
- Email alerting is simulated to a local JSON-lines file; SMTP settings are
  read from config but nothing sends mail.
- The Governance "Apply Thresholds" button scopes to the session only; edit
  `config/config.prod.yaml` to persist policy thresholds.
- The API and the Streamlit app maintain separate Prometheus registries, since
  they are separate processes. The worker records metrics but is not scraped.
- The dashboard login lives in Streamlit's session state, so a **full browser
  refresh signs the user out**. Navigating with the sidebar links stays in the
  session. Surviving a reload would require cookie- or token-backed sessions,
  which Streamlit does not provide natively.
