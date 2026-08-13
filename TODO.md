# Production Deployment Verification - Task List

## Fixes
- [x] 1. Fix `docker-compose.yml` environment blocks (mixed merge+list YAML error)
- [x] 2. Create `.dockerignore`
- [x] 3. Fix `config_loader.py` to respect CONFIG_PATH env var
- [x] 4. Fix `alertmanager.yml` env substitution + webhook URL
- [x] 5. Fix `docker-compose.prod.yml` remove Swarm-only directives
- [x] 5a. Rewrite `docker-compose.prod.yml` as standalone with all 8 services
- [x] 5b. Fix `schema.py` COALESCE casts for PostgreSQL TEXT columns
- [x] 5c. Fix `connection.py` closeall() -> close() (psycopg_pool API)
- [x] 5d. Add retry logic for PostgreSQL schema initialization
- [x] 5e. Reduce uvicorn workers to 1 (prevent PG init deadlocks)
- [x] 5f. Add `psycopg[binary]` to requirements.txt
- [x] 5g. Remove optional service upstreams from nginx config
- [x] 5h. Add worker healthcheck (os.kill via Python)

## Validation & Build
- [x] 6. Validate compose config (`docker compose config`)
- [x] 6a. Create `.env` / `.env.example` with production variables
- [x] 6b. Fix xgboost pin to avoid NVIDIA CUDA dependency (pinned to <2.1.0)
- [x] 7. Build all images from scratch (`docker compose build --no-cache`)

## Deployment & Verification
- [x] 8. Start the production stack (`docker compose up -d`)
- [x] 9. Verify all services healthy (8/8 healthy)
- [x] 10. Verify database + migrations (5 core tables created)
- [x] 11. Verify API health endpoints (/health, /health/ready, /health/detailed)
- [x] 12. Verify Streamlit (200 OK, 10951 bytes HTML)
- [x] 13. Verify Worker + Redis (Redis PONG, worker polling with concurrency=4)
- [x] 14. Verify Prometheus scrape (targets up, ml_api_requests_total metrics)
- [x] 15. Verify Grafana (Prometheus datasource + 4 dashboards provisioned)
- [x] 16. Verify Nginx routing (200 OK through port 80)
- [x] 17. Verify inter-service networking (DNS resolution working)
- [x] 18. Verify environment variables (PIPELINE_DB_DSN, JWT_SECRET, etc.)
- [x] 19. Security check (auth enforced, no secrets in logs, non-root user)
- [x] 20. Resource/reliability check (150MB RSS, 18 threads)
- [x] 21. Restart/shutdown recovery test (down + up, all 8 healthy after restart)
- [x] 22. Run application tests (100/100 unit tests passed)

## Cleanup & Report
- [x] 23. Final cleanup (all committed, clean git status)
- [x] 24. Commit fixes (5 atomic commits)
- [x] 25. Final report
