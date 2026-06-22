# Operations Runbook

## Incident Response

### Severity Levels
| Level | Description | Response Time |
|---|---|---|
| P1 | Production down, no inference | 15 minutes |
| P2 | Major feature degraded | 1 hour |
| P3 | Minor issue, workaround available | 4 hours |
| P4 | Enhancement / bug | Next sprint |

### Common Incidents

#### 1. API returning 503
**Symptoms**: `/health/ready` returns `not_ready`, predictions failing  
**Diagnosis**:
```bash
curl -s http://localhost:8000/health/detailed | jq .
```
**Resolution**:
- Check database connectivity (`psql $PIPELINE_DB_DSN -c "SELECT 1"`)
- Restart API pod/service
- Check Prometheus for error spikes

#### 2. Model predictions failing
**Symptoms**: 500 errors on `/v1/predict`  
**Diagnosis**:
```bash
# Check logs
docker-compose logs api | grep -i error

# Verify model artifacts exist
ls -la artifacts/models/
```
**Resolution**:
- Re-promote last known good model
- Re-run pipeline for affected dataset

#### 3. Database full
**Symptoms**: Insert failures, connection errors  
**Diagnosis**:
```bash
psql $PIPELINE_DB_DSN -c "SELECT pg_size_pretty(pg_database_size(current_database()));"
```
**Resolution**:
- Run backup immediately
- Purge old predictions/logs per retention policy
- Scale up storage

#### 4. Drift alerts firing
**Symptoms**: Alerts in #ml-pipeline-monitor Slack channel  
**Diagnosis**:
```bash
# Check latest drift reports
curl -s http://localhost:8000/health/detailed
```
**Resolution**:
- Evaluate if retraining is needed
- Update drift thresholds if false positive
- Archive old drift references

## Routine Operations

### Daily
- [ ] Verify backup completion
- [ ] Check alert queue for new incidents
- [ ] Review Grafana dashboards for anomalies

### Weekly
- [ ] Review and archive old drift references
- [ ] Clean up old predictions (> 30 days)
- [ ] Verify security scan results

### Monthly
- [ ] Rotate JWT secrets (requires service restart)
- [ ] Review and update retention policies
- [ ] Test backup restoration procedure
- [ ] Update dependencies (security patches first)

### Quarterly
- [ ] Load test API endpoints
- [ ] Review and update runbooks
- [ ] Audit user access and API keys

## Useful Commands

```bash
# View logs
docker-compose logs -f api
docker-compose logs -f worker

# Database backup
python -m scripts.backup backup postgres mlmonitor --dsn "$PIPELINE_DB_DSN"

# Run migrations
docker-compose exec app alembic upgrade head

# Check Prometheus targets
curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets'

# Trigger manual pipeline run
curl -X POST http://localhost:8000/v1/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"trigger":"manual"}'

# Celery worker status
docker-compose exec worker celery -A services.worker inspect active
```

## Performance Baselines

| Metric | Target | Alert Threshold |
|---|---|---|
| API latency (p99) | < 200ms | > 500ms |
| Pipeline duration | < 5 min | > 15 min |
| DB connection usage | < 70% | > 90% |
| Memory usage | < 80% | > 90% |
| Disk usage | < 80% | > 90% |

## Escalation

1. **On-call engineer** — first responder
2. **MLOps team lead** — if unresolved after 30 min (P1/P2)
3. **Engineering manager** — for customer-facing impact

## Contact

- **Slack**: #ml-pipeline-monitor
- **Email**: mlops-oncall@example.com
- **PagerDuty**: ml-pipeline-monitor service
