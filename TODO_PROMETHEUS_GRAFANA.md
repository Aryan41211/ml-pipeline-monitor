# TODO — Prometheus + Grafana monitoring

- [ ] Add Prometheus metrics endpoint to FastAPI service (`services/api/app.py`)
  - [ ] Request count + error count
  - [ ] Prediction latency histogram/summary
- [ ] Instrument pipeline execution
  - [ ] Training duration histogram
  - [ ] Pipeline success/failure counters
- [ ] Instrument drift detection
  - [ ] Drift detection counter
- [ ] Instrument system monitoring
  - [ ] CPU/RAM/Disk usage gauges (process + host)
- [ ] Add dashboards JSON (Grafana) under `monitoring/grafana/dashboards/`
  - [ ] Pipeline Metrics
  - [ ] API Metrics
  - [ ] System Health
  - [ ] Drift Monitoring
- [ ] Add Prometheus config under `monitoring/prometheus/prometheus.yml`
- [ ] Update `docker-compose.yml` with Prometheus + Grafana services + volumes
- [ ] Update `requirements.txt` with `prometheus-client` and any Grafana tooling as needed
- [ ] Add/update tests (unit tests around metric registration + update functions)
- [ ] Run `pytest -q` and fix failures
- [ ] Commit: `feat: add prometheus and grafana monitoring`
- [ ] Push changes to GitHub

