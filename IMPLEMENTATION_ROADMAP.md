# IMPLEMENTATION_ROADMAP.md — ML Pipeline Monitor

**Project:** ML Pipeline Monitor  
**Roadmap date:** 2026-11-24

## Completed

| Task | Status | Description |
|---|---|---|
| Fix 18 failing API tests | DONE | Body parameter detection fix |
| Fix test isolation issue | DONE | Accept 429 for rate-limited login |
| Add drift detector unit tests | DONE | 19 tests, 10 controlled scenarios |
| Validate all 12 algorithms | DONE | All execute successfully |
| Verify PSI/KS correctness | DONE | Against controlled distributions |
| Validate Docker Compose config | DONE | `docker compose config --quiet` passes |

## In Progress

| Task | Status | Description |
|---|---|---|
| Measure `/predict` API latency | PLANNED | Requires auth + model setup |
| Docker health verification | BLOCKED | Docker Desktop unavailable |
| Measure drift detection time | PLANNED | Standalone timing |

## Not Started

| Task | Status |
|---|---|
| CI/CD pipeline | NOT STARTED |
| Background job failure rate testing | NOT STARTED |
| Model governance testing | NOT STARTED |
| Observability dashboards | NOT STARTED |

## Not Applicable

| Item | Reason |
|---|---|
| Celery integration | Polling worker used instead (architectural decision) |

## Acceptance Criteria

| Criterion | Status |
|---|---|
| All critical failing tests resolved | DONE |
| Algorithm execution validated | DONE |
| Drift correctness verified | DONE |
| Background processing reliability | PARTIAL |
| Registry transitions tested | NOT STARTED |
| Monitoring data verified | NOT STARTED |
| Docker deployment verified | BLOCKED |
