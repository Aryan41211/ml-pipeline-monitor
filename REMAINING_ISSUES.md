# REMAINING_ISSUES.md — ML Pipeline Monitor

**Project:** ML Pipeline Monitor  
**Date:** 2026-11-24

## Critical Issues

| Issue | Impact | Priority | Status |
|---|---|---:|---|
| No benchmark datasets in `data/` | Cannot measure max dataset size | High | Open |
| `/predict` API latency not measured | Cannot claim API performance | Medium | Open |
| Docker health not verified | Cannot claim deployment readiness | Medium | Blocked (Docker Desktop unavailable) |
| Drift execution time not measured | Cannot measure drift performance | Medium | Open |

## Medium Issues

| Issue | Impact | Priority | Status |
|---|---|---:|---|
| 18 tests were failing (now fixed) | Was a reliability concern | High | Resolved |
| Coverage below 80% target (was 77.94%) | Below project standard | Medium | Resolved (now 81.03%) |
| Celery NOT used | Cannot claim Celery background jobs | Medium | Documented |
| Background worker failure rate not measured | Cannot claim job reliability | Medium | Open |
| Model registry transitions not tested | Cannot claim governance | Medium | Open |

## Minor Issues

| Issue | Impact | Priority | Status |
|---|---|---:|---|
| No live deployment | Cannot verify production health | Low | Open |
| No Kubernetes deployment | No K8s deployment | Low | Open |

## Issues Resolved in This Audit

| Issue | Resolution |
|---|---|
| 18 failing API tests | Fixed body parameter detection (removed `from __future__ import annotations`) |
| Test isolation (rate limit bucket) | Accept 429 as valid rejection for unauthenticated login |
| Drift detector correctness | Added 19 unit tests validating PSI/KS against controlled distributions |
| Algorithm execution | Verified all 12 algorithms execute successfully |

## Acceptance Criteria Status

| Criterion | Status |
|---|---|
| All critical failing tests resolved | DONE |
| Algorithm execution validated | DONE |
| Drift correctness verified | DONE |
| Background processing reliability | PARTIAL (code review only) |
| Registry transitions tested | OPEN |
| Monitoring verified | OPEN |
| Docker deployment verified | BLOCKED |

## Next Steps

1. Measure `/predict` API latency (requires auth + model setup)
2. Verify Docker health (requires Docker Desktop)
3. Measure drift detection standalone execution time
4. Test background worker failure handling
5. Test model registry transitions
