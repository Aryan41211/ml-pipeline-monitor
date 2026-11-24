# STAGE 0 — DISCOVERY AND BASELINE (CONTEXT SUMMARY)

**Date:** 2026-11-24  
**Stage:** 0 of 6 (Discovery and Baseline)  
**Status:** COMPLETE

## Repositories Located (all on `main`, clean working trees)

| # | Project | Path | Branch | Status |
|---|---------|------|--------|--------|
| 1 | NyaySaathi | `C:\projects\NyaySaathi` | main | Accessible |
| 2 | Retrieval Intelligence Platform | `C:\projects\Retrieval Intelligence Platform` | main | Accessible |
| 3 | ML Pipeline Monitor | `C:\projects\ML-pipeline-monitor` | main | Accessible |
| 4 | EquiPilot AI | `C:\projects\Equipilot AI` | main | Accessible |

*(Pratinik Infotech remains inaccessible — reported in prior audit.)*

## Baseline Test Results

| Project | Passed | Failed | Coverage | Notes |
|---|---:|---:|---:|---|
| NyaySaathi | 3 | 0 | — | 3 API smoke/health tests; model load dominates runtime (67s) |
| ML Pipeline Monitor | 273 | 18 | 77.94% | Below 80% target; 18 failures concentrated in API/auth/predict paths |
| Retrieval Platform | 533 | 0 | — | Full suite green |
| EquiPilot AI | 257 | 3 | — | 3 failures: deploy-config doc checks + frontend connection test (environment-related) |

## ML Pipeline Monitor — 18 Failing Tests (Stage 1 Target)

All 18 failures are in `tests/integration/`:
- `test_api_coverage.py` (7): auth paths, predict error paths, login/refresh
- `test_api_integration.py` (7): auth v1, predict v1, legacy predict
- `test_api_security.py` (3): API key enforcement, login rate limit
- `test_api_service.py` (2): predict endpoint success + validation error

Common theme: API authentication, API key enforcement, `/predict` endpoint, token refresh, and rate limiting. Root causes need reproduction.

## Prior Audit Findings (carried forward)

- ML Pipeline Monitor uses a **polling worker, NOT Celery** (CLAUDE.md §11). Do not claim Celery.
- `data/raw`, `data/sample`, `data/processed` are empty — no benchmark datasets.
- Drift detector (PSI + KS) verified correct vs controlled distributions.
- 7 pipeline stages confirmed; 12 algorithms (6 clf + 6 reg).

## Execution Plan (remaining stages)

1. **Stage 1:** ML Pipeline Monitor — fix 18 failing tests, validate drift engine, Docker health.
2. **Stage 2:** Retrieval Platform — Ragas evaluation, grounding validation, security, load testing.
3. **Stage 3:** NyaySaathi — corpus engineering, retrieval evaluation, ranking quality.
4. **Stage 4:** EquiPilot AI — connect SynthesisAgent, citations, evaluation.
5. **Stage 5:** Cross-project audit + CI consistency.
6. **Stage 6:** Final engineering review + deliverables.

## Git Workflow Note

Per instructions: work directly on `main`, no feature branches, frequent conventional commits after each coherent unit, do not push without approval.
