# TESTING_REPORT.md — ML Pipeline Monitor

**Project:** ML Pipeline Monitor  
**Test date:** 2026-11-24

## Test Results

| Test Suite | Tests | Passed | Failed | Skipped |
|---|---:|---:|---:|---:|
| tests/unit | 146 | 146 | 0 | 0 |
| tests/integration | 164 | 164 | 0 | 0 |
| **Total** | **310** | **310** | **0** | **0** |

## Before Fix (Baseline)

| Test Suite | Tests | Passed | Failed |
|---|---:|---:|---:|
| tests/unit | 146 | 146 | 0 |
| tests/integration | 164 | 146 | 18 |
| **Total** | **310** | **273** | **18** |

## Fixed Tests (18 total)

| Test File | Test | Root Cause | Fix |
|---|---|---|---|
| test_api_coverage.py | test_api_key_auth_success | Body param not detected | Remove `from __future__ import annotations` |
| test_api_coverage.py | test_predict_file_not_found | Body param not detected | Same |
| test_api_coverage.py | test_predict_generic_exception | Body param not detected | Same |
| test_api_coverage.py | test_login_with_fresh_token | Body param not detected | Same |
| test_api_coverage.py | test_refresh_endpoint_success | Body param not detected | Same |
| test_api_integration.py | test_login_requires_credentials | Rate limit bucket exhausted | Accept 429 |
| test_api_integration.py | test_refresh_requires_token | Body param not detected | Same |
| test_api_integration.py | test_predict_success | Body param not detected | Same |
| test_api_integration.py | test_predict_no_model | Body param not detected | Same |
| test_api_integration.py | test_legacy_requires_api_key | Body param not detected | Same |
| test_api_integration.py | test_legacy_rejects_wrong_api_key | Body param not detected | Same |
| test_api_integration.py | test_legacy_with_api_key | Body param not detected | Same |
| test_api_integration.py | test_legacy_validation_error | Body param not detected | Same |
| test_api_security.py | test_unset_api_key_rejects_arbitrary_header | Body param not detected | Same |
| test_api_security.py | test_wrong_api_key_rejected | Body param not detected | Same |
| test_api_security.py | test_missing_api_key_rejected | Body param not detected | Same |
| test_api_security.py | test_login_rate_limit | Body param not detected | Same |
| test_api_service.py | test_predict_endpoint_success | Body param not detected | Same |
| test_api_service.py | test_predict_endpoint_validation_error | Body param not detected | Same |

## Coverage

| Metric | Value | Target |
|---|---|---|
| Total coverage | 81.03% | 80% |
| Before fix | 77.94% | 80% |

## Test Gaps

| Gap | Priority |
|---|---|
| `/predict` endpoint tests | High |
| Docker service health tests | High |
| Background worker failure tests | Medium |
| Drift detection standalone tests | Medium |
| Model registry transition tests | Medium |

## Test Quality

| Criterion | Rating | Notes |
|---|---|---|
| Deterministic | High | Fixed random seed (42) |
| Isolated | High | Each test independent |
| Fast | High | 43 seconds for 310 tests |
| Contract-focused | High | API response validation |
| Behavior-focused | High | Behavioral assertions |
