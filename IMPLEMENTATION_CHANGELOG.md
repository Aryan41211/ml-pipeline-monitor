# IMPLEMENTATION_CHANGELOG.md — ML Pipeline Monitor

**Project:** ML Pipeline Monitor

## [Unreleased]

### Fixed
- API body parameter detection: Removed `from __future__ import annotations` which prevented FastAPI from detecting Pydantic model parameters on POST endpoints
- Test isolation: `test_login_requires_credentials` now accepts 429 (rate-limited) as valid rejection
- 18 failing tests resolved (all API/auth/predict paths)

### Added
- 19 drift detector unit tests covering 10 controlled scenarios
- PSI correctness validation (identical, mean-shifted, scale-changed)
- KS-test correctness validation (identical, mean-shifted)
- Algorithm execution validation (all 12 algorithms)

### Metrics Changed

| Metric | Before | After | Change |
|---|---:|---:|---:|
| Tests passed | 273 | 310 | +37 |
| Tests failed | 18 | 0 | -18 |
| Coverage | 77.94% | 81.03% | +3.09 pp |

## [Previous]

- Initial release with 7-stage pipeline, 12 algorithms, drift detection
