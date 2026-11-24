# ML Pipeline Monitor — Phase 2 Reproduction Guide

**Project:** ML Pipeline Monitor  
**Git SHA:** `31d2281`  
**Branch:** `main`  
**Date:** 2026-11-24  
**Environment:** 16 CPU cores, 15.65 GB RAM, CPU-only

---

## 1. Environment Setup

```bash
cd C:\projects\ML-pipeline-monitor

# Verify Python version
python --version  # 3.12

# Install dependencies
pip install -r requirements.txt

# Key dependencies:
# - scikit-learn, xgboost, scipy, pandas, numpy
# - FastAPI, SQLAlchemy, Streamlit
# - pytest (for testing)
```

---

## 2. Run Tests

```bash
cd C:\projects\ML-pipeline-monitor

# Run all unit/integration tests (excludes E2E Playwright)
python -m pytest tests -q -p no:cacheprovider --ignore=tests/e2e

# Expected: 310 passed, 0 failed

# Run E2E tests (requires Playwright browser - will error)
python -m pytest tests/e2e -q -p no:cacheprovider

# Expected: 27 errors (no browser available)
```

---

## 3. Algorithm Benchmark

```bash
cd C:\projects\ML-pipeline-monitor

python scripts/mlpm_algorithm_benchmark.py

# Output: experiment_results/mlpm_algorithm_benchmark.json
# CSV: MLPM_MODEL_BENCHMARK.csv
# Runtime: ~2-3 minutes
# Tests: 12 algorithms × 5 datasets × 3 runs
```

### Datasets Used

| Task | Dataset | Samples | Features | Classes |
|------|---------|---------|----------|---------|
| Classification | iris | 150 | 4 | 3 |
| Classification | breast_cancer | 569 | 30 | 2 |
| Classification | wine | 178 | 13 | 3 |
| Regression | diabetes | 442 | 10 | - |
| Regression | california_housing | 20,640 | 8 | - |

---

## 4. Drift Detection Benchmark

```bash
cd C:\projects\ML-pipeline-monitor

python scripts/mlpm_drift_benchmark.py

# Output: experiment_results/mlpm_drift_benchmark.json
# CSV: MLPM_DRIFT_BENCHMARK.csv
# Runtime: ~30 seconds
# Tests: 4 methods × 8 drift types × 5 trials = 160 tests
```

### Drift Scenarios

| Scenario | Description | Expected Drift |
|----------|-------------|----------------|
| no_drift | Identical normal distributions | False |
| small_drift | Mean shift of 0.3 | True |
| moderate_drift | Mean shift of 0.7 | True |
| severe_drift | Mean shift of 1.5 | True |
| scale_change | Variance 1 → 3 | True |
| covariate_shift | Mean shift of 2.0 | True |
| feature_level_drift | Only first feature drifts | True |
| class_imbalance | Class balance 0.5 → 0.8 | True |

---

## 5. Key Metrics Summary

### Algorithm Benchmark (Best Results)

| Task | Dataset | Best Model | Metric | Value |
|------|---------|------------|--------|-------|
| Classification | iris | Gradient Boosting | Accuracy | 0.9667 |
| Classification | breast_cancer | Gradient Boosting | Accuracy | 0.9620 |
| Classification | wine | Random Forest / XGBoost | Accuracy | 0.9907 |
| Regression | diabetes | Ridge Regression | R² | 0.5068 |
| Regression | california_housing | XGBoost | R² | 0.8391 |

### Drift Detection (Best Method)

| Method | Precision | Recall | F1 | FPR |
|--------|-----------|--------|-----|-----|
| KS Test | 1.0 | 1.0 | 1.0 | 0.0 |

---

## 6. Docker (Blocked)

Docker Desktop is not available on this machine. Docker Compose configuration files exist and are syntactically valid:

- docker-compose.yml
- docker-compose.dev.yml
- docker-compose.prod.yml
- docker-compose.flower.yml
- docker-compose.override.yml

**Cannot verify:** Image build, service startup, health checks, integration tests in Docker.

---

## 7. Key Files

| File | Purpose |
|------|---------|
| `src/ml_pipeline_monitor/ml/pipeline.py` | MLPipeline class, 7 stages, 12 algorithms |
| `src/ml_pipeline_monitor/ml/drift_detector.py` | PSI + KS drift detection |
| `src/ml_pipeline_monitor/ml/data_loader.py` | Dataset loading and splitting |
| `scripts/mlpm_algorithm_benchmark.py` | Algorithm benchmark runner |
| `scripts/mlpm_drift_benchmark.py` | Drift detection benchmark runner |
| `experiment_results/mlpm_algorithm_benchmark.json` | Raw algorithm results |
| `experiment_results/mlpm_drift_benchmark.json` | Raw drift results |
| `experiment_results/baseline_snapshot.json` | Baseline environment/metrics |
| `MLPM_MODEL_BENCHMARK.csv` | CSV of algorithm results |
| `MLPM_DRIFT_BENCHMARK.csv` | CSV of drift results |
| `MLPM_PIPELINE_PERFORMANCE.json` | Pipeline performance summary |
| `MLPM_RELIABILITY_REPORT.md` | Reliability assessment |

---

## 8. Git Status

- All changes committed to `main` (SHA: `31d2281`)
- No branches created or switched
- No commits pushed (pending approval)
