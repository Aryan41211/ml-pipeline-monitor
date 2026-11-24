# PERFORMANCE_BENCHMARKS.md — ML Pipeline Monitor

**Project:** ML Pipeline Monitor  
**Benchmark date:** 2026-11-24  
**Hardware:** Developer workstation, 16 CPU cores, 15.65 GB RAM, CPU-only  
**Software:** Python 3.12.10, scikit-learn 1.8.0, XGBoost 2.x

## Pipeline Execution (Synthetic Data)

| Metric | Value | Unit |
|---|---:|---|
| Dataset size | 2,000 | rows |
| Features | 5 | count |
| Train/test split | 1,600 / 400 | rows |
| Random seed | 42 | |
| Pipeline status | completed | |
| Total duration | 2.83 | seconds |

## Stage Durations

| Stage | Duration (sec) |
|---|---:|
| Data Validation | 0.0139 |
| Preprocessing | 0.0016 |
| Feature Analysis | 0.0010 |
| Cross-Validation | 2.634 |
| Training | 0.1109 |
| Evaluation | 0.0360 |
| Feature Importance | 0.0296 |

## Model Metrics (Synthetic Classification, Random Forest)

| Metric | Value |
|---|---:|
| Accuracy | 0.965 |
| Precision | 0.9652 |
| Recall | 0.965 |
| F1 Score | 0.965 |
| ROC-AUC | 0.9969 |
| CV Mean | 0.9819 |
| CV Std | 0.0091 |

> **NOTE:** These are synthetic demonstration metrics on a learnable task. NOT production model performance.

## API Latency

| Endpoint | p50 | p95 | p99 | Mean | n |
|---|---:|---:|---:|---:|---:|
| /health/live | 1.56ms | 22.51ms | 24.24ms | 7.49ms | 50 |

## Algorithm Execution (All 12 Algorithms Verified)

| Algorithm | Task | Status |
|---|---|---|
| Random Forest | Classification | ✅ |
| XGBoost | Classification | ✅ |
| Gradient Boosting | Classification | ✅ |
| Logistic Regression | Classification | ✅ |
| SVM | Classification | ✅ |
| Decision Tree | Classification | ✅ |
| Random Forest | Regression | ✅ |
| XGBoost | Regression | ✅ |
| Gradient Boosting | Regression | ✅ |
| Ridge Regression | Regression | ✅ |
| SVR | Regression | ✅ |
| Decision Tree | Regression | ✅ |

## Drift Detection

| Scenario | PSI | KS p-value | Drift Detected | Severity |
|---|---:|---:|---:|---:|
| Identical distributions | 0.0 | 1.0 | No | none |
| Mean-shifted (N(1.5,1)) | 2.26 | 0.0 | Yes | significant |
| Scale-changed (N(0,3)) | 1.74 | Not measured | Yes | significant |

## Reproducibility

- Random seed: 42
- Script: `ml_pipeline_monitor_benchmark.py`
- Results: `ml_pipeline_monitor_results.json`
- Synthetic data only (no benchmark datasets in repository)

## Evaluation Gaps

| Gap | Impact | Priority |
|---|---|---:|
| No benchmark datasets | Cannot measure max dataset size | High |
| `/predict` API latency not measured | Cannot claim API performance | Medium |
| Docker health not verified | Cannot claim deployment readiness | Medium |
| Drift execution time not measured | Cannot measure drift performance | Medium |
