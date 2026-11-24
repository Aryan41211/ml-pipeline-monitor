# METRICS_AND_EVALUATION.md — ML Pipeline Monitor

**Project:** ML Pipeline Monitor  
**Date:** 2026-11-24

## Pipeline Metrics

| Metric | Value | Unit | Evidence |
|---|---:|---|---|
| Pipeline stages | 7 | count | `ml/pipeline.py` |
| Classification algorithms | 6 | count | `CLF_REGISTRY` |
| Regression algorithms | 6 | count | `REG_REGISTRY` |
| Total algorithms | 12 | count | Registries |
| CV strategy | StratifiedKFold / KFold | | `ml/pipeline.py` |
| Random seed | 42 | | `config.yaml` |
| Pipeline duration (n=2000) | 2.83 | sec | Benchmark |
| Pipeline status | completed | | Benchmark |

## Drift Detection Metrics

| Scenario | PSI | KS p-value | Drift Detected | Severity | Evidence |
|---|---:|---:|---:|---:|---|
| Identical distributions | 0.0 | 1.0 | No | none | Benchmark |
| Mean-shifted (N(1.5,1)) | 2.26 | 0.0 | Yes | significant | Benchmark |
| Scale-changed (N(0,3)) | 1.74 | Not measured | Yes | significant | Benchmark |

## Model Performance (Synthetic)

| Metric | Value | Note |
|---|---:|---|
| Accuracy | 0.965 | Synthetic task |
| Precision | 0.9652 | Synthetic task |
| Recall | 0.965 | Synthetic task |
| F1 Score | 0.965 | Synthetic task |
| ROC-AUC | 0.9969 | Synthetic task |
| CV Mean | 0.9819 | 5-fold CV |
| CV Std | 0.0091 | 5-fold CV |

> **WARNING:** These are synthetic demonstration metrics. NOT production model performance.

## Test Metrics

| Metric | Value | Target |
|---|---:|---:|
| Tests passed | 310 | |
| Tests failed | 0 | |
| Coverage | 81.03% | 80% |

## API Metrics

| Endpoint | p50 | p95 | p99 | Mean | n |
|---|---:|---:|---:|---:|---:|
| /health/live | 1.56ms | 22.51ms | 24.24ms | 7.49ms | 50 |

## Evaluation Gaps

| Gap | Impact | Priority |
|---|---|---:|
| No benchmark datasets | Cannot measure max dataset size | High |
| `/predict` latency not measured | Cannot claim API performance | Medium |
| Docker health not verified | Cannot claim deployment readiness | Medium |
| Drift execution time not measured | Cannot measure drift performance | Medium |
