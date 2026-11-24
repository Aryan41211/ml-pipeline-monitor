# PHASE 4 — ML Pipeline Monitor Generalization Report

**Project:** ML Pipeline Monitor  
**Phase:** 4 — Benchmark Breakthrough  
**Date:** 2026-11-25  
**Git SHA:** 852acb4

---

## 1. Generalization Assessment

The reported 0.9907 accuracy (Phase 2) was on the wine dataset (178 samples, 3 classes), which is an easy dataset. Phase 4 evaluates on 13 diverse datasets.

### 1.1 Accuracy by Dataset Difficulty

| Difficulty | Dataset | Best Accuracy | Best F1 | Notes |
|------------|---------|---------------|---------|-------|
| Easy | wine | 1.0000 | 1.0000 | 178 samples, not production-representative |
| Easy | iris | 0.9667 | 0.9666 | 150 samples, 4 features |
| Standard | breast_cancer | 0.9825 | 0.9825 | 569 samples, 30 features |
| Multi-class | digits | 0.9750 | 0.9749 | 10 classes |
| Imbalanced | imbalanced_90_10 | 0.9850 | 0.9847 | Accuracy inflated (90% majority) |
| Noisy | noisy_20pct | 0.8850 | 0.8850 | 20% label noise |
| High-dimensional | high_dimensional_100f | 0.9100 | 0.9100 | 100 features |
| Temporal drift | temporal_drift | 0.8065 | 0.8065 | Concept drift degrades performance |
| Missing values | missing_values | 0.9350 | 0.9350 | 5% NaN, mean imputation |

### 1.2 Regression Generalization

| Dataset | Best Model | Best R² | Best RMSE |
|---------|-----------|---------|-----------|
| diabetes | Extra Trees | 0.4685 | 53.07 |
| california_housing | XGBoost | 0.8266 | 0.4767 |
| noisy_regression | Ridge Regression | 1.0000 | 0.54 |
| high_dimensional_50f | Ridge Regression | 1.0000 | 0.54 |

### 1.3 Calibration Summary

| Dataset | Best Brier Score | Best ECE |
|---------|-----------------|----------|
| iris | 0.0285 (Extra Trees) | N/A |
| breast_cancer | N/A | N/A |
| imbalanced_90_10 | 0.0186 (Gradient Boosting) | 0.0186 |
| noisy_20pct | 0.0442 (XGBoost) | 0.0442 |

## 2. Drift Detection Generalization

| Drift Type | KS Detected | PSI Detected | ADWIN Detected |
|------------|-------------|--------------|----------------|
| no_drift | No | No | No |
| small_drift | Yes | No | Yes |
| moderate_drift | Yes | Yes | Yes |
| severe_drift | Yes | Yes | Yes |
| scale_change | Yes | Yes | Yes |
| covariate_shift | Yes | Yes | Yes |
| feature_level_drift | Yes | Yes | Yes |
| class_imbalance | Yes | No | Yes |
| gradual_drift | Yes | Yes | Yes |
| concept_drift | Yes | Yes | Yes |
| recurring_drift | Yes | Yes | Yes |

**KS Test detects all drift types** — F1=1.0, FPR=0.0.

## 3. Limitations

- All datasets are from scikit-learn or synthetic — no real-world production data
- No time-series cross-validation for temporal datasets
- No group-aware validation for grouped data
- No nested cross-validation for all models (only wine in Phase 3)

## 4. Reproduction

```bash
cd C:\projects\ML-pipeline-monitor
python scripts/phase4/mlpm_phase4_benchmark.py
```
