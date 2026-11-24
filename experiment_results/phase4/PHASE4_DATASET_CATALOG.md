# PHASE 4 — ML Pipeline Monitor Dataset Catalog

**Project:** ML Pipeline Monitor  
**Phase:** 4 — Benchmark Breakthrough  
**Date:** 2026-11-25  
**Git SHA:** 852acb4

---

## 1. Dataset Overview

| # | Dataset | Task | Type | Samples | Features | Source | License |
|---|---------|------|------|---------|----------|--------|---------|
| 1 | iris | classification | balanced_3class | 150 | 4 | scikit-learn | BSD |
| 2 | breast_cancer | classification | balanced_2class | 569 | 30 | scikit-learn | BSD |
| 3 | wine | classification | easy_3class | 178 | 13 | scikit-learn | BSD |
| 4 | digits | classification | multi_class_10 | 1797 | 64 | scikit-learn | BSD |
| 5 | imbalanced_90_10 | classification | imbalanced_2class | 2000 | 20 | synthetic (sklearn) | BSD |
| 6 | noisy_20pct | classification | noisy_2class | 2000 | 20 | synthetic (sklearn) | BSD |
| 7 | high_dimensional_100f | classification | high_dimensional_2class | 1000 | 100 | synthetic (sklearn) | BSD |
| 8 | temporal_drift | classification | temporal_concept_drift | 2000 | 10 | synthetic (sklearn) | BSD |
| 9 | missing_values | classification | missing_values_2class | 1000 | 15 | synthetic (sklearn) | BSD |
| 10 | diabetes | regression | standard_10f | 442 | 10 | scikit-learn | BSD |
| 11 | california_housing | regression | large_8f | 20640 | 8 | scikit-learn | BSD |
| 12 | noisy_regression | regression | noisy_20f | 2000 | 20 | synthetic (sklearn) | BSD |
| 13 | high_dimensional_50f | regression | high_dimensional_50f | 1000 | 50 | synthetic (sklearn) | BSD |

---

## 2. Dataset Coverage

| Requirement | Status | Details |
|-------------|--------|---------|
| Binary classification | ✅ | breast_cancer, imbalanced_90_10, noisy_20pct, high_dimensional_100f, temporal_drift, missing_values |
| Multiclass classification | ✅ | iris (3), wine (3), digits (10) |
| Imbalanced classification | ✅ | imbalanced_90_10 (90/10 split) |
| Regression | ✅ | diabetes, california_housing, noisy_regression, high_dimensional_50f |
| High-dimensional data | ✅ | high_dimensional_100f (100f), high_dimensional_50f (50f) |
| Noisy data | ✅ | noisy_20pct (20% label noise), noisy_regression |
| Missing values | ✅ | missing_values (5% random NaN) |
| Temporal data | ✅ | temporal_drift (concept drift over time) |
| Distribution shift | ✅ | temporal_drift (concept drift) |

Total: 13 datasets covering all required categories.

---

## 3. Split Strategy

- All datasets: 80% train / 20% test split
- Classification: Stratified split preserving class distribution
- Random seed: 42 for reproducibility
- Missing values: Imputed with mean before scaling

---

## 4. Reproduction

```bash
cd C:\projects\ML-pipeline-monitor
python scripts/phase4/mlpm_phase4_benchmark.py

# Output: experiment_results/phase4/mlpm_phase4_results.json
```
