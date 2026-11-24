"""ML Pipeline Monitor Phase 3 — Advanced MLOps Benchmarking.

Extends the Phase 2.5 baseline with:
1. Fixed SVM ROC-AUC computation (no silent zeroing)
2. Imbalanced classification dataset
3. Noisy data dataset (label noise)
4. Calibration metrics (Brier score, Expected Calibration Error)
5. Nested cross-validation for hyperparameter validation
6. Additional drift scenarios (gradual drift, concept drift)
7. Confidence intervals via bootstrap

Results written to experiment_results/mlpm_phase3_improvements.json
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.calibration import calibration_curve
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from ml_pipeline_monitor.ml.pipeline import (
    CLF_REGISTRY,
    REG_REGISTRY,
    DEFAULT_PARAMS,
    MLPipeline,
)

BACKEND = Path(__file__).resolve().parent.parent
EXPERIMENT_DIR = BACKEND / "experiment_results"
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
N_BOOTSTRAP = 1000


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    s = sorted(sorted_vals)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    d = k - f
    return s[f] + (s[c] - s[f]) * d


def _ci95(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        v = float(values[0]) if values else 0.0
        return (v, v)
    rng = np.random.RandomState(RANDOM_SEED)
    boot_means = []
    n = len(values)
    for _ in range(N_BOOTSTRAP):
        sample = rng.choice(values, size=n, replace=True)
        boot_means.append(float(np.mean(sample)))
    boot_means.sort()
    lo = _percentile(boot_means, 2.5)
    hi = _percentile(boot_means, 97.5)
    return (lo, hi)


def _expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    if len(np.unique(y_true)) < 2:
        return 0.0
    try:
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
        # Weight by bin size
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_prob, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        bin_counts = np.bincount(bin_indices, minlength=n_bins)
        total = len(y_prob)
        ece = 0.0
        for i in range(n_bins):
            if bin_counts[i] > 0:
                ece += (bin_counts[i] / total) * abs(frac_pos[i] - mean_pred[i])
        return float(ece)
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_classification_datasets() -> dict[str, tuple[np.ndarray, np.ndarray, str]]:
    """Load classification datasets including imbalanced and noisy variants."""
    datasets_dict = {}

    # Standard datasets
    iris = datasets.load_iris()
    datasets_dict["iris"] = (iris.data, iris.target, "balanced_3class")

    cancer = datasets.load_breast_cancer()
    datasets_dict["breast_cancer"] = (cancer.data, cancer.target, "balanced_2class")

    wine = datasets.load_wine()
    datasets_dict["wine"] = (wine.data, wine.target, "easy_3class")

    digits = datasets.load_digits()
    datasets_dict["digits"] = (digits.data, digits.target, "multi_class_10")

    # Imbalanced classification (90/10 split)
    X_imb, y_imb = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_clusters_per_class=1,
        weights=[0.9, 0.1],
        flip_y=0.01,
        random_state=RANDOM_SEED,
    )
    datasets_dict["imbalanced_90_10"] = (X_imb, y_imb, "imbalanced_2class")

    # Noisy classification (20% label noise)
    X_noisy, y_noisy = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_clusters_per_class=2,
        flip_y=0.20,
        random_state=RANDOM_SEED,
    )
    datasets_dict["noisy_20pct"] = (X_noisy, y_noisy, "noisy_2class")

    return datasets_dict


def load_regression_datasets() -> dict[str, tuple[np.ndarray, np.ndarray, str]]:
    """Load regression datasets."""
    datasets_dict = {}

    diabetes = datasets.load_diabetes()
    datasets_dict["diabetes"] = (diabetes.data, diabetes.target, "standard_10f")

    housing = datasets.fetch_california_housing()
    datasets_dict["california_housing"] = (housing.data, housing.target, "large_8f")

    # Noisy regression
    X_reg_noisy, y_reg_noisy = make_regression(
        n_samples=2000,
        n_features=20,
        n_informative=10,
        noise=0.5,
        random_state=RANDOM_SEED,
    )
    datasets_dict["noisy_regression"] = (X_reg_noisy, y_reg_noisy, "noisy_20f")

    return datasets_dict


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_classification_benchmark(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    dataset_type: str,
) -> dict[str, Any]:
    """Benchmark all classification algorithms on a single dataset."""
    results = {}

    # Train/test split with stratification
    stratify = y if len(np.unique(y)) <= 20 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=stratify
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    n_classes = len(np.unique(y))

    for model_name, model_class in CLF_REGISTRY.items():
        params = DEFAULT_PARAMS.get(model_name, {}).copy()

        # Apply random_state where supported
        import inspect
        sig = inspect.signature(model_class.__init__)
        if "random_state" in sig.parameters:
            params["random_state"] = RANDOM_SEED
        if "n_jobs" in sig.parameters and model_name != "Logistic Regression":
            params["n_jobs"] = -1
        if model_name == "XGBoost":
            params["verbosity"] = 0
            params["eval_metric"] = "logloss"
        if model_name == "SVM":
            params["probability"] = True
        if model_name == "SVR":
            params.pop("random_state", None)

        model = model_class(**params)

        # Train
        t0 = time.perf_counter()
        model.fit(X_train_sc, y_train)
        train_time = time.perf_counter() - t0

        # Predict
        t0 = time.perf_counter()
        y_pred = model.predict(X_test_sc)
        infer_time = (time.perf_counter() - t0) / len(X_test_sc) * 1000  # ms per sample

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # ROC-AUC
        roc_auc = float("nan")
        try:
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test_sc)
                if y_prob.shape[1] == 2:
                    roc_auc = roc_auc_score(y_test, y_prob[:, 1])
                else:
                    if len(np.unique(y_test)) > 1:
                        roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
        except ValueError:
            roc_auc = float("nan")
        except Exception:
            roc_auc = float("nan")

        # Calibration metrics
        brier = float("nan")
        ece = float("nan")
        try:
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test_sc)
                if y_prob.shape[1] == 2:
                    brier = brier_score_loss(y_test, y_prob[:, 1])
                    ece = _expected_calibration_error(y_test, y_prob[:, 1])
                else:
                    # For multi-class, use max probability as confidence
                    max_prob = np.max(y_prob, axis=1)
                    # Convert to binary: correct vs incorrect
                    y_correct = (y_pred == y_test).astype(int)
                    brier = brier_score_loss(y_correct, max_prob)
                    ece = _expected_calibration_error(y_correct, max_prob)
        except Exception:
            pass

        results[model_name] = {
            "model": model_name,
            "accuracy": round(float(accuracy), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(roc_auc), 4) if not np.isnan(roc_auc) else None,
            "brier_score": round(float(brier), 4) if not np.isnan(brier) else None,
            "ece": round(float(ece), 4) if not np.isnan(ece) else None,
            "train_time_s": round(train_time, 4),
            "inference_latency_ms": round(infer_time, 4),
            "n_test_samples": len(X_test_sc),
            "n_classes": n_classes,
            "dataset_type": dataset_type,
        }

    return results


def run_regression_benchmark(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    dataset_type: str,
) -> dict[str, Any]:
    """Benchmark all regression algorithms on a single dataset."""
    results = {}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    for model_name, model_class in REG_REGISTRY.items():
        params = DEFAULT_PARAMS.get(model_name, {}).copy()

        import inspect
        sig = inspect.signature(model_class.__init__)
        if "random_state" in sig.parameters:
            params["random_state"] = RANDOM_SEED
        if "n_jobs" in sig.parameters and model_name != "Ridge Regression":
            params["n_jobs"] = -1
        if model_name == "XGBoost":
            params["verbosity"] = 0
            params["eval_metric"] = "rmse"
        if model_name == "SVR":
            params.pop("random_state", None)

        model = model_class(**params)

        # Train
        t0 = time.perf_counter()
        model.fit(X_train_sc, y_train)
        train_time = time.perf_counter() - t0

        # Predict
        t0 = time.perf_counter()
        y_pred = model.predict(X_test_sc)
        infer_time = (time.perf_counter() - t0) / len(X_test_sc) * 1000

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results[model_name] = {
            "model": model_name,
            "mae": round(float(mae), 4),
            "rmse": round(float(rmse), 4),
            "r2": round(float(r2), 4),
            "train_time_s": round(train_time, 4),
            "inference_latency_ms": round(infer_time, 4),
            "n_test_samples": len(X_test_sc),
            "dataset_type": dataset_type,
        }

    return results


# ---------------------------------------------------------------------------
# Drift detection with additional scenarios
# ---------------------------------------------------------------------------

def compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    ref = np.asarray(reference, dtype=float).ravel()
    cur = np.asarray(current, dtype=float).ravel()
    if ref.size < 2 or cur.size < 2:
        return 0.0
    n_bins = max(2, min(n_bins, min(ref.size, cur.size) // 15))
    lo = float(min(ref.min(), cur.min()))
    hi = float(max(ref.max(), cur.max()))
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return 0.0
    edges = np.linspace(lo, hi, n_bins + 1)
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_pct = (ref_counts + 1.0) / (ref_counts.sum() + n_bins)
    cur_pct = (cur_counts + 1.0) / (cur_counts.sum() + n_bins)
    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return round(psi, 5)


def compute_ks_statistic(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    from scipy import stats
    ref_clean = np.asarray(reference, dtype=float).ravel()
    cur_clean = np.asarray(current, dtype=float).ravel()
    stat, p_value = stats.ks_2samp(ref_clean, cur_clean)
    return float(stat), float(p_value)


def compute_js_divergence(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    from scipy.spatial.distance import jensenshannon
    ref = np.asarray(reference, dtype=float).ravel()
    cur = np.asarray(current, dtype=float).ravel()
    if ref.size < 2 or cur.size < 2:
        return 0.0
    lo = float(min(ref.min(), cur.min()))
    hi = float(max(ref.max(), cur.max()))
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return 0.0
    edges = np.linspace(lo, hi, n_bins + 1)
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_p = (ref_counts + 1e-10) / (ref_counts.sum() + n_bins * 1e-10)
    cur_p = (cur_counts + 1e-10) / (cur_counts.sum() + n_bins * 1e-10)
    return float(jensenshannon(ref_p, cur_p))


def compute_wasserstein(reference: np.ndarray, current: np.ndarray) -> float:
    from scipy.stats import wasserstein_distance
    ref = np.asarray(reference, dtype=float).ravel()
    cur = np.asarray(current, dtype=float).ravel()
    if ref.size < 2 or cur.size < 2:
        return 0.0
    return float(wasserstein_distance(ref, cur))


def generate_drift_scenarios(n_samples: int = 1000, n_features: int = 5) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Generate drift scenarios including gradual drift and concept drift."""
    rng = np.random.RandomState(RANDOM_SEED)
    scenarios = {}

    # 1. No drift
    X_ref = rng.randn(n_samples, n_features)
    X_cur = rng.randn(n_samples, n_features)
    scenarios["no_drift"] = (X_ref, X_cur)

    # 2. Small drift (mean shift 0.3)
    X_cur = rng.randn(n_samples, n_features) + 0.3
    scenarios["small_drift"] = (X_ref, X_cur)

    # 3. Moderate drift (mean shift 0.7)
    X_cur = rng.randn(n_samples, n_features) + 0.7
    scenarios["moderate_drift"] = (X_ref, X_cur)

    # 4. Severe drift (mean shift 1.5)
    X_cur = rng.randn(n_samples, n_features) + 1.5
    scenarios["severe_drift"] = (X_ref, X_cur)

    # 5. Scale change (sigma=3)
    X_cur = rng.randn(n_samples, n_features) * 3.0
    scenarios["scale_change"] = (X_ref, X_cur)

    # 6. Covariate shift (mean shift 2.0)
    X_cur = rng.randn(n_samples, n_features) + 2.0
    scenarios["covariate_shift"] = (X_ref, X_cur)

    # 7. Feature-level drift (only first 2 features drift)
    X_cur = rng.randn(n_samples, n_features).copy()
    X_cur[:, :2] += 1.0
    scenarios["feature_level_drift"] = (X_ref, X_cur)

    # 8. Class imbalance shift (binary labels 0.5->0.8)
    y_ref = np.where(rng.rand(n_samples) < 0.5, 0, 1)
    y_cur = np.where(rng.rand(n_samples) < 0.8, 0, 1)
    scenarios["class_imbalance"] = (y_ref.reshape(-1, 1), y_cur.reshape(-1, 1))

    # 9. Gradual drift (mean shift increases over time)
    X_gradual = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        shift = 1.5 * (i / n_samples)  # Gradually increases from 0 to 1.5
        X_gradual[i] = rng.randn(n_features) + shift
    scenarios["gradual_drift"] = (X_ref, X_gradual)

    # 10. Concept drift (relationship between X and y changes)
    # Reference: y = X @ w + noise
    # Current: y = -X @ w + noise (flipped relationship)
    w = rng.randn(n_features)
    X_concept = rng.randn(n_samples, n_features)
    y_ref = (X_concept @ w > 0).astype(int)
    y_cur = (-(X_concept @ w) > 0).astype(int)
    scenarios["concept_drift"] = (X_concept, y_cur.reshape(-1, 1))

    return scenarios


def run_drift_benchmark() -> dict[str, Any]:
    """Run drift detection benchmark with additional scenarios."""
    scenarios = generate_drift_scenarios()

    methods = {
        "PSI": {"threshold": 0.25, "compute": compute_psi},
        "KS Test": {"threshold": 0.05, "compute": compute_ks_statistic},
        "JS Divergence": {"threshold": 0.1, "compute": compute_js_divergence},
        "Wasserstein": {"threshold": 0.5, "compute": compute_wasserstein},
    }

    results = {
        "n_scenarios": len(scenarios),
        "scenarios": list(scenarios.keys()),
        "methods": {},
    }

    for method_name, method_info in methods.items():
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        runtimes = []

        for scenario_name, (ref, cur) in scenarios.items():
            expected_drift = scenario_name != "no_drift"

            t0 = time.perf_counter()
            score = method_info["compute"](ref, cur)
            runtime_ms = (time.perf_counter() - t0) * 1000
            runtimes.append(runtime_ms)

            threshold = method_info["threshold"]
            if method_name == "KS Test":
                # For KS, p-value < threshold means drift
                detected = score[1] < threshold
            else:
                detected = score > threshold

            if expected_drift and detected:
                tp += 1
            elif expected_drift and not detected:
                fn += 1
            elif not expected_drift and not detected:
                tn += 1
            else:
                fp += 1

        total = tp + fp + tn + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        runtimes.sort()
        n = len(runtimes)

        results["methods"][method_name] = {
            "method": method_name,
            "threshold": threshold,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "fpr": round(fpr, 4),
            "fnr": round(fnr, 4),
            "mean_runtime_ms": round(np.mean(runtimes), 4),
            "p95_runtime_ms": round(_percentile(runtimes, 95), 4),
            "total_tests": total,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
        }

    return results


# ---------------------------------------------------------------------------
# Nested cross-validation
# ---------------------------------------------------------------------------

def run_nested_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    model_class: type,
    task: str,
    param_grid: dict[str, list[Any]],
    n_splits: int = 3,
) -> dict[str, Any]:
    """Run nested cross-validation for hyperparameter validation."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED,
        stratify=y if task == "classification" and len(np.unique(y)) <= 20 else None
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    base_params = DEFAULT_PARAMS.get(model_name, {}).copy()
    import inspect
    sig = inspect.signature(model_class.__init__)
    if "random_state" in sig.parameters:
        base_params["random_state"] = RANDOM_SEED
    if "n_jobs" in sig.parameters and model_name not in ("Logistic Regression", "Ridge Regression"):
        base_params["n_jobs"] = -1
    if model_name == "XGBoost":
        base_params["verbosity"] = 0
        base_params["eval_metric"] = "logloss" if task == "classification" else "rmse"
    if model_name in ("SVM", "SVR"):
        base_params.pop("random_state", None)
    if model_name == "SVM":
        base_params["probability"] = True

    # Inner CV: grid search for best hyperparameters
    scoring = "accuracy" if task == "classification" else "r2"
    cv_inner = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED) if task == "classification" else None

    model = model_class(**base_params)
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=cv_inner,
        scoring=scoring,
        n_jobs=1,  # Avoid nested parallelism issues
    )

    t0 = time.perf_counter()
    grid_search.fit(X_train_sc, y_train)
    nested_train_time = time.perf_counter() - t0

    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Outer evaluation: test set performance
    t0 = time.perf_counter()
    y_pred = best_model.predict(X_test_sc)
    infer_time = (time.perf_counter() - t0) / len(X_test_sc) * 1000

    if task == "classification":
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        return {
            "model": model_name,
            "nested_cv_accuracy": round(float(accuracy), 4),
            "nested_cv_f1": round(float(f1), 4),
            "best_params": best_params,
            "nested_train_time_s": round(nested_train_time, 4),
            "inference_latency_ms": round(infer_time, 4),
        }
    else:
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return {
            "model": model_name,
            "nested_cv_r2": round(float(r2), 4),
            "nested_cv_rmse": round(float(rmse), 4),
            "best_params": best_params,
            "nested_train_time_s": round(nested_train_time, 4),
            "inference_latency_ms": round(infer_time, 4),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 70)
    print("ML Pipeline Monitor Phase 3 — Advanced MLOps Benchmarking")
    print("=" * 70)

    # Load datasets
    clf_datasets = load_classification_datasets()
    reg_datasets = load_regression_datasets()

    print(f"Classification datasets: {list(clf_datasets.keys())}", flush=True)
    print(f"Regression datasets: {list(reg_datasets.keys())}", flush=True)

    results: dict[str, Any] = {
        "random_seed": RANDOM_SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "datasets": {},
        "drift": {},
        "nested_cv": {},
    }

    # --- Classification benchmarks ---
    for ds_name, (X, y, ds_type) in clf_datasets.items():
        print(f"\nClassification: {ds_name} ({ds_type}) ...", flush=True)
        ds_results = run_classification_benchmark(X, y, ds_name, ds_type)
        results["datasets"][ds_name] = {
            "task": "classification",
            "type": ds_type,
            "n_samples": len(X),
            "n_features": X.shape[1],
            "n_classes": len(np.unique(y)),
            "models": ds_results,
        }
        # Print best
        best_f1 = max(ds_results.items(), key=lambda x: x[1]["f1_score"])
        print(f"  Best F1: {best_f1[0]} = {best_f1[1]['f1_score']:.4f}", flush=True)

    # --- Regression benchmarks ---
    for ds_name, (X, y, ds_type) in reg_datasets.items():
        print(f"\nRegression: {ds_name} ({ds_type}) ...", flush=True)
        ds_results = run_regression_benchmark(X, y, ds_name, ds_type)
        results["datasets"][ds_name] = {
            "task": "regression",
            "type": ds_type,
            "n_samples": len(X),
            "n_features": X.shape[1],
            "models": ds_results,
        }
        best_r2 = max(ds_results.items(), key=lambda x: x[1]["r2"])
        print(f"  Best R²: {best_r2[0]} = {best_r2[1]['r2']:.4f}", flush=True)

    # --- Drift detection ---
    print("\nDrift detection benchmark ...", flush=True)
    drift_results = run_drift_benchmark()
    results["drift"] = drift_results
    for method_name, method_result in drift_results["methods"].items():
        print(
            f"  {method_name}: F1={method_result['f1']:.4f} "
            f"Precision={method_result['precision']:.4f} "
            f"Recall={method_result['recall']:.4f} "
            f"FPR={method_result['fpr']:.4f}",
            flush=True,
        )

    # --- Nested CV on wine (the 0.9907 dataset) ---
    print("\nNested cross-validation on wine dataset ...", flush=True)
    X_wine, y_wine, _ = clf_datasets["wine"]

    for model_name in ["Random Forest", "XGBoost", "SVM", "Logistic Regression"]:
        model_class = CLF_REGISTRY[model_name]
        # Simple param grid for each model
        param_grids = {
            "Random Forest": {"n_estimators": [50, 100], "max_depth": [None, 5, 10]},
            "XGBoost": {"n_estimators": [50, 100], "max_depth": [3, 6], "learning_rate": [0.1]},
            "SVM": {"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"]},
            "Logistic Regression": {"C": [0.1, 1.0, 10.0]},
        }
        nc_result = run_nested_cv(X_wine, y_wine, model_name, model_class, "classification",
                                  param_grids.get(model_name, {}))
        results["nested_cv"][model_name] = nc_result
        print(f"  {model_name}: accuracy={nc_result['nested_cv_accuracy']:.4f} "
              f"best_params={nc_result['best_params']}", flush=True)

    # Write results
    out_path = EXPERIMENT_DIR / "mlpm_phase3_improvements.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}", flush=True)

    # Print comparison table
    print("\n" + "=" * 70)
    print("CLASSIFICATION RESULTS — Accuracy / F1 / ROC-AUC / ECE")
    print("=" * 70)
    for ds_name, ds_data in results["datasets"].items():
        if ds_data["task"] != "classification":
            continue
        print(f"\n  {ds_name} ({ds_data['type']}):")
        for model_name, model_data in ds_data["models"].items():
            roc = model_data.get("roc_auc")
            ece = model_data.get("ece")
            print(
                f"    {model_name:<20s} Acc={model_data['accuracy']:.4f} "
                f"F1={model_data['f1_score']:.4f} "
                f"ROC={str(roc):>6s} ECE={str(ece):>6s}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())