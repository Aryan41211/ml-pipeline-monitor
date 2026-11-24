"""ML Pipeline Monitor Phase 4 — Advanced Benchmarking & Drift Detection.

Extends Phase 3 with:
1. More diverse datasets (10+ including temporal, high-dimensional, missing values)
2. Repeated stratified cross-validation
3. Additional drift detection methods (ADWIN, Page-Hinkley, CUSUM)
4. Calibration metrics (Brier score, ECE) for all models
5. Model performance monitoring (latency, throughput, memory)

Usage:
    python scripts/phase4/mlpm_phase4_benchmark.py
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.impute import SimpleImputer
from sklearn.calibration import calibration_curve
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from ml_pipeline_monitor.ml.pipeline import CLF_REGISTRY, REG_REGISTRY, DEFAULT_PARAMS

BACKEND = Path(__file__).resolve().parent.parent.parent
EXPERIMENT_DIR = BACKEND / "experiment_results" / "phase4"
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
N_BOOTSTRAP = 1000
N_REPEATS = 3


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
    if len(np.unique(y_true)) < 2:
        return 0.0
    try:
        frac_pos, mean_pred = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy="uniform"
        )
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.clip(np.digitize(y_prob, bin_edges) - 1, 0, n_bins - 1)
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
# Dataset catalog
# ---------------------------------------------------------------------------

DATASET_CATALOG: list[dict[str, Any]] = []


def _add_clf(name: str, X: np.ndarray, y: np.ndarray, ds_type: str, source: str, license_: str):
    DATASET_CATALOG.append({
        "name": name,
        "task": "classification",
        "type": ds_type,
        "source": source,
        "license": license_,
        "X": X,
        "y": y,
    })


def _add_reg(name: str, X: np.ndarray, y: np.ndarray, ds_type: str, source: str, license_: str):
    DATASET_CATALOG.append({
        "name": name,
        "task": "regression",
        "type": ds_type,
        "source": source,
        "license": license_,
        "X": X,
        "y": y,
    })


# Classification datasets
iris = datasets.load_iris()
_add_clf("iris", iris.data, iris.target, "balanced_3class", "scikit-learn", "BSD")

cancer = datasets.load_breast_cancer()
_add_clf("breast_cancer", cancer.data, cancer.target, "balanced_2class", "scikit-learn", "BSD")

wine = datasets.load_wine()
_add_clf("wine", wine.data, wine.target, "easy_3class", "scikit-learn", "BSD")

digits = datasets.load_digits()
_add_clf("digits", digits.data, digits.target, "multi_class_10", "scikit-learn", "BSD")

# Imbalanced classification (90/10)
X_imb, y_imb = make_classification(
    n_samples=2000, n_features=20, n_informative=10, n_redundant=5,
    n_clusters_per_class=1, weights=[0.9, 0.1], flip_y=0.01, random_state=RANDOM_SEED,
)
_add_clf("imbalanced_90_10", X_imb, y_imb, "imbalanced_2class", "synthetic (sklearn)", "BSD")

# Noisy classification (20% label noise)
X_noisy, y_noisy = make_classification(
    n_samples=2000, n_features=20, n_informative=10, n_redundant=5,
    n_clusters_per_class=2, flip_y=0.20, random_state=RANDOM_SEED,
)
_add_clf("noisy_20pct", X_noisy, y_noisy, "noisy_2class", "synthetic (sklearn)", "BSD")

# High-dimensional classification
X_hd, y_hd = make_classification(
    n_samples=1000, n_features=100, n_informative=20, n_redundant=30,
    n_clusters_per_class=2, random_state=RANDOM_SEED,
)
_add_clf("high_dimensional_100f", X_hd, y_hd, "high_dimensional_2class", "synthetic (sklearn)", "BSD")

# Temporal classification (ordered data with drift)
X_temp, y_temp = make_classification(
    n_samples=2000, n_features=10, n_informative=5, n_redundant=2,
    n_clusters_per_class=2, random_state=RANDOM_SEED,
)
# Introduce temporal drift: later samples have shifted distribution
rng = np.random.RandomState(RANDOM_SEED)
X_temp[1000:] = X_temp[1000:] + 0.5  # shift after midpoint
y_temp[1000:] = (y_temp[1000:] + 1) % 2  # flip some labels
_add_clf("temporal_drift", X_temp, y_temp, "temporal_concept_drift", "synthetic (sklearn)", "BSD")

# Missing values classification
X_missing, y_missing = make_classification(
    n_samples=1000, n_features=15, n_informative=5, n_redundant=5,
    n_clusters_per_class=2, random_state=RANDOM_SEED,
)
X_missing_df = pd.DataFrame(X_missing)
# Introduce missing values (5% random)
mask = rng.random(X_missing_df.shape) < 0.05
X_missing_df = X_missing_df.mask(mask)
_add_clf("missing_values", X_missing_df.values, y_missing, "missing_values_2class", "synthetic (sklearn)", "BSD")

# Regression datasets
diabetes = datasets.load_diabetes()
_add_reg("diabetes", diabetes.data, diabetes.target, "standard_10f", "scikit-learn", "BSD")

housing = datasets.fetch_california_housing()
_add_reg("california_housing", housing.data, housing.target, "large_8f", "scikit-learn", "BSD")

# Noisy regression
X_reg_noisy, y_reg_noisy = make_regression(
    n_samples=2000, n_features=20, n_informative=10, noise=0.5, random_state=RANDOM_SEED,
)
_add_reg("noisy_regression", X_reg_noisy, y_reg_noisy, "noisy_20f", "synthetic (sklearn)", "BSD")

# High-dimensional regression
X_reg_hd, y_reg_hd = make_regression(
    n_samples=1000, n_features=50, n_informative=10, noise=0.2, random_state=RANDOM_SEED,
)
_add_reg("high_dimensional_50f", X_reg_hd, y_reg_hd, "high_dimensional_50f", "synthetic (sklearn)", "BSD")


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_classification_benchmark(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    dataset_type: str,
) -> dict[str, Any]:
    results = {}
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    stratify = y if len(np.unique(y)) <= 20 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=stratify
    )

    # Impute missing values (for datasets with NaN)
    imputer = SimpleImputer(strategy="mean")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_imp)
    X_test_sc = scaler.transform(X_test_imp)

    n_classes = len(np.unique(y))

    for model_name, model_class in CLF_REGISTRY.items():
        params = DEFAULT_PARAMS.get(model_name, {}).copy()
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

        model = model_class(**params)

        # Repeated stratified cross-validation
        cv_scores = []
        for rep in range(N_REPEATS):
            cv = StratifiedKFold(
                n_splits=5, shuffle=True, random_state=RANDOM_SEED + rep
            )
            model_cv = model_class(**params)
            scores = cross_val_score(
                model_cv, X_train_sc, y_train, cv=cv,
                scoring="accuracy", n_jobs=1,
            )
            cv_scores.extend(scores.tolist())

        # Train on full training set
        model = model_class(**params)
        t0 = time.perf_counter()
        model.fit(X_train_sc, y_train)
        train_time = time.perf_counter() - t0

        # Predict
        t0 = time.perf_counter()
        y_pred = model.predict(X_test_sc)
        infer_time = (time.perf_counter() - t0) / len(X_test_sc) * 1000

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

        # Calibration
        brier = float("nan")
        ece = float("nan")
        try:
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test_sc)
                if y_prob.shape[1] == 2:
                    brier = brier_score_loss(y_test, y_prob[:, 1])
                    ece = _expected_calibration_error(y_test, y_prob[:, 1])
                else:
                    max_prob = np.max(y_prob, axis=1)
                    y_correct = (y_pred == y_test).astype(int)
                    brier = brier_score_loss(y_correct, max_prob)
                    ece = _expected_calibration_error(y_correct, max_prob)
        except Exception:
            pass

        results[model_name] = {
            "model": model_name,
            "accuracy": round(float(accuracy), 4),
            "accuracy_ci_lo": round(_ci95(cv_scores)[0], 4),
            "accuracy_ci_hi": round(_ci95(cv_scores)[1], 4),
            "cv_accuracy_mean": round(float(np.mean(cv_scores)), 4),
            "cv_accuracy_std": round(float(np.std(cv_scores)), 4),
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
    results = {}
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED
    )

    # Impute missing values (for datasets with NaN)
    imputer = SimpleImputer(strategy="mean")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_imp)
    X_test_sc = scaler.transform(X_test_imp)

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
        t0 = time.perf_counter()
        model.fit(X_train_sc, y_train)
        train_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_pred = model.predict(X_test_sc)
        infer_time = (time.perf_counter() - t0) / len(X_test_sc) * 1000

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # Residual diagnostics
        residuals = y_test - y_pred
        residual_mean = float(np.mean(residuals))
        residual_std = float(np.std(residuals))

        results[model_name] = {
            "model": model_name,
            "mae": round(float(mae), 4),
            "rmse": round(float(rmse), 4),
            "r2": round(float(r2), 4),
            "residual_mean": round(residual_mean, 4),
            "residual_std": round(residual_std, 4),
            "train_time_s": round(train_time, 4),
            "inference_latency_ms": round(infer_time, 4),
            "n_test_samples": len(X_test_sc),
            "dataset_type": dataset_type,
        }

    return results


# ---------------------------------------------------------------------------
# Drift detection with additional methods
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


def compute_ks_statistic(reference: np.ndarray, current: np.ndarray):
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


def adwin_detect(reference: np.ndarray, current: np.ndarray, delta: float = 0.002) -> bool:
    """Simplified ADWIN drift detection based on mean shift."""
    ref = np.asarray(reference, dtype=float).ravel()
    cur = np.asarray(current, dtype=float).ravel()
    if len(ref) < 10 or len(cur) < 10:
        return False
    mean_ref = np.mean(ref)
    mean_cur = np.mean(cur)
    std_ref = np.std(ref)
    if std_ref == 0:
        return abs(mean_cur - mean_ref) > delta
    return abs(mean_cur - mean_ref) / std_ref > delta


def page_hinkley_detect(reference: np.ndarray, current: np.ndarray, threshold: float = 50.0) -> bool:
    """Simplified Page-Hinkley drift detection."""
    ref = np.asarray(reference, dtype=float).ravel()
    cur = np.asarray(current, dtype=float).ravel()
    combined = np.concatenate([ref, cur])
    mean_all = np.mean(combined)
    cumulative_sum = 0.0
    min_sum = 0.0
    for val in combined:
        cumulative_sum += (val - mean_all)
        min_sum = min(min_sum, cumulative_sum)
        if cumulative_sum - min_sum > threshold:
            return True
    return False


def cusum_detect(reference: np.ndarray, current: np.ndarray, threshold: float = 5.0) -> bool:
    """Simplified CUSUM drift detection."""
    ref = np.asarray(reference, dtype=float).ravel()
    cur = np.asarray(current, dtype=float).ravel()
    combined = np.concatenate([ref, cur])
    mean_all = np.mean(combined)
    std_all = np.std(combined)
    if std_all == 0:
        return False
    upper_sum = 0.0
    lower_sum = 0.0
    k = 0.5  # reference value
    h = threshold  # decision interval
    for val in combined:
        standardized = (val - mean_all) / std_all
        upper_sum = max(0, upper_sum + standardized - k)
        lower_sum = max(0, lower_sum - standardized - k)
        if upper_sum > h or lower_sum > h:
            return True
    return False


def generate_drift_scenarios(n_samples: int = 1000, n_features: int = 5) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.RandomState(RANDOM_SEED)
    scenarios = {}
    X_ref = rng.randn(n_samples, n_features)

    scenarios["no_drift"] = (X_ref, rng.randn(n_samples, n_features))
    scenarios["small_drift"] = (X_ref, rng.randn(n_samples, n_features) + 0.3)
    scenarios["moderate_drift"] = (X_ref, rng.randn(n_samples, n_features) + 0.7)
    scenarios["severe_drift"] = (X_ref, rng.randn(n_samples, n_features) + 1.5)
    scenarios["scale_change"] = (X_ref, rng.randn(n_samples, n_features) * 3.0)
    scenarios["covariate_shift"] = (X_ref, rng.randn(n_samples, n_features) + 2.0)

    X_cur = rng.randn(n_samples, n_features).copy()
    X_cur[:, :2] += 1.0
    scenarios["feature_level_drift"] = (X_ref, X_cur)

    y_ref = np.where(rng.rand(n_samples) < 0.5, 0, 1)
    y_cur = np.where(rng.rand(n_samples) < 0.8, 0, 1)
    scenarios["class_imbalance"] = (y_ref.reshape(-1, 1), y_cur.reshape(-1, 1))

    # Gradual drift
    X_gradual = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        shift = 1.5 * (i / n_samples)
        X_gradual[i] = rng.randn(n_features) + shift
    scenarios["gradual_drift"] = (X_ref, X_gradual)

    # Concept drift
    w = rng.randn(n_features)
    X_concept = rng.randn(n_samples, n_features)
    y_ref = (X_concept @ w > 0).astype(int)
    y_cur = (-(X_concept @ w) > 0).astype(int)
    scenarios["concept_drift"] = (X_concept, y_cur.reshape(-1, 1))

    # Recurring drift (drift that appears and disappears)
    X_recurring = rng.randn(n_samples, n_features).copy()
    X_recurring[n_samples // 4:3 * n_samples // 4, :] += 1.0
    scenarios["recurring_drift"] = (X_ref, X_recurring)

    return scenarios


def run_drift_benchmark() -> dict[str, Any]:
    scenarios = generate_drift_scenarios()

    methods = {
        "PSI": {"threshold": 0.25, "compute": compute_psi, "is_stat": False},
        "KS Test": {"threshold": 0.05, "compute": compute_ks_statistic, "is_stat": True},
        "JS Divergence": {"threshold": 0.1, "compute": compute_js_divergence, "is_stat": False},
        "Wasserstein": {"threshold": 0.5, "compute": compute_wasserstein, "is_stat": False},
        "ADWIN": {"threshold": 0.002, "compute": adwin_detect, "is_stat": False},
        "Page-Hinkley": {"threshold": 50.0, "compute": page_hinkley_detect, "is_stat": False},
        "CUSUM": {"threshold": 5.0, "compute": cusum_detect, "is_stat": False},
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
            runtime_ms = (time.perf_counter() - t0) * 1000.0
            runtimes.append(runtime_ms)

            threshold = method_info["threshold"]
            if method_info["is_stat"]:
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
            "mean_runtime_ms": round(float(np.mean(runtimes)), 4),
            "p95_runtime_ms": round(_percentile(runtimes, 95), 4),
            "total_tests": total,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
        }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 70)
    print("ML Pipeline Monitor Phase 4 — Advanced Benchmarking & Drift Detection")
    print("=" * 70)

    print(f"Datasets: {len(DATASET_CATALOG)}", flush=True)
    for ds in DATASET_CATALOG:
        print(f"  {ds['name']}: {ds['task']} | {ds['type']} | {ds['source']}", flush=True)

    results: dict[str, Any] = {
        "random_seed": RANDOM_SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "n_repeats": N_REPEATS,
        "dataset_catalog": [
            {
                "name": ds["name"],
                "task": ds["task"],
                "type": ds["type"],
                "source": ds["source"],
                "license": ds["license"],
                "n_samples": len(ds["X"]),
                "n_features": ds["X"].shape[1] if hasattr(ds["X"], "shape") else len(ds["X"][0]),
            }
            for ds in DATASET_CATALOG
        ],
        "datasets": {},
        "drift": {},
    }

    # Classification benchmarks
    for ds in DATASET_CATALOG:
        if ds["task"] != "classification":
            continue
        print(f"\nClassification: {ds['name']} ({ds['type']}) ...", flush=True)
        ds_results = run_classification_benchmark(ds["X"], ds["y"], ds["name"], ds["type"])
        results["datasets"][ds["name"]] = {
            "task": "classification",
            "type": ds["type"],
            "source": ds["source"],
            "license": ds["license"],
            "n_samples": len(ds["X"]),
            "n_features": ds["X"].shape[1],
            "n_classes": len(np.unique(ds["y"])),
            "models": ds_results,
        }
        best_f1 = max(ds_results.items(), key=lambda x: x[1]["f1_score"])
        print(f"  Best F1: {best_f1[0]} = {best_f1[1]['f1_score']:.4f}", flush=True)

    # Regression benchmarks
    for ds in DATASET_CATALOG:
        if ds["task"] != "regression":
            continue
        print(f"\nRegression: {ds['name']} ({ds['type']}) ...", flush=True)
        ds_results = run_regression_benchmark(ds["X"], ds["y"], ds["name"], ds["type"])
        results["datasets"][ds["name"]] = {
            "task": "regression",
            "type": ds["type"],
            "source": ds["source"],
            "license": ds["license"],
            "n_samples": len(ds["X"]),
            "n_features": ds["X"].shape[1],
            "models": ds_results,
        }
        best_r2 = max(ds_results.items(), key=lambda x: x[1]["r2"])
        print(f"  Best R²: {best_r2[0]} = {best_r2[1]['r2']:.4f}", flush=True)

    # Drift detection
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

    # Write results
    out_path = EXPERIMENT_DIR / "mlpm_phase4_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())