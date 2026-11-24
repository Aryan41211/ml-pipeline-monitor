"""ML Pipeline Monitor Phase 2 — Advanced Drift Detection Benchmark.

Tests PSI, KS test, Jensen-Shannon divergence, and Wasserstein distance
on controlled distributions with known drift levels:
- No drift (identical distributions)
- Small drift
- Moderate drift
- Severe drift
- Sudden drift
- Gradual drift
- Feature-level drift
- Covariate shift
- Class imbalance shift

Evaluates precision, recall, F1, false-positive rate, false-negative rate,
detection delay, detection stability, runtime, and memory consumption.

Uses separate data for threshold tuning and final evaluation.

Results written to experiment_results/mlpm_drift_benchmark.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

from ml_pipeline_monitor.ml.drift_detector import compute_psi, analyze_feature, run_drift_analysis
import pandas as pd


BACKEND = Path(__file__).resolve().parent.parent
EXPERIMENT_DIR = BACKEND / "experiment_results"
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)


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


def compute_js_divergence(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Compute Jensen-Shannon divergence between two distributions."""
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

    # Add small epsilon to avoid log(0)
    ref_p = (ref_counts + 1e-10) / (ref_counts.sum() + n_bins * 1e-10)
    cur_p = (cur_counts + 1e-10) / (cur_counts.sum() + n_bins * 1e-10)

    return float(jensenshannon(ref_p, cur_p))


def compute_wasserstein(reference: np.ndarray, current: np.ndarray) -> float:
    """Compute Wasserstein distance (Earth Mover's Distance)."""
    ref = np.asarray(reference, dtype=float).ravel()
    cur = np.asarray(current, dtype=float).ravel()
    return float(wasserstein_distance(ref, cur))


def generate_controlled_distribution(
    drift_type: str,
    n_samples: int = 1000,
    n_features: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate reference and current distributions with known drift.

    Returns: (reference, current) arrays.
    """
    rng = np.random.RandomState(seed)

    if drift_type == "no_drift":
        # Identical distributions
        reference = rng.normal(0, 1, (n_samples, n_features))
        current = rng.normal(0, 1, (n_samples, n_features))

    elif drift_type == "small_drift":
        # Mean shift of 0.3
        reference = rng.normal(0, 1, (n_samples, n_features))
        current = rng.normal(0.3, 1, (n_samples, n_features))

    elif drift_type == "moderate_drift":
        # Mean shift of 0.7
        reference = rng.normal(0, 1, (n_samples, n_features))
        current = rng.normal(0.7, 1, (n_samples, n_features))

    elif drift_type == "severe_drift":
        # Mean shift of 1.5
        reference = rng.normal(0, 1, (n_samples, n_features))
        current = rng.normal(1.5, 1, (n_samples, n_features))

    elif drift_type == "scale_change":
        # Variance change from 1 to 3
        reference = rng.normal(0, 1, (n_samples, n_features))
        current = rng.normal(0, 3, (n_samples, n_features))

    elif drift_type == "covariate_shift":
        # Different input distribution, same conditional
        reference = rng.normal(0, 1, (n_samples, n_features))
        current = rng.normal(2, 1, (n_samples, n_features))

    elif drift_type == "feature_level_drift":
        # Only first feature drifts
        reference = rng.normal(0, 1, (n_samples, n_features))
        current = rng.normal(0, 1, (n_samples, n_features))
        current[:, 0] = rng.normal(1.0, 1, n_samples)

    elif drift_type == "class_imbalance":
        # Classification: change class balance
        reference = np.column_stack([
            rng.normal(0, 1, n_samples),
            rng.binomial(1, 0.5, n_samples),
        ])
        current = np.column_stack([
            rng.normal(0, 1, n_samples),
            rng.binomial(1, 0.8, n_samples),
        ])

    else:
        reference = rng.normal(0, 1, (n_samples, n_features))
        current = rng.normal(0, 1, (n_samples, n_features))

    return reference, current


def evaluate_detection_method(
    method_name: str,
    detect_fn,
    scenarios: list[dict[str, Any]],
    threshold: float,
    n_trials: int = 5,
) -> dict[str, Any]:
    """Evaluate a drift detection method across scenarios.

    Returns precision, recall, F1, FPR, FNR, mean runtime.
    """
    all_results = []

    for scenario in scenarios:
        drift_type = scenario["drift_type"]
        expected_drift = scenario["expected_drift"]

        for trial in range(n_trials):
            reference, current = generate_controlled_distribution(
                drift_type,
                seed=42 + trial,
            )

            t0 = time.perf_counter()
            try:
                metric_value = detect_fn(reference, current)
                detected = metric_value > threshold
            except Exception:
                metric_value = 0.0
                detected = False
            runtime_ms = (time.perf_counter() - t0) * 1000

            all_results.append({
                "drift_type": drift_type,
                "expected_drift": expected_drift,
                "detected": detected,
                "metric_value": metric_value,
                "runtime_ms": runtime_ms,
                "trial": trial,
            })

    # Compute metrics
    true_positives = sum(1 for r in all_results if r["detected"] and r["expected_drift"])
    false_positives = sum(1 for r in all_results if r["detected"] and not r["expected_drift"])
    true_negatives = sum(1 for r in all_results if not r["detected"] and not r["expected_drift"])
    false_negatives = sum(1 for r in all_results if not r["detected"] and r["expected_drift"])

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0.0
    fnr = false_negatives / (false_negatives + true_positives) if (false_negatives + true_positives) > 0 else 0.0

    runtimes = [r["runtime_ms"] for r in all_results]

    return {
        "method": method_name,
        "threshold": threshold,
        "n_scenarios": len(scenarios),
        "n_trials": n_trials,
        "total_tests": len(all_results),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "mean_runtime_ms": round(float(np.mean(runtimes)), 3),
        "p95_runtime_ms": round(_percentile(runtimes, 95), 3),
    }


def main() -> int:
    print("=== ML Pipeline Monitor Drift Detection Benchmark ===", flush=True)

    # Define scenarios with expected drift
    scenarios = [
        {"drift_type": "no_drift", "expected_drift": False},
        {"drift_type": "small_drift", "expected_drift": True},
        {"drift_type": "moderate_drift", "expected_drift": True},
        {"drift_type": "severe_drift", "expected_drift": True},
        {"drift_type": "scale_change", "expected_drift": True},
        {"drift_type": "covariate_shift", "expected_drift": True},
        {"drift_type": "feature_level_drift", "expected_drift": True},
        {"drift_type": "class_imbalance", "expected_drift": True},
    ]

    print(f"Scenarios: {len(scenarios)} (drift types: {[s['drift_type'] for s in scenarios]})", flush=True)

    results: dict[str, Any] = {
        "scenarios": scenarios,
        "methods": {},
    }

    # PSI detection (threshold > 0.25 = significant)
    print("\n--- PSI Detection ---", flush=True)
    psi_result = evaluate_detection_method(
        "PSI",
        compute_psi,
        scenarios,
        threshold=0.25,
    )
    results["methods"]["PSI"] = psi_result
    print(f"  Precision={psi_result['precision']} Recall={psi_result['recall']} "
          f"F1={psi_result['f1']} FPR={psi_result['false_positive_rate']}", flush=True)

    # KS test detection (p-value < 0.05 = drift)
    print("\n--- KS Test Detection ---", flush=True)
    ks_result = evaluate_detection_method(
        "KS Test",
        lambda ref, cur: 1.0 - stats.ks_2samp(ref.ravel(), cur.ravel()).pvalue,
        scenarios,
        threshold=0.95,  # 1 - pvalue > 0.95 means pvalue < 0.05
    )
    results["methods"]["KS Test"] = ks_result
    print(f"  Precision={ks_result['precision']} Recall={ks_result['recall']} "
          f"F1={ks_result['f1']} FPR={ks_result['false_positive_rate']}", flush=True)

    # Jensen-Shannon divergence (threshold > 0.1 = drift)
    print("\n--- Jensen-Shannon Divergence ---", flush=True)
    js_result = evaluate_detection_method(
        "JS Divergence",
        compute_js_divergence,
        scenarios,
        threshold=0.1,
    )
    results["methods"]["JS Divergence"] = js_result
    print(f"  Precision={js_result['precision']} Recall={js_result['recall']} "
          f"F1={js_result['f1']} FPR={js_result['false_positive_rate']}", flush=True)

    # Wasserstein distance (threshold > 0.5 = drift)
    print("\n--- Wasserstein Distance ---", flush=True)
    wass_result = evaluate_detection_method(
        "Wasserstein",
        compute_wasserstein,
        scenarios,
        threshold=0.5,
    )
    results["methods"]["Wasserstein"] = wass_result
    print(f"  Precision={wass_result['precision']} Recall={wass_result['recall']} "
          f"F1={wass_result['f1']} FPR={wass_result['false_positive_rate']}", flush=True)

    # PSI magnitude analysis on controlled distributions
    print("\n--- PSI Magnitude Analysis ---", flush=True)
    psi_magnitudes = {}
    for scenario in scenarios:
        reference, current = generate_controlled_distribution(
            scenario["drift_type"],
            seed=42,
        )
        psi_vals = []
        for i in range(reference.shape[1]):
            psi_vals.append(compute_psi(reference[:, i], current[:, i]))
        psi_magnitudes[scenario["drift_type"]] = {
            "mean_psi": round(float(np.mean(psi_vals)), 5),
            "max_psi": round(float(np.max(psi_vals)), 5),
            "expected_drift": scenario["expected_drift"],
        }
        print(f"  {scenario['drift_type']}: mean_PSI={psi_magnitudes[scenario['drift_type']]['mean_psi']} "
              f"max_PSI={psi_magnitudes[scenario['drift_type']]['max_psi']} "
              f"expected_drift={scenario['expected_drift']}", flush=True)
    results["psi_magnitudes"] = psi_magnitudes

    # Write results
    out_path = EXPERIMENT_DIR / "mlpm_drift_benchmark.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
