"""
Statistical data drift detection.

Implements the Kolmogorov-Smirnov two-sample test for continuous features
and the Population Stability Index (PSI) as a complementary measure.
A feature is flagged as drifted when either the KS p-value falls below the
significance level or the PSI exceeds the moderate-change threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# PSI
# ---------------------------------------------------------------------------

def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Population Stability Index between two continuous distributions.

    PSI interpretation:
      < 0.10  — no significant change
      0.10–0.25 — moderate shift; worth monitoring
      > 0.25  — significant distribution change

    Parameters
    ----------
    reference : array-like
        Baseline distribution (e.g. training data).
    current : array-like
        Observed distribution to compare against baseline.
    n_bins : int
        Number of histogram bins used for discretisation.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)

    lo = min(ref.min(), cur.min())
    hi = max(ref.max(), cur.max())
    edges = np.linspace(lo, hi, n_bins + 1)

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)

    ref_pct = ref_counts / ref_counts.sum()
    cur_pct = cur_counts / cur_counts.sum()

    # Avoid log(0) or division by zero
    ref_pct = np.where(ref_pct == 0, 1e-6, ref_pct)
    cur_pct = np.where(cur_pct == 0, 1e-6, cur_pct)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return round(psi, 5)


# ---------------------------------------------------------------------------
# Per-feature result
# ---------------------------------------------------------------------------

@dataclass
class FeatureDriftResult:
    feature: str
    ks_statistic: float
    p_value: float
    psi: float
    drift_detected: bool
    severity: str     # 'none' | 'moderate' | 'significant'


def _classify_severity(p_value: float, psi: float, alpha: float) -> str:
    if psi > 0.25 or p_value < 0.01:
        return "significant"
    if psi > 0.10 or p_value < alpha:
        return "moderate"
    return "none"


def analyze_feature(
    reference: pd.Series,
    current: pd.Series,
    alpha: float = 0.05,
) -> FeatureDriftResult:
    """
    Run KS test + PSI for a single numeric feature.

    Parameters
    ----------
    reference : pd.Series
        Baseline observations.
    current : pd.Series
        New observations to test.
    alpha : float
        Significance level for the KS test.
    """
    ref_clean = reference.dropna().values
    cur_clean = current.dropna().values

    ks_stat, p_value = stats.ks_2samp(ref_clean, cur_clean)
    psi = compute_psi(ref_clean, cur_clean)

    drift_detected = (p_value < alpha) or (psi > 0.10)
    severity = _classify_severity(p_value, psi, alpha)

    return FeatureDriftResult(
        feature=str(reference.name),
        ks_statistic=round(float(ks_stat), 5),
        p_value=round(float(p_value), 5),
        psi=psi,
        drift_detected=drift_detected,
        severity=severity,
    )


# ---------------------------------------------------------------------------
# Full dataset drift report
# ---------------------------------------------------------------------------

def run_drift_analysis(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Compare two DataFrames column by column and return a drift report.

    Returns
    -------
    dict with keys:
        features_analyzed  : int
        features_drifted   : int
        drift_ratio        : float  (0–1)
        overall_drift      : bool
        average_psi        : float
        feature_results    : list[dict]
    """
    common = [c for c in reference.columns if c in current.columns]
    results: List[Dict[str, Any]] = []

    for col in common:
        r = analyze_feature(reference[col], current[col], alpha=alpha)
        results.append(
            {
                "feature": r.feature,
                "ks_statistic": r.ks_statistic,
                "p_value": r.p_value,
                "psi": r.psi,
                "drift_detected": r.drift_detected,
                "severity": r.severity,
            }
        )

    n_drifted = sum(1 for r in results if r["drift_detected"])
    drift_ratio = n_drifted / len(results) if results else 0.0
    avg_psi = float(np.mean([r["psi"] for r in results])) if results else 0.0

    # Dataset-level drift: >20 % of features show drift
    overall_drift = drift_ratio > 0.20

    return {
        "features_analyzed": len(results),
        "features_drifted": n_drifted,
        "drift_ratio": round(drift_ratio, 4),
        "overall_drift": overall_drift,
        "average_psi": round(avg_psi, 5),
        "feature_results": results,
    }
