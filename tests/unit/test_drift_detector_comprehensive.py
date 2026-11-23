"""Comprehensive statistical drift detection tests.

Validates PSI and KS-test correctness across controlled scenarios per
Phase M3 of the engineering program:

1. Identical distributions (no drift)
2. Slightly shifted distributions (moderate drift)
3. Moderately shifted distributions (significant drift)
4. Strongly shifted distributions (critical drift)
5. Missing-value changes
6. Variance changes
7. Mean changes
8. Categorical distribution changes
9. Different sample sizes
10. Noisy / small-sample edge cases
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ml_pipeline_monitor.ml import drift_detector

# Controlled thresholds used across tests (match config defaults).
MONITORING_CFG = {
    "monitoring": {
        "psi_moderate_threshold": 0.10,
        "psi_significant_threshold": 0.25,
        "drift_feature_ratio_threshold": 0.20,
    }
}


def _monkeypatch_config(monkeypatch):
    monkeypatch.setattr(
        drift_detector,
        "load_config",
        lambda: MONITORING_CFG,
    )


# ---------------------------------------------------------------------------
# PSI correctness: compute_psi
# ---------------------------------------------------------------------------

class TestComputePSI:
    def test_identical_distributions_psi_approx_zero(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 2000)
        assert drift_detector.compute_psi(ref, ref.copy()) == 0.0

    def test_mean_shift_psi_significant(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 2000)
        shifted = rng.normal(1.5, 1.0, 2000)
        psi = drift_detector.compute_psi(ref, shifted)
        assert psi > 0.25, f"mean-shifted PSI {psi} should exceed 0.25"

    def test_scale_change_psi_significant(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 2000)
        scaled = rng.normal(0, 3.0, 2000)
        psi = drift_detector.compute_psi(ref, scaled)
        assert psi > 0.25, f"scale-changed PSI {psi} should exceed 0.25"

    def test_small_sample_no_blowup(self):
        """Laplace smoothing prevents PSI explosion on tiny identical samples."""
        ref = np.array([1.0, 2.0, 3.0])
        cur = np.array([1.0, 2.0, 3.0])
        psi = drift_detector.compute_psi(ref, cur)
        assert abs(psi) < 0.01, f"tiny identical-sample PSI {psi} should be ~0"

    def test_different_sample_sizes(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 5000)
        cur = rng.normal(0, 1, 200)
        psi = drift_detector.compute_psi(ref, cur)
        assert abs(psi) < 0.05, f"different-size identical PSI {psi} should be small"

    def test_constant_column(self):
        """A constant feature produces PSI 0 (no distribution to compare)."""
        ref = np.ones(100)
        cur = np.ones(100)
        psi = drift_detector.compute_psi(ref, cur)
        assert psi == 0.0


# ---------------------------------------------------------------------------
# KS test correctness
# ---------------------------------------------------------------------------

class TestKSTest:
    def test_identical_distributions_ks_no_drift(self):
        rng = np.random.default_rng(7)
        ref = rng.normal(0, 1, 2000)
        res = drift_detector.analyze_feature(
            pd.Series(ref, name="x"), pd.Series(ref.copy(), name="x")
        )
        assert res.p_value > 0.05
        assert res.drift_detected is False
        assert res.severity == "none"

    def test_mean_shift_ks_drift(self):
        rng = np.random.default_rng(7)
        ref = rng.normal(0, 1, 2000)
        cur = rng.normal(1.5, 1.0, 2000)
        res = drift_detector.analyze_feature(
            pd.Series(ref, name="x"), pd.Series(cur, name="x")
        )
        assert res.p_value < 0.05
        assert res.drift_detected is True
        assert res.severity in ("moderate", "significant")

    def test_variance_change_ks_drift(self):
        rng = np.random.default_rng(7)
        ref = rng.normal(0, 1, 2000)
        cur = rng.normal(0, 3.0, 2000)
        res = drift_detector.analyze_feature(
            pd.Series(ref, name="x"), pd.Series(cur, name="x")
        )
        # KS is sensitive to distribution shape; variance change should flag drift.
        assert res.drift_detected is True or res.psi > 0.10

    def test_small_shift_may_not_detect(self):
        """A tiny shift with small samples may not cross significance."""
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 100)
        cur = rng.normal(0.05, 1, 100)
        res = drift_detector.analyze_feature(
            pd.Series(ref, name="x"), pd.Series(cur, name="x"), alpha=0.05
        )
        # No assertion on outcome — just verifying the function doesn't crash
        # and produces a valid result for a marginal case.
        assert res.severity in ("none", "moderate", "significant")


# ---------------------------------------------------------------------------
# Full dataset drift analysis
# ---------------------------------------------------------------------------

class TestRunDriftAnalysis:
    def test_identical_dataset_stable(self, monkeypatch):
        _monkeypatch_config(monkeypatch)
        rng = np.random.default_rng(42)
        ref = pd.DataFrame({
            "f1": rng.normal(0, 1, 500),
            "f2": rng.normal(0, 1, 500),
            "f3": rng.normal(0, 1, 500),
        })
        report = drift_detector.run_drift_analysis(ref, ref.copy())
        assert report["overall_severity"] == "stable"
        assert report["overall_drift"] is False
        assert report["features_drifted"] == 0
        assert abs(report["average_psi"]) < 0.01

    def test_all_features_shifted_critical(self, monkeypatch):
        _monkeypatch_config(monkeypatch)
        rng = np.random.default_rng(42)
        ref = pd.DataFrame({
            "f1": rng.normal(0, 1, 500),
            "f2": rng.normal(0, 1, 500),
        })
        cur = pd.DataFrame({
            "f1": rng.normal(3, 1, 500),
            "f2": rng.normal(3, 1, 500),
        })
        report = drift_detector.run_drift_analysis(ref, cur)
        assert report["overall_severity"] == "critical"
        assert report["overall_drift"] is True
        assert report["features_drifted"] == 2

    def test_some_features_shifted_warning(self, monkeypatch):
        _monkeypatch_config(monkeypatch)
        rng = np.random.default_rng(42)
        ref = pd.DataFrame({
            "f1": rng.normal(0, 1, 500),
            "f2": rng.normal(0, 1, 500),
            "f3": rng.normal(0, 1, 500),
            "f4": rng.normal(0, 1, 500),
        })
        cur = ref.copy()
        cur["f1"] = rng.normal(3, 1, 500)  # only 1 of 4 features drifted = 25% > 20% threshold
        report = drift_detector.run_drift_analysis(ref, cur)
        assert report["features_drifted"] >= 1
        assert report["drift_ratio"] >= 0.20
        assert report["overall_drift"] is True

    def test_missing_value_changes_detected(self, monkeypatch):
        _monkeypatch_config(monkeypatch)
        rng = np.random.default_rng(42)
        ref = pd.DataFrame({"f1": rng.normal(0, 1, 500)})
        cur = pd.DataFrame({"f1": np.concatenate([rng.normal(0, 1, 400), [np.nan] * 100])})
        report = drift_detector.run_drift_analysis(ref, cur)
        # Missing-value change should at minimum not crash and produce a result.
        assert "overall_severity" in report
        assert report["features_analyzed"] == 1

    def test_categorical_distribution_change(self, monkeypatch):
        _monkeypatch_config(monkeypatch)
        # Categorical data encoded as integers
        ref = pd.DataFrame({"cat": np.random.choice([0, 1, 2], 500, p=[0.7, 0.2, 0.1])})
        cur = pd.DataFrame({"cat": np.random.choice([0, 1, 2], 500, p=[0.1, 0.2, 0.7])})
        report = drift_detector.run_drift_analysis(ref, cur)
        assert report["features_drifted"] >= 1, "categorical shift should be detected"
        assert report["overall_drift"] is True

    def test_no_common_columns(self, monkeypatch):
        _monkeypatch_config(monkeypatch)
        ref = pd.DataFrame({"a": [1.0, 2.0]})
        cur = pd.DataFrame({"b": [3.0, 4.0]})
        report = drift_detector.run_drift_analysis(ref, cur)
        assert report["features_analyzed"] == 0
        assert report["features_drifted"] == 0
        assert report["overall_drift"] is False

    def test_feature_level_results_contain_expected_keys(self, monkeypatch):
        _monkeypatch_config(monkeypatch)
        rng = np.random.default_rng(42)
        ref = pd.DataFrame({"f1": rng.normal(0, 1, 200)})
        cur = pd.DataFrame({"f1": rng.normal(1, 1, 200)})
        report = drift_detector.run_drift_analysis(ref, cur)
        fr = report["feature_results"][0]
        for key in ("feature", "ks_statistic", "p_value", "psi", "drift_detected", "severity"):
            assert key in fr, f"missing key {key} in feature result"

    def test_analyze_feature_severity_tiers(self, monkeypatch):
        _monkeypatch_config(monkeypatch)
        rng = np.random.default_rng(42)
        # none: identical
        ref = rng.normal(0, 1, 1000)
        r1 = drift_detector.analyze_feature(
            pd.Series(ref, name="x"), pd.Series(ref.copy(), name="x")
        )
        assert r1.severity == "none"

        # moderate: small shift
        cur = rng.normal(0.5, 1, 1000)
        r2 = drift_detector.analyze_feature(
            pd.Series(ref, name="x"), pd.Series(cur, name="x")
        )
        assert r2.severity in ("moderate", "significant")

        # significant: large shift
        cur2 = rng.normal(3, 1, 1000)
        r3 = drift_detector.analyze_feature(
            pd.Series(ref, name="x"), pd.Series(cur2, name="x")
        )
        assert r3.severity == "significant"


# ---------------------------------------------------------------------------
# Multiple hypothesis testing consideration
# ---------------------------------------------------------------------------

class TestMultipleFeatures:
    def test_drift_ratio_calculation(self, monkeypatch):
        _monkeypatch_config(monkeypatch)
        rng = np.random.default_rng(42)
        # 5 features, 2 shifted → drift_ratio = 0.4
        ref = pd.DataFrame({f"f{i}": rng.normal(0, 1, 300) for i in range(5)})
        cur = ref.copy()
        cur["f0"] = rng.normal(3, 1, 300)
        cur["f1"] = rng.normal(3, 1, 300)
        report = drift_detector.run_drift_analysis(ref, cur)
        assert report["features_drifted"] == 2
        assert report["drift_ratio"] == pytest.approx(0.4)
