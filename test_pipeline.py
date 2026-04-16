"""
Unit tests for the core pipeline and supporting utilities.

Run with: pytest tests/ -v
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.drift_detector import compute_psi, run_drift_analysis
from src.pipeline import MLPipeline
from src.data_loader import load_dataset, DATASET_OPTIONS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def clf_splits():
    X, y = make_classification(
        n_samples=400, n_features=10, n_informative=6, random_state=0
    )
    cols = [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    target = pd.Series(y, name="target")
    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.25, random_state=0)
    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="module")
def reg_splits():
    X, y = make_regression(n_samples=400, n_features=10, n_informative=6, random_state=0)
    cols = [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    target = pd.Series(y, name="target")
    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.25, random_state=0)
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------

class TestMLPipeline:
    @pytest.mark.parametrize("model_type", ["Random Forest", "Logistic Regression"])
    def test_classification_completes(self, clf_splits, model_type):
        X_train, X_test, y_train, y_test = clf_splits
        pipeline = MLPipeline(
            dataset_name="test",
            model_type=model_type,
            task="classification",
            params={},
            cv_folds=3,
            random_state=42,
        )
        result = pipeline.run(X_train, X_test, y_train, y_test)

        assert result.status == "completed"
        assert 0.0 <= result.metrics["accuracy"] <= 1.0
        assert 0.0 <= result.metrics["f1_score"] <= 1.0
        assert result.cv_scores is not None
        assert len(result.cv_scores) == 3
        assert result.confusion_mat is not None

    @pytest.mark.parametrize("model_type", ["Random Forest", "Ridge Regression"])
    def test_regression_completes(self, reg_splits, model_type):
        X_train, X_test, y_train, y_test = reg_splits
        pipeline = MLPipeline(
            dataset_name="test_reg",
            model_type=model_type,
            task="regression",
            params={},
            cv_folds=3,
            random_state=42,
        )
        result = pipeline.run(X_train, X_test, y_train, y_test)

        assert result.status == "completed"
        assert "rmse" in result.metrics
        assert "r2" in result.metrics
        assert result.cv_scores is not None

    def test_run_id_is_unique(self, clf_splits):
        X_train, X_test, y_train, y_test = clf_splits
        ids = set()
        for _ in range(5):
            p = MLPipeline("test", "Random Forest", "classification", {}, cv_folds=2)
            ids.add(p.run_id)
        assert len(ids) == 5

    def test_feature_importances_present(self, clf_splits):
        X_train, X_test, y_train, y_test = clf_splits
        pipeline = MLPipeline(
            dataset_name="test",
            model_type="Random Forest",
            task="classification",
            params={"n_estimators": 20},
            cv_folds=2,
        )
        result = pipeline.run(X_train, X_test, y_train, y_test)
        assert result.feature_importances is not None
        assert len(result.feature_importances) == X_train.shape[1]
        assert abs(result.feature_importances.sum() - 1.0) < 1e-6

    def test_progress_callback_is_called(self, clf_splits):
        X_train, X_test, y_train, y_test = clf_splits
        calls = []

        def cb(stage, progress, msg):
            calls.append((stage, progress))

        pipeline = MLPipeline(
            "test", "Decision Tree", "classification", {}, cv_folds=2,
            progress_callback=cb,
        )
        pipeline.run(X_train, X_test, y_train, y_test)
        assert len(calls) >= 7
        assert calls[-1][1] == 1.0


# ---------------------------------------------------------------------------
# Drift detector tests
# ---------------------------------------------------------------------------

class TestDriftDetector:
    def test_psi_identical_distributions_is_zero(self):
        rng = np.random.default_rng(1)
        arr = rng.normal(0, 1, 1000)
        psi = compute_psi(arr, arr)
        assert psi < 0.01

    def test_psi_large_shift_is_high(self):
        rng = np.random.default_rng(2)
        ref = rng.normal(0, 1, 1000)
        cur = rng.normal(5, 1, 1000)
        psi = compute_psi(ref, cur)
        assert psi > 0.25

    def test_drift_analysis_no_drift(self):
        rng = np.random.default_rng(3)
        ref = pd.DataFrame(rng.normal(0, 1, (500, 5)), columns=[f"f{i}" for i in range(5)])
        cur = pd.DataFrame(rng.normal(0, 1, (200, 5)), columns=[f"f{i}" for i in range(5)])
        result = run_drift_analysis(ref, cur, alpha=0.05)
        assert result["features_analyzed"] == 5
        # no (or minimal) drift expected from same distribution
        assert result["drift_ratio"] < 0.5

    def test_drift_analysis_detects_shift(self):
        rng = np.random.default_rng(4)
        ref = pd.DataFrame(rng.normal(0, 1, (500, 5)), columns=[f"f{i}" for i in range(5)])
        cur = pd.DataFrame(rng.normal(4, 1, (200, 5)), columns=[f"f{i}" for i in range(5)])
        result = run_drift_analysis(ref, cur, alpha=0.05)
        assert result["overall_drift"] is True
        assert result["features_drifted"] > 0

    def test_drift_analysis_empty_overlap(self):
        rng = np.random.default_rng(5)
        ref = pd.DataFrame(rng.normal(0, 1, (100, 3)), columns=["a", "b", "c"])
        cur = pd.DataFrame(rng.normal(0, 1, (100, 2)), columns=["a", "b"])
        result = run_drift_analysis(ref, cur)
        assert result["features_analyzed"] == 2


# ---------------------------------------------------------------------------
# Data loader tests
# ---------------------------------------------------------------------------

class TestDataLoader:
    @pytest.mark.parametrize("key", ["breast_cancer", "iris", "synthetic_clf"])
    def test_classification_datasets_load(self, key):
        ds = load_dataset(key, test_size=0.20, random_state=0)
        assert ds["task"] == "classification"
        assert len(ds["X_train"]) > 0
        assert len(ds["X_test"]) > 0
        assert ds["stats"]["missing_values"] == 0

    def test_regression_dataset_loads(self):
        ds = load_dataset("synthetic_reg", test_size=0.20, random_state=0)
        assert ds["task"] == "regression"
        assert ds["stats"]["n_features"] == 20

    def test_invalid_key_raises(self):
        with pytest.raises(ValueError):
            load_dataset("nonexistent_dataset")

    def test_train_test_sizes_consistent(self):
        ds = load_dataset("iris", test_size=0.30, random_state=42)
        total = ds["stats"]["train_size"] + ds["stats"]["test_size"]
        assert total == ds["stats"]["n_samples"]
