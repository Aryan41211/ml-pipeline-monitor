"""
Dataset loading and splitting utilities.

Wraps scikit-learn's bundled datasets and supports synthetic generation
so the application works without external data files.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from typing import Any, Dict, List, Optional


# Human-readable labels -> internal key used in config.yaml
DATASET_OPTIONS: Dict[str, str] = {
    "Breast Cancer Wisconsin": "breast_cancer",
    "Wine Recognition": "wine",
    "Iris Species": "iris",
    "Handwritten Digits": "digits",
    "Synthetic Classification": "synthetic_clf",
    "Synthetic Regression": "synthetic_reg",
}


def _make_sklearn_bundle(loader_fn, **kwargs) -> Dict[str, Any]:
    """Return a normalised dict from a sklearn dataset loader."""
    data = loader_fn(as_frame=True, **kwargs)
    X = data.data if isinstance(data.data, pd.DataFrame) else pd.DataFrame(data.data)
    y = data.target
    target_names = (
        list(data.target_names) if hasattr(data, "target_names") else None
    )
    return {"X": X, "y": y, "target_names": target_names}


def load_dataset(
    key: str,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Load a dataset by its internal key and return train/test splits.

    Parameters
    ----------
    key : str
        One of the keys in DATASET_OPTIONS.
    test_size : float
        Fraction of samples held out for evaluation.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    dict with keys: X_train, X_test, y_train, y_test,
                    feature_names, target_names, task, stats
    """
    if key == "breast_cancer":
        bundle = _make_sklearn_bundle(datasets.load_breast_cancer)
        task = "classification"
    elif key == "wine":
        bundle = _make_sklearn_bundle(datasets.load_wine)
        task = "classification"
    elif key == "iris":
        bundle = _make_sklearn_bundle(datasets.load_iris)
        task = "classification"
    elif key == "digits":
        bundle = _make_sklearn_bundle(datasets.load_digits)
        task = "classification"
    elif key == "synthetic_clf":
        X_arr, y_arr = datasets.make_classification(
            n_samples=2_000,
            n_features=20,
            n_informative=12,
            n_redundant=4,
            n_clusters_per_class=2,
            random_state=random_state,
        )
        cols = [f"feature_{i:02d}" for i in range(X_arr.shape[1])]
        bundle = {
            "X": pd.DataFrame(X_arr, columns=cols),
            "y": pd.Series(y_arr, name="target"),
            "target_names": ["class_0", "class_1"],
        }
        task = "classification"
    elif key == "synthetic_reg":
        X_arr, y_arr = datasets.make_regression(
            n_samples=2_000,
            n_features=20,
            n_informative=12,
            noise=0.15,
            random_state=random_state,
        )
        cols = [f"feature_{i:02d}" for i in range(X_arr.shape[1])]
        bundle = {
            "X": pd.DataFrame(X_arr, columns=cols),
            "y": pd.Series(y_arr, name="target"),
            "target_names": None,
        }
        task = "regression"
    else:
        raise ValueError(f"Unknown dataset key: {key!r}")

    X: pd.DataFrame = bundle["X"]
    y: pd.Series = bundle["y"]

    stratify = y if task == "classification" and y.nunique() <= 20 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    stats = _compute_stats(X, y, X_train, X_test, task)

    return {
        "X_train": X_train.reset_index(drop=True),
        "X_test": X_test.reset_index(drop=True),
        "y_train": y_train.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "feature_names": list(X.columns),
        "target_names": bundle["target_names"],
        "task": task,
        "stats": stats,
    }


def _compute_stats(
    X: pd.DataFrame,
    y: pd.Series,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    task: str,
) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "n_samples": len(X),
        "n_features": X.shape[1],
        "train_size": len(X_train),
        "test_size": len(X_test),
        "missing_values": int(X.isnull().sum().sum()),
    }
    if task == "classification":
        stats["n_classes"] = int(y.nunique())
        stats["class_distribution"] = y.value_counts(normalize=True).round(4).to_dict()
    else:
        stats["target_mean"] = round(float(y.mean()), 4)
        stats["target_std"] = round(float(y.std()), 4)
        stats["target_min"] = round(float(y.min()), 4)
        stats["target_max"] = round(float(y.max()), 4)
    return stats


def get_feature_statistics(X: pd.DataFrame) -> pd.DataFrame:
    """Return per-feature descriptive statistics as a DataFrame."""
    desc = X.describe().T.copy()
    desc["missing"] = X.isnull().sum()
    desc["missing_pct"] = (X.isnull().mean() * 100).round(2)
    desc["skewness"] = X.skew().round(4)
    desc["kurtosis"] = X.kurt().round(4)
    return desc.round(4)
