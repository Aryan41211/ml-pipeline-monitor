from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_validation import validate_dataset


def _base_classification_frame(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    x1 = rng.normal(0, 1, size=n)
    x2 = rng.normal(0, 1, size=n)
    # Balanced-ish
    y = np.array([0] * (n // 2) + [1] * (n - n // 2))
    rng.shuffle(y)
    return pd.DataFrame({"f1": x1, "f2": x2, "target": y})


def test_empty_frame_fails():
    df = pd.DataFrame(columns=["f1", "target"])
    res = validate_dataset(df, task="classification", target_name="target")
    assert res.status == "fail"
    assert res.quality_score == 0.0
    assert "empty_frame" in res.fail_reasons


def test_missing_target_column_fails():
    df = pd.DataFrame({"f1": [1, 2, 3]})
    res = validate_dataset(df, task="classification", target_name="target")
    assert res.status == "fail"
    assert res.quality_score == 0.0
    assert "missing_target_column" in res.fail_reasons[0]


def test_missing_values_and_duplicates_trigger_fail_reasons():
    df = _base_classification_frame(40)

    # Missing values
    df.loc[0, "f1"] = np.nan
    df.loc[1, "f2"] = np.nan

    # Duplicate rows (duplicate exact rows)
    df2 = pd.concat([df, df.iloc[[2, 3]].copy()], ignore_index=True)
    df2 = pd.concat([df2, df2.iloc[[2]].copy()], ignore_index=True)

    res = validate_dataset(
        df2,
        task="classification",
        target_name="target",
        missing_total_threshold_pct=0.0001,  # force fail
        duplicates_threshold_pct=0.0001,     # force fail
        quality_score_min=100.0,              # force fail by score too
    )
    assert res.status == "fail"
    # At least one of these should be present
    assert any(
        k in res.fail_reasons
        for k in ("missing_values_too_high", "duplicates_too_high", "quality_score_below_minimum")
    )


def test_constant_features_detected():
    df = _base_classification_frame(60)
    df["f1"] = 1.0  # constant
    res = validate_dataset(
        df,
        task="classification",
        target_name="target",
        constant_features_threshold=0,
        quality_score_min=100.0,
    )
    assert res.status == "fail"
    assert "constant_features_detected" in res.fail_reasons


def test_object_dtype_detection_counts_object_columns():
    df = _base_classification_frame(50).copy()
    df["f2"] = df["f2"].astype(object)
    res = validate_dataset(
        df,
        task="classification",
        target_name="target",
        object_columns_threshold=0,
        quality_score_min=100.0,
    )
    assert res.status == "fail"
    assert "dtype_mismatch_suspected" in res.fail_reasons


def test_outliers_and_invalid_ranges_trigger_reasons():
    # Create numeric features with extreme values
    rng = np.random.default_rng(0)
    n = 200
    f1 = rng.normal(0, 1, size=n)
    f2 = rng.normal(0, 1, size=n)

    # Add outliers
    f1[0] = 50.0
    f1[1] = -50.0

    y = rng.integers(0, 2, size=n)
    df = pd.DataFrame({"f1": f1, "f2": f2, "target": y})

    res = validate_dataset(
        df,
        task="classification",
        target_name="target",
        outlier_method="zscore",
        outlier_z_threshold=2.0,
        outliers_threshold_total=0,     # force outlier fail
        invalid_ranges_features_threshold=0,  # force invalid-range fail
        quantile_low=0.01,
        quantile_high=0.99,
        quality_score_min=100.0,
    )
    assert res.status == "fail"
    assert "outliers_too_many" in res.fail_reasons or "invalid_ranges_detected" in res.fail_reasons


def test_class_imbalance_detection():
    df = _base_classification_frame(100)
    # Make it very imbalanced: mostly zeros
    df["target"] = 0
    df.loc[df.index[:5], "target"] = 1

    res = validate_dataset(
        df,
        task="classification",
        target_name="target",
        imbalance_ratio_threshold=2.0,  # force class imbalance fail
        quality_score_min=100.0,
    )
    assert res.status == "fail"
    assert "class_imbalance_too_high" in res.fail_reasons


def test_high_correlation_detection_records_pairs():
    rng = np.random.default_rng(123)
    n = 120
    x = rng.normal(0, 1, size=n)
    # Strongly correlated features
    df = pd.DataFrame(
        {
            "f1": x,
            "f2": x * 2.0 + rng.normal(0, 0.01, size=n),
            "target": rng.integers(0, 2, size=n),
        }
    )

    res = validate_dataset(
        df,
        task="classification",
        target_name="target",
        high_corr_threshold=0.8,
        quality_score_min=0.0,  # ensure pass by score if everything else is OK
    )
    assert res.status in ("pass", "fail")
    report = res.report
    assert "high_correlation" in report
    assert report["high_correlation"]["high_corr_pairs"] >= 1
