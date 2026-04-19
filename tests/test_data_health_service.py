from pathlib import Path

import pandas as pd
import pytest

from services import data_health_service


def test_missing_value_report_basic(sample_feature_frame):
    out = data_health_service.missing_value_report(sample_feature_frame)
    assert out["total_missing"] == 1
    assert out["total_missing_pct"] > 0
    assert not out["per_column"].empty


def test_missing_value_report_empty_df():
    out = data_health_service.missing_value_report(pd.DataFrame())
    assert out["total_missing"] == 0
    assert out["total_missing_pct"] == 0.0


def test_class_imbalance_report_classification(sample_target_series):
    out = data_health_service.class_imbalance_report(sample_target_series, "classification")
    assert out["enabled"] is True
    assert out["imbalance_ratio"] is not None
    assert not out["distribution"].empty


def test_class_imbalance_report_non_classification(sample_target_series):
    out = data_health_service.class_imbalance_report(sample_target_series, "regression")
    assert out["enabled"] is False


def test_basic_statistics_report_only_numeric(sample_feature_frame):
    out = data_health_service.basic_statistics_report(sample_feature_frame)
    assert {"feature", "mean", "std", "min", "max"}.issubset(set(out.columns))


def test_outlier_report_iqr_and_zscore(sample_feature_frame):
    iqr = data_health_service.outlier_report(sample_feature_frame, method="iqr")
    zsc = data_health_service.outlier_report(sample_feature_frame, method="zscore", z_threshold=2.0)
    assert not iqr.empty
    assert not zsc.empty
    assert "outlier_count" in iqr.columns


def test_compare_schema_with_and_without_baseline():
    no_base = data_health_service.compare_schema(["a", "b"], None)
    assert no_base["has_baseline"] is False

    with_base = data_health_service.compare_schema(["a", "c"], ["a", "b"])
    assert with_base["has_baseline"] is True
    assert with_base["new_columns"] == ["c"]
    assert with_base["missing_columns"] == ["b"]


def test_schema_baseline_save_and_load(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(data_health_service, "get_artifact_dirs", lambda: {"root": tmp_path})

    data_health_service.save_schema_baseline("iris", ["a", "b"])
    loaded = data_health_service.load_schema_baseline("iris")
    assert loaded == ["a", "b"]


def test_schema_baseline_load_handles_invalid_json(monkeypatch, tmp_path: Path):
    baseline_dir = tmp_path / "schema_baselines"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    (baseline_dir / "iris.json").write_text("{not json}", encoding="utf-8")
    monkeypatch.setattr(data_health_service, "get_artifact_dirs", lambda: {"root": tmp_path})

    loaded = data_health_service.load_schema_baseline("iris")
    assert loaded is None


def test_load_health_input_uses_preview(monkeypatch):
    ds = {
        "X_train": pd.DataFrame({"x": [1, 2]}),
        "X_test": pd.DataFrame({"x": [3]}),
        "y_train": pd.Series([0, 1], name="target"),
        "y_test": pd.Series([1], name="target"),
        "task": "classification",
    }
    monkeypatch.setattr(data_health_service, "get_dataset_preview", lambda *args, **kwargs: {"dataset": ds})

    out = data_health_service.load_health_input("iris", test_size=0.2, random_state=42)
    assert out["task"] == "classification"
    assert out["frame"].shape[0] == 3
    assert out["target_name"] == "target"
