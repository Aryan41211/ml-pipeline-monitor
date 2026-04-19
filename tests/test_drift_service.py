import pandas as pd
import pytest

from services import drift_service


def test_get_monitoring_defaults_from_config(monkeypatch):
    monkeypatch.setattr(
        drift_service,
        "load_config",
        lambda: {
            "monitoring": {
                "drift_significance_level": 0.01,
                "psi_moderate_threshold": 0.11,
                "psi_significant_threshold": 0.26,
                "drift_feature_ratio_threshold": 0.33,
            }
        },
    )

    cfg = drift_service.get_monitoring_defaults()
    assert cfg["drift_significance_level"] == 0.01
    assert cfg["psi_moderate_threshold"] == 0.11
    assert cfg["psi_significant_threshold"] == 0.26
    assert cfg["drift_feature_ratio_threshold"] == 0.33


def test_severity_from_report_thresholds():
    assert drift_service._severity_from_report({"drift_ratio": 0.6, "average_psi": 0.01}) == "critical"
    assert drift_service._severity_from_report({"drift_ratio": 0.1, "average_psi": 0.2}) == "warning"
    assert drift_service._severity_from_report({"drift_ratio": 0.05, "average_psi": 0.01}) == "stable"


def test_list_drift_reports_proxy(monkeypatch):
    monkeypatch.setattr(drift_service, "get_drift_reports", lambda limit=50: [{"report_id": "r1"}])
    rows = drift_service.list_drift_reports(limit=10)
    assert rows[0]["report_id"] == "r1"


def test_run_drift_and_persist_success(monkeypatch):
    monkeypatch.setattr(
        drift_service,
        "load_config",
        lambda: {
            "pipeline": {"test_size": 0.4, "random_seed": 42},
            "monitoring": {
                "psi_moderate_threshold": 0.10,
                "psi_significant_threshold": 0.25,
                "drift_feature_ratio_threshold": 0.20,
            },
        },
    )

    monkeypatch.setattr(
        drift_service,
        "load_dataset",
        lambda *args, **kwargs: {
            "X_train": pd.DataFrame({"a": [1.0, 2.0, 3.0]}),
            "X_test": pd.DataFrame({"a": [1.2, 2.1, 3.4]}),
            "feature_names": ["a"],
        },
    )

    monkeypatch.setattr(
        drift_service,
        "run_drift_analysis",
        lambda *args, **kwargs: {
            "features_analyzed": 1,
            "features_drifted": 1,
            "drift_ratio": 1.0,
            "overall_drift": True,
            "average_psi": 0.5,
            "feature_results": [{"feature": "a", "psi": 0.5}],
        },
    )

    saved = {}

    def _fake_save(**kwargs):
        saved.update(kwargs)

    monkeypatch.setattr(drift_service, "save_drift_report", _fake_save)
    monkeypatch.setattr(drift_service, "emit_console_alert", lambda *args, **kwargs: None)

    out = drift_service.run_drift_and_persist(
        dataset_label="Iris",
        dataset_key="iris",
        noise_level=0.0,
        mean_shift=0.0,
        alpha=0.05,
    )

    assert out["report"]["overall_severity"] == "critical"
    assert saved["dataset"] == "Iris"
    assert saved["drift_detected"] is True


def test_run_drift_and_persist_propagates_failures(monkeypatch):
    monkeypatch.setattr(
        drift_service,
        "load_config",
        lambda: {"pipeline": {"test_size": 0.4, "random_seed": 42}, "monitoring": {}},
    )
    monkeypatch.setattr(drift_service, "load_dataset", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(RuntimeError):
        drift_service.run_drift_and_persist(
            dataset_label="Iris",
            dataset_key="iris",
            noise_level=0.0,
            mean_shift=0.0,
            alpha=0.05,
        )
