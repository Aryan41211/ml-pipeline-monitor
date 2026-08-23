import pandas as pd
import pytest

from ml_pipeline_monitor.services import drift_service


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


def test_severity_is_classified_by_the_analyzer(monkeypatch):
    """Severity comes from run_drift_analysis, which uses the configured thresholds."""
    import numpy as np

    from ml_pipeline_monitor.ml import drift_detector

    monkeypatch.setattr(
        drift_detector,
        "load_config",
        lambda: {
            "monitoring": {
                "psi_moderate_threshold": 0.10,
                "psi_significant_threshold": 0.25,
                "drift_feature_ratio_threshold": 0.20,
            }
        },
    )

    rng = np.random.default_rng(0)
    reference = pd.DataFrame({"a": rng.normal(0, 1, 500), "b": rng.normal(0, 1, 500)})

    identical = drift_detector.run_drift_analysis(reference, reference.copy())
    assert identical["overall_severity"] == "stable"
    assert identical["overall_drift"] is False

    shifted = pd.DataFrame({"a": rng.normal(6, 1, 500), "b": rng.normal(6, 1, 500)})
    drifted = drift_detector.run_drift_analysis(reference, shifted)
    assert drifted["overall_severity"] == "critical"
    assert drifted["overall_drift"] is True


def test_align_reference_rejects_stale_schema():
    """A baseline missing current features must fail loudly, not compare NaNs."""
    stale = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(ValueError, match="missing features"):
        drift_service._align_reference(stale, ["a", "b"], "Iris")


def test_align_reference_reorders_to_current_schema():
    reference = pd.DataFrame({"b": [1.0], "a": [2.0]})
    aligned = drift_service._align_reference(reference, ["a", "b"], "Iris")
    assert list(aligned.columns) == ["a", "b"]


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
            "overall_severity": "critical",
            "average_psi": 0.5,
            "feature_results": [{"feature": "a", "psi": 0.5}],
        },
    )

    saved = {}

    def _fake_save(**kwargs):
        saved.update(kwargs)

    monkeypatch.setattr(drift_service, "save_drift_report", _fake_save)
    monkeypatch.setattr(drift_service, "emit_console_alert", lambda *args, **kwargs: None)
    monkeypatch.setattr(drift_service, "get_drift_reference", lambda dataset: None)
    monkeypatch.setattr(drift_service, "save_drift_reference", lambda *args, **kwargs: None)

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
