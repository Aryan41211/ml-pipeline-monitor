"""Drift service for running, classifying, and persisting drift reports."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.alerts import emit_console_alert, emit_email_alert
from src.config_loader import load_config
from src.data_loader import DATASET_OPTIONS, load_dataset
from src.database import get_drift_reports, get_drift_reference, save_drift_reference, save_drift_report
from src.drift_detector import run_drift_analysis
from src.logger import get_app_logger

LOGGER = get_app_logger("drift_service")


def get_dataset_options() -> Dict[str, str]:
    """Expose dataset options to UI via service layer."""
    return dict(DATASET_OPTIONS)


def list_drift_reports(limit: int = 50) -> list[Dict[str, Any]]:
    """Return persisted drift reports."""
    return get_drift_reports(limit=limit)


def get_drift_preview_dataset(dataset_key: str) -> Dict[str, Any]:
    """Return baseline dataset preview used in drift UI."""
    cfg = load_config()
    pipeline_cfg = cfg.get("pipeline", {})
    return load_dataset(
        dataset_key,
        test_size=float(pipeline_cfg.get("test_size", 0.40)),
        random_state=int(pipeline_cfg.get("random_seed", 42)),
    )


def get_monitoring_defaults() -> Dict[str, Any]:
    """Return monitoring/drift threshold defaults from config."""
    cfg = load_config().get("monitoring", {})
    return {
        "drift_significance_level": float(cfg.get("drift_significance_level", 0.05)),
        "psi_moderate_threshold": float(cfg.get("psi_moderate_threshold", 0.10)),
        "psi_significant_threshold": float(cfg.get("psi_significant_threshold", 0.25)),
        "drift_feature_ratio_threshold": float(cfg.get("drift_feature_ratio_threshold", 0.20)),
    }


def _severity_from_report(report: Dict[str, Any]) -> str:
    """Classify overall drift severity for UI alerts."""
    ratio = float(report.get("drift_ratio", 0.0))
    avg_psi = float(report.get("average_psi", 0.0))

    if ratio >= 0.50 or avg_psi >= 0.25:
        return "critical"
    if ratio >= 0.20 or avg_psi >= 0.10:
        return "warning"
    return "stable"


def _validate_drift_inputs(
    dataset_label: str,
    dataset_key: str,
    noise_level: float,
    mean_shift: float,
    alpha: float,
) -> None:
    """Validate drift analysis input parameters."""
    if not dataset_label or not dataset_label.strip():
        raise ValueError("dataset_label is required")
    if not dataset_key or dataset_key not in DATASET_OPTIONS.values():
        raise ValueError(f"Invalid dataset_key: {dataset_key}")
    if noise_level < 0 or noise_level > 10:
        raise ValueError("noise_level must be between 0 and 10")
    if mean_shift < 0 or mean_shift > 10:
        raise ValueError("mean_shift must be between 0 and 10")
    if alpha not in {0.01, 0.05, 0.10}:
        raise ValueError("alpha must be one of: 0.01, 0.05, 0.10")


def run_drift_and_persist(
    *,
    dataset_label: str,
    dataset_key: str,
    noise_level: float,
    mean_shift: float,
    alpha: float,
) -> Dict[str, Any]:
    """Run drift analysis and persist report in one service call.
    
    Compares current production data against stored reference distribution
    (from training data at model promotion time).
    """
    _validate_drift_inputs(dataset_label, dataset_key, noise_level, mean_shift, alpha)

    cfg = load_config()
    pipeline_cfg = cfg.get("pipeline", {})
    monitoring_cfg = cfg.get("monitoring", {})

    # Load current production data (simulated with test split + noise/shift for demo)
    ds = load_dataset(
        dataset_key,
        test_size=float(pipeline_cfg.get("test_size", 0.20)),
        random_state=int(pipeline_cfg.get("random_seed", 42)),
    )

    current = ds["X_test"].copy()

    rng = np.random.default_rng(seed=7)
    if noise_level > 0:
        current = current + rng.normal(0, noise_level, size=current.shape)
    if mean_shift > 0:
        current = current + mean_shift

    # Get stored reference distribution
    ref_record = get_drift_reference(dataset_label)
    if ref_record is None:
        # No reference stored yet - use training data as reference and store it
        reference = ds["X_train"].copy()
        save_drift_reference(dataset_label, ds["feature_names"], reference.values)
        LOGGER.info("Stored initial reference distribution for dataset=%s", dataset_label)
    else:
        reference = pd.DataFrame(ref_record["reference_data"], columns=ref_record["feature_names"])

    report = run_drift_analysis(
        pd.DataFrame(reference, columns=ds["feature_names"]),
        pd.DataFrame(current, columns=ds["feature_names"]),
        alpha=alpha,
        moderate_threshold=float(monitoring_cfg.get("psi_moderate_threshold", 0.10)),
        significant_threshold=float(monitoring_cfg.get("psi_significant_threshold", 0.25)),
        feature_ratio_threshold=float(monitoring_cfg.get("drift_feature_ratio_threshold", 0.20)),
    )

    report["overall_severity"] = _severity_from_report(report)

    report_id = str(uuid.uuid4())[:8].upper()
    save_drift_report(
        report_id=report_id,
        dataset=dataset_label,
        reference_size=len(reference),
        current_size=len(current),
        drift_detected=report["overall_drift"],
        drift_score=report["average_psi"],
        features_drifted=report["features_drifted"],
        feature_results=report["feature_results"],
    )

    if report["overall_drift"]:
        emit_console_alert(
            report.get("overall_severity", "warning"),
            f"Data drift detected for dataset={dataset_label}; ratio={report['drift_ratio']:.2f}",
        )
        emit_email_alert(
            report.get("overall_severity", "warning"),
            subject="Data drift alert",
            message=f"Drift detected for dataset={dataset_label}",
            metadata={
                "dataset": dataset_label,
                "drift_ratio": report.get("drift_ratio"),
                "average_psi": report.get("average_psi"),
                "features_drifted": report.get("features_drifted"),
            },
        )
        LOGGER.warning(
            "Drift detected dataset=%s severity=%s features_drifted=%s",
            dataset_label,
            report.get("overall_severity", "warning"),
            report.get("features_drifted"),
        )
    else:
        LOGGER.info("No significant drift dataset=%s", dataset_label)

    return {
        "report": report,
        "reference": reference,
        "current": current,
        "feature_names": ds["feature_names"],
        "report_id": report_id,
    }
