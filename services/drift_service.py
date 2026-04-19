"""Drift service for running, classifying, and persisting drift reports."""

from __future__ import annotations

import uuid
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.alerts import emit_console_alert
from src.config_loader import load_config
from src.data_loader import load_dataset
from src.database import save_drift_report
from src.drift_detector import run_drift_analysis
from src.logger import get_app_logger

LOGGER = get_app_logger("drift_service")


def _severity_from_report(report: Dict[str, Any]) -> str:
    """Classify overall drift severity for UI alerts."""
    ratio = float(report.get("drift_ratio", 0.0))
    avg_psi = float(report.get("average_psi", 0.0))

    if ratio >= 0.50 or avg_psi >= 0.25:
        return "critical"
    if ratio >= 0.20 or avg_psi >= 0.10:
        return "warning"
    return "stable"


def run_drift_and_persist(
    *,
    dataset_label: str,
    dataset_key: str,
    noise_level: float,
    mean_shift: float,
    alpha: float,
) -> Dict[str, Any]:
    """Run drift analysis and persist report in one service call."""
    cfg = load_config()
    pipeline_cfg = cfg.get("pipeline", {})
    monitoring_cfg = cfg.get("monitoring", {})

    ds = load_dataset(
        dataset_key,
        test_size=float(pipeline_cfg.get("test_size", 0.40)),
        random_state=int(pipeline_cfg.get("random_seed", 42)),
    )

    reference = ds["X_train"].copy()
    current = ds["X_test"].copy()

    rng = np.random.default_rng(seed=7)
    if noise_level > 0:
        current = current + rng.normal(0, noise_level, size=current.shape)
    if mean_shift > 0:
        current = current + mean_shift

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
