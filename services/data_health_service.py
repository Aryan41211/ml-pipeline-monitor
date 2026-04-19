"""Data health service: quality checks, schema comparison, and outlier summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from services.pipeline_service import get_dataset_options, get_dataset_preview, get_pipeline_defaults
from src.config_loader import get_artifact_dirs


def load_health_input(dataset_key: str, *, test_size: float, random_state: int) -> Dict[str, Any]:
    """Load dataset preview and return unified frame for quality checks."""
    payload = get_dataset_preview(dataset_key, test_size=test_size, random_state=random_state)
    ds = payload["dataset"]

    x_all = pd.concat([ds["X_train"], ds["X_test"]], axis=0, ignore_index=True)
    y_all = pd.concat([ds["y_train"], ds["y_test"]], axis=0, ignore_index=True)

    target_name = "target"
    if hasattr(y_all, "name") and y_all.name:
        target_name = str(y_all.name)

    frame = x_all.copy()
    frame[target_name] = y_all

    return {
        "dataset": ds,
        "frame": frame,
        "feature_frame": x_all,
        "target": y_all,
        "task": ds.get("task", "classification"),
        "target_name": target_name,
    }


def missing_value_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Return per-column and global missing-value statistics."""
    missing_count = df.isna().sum()
    total_cells = df.shape[0] * df.shape[1] if df.shape[0] and df.shape[1] else 0
    total_missing = int(missing_count.sum())
    total_missing_pct = (total_missing / total_cells * 100.0) if total_cells else 0.0

    per_col = pd.DataFrame(
        {
            "column": missing_count.index,
            "missing_count": missing_count.values,
            "missing_pct": (missing_count.values / len(df) * 100.0) if len(df) else np.zeros(len(missing_count)),
        }
    ).sort_values(["missing_count", "column"], ascending=[False, True])

    return {
        "total_missing": total_missing,
        "total_missing_pct": float(total_missing_pct),
        "per_column": per_col,
    }


def class_imbalance_report(target: pd.Series, task: str) -> Dict[str, Any]:
    """Return class distribution and imbalance ratio for classification tasks."""
    if task != "classification":
        return {"enabled": False, "distribution": pd.DataFrame(), "imbalance_ratio": None}

    counts = target.value_counts(dropna=False).sort_values(ascending=False)
    dist = pd.DataFrame(
        {
            "class": counts.index.astype(str),
            "count": counts.values,
            "pct": (counts.values / counts.sum() * 100.0) if counts.sum() else np.zeros(len(counts)),
        }
    )

    imbalance_ratio = None
    if len(counts) >= 2 and counts.iloc[-1] > 0:
        imbalance_ratio = float(counts.iloc[0] / counts.iloc[-1])

    return {
        "enabled": True,
        "distribution": dist,
        "imbalance_ratio": imbalance_ratio,
    }


def basic_statistics_report(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Return basic stats for numeric features."""
    numeric = feature_df.select_dtypes(include=[np.number]).copy()
    if numeric.empty:
        return pd.DataFrame(columns=["feature", "mean", "std", "min", "max"])

    summary = numeric.agg(["mean", "std", "min", "max"]).T.reset_index()
    summary.columns = ["feature", "mean", "std", "min", "max"]
    return summary.sort_values("feature")


def outlier_report(feature_df: pd.DataFrame, *, method: str = "iqr", z_threshold: float = 3.0) -> pd.DataFrame:
    """Return outlier count per numeric feature using IQR or z-score."""
    numeric = feature_df.select_dtypes(include=[np.number]).copy()
    if numeric.empty:
        return pd.DataFrame(columns=["feature", "outlier_count", "outlier_pct"]) 

    rows: List[Dict[str, Any]] = []
    n = len(numeric)

    for col in numeric.columns:
        s = numeric[col].dropna()
        if s.empty:
            out_count = 0
        elif method == "zscore":
            std = float(s.std())
            if std == 0:
                out_count = 0
            else:
                z = ((s - s.mean()) / std).abs()
                out_count = int((z > z_threshold).sum())
        else:
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            out_count = int(((s < lower) | (s > upper)).sum())

        out_pct = (out_count / n * 100.0) if n else 0.0
        rows.append({"feature": str(col), "outlier_count": out_count, "outlier_pct": out_pct})

    return pd.DataFrame(rows).sort_values(["outlier_count", "feature"], ascending=[False, True])


def _schema_dir() -> Path:
    root = get_artifact_dirs()["root"]
    d = root / "schema_baselines"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_schema_baseline(dataset_key: str) -> List[str] | None:
    """Load saved baseline columns for a dataset, if available."""
    path = _schema_dir() / f"{dataset_key}.json"
    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    cols = payload.get("columns") if isinstance(payload, dict) else None
    if isinstance(cols, list):
        return [str(c) for c in cols]
    return None


def save_schema_baseline(dataset_key: str, columns: List[str]) -> None:
    """Persist baseline schema for future comparisons."""
    path = _schema_dir() / f"{dataset_key}.json"
    payload = {"dataset_key": dataset_key, "columns": list(columns)}
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def compare_schema(current_columns: List[str], baseline_columns: List[str] | None) -> Dict[str, Any]:
    """Return schema-drift comparison against saved baseline."""
    if baseline_columns is None:
        return {"has_baseline": False, "new_columns": current_columns, "missing_columns": []}

    cur_set = set(current_columns)
    base_set = set(baseline_columns)

    new_cols = sorted(cur_set - base_set)
    missing_cols = sorted(base_set - cur_set)

    return {
        "has_baseline": True,
        "new_columns": new_cols,
        "missing_columns": missing_cols,
    }


def data_health_defaults() -> Dict[str, Any]:
    """Expose default controls for the Data Health page."""
    p = get_pipeline_defaults()
    return {
        "test_size": float(p.get("test_size", 0.20)),
        "random_seed": int(p.get("random_seed", 42)),
        "dataset_options": get_dataset_options(),
    }
