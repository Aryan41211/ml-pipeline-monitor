from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ValidationResult:
    quality_score: float  # 0-100
    status: str  # "pass" | "fail"
    report: Dict[str, Any]
    recommendations: List[str]
    fail_reasons: List[str]


class DataQualityFailed(Exception):
    """Raised when dataset validation fails and training must stop."""

    def __init__(
        self,
        *,
        dataset: str,
        quality_score: float,
        min_quality_score: float,
        validation_result: ValidationResult,
        fail_stage: str = "Data Validation",
    ) -> None:
        super().__init__(
            f"Data quality failed: {quality_score:.2f} < {min_quality_score:.2f} ({dataset})"
        )
        self.dataset = dataset
        self.quality_score = float(quality_score)
        self.min_quality_score = float(min_quality_score)
        self.validation_result = validation_result
        self.fail_stage = fail_stage


def _safe_jsonable(obj: Any) -> Any:
    """Convert common numpy/pandas scalars to plain JSON-safe primitives."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if v != v:  # NaN
            return None
        return v
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj


def _missing_value_summary(df: pd.DataFrame) -> Dict[str, Any]:
    per_col_missing = df.isna().sum()
    per_col_missing_pct = per_col_missing / len(df) * 100.0 if len(df) else 0.0
    total_missing = int(per_col_missing.sum())
    total_cells = int(df.shape[0] * df.shape[1]) if df.shape[0] and df.shape[1] else 0
    total_missing_pct = float(total_missing / total_cells * 100.0) if total_cells else 0.0

    per_col = pd.DataFrame(
        {
            "column": per_col_missing.index.astype(str),
            "missing_count": per_col_missing.values.astype(int),
            "missing_pct": per_col_missing_pct.values.astype(float),
        }
    ).sort_values(["missing_count", "column"], ascending=[False, True])

    return {
        "total_missing": total_missing,
        "total_missing_pct": total_missing_pct,
        "per_column": per_col.to_dict(orient="records"),
    }


def _duplicate_report(df: pd.DataFrame) -> Dict[str, Any]:
    if len(df) == 0:
        return {"duplicate_rows": 0, "duplicate_rows_pct": 0.0}
    dup_mask = df.duplicated(keep=False)
    dup_count = int(dup_mask.sum())
    dup_pct = float(dup_count / len(df) * 100.0)
    return {"duplicate_rows": dup_count, "duplicate_rows_pct": dup_pct}


def _constant_feature_report(feature_df: pd.DataFrame) -> Dict[str, Any]:
    if feature_df.empty:
        return {"constant_features": 0, "constant_feature_names": []}

    numeric = feature_df.select_dtypes(include=[np.number])
    if numeric.empty:
        # For non-numeric, treat as constant if all values identical incl NaNs
        const_cols = []
        for col in feature_df.columns:
            s = feature_df[col]
            nunique = int(s.nunique(dropna=False))
            if nunique <= 1:
                const_cols.append(str(col))
        return {
            "constant_features": len(const_cols),
            "constant_feature_names": const_cols,
        }

    variances = numeric.var(numeric_only=True)
    const_cols = [str(c) for c in variances.index if float(variances[c]) == 0.0]
    return {
        "constant_features": len(const_cols),
        "constant_feature_names": const_cols,
    }


def _type_mismatch_report(feature_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Best-effort checks:
    - numeric features should be numeric dtype
    - object columns are treated as potential mismatches
    """
    if feature_df.empty:
        return {"object_columns": 0, "object_column_names": []}

    object_cols = [
        str(c)
        for c in feature_df.columns
        if feature_df[c].dtype == "object" or str(feature_df[c].dtype).lower() == "string"
    ]
    return {"object_columns": len(object_cols), "object_column_names": object_cols}


def _outlier_report(feature_df: pd.DataFrame, *, method: str, z_threshold: float) -> Dict[str, Any]:
    numeric = feature_df.select_dtypes(include=[np.number]).copy()
    if numeric.empty:
        return {"outliers_detected": 0, "outlier_by_feature": []}

    outlier_by_feature: List[Dict[str, Any]] = []
    n_rows = len(numeric)

    for col in numeric.columns:
        s = numeric[col].dropna()
        if s.empty:
            out_count = 0
        else:
            if method == "zscore":
                std = float(s.std())
                if std == 0.0:
                    out_count = 0
                else:
                    z = ((s - float(s.mean())) / std).abs()
                    out_count = int((z > z_threshold).sum())
            else:
                q1 = float(s.quantile(0.25))
                q3 = float(s.quantile(0.75))
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                out_count = int(((s < lower) | (s > upper)).sum())

        out_pct = float(out_count / n_rows * 100.0) if n_rows else 0.0
        outlier_by_feature.append(
            {"feature": str(col), "outlier_count": out_count, "outlier_pct": out_pct}
        )

    outlier_by_feature.sort(key=lambda r: (r["outlier_count"], r["feature"]), reverse=True)
    total_outliers = int(sum(r["outlier_count"] for r in outlier_by_feature))
    return {
        "outliers_detected": total_outliers,
        "outlier_by_feature": outlier_by_feature,
    }


def _invalid_range_report(feature_df: pd.DataFrame, *, quantile_low: float = 0.001, quantile_high: float = 0.999) -> Dict[str, Any]:
    """
    Use robust quantile-based invalid ranges:
    values outside very low/high quantiles are considered invalid-ish.
    """
    numeric = feature_df.select_dtypes(include=[np.number]).copy()
    if numeric.empty:
        return {"features_with_outside_range": 0, "invalid_by_feature": []}

    invalid_by_feature: List[Dict[str, Any]] = []
    n_rows = len(numeric)

    for col in numeric.columns:
        s = numeric[col].dropna()
        if s.empty:
            invalid_count = 0
        else:
            lo = float(s.quantile(quantile_low))
            hi = float(s.quantile(quantile_high))
            invalid_count = int(((s < lo) | (s > hi)).sum())
        invalid_pct = float(invalid_count / n_rows * 100.0) if n_rows else 0.0
        invalid_by_feature.append(
            {"feature": str(col), "invalid_count": invalid_count, "invalid_pct": invalid_pct}
        )

    invalid_by_feature.sort(key=lambda r: (r["invalid_count"], r["feature"]), reverse=True)
    features_with_outside_range = sum(1 for r in invalid_by_feature if r["invalid_count"] > 0)

    return {
        "features_with_outside_range": int(features_with_outside_range),
        "invalid_by_feature": invalid_by_feature,
    }


def _class_imbalance_report(target: pd.Series) -> Dict[str, Any]:
