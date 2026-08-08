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
    per_col_missing_pct = (
        (per_col_missing.astype(float) / float(len(df))) * 100.0 if len(df) else np.zeros(len(per_col_missing))
    )
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
    counts = target.value_counts(dropna=False).sort_values(ascending=False)
    total = int(counts.sum())
    if total == 0:
        return {"imbalance_ratio": None, "entropy": None, "distribution": []}

    # imbalance ratio: most common / least common non-zero
    nonzero = counts[counts > 0]
    if len(nonzero) < 2:
        imbalance_ratio = None
    else:
        imbalance_ratio = float(nonzero.iloc[0] / nonzero.iloc[-1])

    probs = counts / total
    entropy = float(-(probs * np.log2(probs.replace(0, np.nan))).sum(skipna=True))

    dist = (
        pd.DataFrame({"class": counts.index.astype(str), "count": counts.values, "pct": (probs.values * 100.0)})
        .to_dict(orient="records")
    )

    return {"imbalance_ratio": imbalance_ratio, "entropy": entropy, "distribution": dist}


def _high_correlation_report(feature_df: pd.DataFrame, *, threshold: float) -> Dict[str, Any]:
    numeric = feature_df.select_dtypes(include=[np.number]).copy()
    if numeric.shape[1] < 2:
        return {"high_corr_pairs": 0, "high_corr_features": []}

    corr = numeric.corr().abs()
    # upper triangle without diagonal
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    upper = corr.where(mask)

    high_pairs_df = (
        upper.stack()
        .reset_index()
        .rename(columns={"level_0": "feature_a", "level_1": "feature_b", 0: "abs_corr"})
    )
    high_pairs = high_pairs_df[high_pairs_df["abs_corr"] > float(threshold)]

    high_features = sorted(
        {str(a) for a in high_pairs["feature_a"].tolist()} | {str(b) for b in high_pairs["feature_b"].tolist()}
    )

    return {
        "high_corr_pairs": int(len(high_pairs)),
        "high_corr_features": high_features,
        "high_corr_pairs_list": [
            {"a": str(r.feature_a), "b": str(r.feature_b), "abs_corr": float(r.abs_corr)}
            for r in high_pairs.itertuples(index=False)
        ],
    }


def _compute_quality_score(
    *,
    missing_total_pct: float,
    duplicates_pct: float,
    outliers_total: int,
    constant_features: int,
    object_columns: int,
    high_corr_pairs: int,
    imbalance_ratio: Optional[float],
    invalid_features_with_outside_range: int,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, List[str]]:
    """
    Convert findings into a 0-100 score with penalty-based reduction.
    """
    w = weights or {
        "missing": 0.28,
        "duplicates": 0.12,
        "outliers": 0.20,
        "types": 0.12,
        "ranges": 0.10,
        "imbalance": 0.10,
        "constant": 0.05,
        "correlation": 0.03,
    }

    penalties: List[Tuple[str, float]] = []

    # Missing: 0% -> 0 penalty, 20% -> 1 penalty
    missing_pen = min(1.0, missing_total_pct / 20.0) * w["missing"]
    penalties.append(("missing", missing_pen))

    dup_pen = min(1.0, duplicates_pct / 10.0) * w["duplicates"]
    penalties.append(("duplicates", dup_pen))

    # Outliers: normalize via count relative to feature count * rows (best-effort)
    out_pen = min(1.0, outliers_total / 200.0) * w["outliers"]
    penalties.append(("outliers", out_pen))

    type_pen = min(1.0, object_columns / 10.0) * w["types"]
    penalties.append(("types", type_pen))

    ranges_pen = min(1.0, invalid_features_with_outside_range / 10.0) * w["ranges"]
    penalties.append(("ranges", ranges_pen))

    constant_pen = min(1.0, constant_features / 10.0) * w["constant"]
    penalties.append(("constant", constant_pen))

    corr_pen = min(1.0, high_corr_pairs / 100.0) * w["correlation"]
    penalties.append(("correlation", corr_pen))

    if imbalance_ratio is None:
        imb_pen = 0.0
    else:
        # imbalance_ratio=1 -> 0, 10 -> 1
        imb_pen = min(1.0, (imbalance_ratio - 1.0) / 9.0) * w["imbalance"]
    penalties.append(("imbalance", imb_pen))

    total_penalty = sum(p for _, p in penalties)
    total_penalty = min(1.0, float(total_penalty))

    score = 100.0 * (1.0 - total_penalty)

    # Fail reasons are computed later by thresholding; return placeholder list here.
    return float(max(0.0, min(100.0, score))), []


def validate_dataset(
    frame: pd.DataFrame,
    *,
    task: str,
    target_name: str,
    high_corr_threshold: float = 0.85,
    outlier_method: str = "iqr",
    outlier_z_threshold: float = 3.0,
    quantile_low: float = 0.001,
    quantile_high: float = 0.999,
    missing_total_threshold_pct: float = 5.0,
    duplicates_threshold_pct: float = 1.0,
    outliers_threshold_total: int = 10,
    constant_features_threshold: int = 0,
    object_columns_threshold: int = 0,
    invalid_ranges_features_threshold: int = 0,
    imbalance_ratio_threshold: float = 5.0,
    quality_score_min: float = 75.0,
) -> ValidationResult:
    """
    Validate dataset frame (must include target column).
    Returns quality score + recommendations.
    """
    if frame is None or frame.empty:
        report = {
            "error": "empty_frame",
            "missing": {"total_missing": 0, "total_missing_pct": 0.0},
            "duplicates": {"duplicate_rows": 0, "duplicate_rows_pct": 0.0},
            "outliers": {"outliers_detected": 0, "outlier_by_feature": []},
            "type_mismatch": {"object_columns": 0, "object_column_names": []},
            "invalid_ranges": {"features_with_outside_range": 0, "invalid_by_feature": []},
            "class_imbalance": {},
            "constant_features": {"constant_features": 0, "constant_feature_names": []},
            "high_correlation": {"high_corr_pairs": 0, "high_corr_features": []},
        }
        return ValidationResult(
            quality_score=0.0,
            status="fail",
            report=report,
            recommendations=["Provide non-empty training data before validation."],
            fail_reasons=["empty_frame"],
        )

    if target_name not in frame.columns:
        report = {
            "error": "missing_target_column",
            "target_name": target_name,
            "available_columns": [str(c) for c in frame.columns],
        }
        return ValidationResult(
            quality_score=0.0,
            status="fail",
            report=report,
            recommendations=["Ensure target column is present for validation."],
            fail_reasons=["missing_target_column"],
        )

    # Separate features/target
    target = frame[target_name]
    feature_df = frame.drop(columns=[target_name])

    # Core checks
    missing = _missing_value_summary(frame.drop(columns=[target_name], errors="ignore"))
    # duplicate rows: consider full frame including target to be conservative
    dup = _duplicate_report(frame)

    constant = _constant_feature_report(feature_df)
    type_mismatch = _type_mismatch_report(feature_df)

    # Outliers / invalid ranges operate on numeric subset
    outliers = _outlier_report(
        feature_df,
        method="zscore" if outlier_method.lower() == "zscore" else "iqr",
        z_threshold=outlier_z_threshold,
    )
    invalid_ranges = _invalid_range_report(feature_df, quantile_low=quantile_low, quantile_high=quantile_high)

    # Correlation on numeric features
    high_corr = _high_correlation_report(feature_df, threshold=high_corr_threshold)

    # Class imbalance
    imbalance = {}
    imbalance_ratio: Optional[float] = None
    if task == "classification":
        imbalance = _class_imbalance_report(target)
        imbalance_ratio = imbalance.get("imbalance_ratio")

    # Recommendations + fail reasons (threshold-based)
    fail_reasons: List[str] = []
    recommendations: List[str] = []

    missing_total_pct = float(missing.get("total_missing_pct", 0.0))
    duplicates_pct = float(dup.get("duplicate_rows_pct", 0.0))
    outliers_total = int(outliers.get("outliers_detected", 0))
    constant_features = int(constant.get("constant_features", 0))
    object_columns = int(type_mismatch.get("object_columns", 0))
    high_corr_pairs = int(high_corr.get("high_corr_pairs", 0))
    invalid_features_with_outside_range = int(
        invalid_ranges.get("features_with_outside_range", 0) or 0
    )

    if missing_total_pct > missing_total_threshold_pct:
        fail_reasons.append("missing_values_too_high")
        recommendations.append("Impute missing values or remove highly sparse columns/rows.")
    if duplicates_pct > duplicates_threshold_pct:
        fail_reasons.append("duplicates_too_high")
        recommendations.append("Deduplicate rows prior to training (or use aggregation).")
    if outliers_total > outliers_threshold_total:
        fail_reasons.append("outliers_too_many")
        recommendations.append("Cap/winsorize outliers or use robust scaling for affected features.")
    if object_columns > object_columns_threshold:
        fail_reasons.append("dtype_mismatch_suspected")
        recommendations.append("Cast features to numeric types and ensure schema matches training expectations.")
    if invalid_features_with_outside_range > invalid_ranges_features_threshold:
        fail_reasons.append("invalid_ranges_detected")
        recommendations.append("Validate feature ranges; investigate upstream data generation/units.")
    if task == "classification" and imbalance_ratio is not None and imbalance_ratio > imbalance_ratio_threshold:
        fail_reasons.append("class_imbalance_too_high")
        recommendations.append("Use stratified sampling, class weights, or resampling to address imbalance.")
    if constant_features > constant_features_threshold:
        fail_reasons.append("constant_features_detected")
        recommendations.append("Remove constant (zero-variance) features to improve model reliability.")
    if high_corr_pairs > 0:
        # For now, we treat any pair > threshold as a recommendation; strictness is handled via quality score.
        # Still include as a reason if strong.
        if high_corr_pairs > 0 and float(high_corr_pairs) > (high_corr_pairs * 0 + 0):  # keep stable
            if high_corr_pairs > 0 and high_corr_pairs >= 1:
                recommendations.append("Consider removing one of highly correlated features to reduce multicollinearity.")
                if high_corr_pairs >= 20:
                    fail_reasons.append("high_correlation_too_many_pairs")

    quality_score, _ = _compute_quality_score(
        missing_total_pct=missing_total_pct,
        duplicates_pct=duplicates_pct,
        outliers_total=outliers_total,
        constant_features=constant_features,
        object_columns=object_columns,
        high_corr_pairs=high_corr_pairs,
        imbalance_ratio=imbalance_ratio,
        invalid_features_with_outside_range=invalid_features_with_outside_range,
    )

    status = "pass" if quality_score >= quality_score_min else "fail"

    report = {
        "missing": missing,
        "duplicates": dup,
        "outliers": outliers,
        "type_mismatch": type_mismatch,
        "invalid_ranges": invalid_ranges,
        "class_imbalance": imbalance,
        "constant_features": constant,
        "high_correlation": high_corr,
        "parameters": {
            "high_corr_threshold": high_corr_threshold,
            "outlier_method": outlier_method,
            "outlier_z_threshold": outlier_z_threshold,
            "quantile_low": quantile_low,
            "quantile_high": quantile_high,
            "missing_total_threshold_pct": missing_total_threshold_pct,
            "duplicates_threshold_pct": duplicates_threshold_pct,
            "outliers_threshold_total": outliers_threshold_total,
            "constant_features_threshold": constant_features_threshold,
            "object_columns_threshold": object_columns_threshold,
            "invalid_ranges_features_threshold": invalid_ranges_features_threshold,
            "imbalance_ratio_threshold": imbalance_ratio_threshold,
            "quality_score_min": quality_score_min,
        },
    }

    # If it fails, ensure there is at least one fail reason.
    if status == "fail" and not fail_reasons:
        fail_reasons = ["quality_score_below_minimum"]

    if status == "pass" and not recommendations:
        recommendations = ["Validation passed. Ready for training."]

    return ValidationResult(
        quality_score=quality_score,
        status=status,
        report=report,
        recommendations=recommendations,
        fail_reasons=fail_reasons,
    )
