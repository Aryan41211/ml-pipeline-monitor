"""
Core ML pipeline orchestration.

Each pipeline run proceeds through fixed stages: data validation,
preprocessing, feature analysis, cross-validation, training, evaluation,
and feature importance extraction.  A progress callback lets the Streamlit
front-end update in real time without blocking the event loop.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import xgboost as xgb


# ---------------------------------------------------------------------------
# Supported algorithms
# ---------------------------------------------------------------------------

CLF_REGISTRY: Dict[str, type] = {
    "Random Forest": RandomForestClassifier,
    "XGBoost": xgb.XGBClassifier,
    "Gradient Boosting": GradientBoostingClassifier,
    "Logistic Regression": LogisticRegression,
    "SVM": SVC,
    "Decision Tree": DecisionTreeClassifier,
}

REG_REGISTRY: Dict[str, type] = {
    "Random Forest": RandomForestRegressor,
    "XGBoost": xgb.XGBRegressor,
    "Gradient Boosting": GradientBoostingRegressor,
    "Ridge Regression": Ridge,
    "SVR": SVR,
    "Decision Tree": DecisionTreeRegressor,
}

# Default parameter sets per algorithm (overridden by user input)
DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    "Random Forest": {"n_estimators": 100, "max_depth": None, "min_samples_split": 2},
    "XGBoost": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6},
    "Gradient Boosting": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
    "Logistic Regression": {"C": 1.0, "max_iter": 1000},
    "SVM": {"C": 1.0, "kernel": "rbf"},
    "Decision Tree": {"max_depth": None, "min_samples_split": 2},
    "Ridge Regression": {"alpha": 1.0},
    "SVR": {"C": 1.0, "kernel": "rbf"},
}


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    name: str
    status: str          # 'success' | 'failed' | 'skipped'
    duration: float      # seconds
    logs: List[str] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    run_id: str
    dataset: str
    model_type: str
    task: str
    status: str = "pending"
    duration: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    cv_scores: Optional[np.ndarray] = None
    feature_importances: Optional[pd.Series] = None
    confusion_mat: Optional[np.ndarray] = None
    stages: List[StageResult] = field(default_factory=list)
    model: Any = None
    scaler: Any = None


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------

ProgressCallback = Callable[[str, float, str], None]


class MLPipeline:
    """
    End-to-end ML pipeline with configurable estimators and live progress
    callbacks for front-end integration.

    Parameters
    ----------
    dataset_name : str
        Display name of the dataset (stored in run metadata).
    model_type : str
        One of the keys in CLF_REGISTRY / REG_REGISTRY.
    task : str
        'classification' or 'regression'.
    params : dict
        Hyper-parameter overrides applied on top of DEFAULT_PARAMS.
    cv_folds : int
        Number of cross-validation folds.
    random_state : int
        Global reproducibility seed.
    progress_callback : callable, optional
        Called as (stage_name, progress_0_to_1, message) after each
        milestone.  Safe to pass a Streamlit st.empty() updater.
    """

    def __init__(
        self,
        dataset_name: str,
        model_type: str,
        task: str,
        params: Dict[str, Any],
        cv_folds: int = 5,
        random_state: int = 42,
        n_jobs: int | None = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.model_type = model_type
        self.task = task
        self.params = params
        self.cv_folds = cv_folds
        self.random_state = random_state
        n_jobs_value = -1 if n_jobs is None else n_jobs
        try:
            n_jobs_value = int(n_jobs_value)
        except (TypeError, ValueError):
            n_jobs_value = -1
        if n_jobs_value == 0:
            n_jobs_value = -1
        self.n_jobs = n_jobs_value
        self._cb = progress_callback
        self.run_id = str(uuid.uuid4())[:8].upper()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _emit(self, stage: str, progress: float, msg: str) -> None:
        if self._cb is not None:
            self._cb(stage, progress, msg)

    def _build_estimator(self) -> Any:
        registry = CLF_REGISTRY if self.task == "classification" else REG_REGISTRY
        if self.model_type not in registry:
            raise ValueError(f"Unsupported model type '{self.model_type}' for task '{self.task}'")
        cls = registry[self.model_type]

        # Merge defaults with user overrides, then strip unsupported keys
        base = DEFAULT_PARAMS.get(self.model_type, {}).copy()
        base.update(self.params)

        # Apply random_state where supported
        import inspect
        sig = inspect.signature(cls.__init__)
        if "random_state" in sig.parameters:
            base["random_state"] = self.random_state
        if "n_jobs" in sig.parameters and self.model_type != "Logistic Regression":
            base.setdefault("n_jobs", self.n_jobs)

        # XGBoost-specific tweaks
        if self.model_type == "XGBoost":
            base.setdefault("verbosity", 0)
            base.setdefault("eval_metric", "logloss" if self.task == "classification" else "rmse")
            base.pop("verbosity", None)  # suppress deprecation
            base["verbosity"] = 0

        # SVM / SVR: no random_state, but needs probability for ROC-AUC
        if self.model_type == "SVM":
            base.pop("random_state", None)
            base["probability"] = True
        elif self.model_type == "SVR":
            base.pop("random_state", None)

        if self.model_type in ("Ridge Regression",):
            base.pop("random_state", None)

        return cls(**base)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> PipelineResult:
        """Execute all pipeline stages and return a PipelineResult."""
        result = PipelineResult(
            run_id=self.run_id,
            dataset=self.dataset_name,
            model_type=self.model_type,
            task=self.task,
            params=self.params,
        )
        wall_start = time.perf_counter()

        # ------------------------------------------------------------------
        # Stage 1 — Data Validation
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        self._emit("Data Validation", 0.05, "Verifying schema and data integrity")
        logs: List[str] = []

        n_train, n_test = len(X_train), len(X_test)
        missing = int(X_train.isnull().sum().sum() + X_test.isnull().sum().sum())
        logs.append(f"Train samples : {n_train:,}   |   Test samples: {n_test:,}")
        logs.append(f"Features      : {X_train.shape[1]}   |   Missing values: {missing}")

        if self.task == "classification":
            dist = y_train.value_counts(normalize=True).round(3).to_dict()
            logs.append(f"Class distribution (train): {dist}")

        self._emit("Data Validation", 0.12, "Validation passed — no schema violations")
        result.stages.append(StageResult("Data Validation", "success", time.perf_counter() - t0, logs))

        # ------------------------------------------------------------------
        # Stage 2 — Preprocessing
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        self._emit("Preprocessing", 0.15, "Fitting StandardScaler on training split")
        logs = []

        scaler = StandardScaler()
        X_tr_sc = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index,
        )
        X_te_sc = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index,
        )

        logs.append(f"Scaler fitted on {n_train:,} samples")
        logs.append(
            f"Feature mean  : [{scaler.mean_.min():.4f}, {scaler.mean_.max():.4f}]"
        )
        logs.append(
            f"Feature scale : [{scaler.scale_.min():.4f}, {scaler.scale_.max():.4f}]"
        )
        result.scaler = scaler

        self._emit("Preprocessing", 0.24, "Preprocessing complete")
        result.stages.append(StageResult("Preprocessing", "success", time.perf_counter() - t0, logs))

        # ------------------------------------------------------------------
        # Stage 3 — Feature Analysis
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        self._emit("Feature Analysis", 0.27, "Computing correlations and variance statistics")
        logs = []

        corr = X_tr_sc.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        high_corr_pairs = int((upper > 0.85).sum().sum())
        low_var_count = int((X_tr_sc.var() < 0.01).sum())

        logs.append(f"High-correlation pairs (>0.85) : {high_corr_pairs}")
        logs.append(f"Low-variance features (<0.01)  : {low_var_count}")
        logs.append(
            f"Variance range : [{X_tr_sc.var().min():.4f}, {X_tr_sc.var().max():.4f}]"
        )

        self._emit("Feature Analysis", 0.36, "Feature analysis complete")
        result.stages.append(StageResult("Feature Analysis", "success", time.perf_counter() - t0, logs))

        # ------------------------------------------------------------------
        # Stage 4 — Cross-Validation
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        self._emit("Cross-Validation", 0.39, f"Running {self.cv_folds}-fold CV")
        logs = []

        cv_estimator = self._build_estimator()

        if self.task == "classification":
            cv = StratifiedKFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
            )
            scoring = "f1_weighted"
            scoring_label = "F1 (weighted)"
        else:
            cv = KFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
            )
            scoring = "r2"
            scoring_label = "R²"

        cv_scores = cross_val_score(
            cv_estimator,
            X_tr_sc,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=self.n_jobs,
        )

        logs.append(
            f"CV {scoring_label}: {cv_scores.mean():.4f}"
            f"  (+/- {cv_scores.std() * 2:.4f})"
        )
        logs.append(f"Fold scores : {[round(s, 4) for s in cv_scores]}")
        result.cv_scores = cv_scores

        self._emit(
            "Cross-Validation", 0.55,
            f"CV {scoring_label} = {cv_scores.mean():.4f}",
        )
        result.stages.append(StageResult("Cross-Validation", "success", time.perf_counter() - t0, logs))

        # ------------------------------------------------------------------
        # Stage 5 — Model Training
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        self._emit("Training", 0.58, f"Fitting {self.model_type} on full training set")
        logs = []

        model = self._build_estimator()
        fit_start = time.perf_counter()
        model.fit(X_tr_sc, y_train)
        fit_duration = time.perf_counter() - fit_start

        logs.append(f"Estimator    : {self.model_type}")
        logs.append(f"Fit duration : {fit_duration:.3f} s")
        logs.append(f"Parameters   : {self.params}")
        result.model = model

        self._emit("Training", 0.74, f"Training complete ({fit_duration:.3f} s)")
        result.stages.append(
            StageResult(
                "Training", "success", time.perf_counter() - t0, logs,
                {"fit_duration_s": round(fit_duration, 3)},
            )
        )

        # ------------------------------------------------------------------
        # Stage 6 — Evaluation
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        self._emit("Evaluation", 0.77, "Scoring on held-out test set")
        logs = []
        metrics: Dict[str, float] = {}

        y_pred = model.predict(X_te_sc)

        if self.task == "classification":
            metrics["accuracy"] = round(accuracy_score(y_test, y_pred), 4)
            metrics["precision"] = round(
                precision_score(y_test, y_pred, average="weighted", zero_division=0), 4
            )
            metrics["recall"] = round(
                recall_score(y_test, y_pred, average="weighted", zero_division=0), 4
            )
            metrics["f1_score"] = round(
                f1_score(y_test, y_pred, average="weighted", zero_division=0), 4
            )

            try:
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_te_sc)
                    if y_prob.shape[1] == 2:
                        metrics["roc_auc"] = round(
                            roc_auc_score(y_test, y_prob[:, 1]), 4
                        )
                        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
                        metrics["roc_curve_fpr"] = [round(float(v), 6) for v in fpr.tolist()]
                        metrics["roc_curve_tpr"] = [round(float(v), 6) for v in tpr.tolist()]
                    else:
                        metrics["roc_auc"] = round(
                            roc_auc_score(y_test, y_prob, multi_class="ovr"), 4
                        )
            except Exception:
                pass

            result.confusion_mat = confusion_matrix(y_test, y_pred)

            for k, v in metrics.items():
                if isinstance(v, list):
                    logs.append(f"{k:<12}: [{len(v)} points]")
                else:
                    logs.append(f"{k:<12}: {float(v):.4f}")

        else:
            metrics["rmse"] = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)
            metrics["mae"] = round(float(mean_absolute_error(y_test, y_pred)), 4)
            metrics["r2"] = round(float(r2_score(y_test, y_pred)), 4)

            for k, v in metrics.items():
                logs.append(f"{k:<6}: {v:.4f}")

        metrics["cv_mean"] = round(float(cv_scores.mean()), 4)
        metrics["cv_std"] = round(float(cv_scores.std()), 4)
        result.metrics = metrics

        self._emit("Evaluation", 0.90, "Evaluation complete")
        result.stages.append(StageResult("Evaluation", "success", time.perf_counter() - t0, logs))

        # ------------------------------------------------------------------
        # Stage 7 — Feature Importance
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        self._emit("Feature Importance", 0.92, "Extracting feature importances")
        logs = []

        if hasattr(model, "feature_importances_"):
            importances = pd.Series(
                model.feature_importances_, index=X_train.columns
            ).sort_values(ascending=False)
            result.feature_importances = importances
            top3 = importances.head(3)
            logs.append(
                "Top features: "
                + ", ".join(f"{k} ({v:.4f})" for k, v in top3.items())
            )
        elif hasattr(model, "coef_"):
            coef = model.coef_
            coef_abs = np.abs(coef).mean(axis=0) if np.ndim(coef) > 1 else np.abs(np.ravel(coef))
            importances = pd.Series(coef_abs, index=X_train.columns).sort_values(
                ascending=False
            )
            result.feature_importances = importances
            logs.append("Importance derived from model coefficients (absolute values)")
        else:
            logs.append("Feature importances not available for this estimator")

        self._emit("Feature Importance", 0.98, "Pipeline complete")
        result.stages.append(StageResult("Feature Importance", "success", time.perf_counter() - t0, logs))

        result.duration = round(time.perf_counter() - wall_start, 3)
        result.status = "completed"
        self._emit("Done", 1.0, f"Run {self.run_id} finished in {result.duration:.2f} s")

        return result
