"""Pipeline Runner page for end-to-end model training and persistence."""

from datetime import datetime
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st

from services.pipeline_service import (
    compute_next_run_ts,
    run_pipeline_and_persist,
    should_trigger_scheduled_run,
)
from src.auth import is_authenticated, render_auth_controls
from src.config_loader import load_config
from src.data_loader import DATASET_OPTIONS, load_dataset, get_feature_statistics
from src.database import initialize_db
from src.pipeline import CLF_REGISTRY, REG_REGISTRY

st.set_page_config(page_title="Pipeline Runner | ML Monitor", layout="wide")
initialize_db()

APP_CONFIG = load_config()
PIPELINE_CONFIG = APP_CONFIG.get("pipeline", {})
DATASET_CONFIG = APP_CONFIG.get("datasets", {})


def _get_automation_state() -> dict:
    if "automation_state" not in st.session_state:
        st.session_state["automation_state"] = {
            "enabled": False,
            "interval_minutes": 30,
            "next_run_at": None,
            "last_run_at": None,
            "last_run_id": None,
            "config": None,
            "history": [],
        }
    return st.session_state["automation_state"]


def _trigger_automation_if_due() -> None:
    state = _get_automation_state()
    if not should_trigger_scheduled_run(state.get("enabled", False), state.get("next_run_at")):
        return

    cfg = state.get("config")
    if not cfg:
        state["enabled"] = False
        return

    try:
        payload = run_pipeline_and_persist(**cfg)
        result = payload["result"]
        now = datetime.utcnow()

        state["last_run_at"] = now
        state["last_run_id"] = result.run_id
        state["next_run_at"] = compute_next_run_ts(state.get("interval_minutes", 30))
        state["history"] = [
            {
                "run_id": result.run_id,
                "dataset": cfg["dataset_label"],
                "model": cfg["model_type"],
                "at": now.isoformat(timespec="seconds"),
            }
        ] + state.get("history", [])[:9]
    except Exception:
        state["enabled"] = False


_trigger_automation_if_due()

# ---------------------------------------------------------------------------
# CSS (shared with home)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
        [data-testid="metric-container"] {
            background: #f8fafc; border: 1px solid #e2e8f0;
            border-radius: 10px; padding: 18px 20px;
        }
        [data-testid="stMetricValue"] { font-size: 1.9rem !important; font-weight: 700; color: #0f172a; }
        .page-header { padding: 8px 0 24px 0; border-bottom: 2px solid #e2e8f0; margin-bottom: 28px; }
        .page-header h1 { font-size: 1.75rem; font-weight: 700; color: #0f172a; margin: 0; }
        .page-header p  { color: #64748b; margin: 4px 0 0 0; font-size: 0.9rem; }
        .section-title  { font-size: 1rem; font-weight: 600; color: #1e293b;
                          margin-bottom: 14px; padding-bottom: 6px; border-bottom: 1px solid #f1f5f9; }
        .stage-row { display: flex; align-items: center; gap: 10px;
                     padding: 8px 12px; border-radius: 6px; margin-bottom: 4px;
                     background: #f8fafc; border: 1px solid #e2e8f0; }
        .stage-dot { width:10px; height:10px; border-radius:50%; flex-shrink:0; }
        .stage-name { font-size:0.85rem; font-weight:600; color:#1e293b; flex:1; }
        .stage-dur  { font-size:0.78rem; color:#64748b; }
        [data-testid="stSidebar"] { background: #f8fafc; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="page-header">
      <h1>Pipeline Runner</h1>
      <p>Configure, execute, and inspect an end-to-end ML training pipeline.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Helper: hyperparameter widgets per algorithm
# ---------------------------------------------------------------------------
def _clf_param_widgets(model_type: str) -> dict:
    params = {}
    if model_type == "Random Forest":
        params["n_estimators"] = st.slider("n_estimators", 50, 500, 100, 50)
        params["max_depth"]    = st.select_slider(
            "max_depth", options=["None", 5, 10, 15, 20, 30], value="None"
        )
        if params["max_depth"] == "None":
            params["max_depth"] = None
        else:
            params["max_depth"] = int(params["max_depth"])
        params["min_samples_split"] = st.slider("min_samples_split", 2, 20, 2)

    elif model_type == "XGBoost":
        params["n_estimators"]  = st.slider("n_estimators",  50, 500, 100, 50)
        params["learning_rate"] = st.select_slider(
            "learning_rate", options=[0.01, 0.05, 0.1, 0.2, 0.3], value=0.1
        )
        params["max_depth"]     = st.slider("max_depth", 2, 12, 6)

    elif model_type == "Gradient Boosting":
        params["n_estimators"]  = st.slider("n_estimators",  50, 300, 100, 50)
        params["learning_rate"] = st.select_slider(
            "learning_rate", options=[0.01, 0.05, 0.1, 0.2, 0.3], value=0.1
        )
        params["max_depth"]     = st.slider("max_depth", 1, 8, 3)

    elif model_type == "Logistic Regression":
        params["C"]        = st.select_slider("C (regularisation)", [0.001, 0.01, 0.1, 1.0, 10.0, 100.0], value=1.0)
        params["max_iter"] = st.slider("max_iter", 100, 2000, 1000, 100)

    elif model_type == "SVM":
        params["C"]      = st.select_slider("C", [0.01, 0.1, 1.0, 10.0, 100.0], value=1.0)
        params["kernel"] = st.selectbox("kernel", ["rbf", "linear", "poly"])

    elif model_type == "Decision Tree":
        params["max_depth"] = st.select_slider(
            "max_depth", options=["None", 3, 5, 8, 12, 20], value="None"
        )
        if params["max_depth"] == "None":
            params["max_depth"] = None
        else:
            params["max_depth"] = int(params["max_depth"])
        params["min_samples_split"] = st.slider("min_samples_split", 2, 20, 2)

    return params


def _reg_param_widgets(model_type: str) -> dict:
    params = {}
    if model_type == "Random Forest":
        params["n_estimators"] = st.slider("n_estimators", 50, 500, 100, 50)
        params["max_depth"]    = st.select_slider(
            "max_depth", options=["None", 5, 10, 15, 20], value="None"
        )
        params["max_depth"] = None if params["max_depth"] == "None" else int(params["max_depth"])

    elif model_type == "XGBoost":
        params["n_estimators"]  = st.slider("n_estimators",  50, 500, 100, 50)
        params["learning_rate"] = st.select_slider(
            "learning_rate", options=[0.01, 0.05, 0.1, 0.2, 0.3], value=0.1
        )
        params["max_depth"]     = st.slider("max_depth", 2, 12, 6)

    elif model_type == "Gradient Boosting":
        params["n_estimators"]  = st.slider("n_estimators", 50, 300, 100, 50)
        params["learning_rate"] = st.select_slider(
            "learning_rate", options=[0.01, 0.05, 0.1, 0.2, 0.3], value=0.1
        )
        params["max_depth"]     = st.slider("max_depth", 1, 8, 3)

    elif model_type == "Ridge Regression":
        params["alpha"] = st.select_slider("alpha", [0.001, 0.01, 0.1, 1.0, 10.0, 100.0], value=1.0)

    elif model_type == "SVR":
        params["C"]      = st.select_slider("C", [0.01, 0.1, 1.0, 10.0, 100.0], value=1.0)
        params["kernel"] = st.selectbox("kernel", ["rbf", "linear", "poly"])

    elif model_type == "Decision Tree":
        params["max_depth"] = st.select_slider(
            "max_depth", options=["None", 3, 5, 8, 12], value="None"
        )
        params["max_depth"] = None if params["max_depth"] == "None" else int(params["max_depth"])
        params["min_samples_split"] = st.slider("min_samples_split", 2, 20, 2)

    return params


# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Navigation")
    st.page_link("app.py", label="Overview")
    st.page_link("pages/1_Pipeline_Runner.py", label="Pipeline Runner")
    st.page_link("pages/2_Experiment_Tracking.py", label="Experiment Tracking")
    st.page_link("pages/3_Model_Registry.py", label="Model Registry")
    st.page_link("pages/4_Data_Drift.py", label="Data Drift")
    st.divider()

    st.markdown("### Configuration")
    st.divider()

    dataset_label = st.selectbox("Dataset", list(DATASET_OPTIONS.keys()))
    dataset_key   = DATASET_OPTIONS[dataset_label]

    default_test_size = float(PIPELINE_CONFIG.get("test_size", 0.20))
    default_cv_folds = int(PIPELINE_CONFIG.get("cv_folds", 5))
    default_seed = int(PIPELINE_CONFIG.get("random_seed", 42))

    test_size = st.slider("Test split", 0.10, 0.40, default_test_size, 0.05)
    cv_folds = st.slider("CV folds", 2, 10, default_cv_folds)
    random_state = st.number_input("Random seed", min_value=0, max_value=9999, value=default_seed)

    st.divider()
    st.markdown("**Algorithm**")

    # Determine available algorithms after dataset load attempt
    # (task is known from DATASET_OPTIONS config)
    task = DATASET_CONFIG.get(dataset_key, {}).get("task")
    if task not in {"classification", "regression"}:
        clf_datasets = {"breast_cancer", "wine", "iris", "digits", "synthetic_clf"}
        task = "classification" if dataset_key in clf_datasets else "regression"

    if task == "classification":
        algo_options = list(CLF_REGISTRY.keys())
    else:
        algo_options = list(REG_REGISTRY.keys())

    model_type = st.selectbox("Model", algo_options)

    st.divider()
    st.markdown("**Hyperparameters**")

    if task == "classification":
        params = _clf_param_widgets(model_type)
    else:
        params = _reg_param_widgets(model_type)

    st.divider()
    auth_ok = render_auth_controls()

    st.divider()
    run_btn = st.button(
        "Run Pipeline",
        type="primary",
        use_container_width=True,
        disabled=not auth_ok,
    )


# ---------------------------------------------------------------------------
# Dataset preview tab
# ---------------------------------------------------------------------------
tab_run, tab_data, tab_auto = st.tabs(
    ["Pipeline Execution", "Dataset Preview", "Automation"]
)

with tab_data:
    with st.spinner("Loading dataset..."):
        try:
            ds = load_dataset(dataset_key, test_size=test_size, random_state=int(random_state))
        except Exception as exc:
            st.error(f"Failed to load dataset: {exc}")
            st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Samples",  f"{ds['stats']['n_samples']:,}")
    c2.metric("Features", ds["stats"]["n_features"])
    c3.metric("Train",    f"{ds['stats']['train_size']:,}")
    c4.metric("Test",     f"{ds['stats']['test_size']:,}")

    if task == "classification":
        st.caption(
            f"Classes: {ds['stats']['n_classes']}   |   "
            f"Missing values: {ds['stats']['missing_values']}"
        )
    else:
        st.caption(
            f"Target mean: {ds['stats']['target_mean']:.4f}   |   "
            f"Target std: {ds['stats']['target_std']:.4f}"
        )

    st.markdown("---")
    st.markdown('<div class="section-title">Feature Statistics</div>', unsafe_allow_html=True)
    feat_stats = get_feature_statistics(ds["X_train"])
    st.dataframe(feat_stats.style.format("{:.4f}", na_rep="—"), use_container_width=True)

    st.markdown('<div class="section-title" style="margin-top:20px">Feature Distributions (first 9)</div>', unsafe_allow_html=True)
    sample_feats = ds["X_train"].columns[:9].tolist()
    n_cols = 3
    rows = [sample_feats[i:i + n_cols] for i in range(0, len(sample_feats), n_cols)]
    for row in rows:
        cols = st.columns(n_cols)
        for col_widget, feat in zip(cols, row):
            with col_widget:
                fig = px.histogram(
                    ds["X_train"],
                    x=feat,
                    nbins=30,
                    color_discrete_sequence=["#2563eb"],
                    labels={feat: feat},
                )
                fig.update_layout(
                    height=180,
                    margin=dict(l=0, r=0, t=24, b=0),
                    showlegend=False,
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    title=dict(text=feat, font=dict(size=11)),
                    xaxis=dict(showgrid=False, title=None),
                    yaxis=dict(showgrid=False, title=None, showticklabels=False),
                )
                st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Pipeline execution tab
# ---------------------------------------------------------------------------
with tab_run:
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None

    if not is_authenticated():
        st.warning("Please login from the sidebar to execute pipeline runs.")

    if run_btn:
        # UI placeholders
        prog_bar    = st.progress(0.0)
        status_ph   = st.empty()
        log_ph      = st.empty()

        all_logs: list  = []

        def _progress_cb(stage: str, progress: float, message: str) -> None:
            prog_bar.progress(progress)
            status_ph.markdown(
                f"<div style='color:#2563eb;font-weight:600;font-size:0.9rem'>"
                f"Running: {stage} — {message}</div>",
                unsafe_allow_html=True,
            )
            ts = time.strftime("%H:%M:%S")
            all_logs.append(f"[{ts}] [{stage}] {message}")
            log_ph.code("\n".join(all_logs[-20:]), language=None)

        try:
            payload = run_pipeline_and_persist(
                dataset_label=dataset_label,
                dataset_key=dataset_key,
                model_type=model_type,
                task=task,
                params=params,
                test_size=float(test_size),
                cv_folds=int(cv_folds),
                random_state=int(random_state),
                progress_callback=_progress_cb,
            )
        except Exception as exc:
            status_ph.error(f"Pipeline failed: {exc}")
            st.stop()

        result = payload["result"]
        ds = payload["dataset"]
        model_record = payload["model_record"]
        model_artifact_path = payload["artifacts"]["model_path"]

        prog_bar.progress(1.0)
        status_ph.success(
            f"Run {result.run_id} completed in {result.duration:.2f} s"
        )

        if model_artifact_path:
            st.caption(
                f"Artifacts saved. Model v{model_record['version']} at {Path(model_artifact_path).as_posix()}"
            )

        st.session_state["last_result"] = result
        st.session_state["last_ds"]     = ds

    # -----------------------------------------------------------------------
    # Results section
    # -----------------------------------------------------------------------
    result = st.session_state.get("last_result")
    ds_res = st.session_state.get("last_ds")

    if result is None:
        st.info(
            "Configure your experiment in the sidebar and click **Run Pipeline** to start."
        )
    else:
        st.markdown("---")
        st.markdown(
            f'<div class="section-title">Results — Run {result.run_id}</div>',
            unsafe_allow_html=True,
        )

        # Metrics
        metrics = result.metrics
        if result.task == "classification":
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Accuracy",  f"{metrics.get('accuracy', 0):.4f}")
            m2.metric("Precision", f"{metrics.get('precision', 0):.4f}")
            m3.metric("Recall",    f"{metrics.get('recall', 0):.4f}")
            m4.metric("F1 Score",  f"{metrics.get('f1_score', 0):.4f}")
            m5.metric("ROC-AUC",   f"{metrics.get('roc_auc', 0):.4f}" if "roc_auc" in metrics else "—")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
            m2.metric("MAE",  f"{metrics.get('mae', 0):.4f}")
            m3.metric("R²",   f"{metrics.get('r2', 0):.4f}")

        cv_mean = metrics.get("cv_mean", 0)
        cv_std  = metrics.get("cv_std", 0)
        st.caption(f"Cross-validation: {cv_mean:.4f} ± {cv_std:.4f}  ({result.cv_scores.shape[0]}-fold)")

        st.markdown("<br>", unsafe_allow_html=True)

        # Stage summary
        st.markdown('<div class="section-title">Stage Summary</div>', unsafe_allow_html=True)
        stage_cols = st.columns(len(result.stages))
        colors = {"success": "#22c55e", "failed": "#ef4444", "skipped": "#94a3b8"}
        for i, stage in enumerate(result.stages):
            with stage_cols[i]:
                dot_color = colors.get(stage.status, "#94a3b8")
                st.markdown(
                    f"""
                    <div style="text-align:center;padding:12px 6px;
                                background:#f8fafc;border:1px solid #e2e8f0;
                                border-radius:8px">
                        <div style="width:10px;height:10px;border-radius:50%;
                                    background:{dot_color};margin:0 auto 6px auto"></div>
                        <div style="font-size:0.72rem;font-weight:600;color:#1e293b;
                                    line-height:1.3">{stage.name}</div>
                        <div style="font-size:0.7rem;color:#64748b;margin-top:3px">
                            {stage.duration:.2f} s</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts
        chart_left, chart_right = st.columns(2, gap="large")

        with chart_left:
            if result.feature_importances is not None:
                st.markdown('<div class="section-title">Feature Importances (Top 15)</div>', unsafe_allow_html=True)
                top_n = result.feature_importances.head(15).sort_values()
                fig = go.Figure(
                    go.Bar(
                        x=top_n.values,
                        y=top_n.index,
                        orientation="h",
                        marker_color="#2563eb",
                        marker_line_width=0,
                    )
                )
                fig.update_layout(
                    height=380,
                    margin=dict(l=0, r=20, t=10, b=0),
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    xaxis=dict(showgrid=True, gridcolor="#f1f5f9", title="Importance"),
                    yaxis=dict(showgrid=False, title=None, tickfont=dict(size=11)),
                )
                st.plotly_chart(fig, use_container_width=True)

            if result.cv_scores is not None:
                st.markdown('<div class="section-title" style="margin-top:8px">CV Score Distribution</div>', unsafe_allow_html=True)
                fold_df = pd.DataFrame({
                    "Fold": [f"Fold {i+1}" for i in range(len(result.cv_scores))],
                    "Score": result.cv_scores,
                })
                fig_cv = px.bar(
                    fold_df,
                    x="Fold",
                    y="Score",
                    color_discrete_sequence=["#6366f1"],
                    text="Score",
                )
                fig_cv.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                fig_cv.update_layout(
                    height=260,
                    margin=dict(l=0, r=0, t=10, b=0),
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    xaxis=dict(showgrid=False),
                    yaxis=dict(
                        showgrid=True, gridcolor="#f1f5f9",
                        range=[
                            max(0, result.cv_scores.min() - 0.05),
                            min(1.0, result.cv_scores.max() + 0.05),
                        ],
                    ),
                    showlegend=False,
                )
                st.plotly_chart(fig_cv, use_container_width=True)

        with chart_right:
            if result.task == "classification" and result.confusion_mat is not None:
                st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
                cm = result.confusion_mat
                labels = (
                    ds_res["target_names"]
                    if ds_res and ds_res.get("target_names")
                    else [str(i) for i in range(cm.shape[0])]
                )
                fig_cm = ff.create_annotated_heatmap(
                    z=cm,
                    x=labels,
                    y=labels,
                    colorscale="Blues",
                    showscale=True,
                )
                fig_cm.update_layout(
                    height=380,
                    margin=dict(l=60, r=20, t=30, b=60),
                    xaxis=dict(title="Predicted", side="bottom"),
                    yaxis=dict(title="Actual", autorange="reversed"),
                    paper_bgcolor="white",
                )
                st.plotly_chart(fig_cm, use_container_width=True)

        # Stage logs expander
        with st.expander("Stage Logs"):
            for stage in result.stages:
                st.markdown(f"**{stage.name}** ({stage.duration:.3f} s)")
                if stage.logs:
                    st.code("\n".join(stage.logs), language=None)
                st.markdown("---")


with tab_auto:
    st.markdown('<div class="section-title">Scheduled Pipeline Runs (Simulated Cron)</div>', unsafe_allow_html=True)
    st.caption(
        "Automation executes whenever this page refreshes and the next run timestamp is due. "
        "This provides cron-like behavior inside Streamlit sessions."
    )

    auto = _get_automation_state()

    c1, c2, c3 = st.columns(3)
    c1.metric("Automation", "Enabled" if auto.get("enabled") else "Disabled")
    c2.metric("Last Auto Run", auto.get("last_run_id") or "—")

    next_run_label = "—"
    if auto.get("next_run_at"):
        next_run_label = auto["next_run_at"].strftime("%Y-%m-%d %H:%M:%S UTC")
    c3.metric("Next Run", next_run_label)

    enabled = st.toggle("Enable automation", value=bool(auto.get("enabled", False)))
    interval = st.slider(
        "Run interval (minutes)",
        min_value=1,
        max_value=180,
        value=int(auto.get("interval_minutes", 30)),
    )

    col_set, col_stop = st.columns(2)
    with col_set:
        set_schedule = st.button("Apply Schedule", type="primary", use_container_width=True)
    with col_stop:
        stop_schedule = st.button("Stop Automation", use_container_width=True)

    if set_schedule:
        if not is_authenticated():
            st.error("Login required to configure automation.")
            st.stop()
        auto["enabled"] = bool(enabled)
        auto["interval_minutes"] = int(interval)
        auto["config"] = {
            "dataset_label": dataset_label,
            "dataset_key": dataset_key,
            "model_type": model_type,
            "task": task,
            "params": params,
            "test_size": float(test_size),
            "cv_folds": int(cv_folds),
            "random_state": int(random_state),
        }
        auto["next_run_at"] = compute_next_run_ts(interval)
        st.success("Automation schedule updated.")

    if stop_schedule:
        if not is_authenticated():
            st.error("Login required to stop automation.")
            st.stop()
        auto["enabled"] = False
        auto["next_run_at"] = None
        st.info("Automation stopped.")

    history = auto.get("history", [])
    if history:
        st.markdown('<div class="section-title" style="margin-top:18px">Recent Automated Runs</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)
    else:
        st.info("No automated runs yet.")
