"""
Experiment Tracking page.

Displays all recorded experiment runs, supports filtering by dataset and
algorithm, and renders metric comparison charts to help identify the best
performing configurations.
"""
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from services.app_service import initialize_application
from src.auth import render_auth_controls
from services.pipeline_service import list_experiments
from services.telemetry_service import track_user_action
from src.ui_theme import (
    apply_plotly_layout,
    apply_ui_theme,
    render_empty_data_explainer,
    render_expandable_rows,
    render_loading_skeleton,
    render_page_header,
    render_page_header_with_action,
    render_section_title,
    render_sidebar_brand,
    render_sidebar_nav,
    render_summary_table,
)

st.set_page_config(page_title="Experiment Tracking | ML Monitor", layout="wide")
initialize_application()
apply_ui_theme()
refresh_clicked = render_page_header_with_action(
    "Experiment tracking",
    "Compare all runs across accuracy, F1, ROC and operational metrics.",
    "Refresh",
    action_key="tracking_refresh",
)
if refresh_clicked:
    track_user_action("experiment_tracking", "refresh")
    st.rerun()

# ---------------------------------------------------------------------------
# Load and parse
# ---------------------------------------------------------------------------
sk = st.empty()
with sk.container():
    render_loading_skeleton(lines=3, key="tracking_load")


@st.cache_data(ttl=20, show_spinner=False)
def _cached_experiments(limit: int):
    return list_experiments(limit=limit)


try:
    raw = _cached_experiments(limit=200)
except Exception as exc:
    sk.empty()
    st.error("Unable to load experiments right now.")
    st.caption(f"Details: {exc}")
    if st.button("Retry", key="tracking_retry", type="primary"):
        st.rerun()
    st.stop()
sk.empty()
if not raw:
    render_empty_data_explainer(
        "No experiment records exist in the local database yet.",
        "Run a pipeline from Pipeline Runner.",
        "This page will auto-populate comparison tables and charts.",
    )
    st.page_link("pages/1_Pipeline_Runner.py", label="Run first pipeline")
    st.stop()


def _safe_json(val):
    if isinstance(val, dict):
        return val
    try:
        return json.loads(val) if val else {}
    except Exception:
        return {}


rows = []
for exp in raw:
    m = _safe_json(exp.get("metrics"))
    p = _safe_json(exp.get("params"))
    rows.append(
        {
            "run_id":    exp.get("run_id", ""),
            "dataset":   exp.get("dataset", ""),
            "algorithm": exp.get("model_type", ""),
            "task":      exp.get("task", "classification"),
            "duration":  round(exp.get("duration_seconds") or 0, 2),
            "created_at": exp.get("created_at", ""),
            "status": exp.get("status", "completed"),
            # classification metrics
            "accuracy":  m.get("accuracy"),
            "precision": m.get("precision"),
            "recall":    m.get("recall"),
            "f1_score":  m.get("f1_score"),
            "roc_auc":   m.get("roc_auc"),
            # regression metrics
            "rmse":      m.get("rmse"),
            "mae":       m.get("mae"),
            "r2":        m.get("r2"),
            # cv
            "cv_mean":   m.get("cv_mean"),
            "cv_std":    m.get("cv_std"),
            "roc_curve_fpr": m.get("roc_curve_fpr") if isinstance(m.get("roc_curve_fpr"), list) else None,
            "roc_curve_tpr": m.get("roc_curve_tpr") if isinstance(m.get("roc_curve_tpr"), list) else None,
        }
    )

df = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
with st.sidebar:
    render_sidebar_brand()
    st.markdown("### Navigation")
    render_sidebar_nav()
    st.divider()

    st.markdown("### Configuration")
    st.divider()

    datasets = ["All"] + (sorted(df["dataset"].unique().tolist()) if not df.empty else [])
    algorithms = ["All"] + (sorted(df["algorithm"].unique().tolist()) if not df.empty else [])
    tasks = ["All"] + (sorted(df["task"].unique().tolist()) if not df.empty else [])

    sel_dataset = st.selectbox("Dataset",   datasets)
    sel_algo    = st.selectbox("Algorithm", algorithms)
    sel_task    = st.selectbox("Task",      tasks)

    st.divider()
    if st.button("Clear filters", width="stretch"):
        track_user_action("experiment_tracking", "clear_filters")
        st.rerun()

    st.divider()
    st.markdown("### Access")
    render_auth_controls()

filtered = df.copy()
if sel_dataset != "All":
    filtered = filtered[filtered["dataset"] == sel_dataset]
if sel_algo != "All":
    filtered = filtered[filtered["algorithm"] == sel_algo]
if sel_task != "All":
    filtered = filtered[filtered["task"] == sel_task]

# ---------------------------------------------------------------------------
# KPI strip
# ---------------------------------------------------------------------------
clf_rows = filtered[filtered["task"] == "classification"]
reg_rows = filtered[filtered["task"] == "regression"]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Runs shown",      len(filtered))
k2.metric(
    "Best accuracy",
    f"{clf_rows['accuracy'].max():.4f}" if not clf_rows.empty and clf_rows["accuracy"].notna().any() else "â€”",
)
k3.metric(
    "Best F1",
    f"{clf_rows['f1_score'].max():.4f}" if not clf_rows.empty and clf_rows["f1_score"].notna().any() else "â€”",
)
k4.metric(
    "Best RÂ²",
    f"{reg_rows['r2'].max():.4f}" if not reg_rows.empty and reg_rows["r2"].notna().any() else "â€”",
)

st.markdown("<br>", unsafe_allow_html=True)

if filtered.empty:
    render_empty_data_explainer(
        "No experiment runs match the current database and filter selection.",
        "Run a pipeline first, or reset filters from the sidebar.",
        "Once a run completes, this page will auto-populate metrics and comparison charts.",
    )
    cta_left, cta_right = st.columns(2)
    with cta_left:
        st.page_link("pages/1_Pipeline_Runner.py", label="Run first pipeline")
    with cta_right:
        st.page_link("pages/3_Model_Registry.py", label="Open model registry")
    st.stop()

# ---------------------------------------------------------------------------
# Run-to-run delta tracking
# ---------------------------------------------------------------------------
render_section_title("Run-to-Run Metric Delta")
st.caption("Select two runs. Delta is calculated as Comparison minus Baseline.")

compare_columns = [
    "run_id",
    "dataset",
    "algorithm",
    "task",
    "created_at",
    "accuracy",
    "f1_score",
    "precision",
    "recall",
    "roc_auc",
    "rmse",
    "mae",
    "r2",
    "cv_mean",
    "cv_std",
]
compare_pool = filtered[[c for c in compare_columns if c in filtered.columns]].copy()
compare_pool = compare_pool.sort_values("created_at", ascending=False)

run_labels = [
    f"{row.run_id} | {row.algorithm} | {row.dataset} | {row.task}"
    for row in compare_pool.itertuples(index=False)
]
label_to_run = dict(zip(run_labels, compare_pool["run_id"].tolist()))

sel_left, sel_right = st.columns(2)
with sel_left:
    baseline_label = st.selectbox(
        "Baseline run",
        options=run_labels,
        key="delta_baseline_run",
    )

baseline_run_id = label_to_run.get(baseline_label)
comparison_options = [lbl for lbl in run_labels if label_to_run[lbl] != baseline_run_id]

with sel_right:
    comparison_label = st.selectbox(
        "Comparison run",
        options=comparison_options,
        key="delta_comparison_run",
    )

comparison_run_id = label_to_run.get(comparison_label)

if baseline_run_id and comparison_run_id:
    base_row = compare_pool[compare_pool["run_id"] == baseline_run_id].iloc[0]
    cmp_row = compare_pool[compare_pool["run_id"] == comparison_run_id].iloc[0]

    st.caption(f"Baseline: {baseline_run_id}   |   Comparison: {comparison_run_id}")

    metric_labels = {
        "accuracy": "Accuracy",
        "f1_score": "F1 Score",
        "precision": "Precision",
        "recall": "Recall",
        "roc_auc": "ROC AUC",
        "rmse": "RMSE",
        "mae": "MAE",
        "r2": "RÂ²",
        "cv_mean": "CV Mean",
        "cv_std": "CV Std",
    }
    priority = ["accuracy", "f1_score"]
    metric_candidates = [
        "accuracy",
        "f1_score",
        "precision",
        "recall",
        "roc_auc",
        "rmse",
        "mae",
        "r2",
        "cv_mean",
        "cv_std",
    ]

    shared = []
    for m in metric_candidates:
        if m not in compare_pool.columns:
            continue
        b_val = base_row.get(m)
        c_val = cmp_row.get(m)
        if pd.notna(b_val) and pd.notna(c_val):
            shared.append(m)

    ordered_shared = [m for m in priority if m in shared] + [m for m in shared if m not in priority]

    if not ordered_shared:
        st.info("No common numeric metrics are available between the selected runs.")
    else:
        metric_cols = st.columns(2)
        for idx, metric in enumerate(ordered_shared):
            baseline_value = float(base_row[metric])
            comparison_value = float(cmp_row[metric])
            delta = comparison_value - baseline_value
            delta_pct = None if baseline_value == 0 else (delta / abs(baseline_value)) * 100.0
            delta_text = f"{delta:+.4f}" if delta_pct is None else f"{delta:+.4f} ({delta_pct:+.2f}%)"

            with metric_cols[idx % 2]:
                st.metric(
                    metric_labels.get(metric, metric.replace("_", " ").title()),
                    f"{comparison_value:.4f}",
                    delta=delta_text,
                    delta_color="normal",
                )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Experiments table
# ---------------------------------------------------------------------------
render_section_title("All Runs")

display = filtered.copy()
display["duration"] = display["duration"].apply(lambda x: f"{x:.2f} s")

# Show relevant metric columns
show_cols = ["run_id", "status", "dataset", "algorithm", "task", "duration", "cv_mean"]
if clf_rows.shape[0] > 0:
    show_cols += ["accuracy", "f1_score", "roc_auc"]
if reg_rows.shape[0] > 0:
    show_cols += ["rmse", "r2"]
show_cols += ["created_at"]
show_cols = [c for c in show_cols if c in display.columns]

table_frame = display[show_cols].rename(columns=lambda c: c.replace("_", " ").title())
table_frame = render_summary_table(
    table_frame,
    key_prefix="exp_runs",
    columns=table_frame.columns.tolist(),
    sort_by="Created At" if "Created At" in table_frame.columns else None,
    filterable_columns=[c for c in ["Dataset", "Algorithm", "Task", "Status"] if c in table_frame.columns],
    max_rows=30,
)

render_section_title("Run Details", margin_top_px=14)
render_expandable_rows(
    table_frame,
    title_col="Run Id",
    detail_cols=[c for c in ["Status", "Dataset", "Algorithm", "Task", "Duration", "Created At"] if c in table_frame.columns],
    badge_col="Status" if "Status" in table_frame.columns else None,
    badge_mode="status",
)

# Export
csv = filtered.to_csv(index=False)
st.download_button(
    label="Export to CSV",
    data=csv,
    file_name="experiments.csv",
    mime="text/csv",
)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
load_charts = st.toggle("Load advanced charts", value=False, help="Enable to render heavy comparison charts.")
if not load_charts:
    st.info("Advanced charts are paused for performance. Toggle 'Load advanced charts' to render them.")
    st.stop()

tab_clf, tab_reg, tab_compare, tab_multi = st.tabs(
    ["Classification Metrics", "Regression Metrics", "Parameter Analysis", "Experiment Comparison"]
)

with tab_clf:
    if clf_rows.empty:
        st.info("No classification experiments in current filter.")
    else:
        c_left, c_right = st.columns(2, gap="large")

        with c_left:
            render_section_title("Accuracy by Algorithm")
            acc_df = clf_rows.dropna(subset=["accuracy"]).copy()
            fig = px.box(
                acc_df,
                x="algorithm",
                y="accuracy",
                color="algorithm",
                color_discrete_sequence=px.colors.qualitative.Set2,
                points="all",
            )
            fig.update_layout(
                showlegend=False,
            )
            apply_plotly_layout(fig, height=360, y_title="Accuracy")
            st.plotly_chart(fig, width="stretch")

        with c_right:
            render_section_title("F1 Score vs Accuracy")
            scatter_df = clf_rows.dropna(subset=["accuracy", "f1_score"])
            fig2 = px.scatter(
                scatter_df,
                x="accuracy",
                y="f1_score",
                color="algorithm",
                hover_data=["run_id", "dataset", "duration"],
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig2.update_layout(
                legend=dict(title=None),
            )
            apply_plotly_layout(fig2, height=360, x_title="Accuracy", y_title="F1 Score")
            st.plotly_chart(fig2, width="stretch")

        render_section_title("Metric Comparison Across Runs")
        metric_cols = ["accuracy", "precision", "recall", "f1_score"]
        available   = [c for c in metric_cols if clf_rows[c].notna().any()]
        melted = clf_rows[["run_id", "algorithm", "dataset"] + available].melt(
            id_vars=["run_id", "algorithm", "dataset"],
            var_name="metric",
            value_name="value",
        ).dropna()

        fig3 = px.bar(
            melted,
            x="run_id",
            y="value",
            color="metric",
            barmode="group",
            hover_data=["algorithm", "dataset"],
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig3.update_layout(
            legend=dict(title=None),
        )
        apply_plotly_layout(fig3, height=340, x_title="Run ID", y_title="Score")
        fig3.update_yaxes(range=[0, 1.05])
        st.plotly_chart(fig3, width="stretch")


with tab_reg:
    if reg_rows.empty:
        st.info("No regression experiments in current filter.")
    else:
        c_left, c_right = st.columns(2, gap="large")
        with c_left:
            render_section_title("RÂ² by Algorithm")
            fig = px.box(
                reg_rows.dropna(subset=["r2"]),
                x="algorithm",
                y="r2",
                color="algorithm",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                points="all",
            )
            fig.update_layout(
                height=360, margin=dict(l=0, r=0, t=10, b=0),
                showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(title=None, showgrid=False),
                yaxis=dict(title="RÂ²", showgrid=True, gridcolor="#f1f5f9"),
            )
            st.plotly_chart(fig, width="stretch")

        with c_right:
            render_section_title("RMSE vs MAE")
            fig2 = px.scatter(
                reg_rows.dropna(subset=["rmse", "mae"]),
                x="rmse",
                y="mae",
                color="algorithm",
                hover_data=["run_id", "dataset"],
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig2.update_layout(
                height=360, margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(title="RMSE", showgrid=True, gridcolor="#f1f5f9"),
                yaxis=dict(title="MAE", showgrid=True, gridcolor="#f1f5f9"),
            )
            st.plotly_chart(fig2, width="stretch")


with tab_compare:
    render_section_title("Training Duration vs Primary Metric")

    plot_df = filtered.copy()
    plot_df["primary_metric"] = (
        plot_df["accuracy"].fillna(plot_df["r2"])
    )
    plot_df = plot_df.dropna(subset=["duration", "primary_metric"])

    if plot_df.empty:
        st.info("Not enough data for comparison.")
    else:
        fig = px.scatter(
            plot_df,
            x="duration",
            y="primary_metric",
            color="algorithm",
            size_max=14,
            hover_data=["run_id", "dataset", "task"],
            labels={"duration": "Duration (s)", "primary_metric": "Primary Metric (Accuracy / RÂ²)"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(
            height=420,
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(title="Training Duration (s)", showgrid=True, gridcolor="#f1f5f9"),
            yaxis=dict(title="Primary Metric", showgrid=True, gridcolor="#f1f5f9"),
        )
        st.plotly_chart(fig, width="stretch")

    render_section_title("Cross-Validation Stability", margin_top_px=16)
    cv_df = filtered.dropna(subset=["cv_mean", "cv_std"]).copy()
    if not cv_df.empty:
        fig_cv = go.Figure()
        for algo in cv_df["algorithm"].unique():
            sub = cv_df[cv_df["algorithm"] == algo]
            fig_cv.add_trace(
                go.Scatter(
                    x=sub["run_id"],
                    y=sub["cv_mean"],
                    error_y=dict(type="data", array=(sub["cv_std"] * 2).tolist(), visible=True),
                    mode="markers+lines",
                    name=algo,
                    marker=dict(size=8),
                )
            )
        fig_cv.update_layout(
            height=340,
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(title="Run ID", showgrid=False),
            yaxis=dict(title="CV Score (mean Â± 2 std)", showgrid=True, gridcolor="#f1f5f9"),
            legend=dict(title="Algorithm"),
        )
        st.plotly_chart(fig_cv, width="stretch")


with tab_multi:
    render_section_title("Compare Multiple Experiments")
    st.caption("Select experiments to compare Accuracy, F1 score, and ROC behavior.")

    clf_candidates = filtered[filtered["task"] == "classification"].copy()
    if clf_candidates.empty:
        st.info("No classification experiments available for comparison under current filters.")
    else:
        labels = [
            f"{row.run_id} | {row.algorithm} | {row.dataset}"
            for row in clf_candidates.itertuples(index=False)
        ]
        label_to_run = dict(zip(labels, clf_candidates["run_id"].tolist()))

        selected_labels = st.multiselect(
            "Experiments",
            options=labels,
            default=labels[: min(4, len(labels))],
            help="Choose two or more runs for side-by-side comparison.",
        )

        selected_ids = [label_to_run[x] for x in selected_labels]
        chosen = clf_candidates[clf_candidates["run_id"].isin(selected_ids)].copy()

        if chosen.empty:
            st.info("Select at least one experiment.")
        else:
            s1, s2, s3 = st.columns(3)
            s1.metric("Selected Runs", len(chosen))
            s2.metric(
                "Best Accuracy",
                f"{chosen['accuracy'].max():.4f}" if chosen["accuracy"].notna().any() else "â€”",
            )
            s3.metric(
                "Best F1",
                f"{chosen['f1_score'].max():.4f}" if chosen["f1_score"].notna().any() else "â€”",
            )

            left, right = st.columns(2, gap="large")

            with left:
                bar_df = chosen[["run_id", "accuracy", "f1_score"]].copy()
                melted = bar_df.melt(id_vars=["run_id"], value_vars=["accuracy", "f1_score"]).dropna()
                if melted.empty:
                    st.info("Selected runs are missing accuracy/F1 metrics.")
                else:
                    fig = px.bar(
                        melted,
                        x="run_id",
                        y="value",
                        color="variable",
                        barmode="group",
                        labels={"value": "Score", "run_id": "Run ID", "variable": "Metric"},
                        color_discrete_sequence=["#2563eb", "#16a34a"],
                    )
                    fig.update_layout(
                        height=360,
                        margin=dict(l=0, r=0, t=10, b=0),
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        yaxis=dict(range=[0, 1.05], showgrid=True, gridcolor="#f1f5f9"),
                        xaxis=dict(showgrid=False),
                    )
                    st.plotly_chart(fig, width="stretch")

            with right:
                roc_rows = chosen.dropna(subset=["roc_curve_fpr", "roc_curve_tpr"]).copy()
                if roc_rows.empty:
                    st.info("ROC curve points unavailable for selected runs.")
                else:
                    fig_roc = go.Figure()
                    for row in roc_rows.itertuples(index=False):
                        fig_roc.add_trace(
                            go.Scatter(
                                x=row.roc_curve_fpr,
                                y=row.roc_curve_tpr,
                                mode="lines",
                                name=f"{row.run_id} ({row.algorithm})",
                            )
                        )

                    fig_roc.add_trace(
                        go.Scatter(
                            x=[0, 1],
                            y=[0, 1],
                            mode="lines",
                            name="Random baseline",
                            line=dict(dash="dash", color="#94a3b8"),
                        )
                    )
                    fig_roc.update_layout(
                        height=360,
                        margin=dict(l=0, r=0, t=10, b=0),
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        xaxis=dict(title="False Positive Rate", showgrid=True, gridcolor="#f1f5f9"),
                        yaxis=dict(title="True Positive Rate", showgrid=True, gridcolor="#f1f5f9"),
                        legend=dict(title=None),
                    )
                    st.plotly_chart(fig_roc, width="stretch")

