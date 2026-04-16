"""
Experiment Tracking page.

Displays all recorded experiment runs, supports filtering by dataset and
algorithm, and renders metric comparison charts to help identify the best
performing configurations.
"""
import json
import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.database import get_experiments, initialize_db

st.set_page_config(page_title="Experiment Tracking | ML Monitor", layout="wide")
initialize_db()

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
        [data-testid="stSidebar"] { background: #f8fafc; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="page-header">
      <h1>Experiment Tracking</h1>
      <p>Browse, filter, and compare all recorded training runs.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load and parse
# ---------------------------------------------------------------------------
raw = get_experiments(limit=200)

if not raw:
    st.info("No experiments found.  Run your first pipeline on the **Pipeline Runner** page.")
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
        }
    )

df = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Filters")
    st.divider()

    datasets   = ["All"] + sorted(df["dataset"].unique().tolist())
    algorithms = ["All"] + sorted(df["algorithm"].unique().tolist())
    tasks      = ["All"] + sorted(df["task"].unique().tolist())

    sel_dataset = st.selectbox("Dataset",   datasets)
    sel_algo    = st.selectbox("Algorithm", algorithms)
    sel_task    = st.selectbox("Task",      tasks)

    st.divider()
    if st.button("Clear filters", use_container_width=True):
        st.rerun()

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
    f"{clf_rows['accuracy'].max():.4f}" if not clf_rows.empty and clf_rows["accuracy"].notna().any() else "—",
)
k3.metric(
    "Best F1",
    f"{clf_rows['f1_score'].max():.4f}" if not clf_rows.empty and clf_rows["f1_score"].notna().any() else "—",
)
k4.metric(
    "Best R²",
    f"{reg_rows['r2'].max():.4f}" if not reg_rows.empty and reg_rows["r2"].notna().any() else "—",
)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Experiments table
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">All Runs</div>', unsafe_allow_html=True)

display = filtered.copy()
display["duration"] = display["duration"].apply(lambda x: f"{x:.2f} s")

# Show relevant metric columns
show_cols = ["run_id", "dataset", "algorithm", "task", "duration", "cv_mean"]
if clf_rows.shape[0] > 0:
    show_cols += ["accuracy", "f1_score", "roc_auc"]
if reg_rows.shape[0] > 0:
    show_cols += ["rmse", "r2"]
show_cols += ["created_at"]
show_cols = [c for c in show_cols if c in display.columns]

st.dataframe(
    display[show_cols].rename(columns=lambda c: c.replace("_", " ").title()),
    use_container_width=True,
    hide_index=True,
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
tab_clf, tab_reg, tab_compare = st.tabs(
    ["Classification Metrics", "Regression Metrics", "Parameter Analysis"]
)

with tab_clf:
    if clf_rows.empty:
        st.info("No classification experiments in current filter.")
    else:
        c_left, c_right = st.columns(2, gap="large")

        with c_left:
            st.markdown('<div class="section-title">Accuracy by Algorithm</div>', unsafe_allow_html=True)
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
                height=360,
                margin=dict(l=0, r=0, t=10, b=0),
                showlegend=False,
                plot_bgcolor="white",
                paper_bgcolor="white",
                xaxis=dict(title=None, showgrid=False),
                yaxis=dict(title="Accuracy", showgrid=True, gridcolor="#f1f5f9"),
            )
            st.plotly_chart(fig, use_container_width=True)

        with c_right:
            st.markdown('<div class="section-title">F1 Score vs Accuracy</div>', unsafe_allow_html=True)
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
                height=360,
                margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor="white",
                paper_bgcolor="white",
                xaxis=dict(title="Accuracy", showgrid=True, gridcolor="#f1f5f9"),
                yaxis=dict(title="F1 Score", showgrid=True, gridcolor="#f1f5f9"),
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="section-title">Metric Comparison Across Runs</div>', unsafe_allow_html=True)
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
            height=340,
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(title="Run ID", showgrid=False),
            yaxis=dict(title="Score", showgrid=True, gridcolor="#f1f5f9", range=[0, 1.05]),
            legend=dict(title=None),
        )
        st.plotly_chart(fig3, use_container_width=True)


with tab_reg:
    if reg_rows.empty:
        st.info("No regression experiments in current filter.")
    else:
        c_left, c_right = st.columns(2, gap="large")
        with c_left:
            st.markdown('<div class="section-title">R² by Algorithm</div>', unsafe_allow_html=True)
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
                yaxis=dict(title="R²", showgrid=True, gridcolor="#f1f5f9"),
            )
            st.plotly_chart(fig, use_container_width=True)

        with c_right:
            st.markdown('<div class="section-title">RMSE vs MAE</div>', unsafe_allow_html=True)
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
            st.plotly_chart(fig2, use_container_width=True)


with tab_compare:
    st.markdown('<div class="section-title">Training Duration vs Primary Metric</div>', unsafe_allow_html=True)

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
            labels={"duration": "Duration (s)", "primary_metric": "Primary Metric (Accuracy / R²)"},
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
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title" style="margin-top:16px">Cross-Validation Stability</div>', unsafe_allow_html=True)
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
            yaxis=dict(title="CV Score (mean ± 2 std)", showgrid=True, gridcolor="#f1f5f9"),
            legend=dict(title="Algorithm"),
        )
        st.plotly_chart(fig_cv, use_container_width=True)
