"""
ML Pipeline Monitor — Home Dashboard

Provides an at-a-glance view of experiment history, model registry health,
and current host resource utilisation.  Use the sidebar navigation to access
the Pipeline Runner, Experiment Tracking, Model Registry, and Data Drift pages.
"""
import json
import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from src.database import get_experiments, get_models, initialize_db
from src.system_monitor import get_system_metrics

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ML Pipeline Monitor",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "ML Pipeline Monitor — production MLOps observability platform."},
)

initialize_db()

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* ---- typography ---- */
        html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

        /* ---- metric cards ---- */
        [data-testid="metric-container"] {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 18px 20px;
        }
        [data-testid="metric-container"] > div:first-child {
            color: #64748b;
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.9rem !important;
            font-weight: 700;
            color: #0f172a;
        }

        /* ---- page header ---- */
        .page-header {
            padding: 8px 0 24px 0;
            border-bottom: 2px solid #e2e8f0;
            margin-bottom: 28px;
        }
        .page-header h1 {
            font-size: 1.75rem;
            font-weight: 700;
            color: #0f172a;
            margin: 0;
        }
        .page-header p {
            color: #64748b;
            margin: 4px 0 0 0;
            font-size: 0.9rem;
        }

        /* ---- section headings ---- */
        .section-title {
            font-size: 1rem;
            font-weight: 600;
            color: #1e293b;
            margin-bottom: 14px;
            padding-bottom: 6px;
            border-bottom: 1px solid #f1f5f9;
        }

        /* ---- status pills ---- */
        .pill {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 600;
        }
        .pill-green  { background: #dcfce7; color: #16a34a; }
        .pill-yellow { background: #fef9c3; color: #854d0e; }
        .pill-blue   { background: #dbeafe; color: #1d4ed8; }
        .pill-red    { background: #fee2e2; color: #dc2626; }
        .pill-gray   { background: #f1f5f9; color: #475569; }

        /* ---- sidebar tweaks ---- */
        [data-testid="stSidebar"] { background: #f8fafc; }
        [data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

        /* ---- dataframe ---- */
        .stDataFrame { border-radius: 8px; overflow: hidden; }

        /* ---- progress bars ---- */
        .resource-bar {
            height: 8px;
            border-radius: 4px;
            background: #e2e8f0;
            overflow: hidden;
            margin-top: 4px;
        }
        .resource-bar-fill {
            height: 100%;
            border-radius: 4px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ML Pipeline Monitor")
    st.caption("v1.0.0  —  MLOps observability")
    st.divider()
    st.markdown("**Navigation**")
    st.page_link("app.py",                             label="Overview",             icon=None)
    st.page_link("pages/1_Pipeline_Runner.py",         label="Pipeline Runner",      icon=None)
    st.page_link("pages/2_Experiment_Tracking.py",     label="Experiment Tracking",  icon=None)
    st.page_link("pages/3_Model_Registry.py",          label="Model Registry",       icon=None)
    st.page_link("pages/4_Data_Drift.py",              label="Data Drift",           icon=None)
    st.divider()
    if st.button("Refresh dashboard", use_container_width=True):
        st.rerun()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="page-header">
      <h1>Overview</h1>
      <p>Real-time summary of experiments, model registry, and infrastructure health.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
experiments = get_experiments(limit=200)
models = get_models(limit=100)

exp_df = pd.DataFrame(experiments) if experiments else pd.DataFrame()
mdl_df = pd.DataFrame(models) if models else pd.DataFrame()

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
total_runs = len(exp_df)
registered_models = len(mdl_df)

best_accuracy = None
avg_duration = None

if not exp_df.empty:
    def _parse_metrics(row):
        try:
            return json.loads(row) if isinstance(row, str) else row
        except Exception:
            return {}

    exp_df["_metrics"] = exp_df["metrics"].apply(_parse_metrics)
    accs = [m.get("accuracy") for m in exp_df["_metrics"] if isinstance(m, dict) and "accuracy" in m]
    if accs:
        best_accuracy = max(accs)
    if "duration_seconds" in exp_df.columns:
        avg_duration = exp_df["duration_seconds"].mean()

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric("Total Experiments", f"{total_runs:,}")
with k2:
    st.metric(
        "Best Accuracy",
        f"{best_accuracy:.4f}" if best_accuracy is not None else "—",
    )
with k3:
    st.metric(
        "Avg. Pipeline Duration",
        f"{avg_duration:.1f} s" if avg_duration is not None else "—",
    )
with k4:
    st.metric("Registered Models", f"{registered_models:,}")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Main content: recent experiments + system health
# ---------------------------------------------------------------------------
left_col, right_col = st.columns([2, 1], gap="large")

with left_col:
    st.markdown('<div class="section-title">Recent Experiments</div>', unsafe_allow_html=True)

    if exp_df.empty:
        st.info(
            "No experiments recorded yet.  "
            "Head to the **Pipeline Runner** page to train your first model."
        )
    else:
        # Parse and format display dataframe
        display_cols = ["run_id", "dataset", "model_type", "task", "duration_seconds", "created_at"]
        display_cols = [c for c in display_cols if c in exp_df.columns]
        recent = exp_df.head(10)[display_cols].copy()

        # Attach primary metric
        primary_metric = []
        for _, row in exp_df.head(10).iterrows():
            m = row.get("_metrics", {}) or {}
            if isinstance(m, str):
                try:
                    m = json.loads(m)
                except Exception:
                    m = {}
            val = m.get("accuracy") or m.get("f1_score") or m.get("r2")
            primary_metric.append(round(val, 4) if val is not None else None)

        recent["primary_metric"] = primary_metric
        if "duration_seconds" in recent.columns:
            recent["duration_seconds"] = recent["duration_seconds"].apply(
                lambda x: f"{x:.2f} s" if pd.notna(x) else "—"
            )
        recent.columns = [c.replace("_", " ").title() for c in recent.columns]
        st.dataframe(recent, use_container_width=True, hide_index=True)

    # Activity chart — experiments over time
    if not exp_df.empty and "created_at" in exp_df.columns:
        st.markdown('<div class="section-title" style="margin-top:24px">Experiment Activity</div>', unsafe_allow_html=True)
        exp_df["date"] = pd.to_datetime(exp_df["created_at"]).dt.date
        activity = exp_df.groupby("date").size().reset_index(name="count")
        fig = px.bar(
            activity,
            x="date",
            y="count",
            labels={"date": "Date", "count": "Experiments"},
            color_discrete_sequence=["#2563eb"],
        )
        fig.update_layout(
            height=240,
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(showgrid=False, title=None),
            yaxis=dict(showgrid=True, gridcolor="#f1f5f9", title=None),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

with right_col:
    # -----------------------------------------------------------------------
    # System health
    # -----------------------------------------------------------------------
    st.markdown('<div class="section-title">System Health</div>', unsafe_allow_html=True)

    try:
        sys_m = get_system_metrics()

        def _bar_color(pct: float) -> str:
            if pct < 60:
                return "#22c55e"
            if pct < 85:
                return "#f59e0b"
            return "#ef4444"

        for label, pct in [
            ("CPU", sys_m["cpu_percent"]),
            ("Memory", sys_m["memory_percent"]),
            ("Disk", sys_m["disk_percent"]),
        ]:
            color = _bar_color(pct)
            st.markdown(
                f"""
                <div style="margin-bottom:14px">
                  <div style="display:flex;justify-content:space-between;
                              font-size:0.82rem;color:#374151;margin-bottom:4px">
                    <span>{label}</span>
                    <span style="font-weight:600">{pct:.1f}%</span>
                  </div>
                  <div class="resource-bar">
                    <div class="resource-bar-fill"
                         style="width:{pct}%;background:{color}"></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.caption(
            f"CPU cores: {sys_m['cpu_logical_cores']} logical / "
            f"{sys_m['cpu_physical_cores']} physical"
        )
        st.caption(
            f"Memory: {sys_m['memory_used_gb']:.1f} GB used / "
            f"{sys_m['memory_total_gb']:.1f} GB total"
        )
        st.caption(
            f"Disk: {sys_m['disk_free_gb']:.1f} GB free / "
            f"{sys_m['disk_total_gb']:.1f} GB total"
        )
    except Exception as exc:
        st.warning(f"System metrics unavailable: {exc}")

    # -----------------------------------------------------------------------
    # Algorithm distribution
    # -----------------------------------------------------------------------
    if not exp_df.empty and "model_type" in exp_df.columns:
        st.markdown(
            '<div class="section-title" style="margin-top:28px">Experiments by Algorithm</div>',
            unsafe_allow_html=True,
        )
        algo_counts = exp_df["model_type"].value_counts().reset_index()
        algo_counts.columns = ["Algorithm", "Count"]
        fig2 = px.pie(
            algo_counts,
            names="Algorithm",
            values="Count",
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.5,
        )
        fig2.update_layout(
            height=260,
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=True,
            legend=dict(orientation="v", font=dict(size=11)),
            paper_bgcolor="white",
        )
        fig2.update_traces(textposition="inside", textinfo="percent")
        st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
# Footer note
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "ML Pipeline Monitor  —  built with Streamlit, scikit-learn, XGBoost, Plotly, and SQLite."
)
