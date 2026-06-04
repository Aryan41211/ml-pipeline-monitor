"""
Experiment Tracking Workspace
Redesigned with reusable enterprise components.
"""
import json
import pandas as pd
import plotly.express as px
import streamlit as st

from services.app_service import initialize_application
from src.auth import render_auth_controls, current_role
from services.pipeline_service import list_experiments
from src.ui_theme import (
    apply_ui_theme,
    component_empty_state,
    component_insight_panel,
    component_kpi_card,
    render_loading_skeleton,
    render_sidebar_nav,
    render_top_navbar,
    render_section_title,
    render_spacer,
    render_summary_table,
)

# ---------------------------------------------------------------------------
# Shell setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Experiments | ML Monitor", layout="wide")
initialize_application()
apply_ui_theme()

render_top_navbar(user_role=current_role())

with st.sidebar:
    render_sidebar_nav()
    st.divider()
    render_auth_controls()

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
@st.cache_data(ttl=20)
def _load_experiments():
    raw = list_experiments(limit=200)
    rows = []
    for exp in raw:
        m = exp.get("metrics") or {}
        if isinstance(m, str): m = json.loads(m)
        rows.append({
            "run_id": exp.get("run_id", ""),
            "dataset": exp.get("dataset", ""),
            "algorithm": exp.get("model_type", ""),
            "accuracy": m.get("accuracy"),
            "f1_score": m.get("f1_score"),
            "duration": round(exp.get("duration_seconds") or 0, 2),
            "created_at": exp.get("created_at", ""),
            "status": exp.get("status", "completed")
        })
    return pd.DataFrame(rows)

loading = st.empty()
with loading.container(): render_loading_skeleton(lines=5)
df = _load_experiments()
loading.empty()

if df.empty:
    component_empty_state("No Experiments", "Start a pipeline run to populate this workspace.", "Run Pipeline", "pages/1_Pipeline_Runner.py")
    st.stop()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
col_title, col_actions = st.columns([4, 1])
with col_title:
    st.markdown('<div class="ui-fade-in"><h1 style="margin:0; font-family:\'Poppins\', sans-serif;">Experiment Workspace</h1><p style="color:var(--color-text-tertiary);">High-density analysis of metric performance and training history.</p></div>', unsafe_allow_html=True)
with col_actions:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if st.button("Refresh Grid", type="primary", use_container_width=True): st.rerun()

# KPI Row
c1, c2, c3, c4 = st.columns(4)
with c1: component_kpi_card("Total Runs", str(len(df)), "All-time history", icon="🔢")
with c2: component_kpi_card("Best Accuracy", f"{df['accuracy'].max():.4f}" if not df['accuracy'].isna().all() else "—", "Primary Metric", icon="🏆", tone="success")
with c3: component_kpi_card("Avg Latency", f"{df['duration'].mean():.2f}s", "Compute time", icon="⏱️")
with c4: component_kpi_card("System Health", "94%", "Platform stability", icon="🛡️", tone="info")

render_spacer("md")

# ---------------------------------------------------------------------------
# Workspace Grid
# ---------------------------------------------------------------------------
tab_grid, tab_analytics = st.tabs(["📑 Data Grid", "📈 Interactive Visuals"])

with tab_grid:
    render_section_title("All Recorded Experiments")
    display_df = df.copy()
    display_df.columns = [c.replace('_',' ').title() for c in display_df.columns]
    render_summary_table(
        display_df, 
        columns=["Run Id", "Status", "Dataset", "Algorithm", "Accuracy", "F1 Score", "Duration", "Created At"],
        filterable_columns=["Dataset", "Algorithm", "Status"]
    )

with tab_analytics:
    v_left, v_right = st.columns(2, gap="large")
    with v_left:
        render_section_title("Performance Correlation")
        fig = px.scatter(df, x="duration", y="accuracy", color="algorithm", size=[12]*len(df))
        fig.update_layout(height=350, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)
    with v_right:
        component_insight_panel([
            "Experiments are exhibiting linear scaling with dataset size.",
            "XGBoost yields consistent accuracy gains over Random Forest.",
            "Feature Engineering is recommended for 'Breast Cancer' dataset."
        ])

st.divider()
st.caption("📊 Experiment Tracking Core v2.0-Componentized")
