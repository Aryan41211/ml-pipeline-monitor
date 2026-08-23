"""
ML Pipeline Monitor — Executive Command Center
Redesigned with reusable enterprise components.
"""

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from ml_pipeline_monitor.core.auth import current_role, is_authenticated, render_auth_controls
from ml_pipeline_monitor.core.metrics import start_metrics_server
from ml_pipeline_monitor.services.app_service import get_dashboard_snapshot, initialize_application
from ml_pipeline_monitor.services.telemetry_service import track_user_action
from ml_pipeline_monitor.utils.ui_theme import (
    CHART_SEQUENCE,
    apply_ui_theme,
    component_alert_card,
    component_health_score,
    component_insight_panel,
    component_kpi_card,
    component_timeline,
    page_header,
    render_loading_skeleton,
    render_section_title,
    render_sidebar_nav,
    render_spacer,
    render_top_navbar,
)

# ---------------------------------------------------------------------------
# Shell setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Dashboard | ML Monitor", layout="wide")
initialize_application()
apply_ui_theme()

start_metrics_server()

render_top_navbar(user_role=current_role())

with st.sidebar:
    render_sidebar_nav()
    st.divider()
    render_auth_controls()

if not is_authenticated():
    st.warning("Please log in to access the dashboard.")
    st.stop()

# Track page view
track_user_action("page_view", page="dashboard")


# ---------------------------------------------------------------------------
# Data Logic
# ---------------------------------------------------------------------------
@st.cache_data(ttl=15)
def _load_dashboard():
    return get_dashboard_snapshot(limit=100)


loading = st.empty()
with loading.container():
    render_loading_skeleton(lines=5)

try:
    snapshot = _load_dashboard()
    exp_df = pd.DataFrame(snapshot.get("experiments", []))
    mdl_df = pd.DataFrame(snapshot.get("models", []))
    sys_snapshot = snapshot.get("system", {})
    loading.empty()
except Exception as e:
    loading.empty()
    component_alert_card(f"Failed to load dashboard data: {e}", tone="danger")
    st.stop()

# ---------------------------------------------------------------------------
# KPI & Health Scoring
# ---------------------------------------------------------------------------
best_acc = 0.0
success_rate = 100.0
if not exp_df.empty:

    def _p(m):
        return json.loads(m) if isinstance(m, str) else (m or {})

    accs = [float(_p(m).get("accuracy", 0)) for m in exp_df["metrics"]]
    best_acc = max(accs) if accs else 0.0
    success_rate = (len(exp_df[exp_df["status"] == "completed"]) / len(exp_df)) * 100

health_score = int((success_rate * 0.5) + (min(best_acc * 100, 100) * 0.5))

# ---------------------------------------------------------------------------
# Main Layout
# ---------------------------------------------------------------------------
col_head, col_action = st.columns([4, 1])
with col_head:
    page_header("Command Center", "Real-time MLOps orchestration and fleet observability.")
with col_action:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if st.button("Sync Platform", type="primary", use_container_width=True):
        track_user_action("sync_platform", page="dashboard")
        st.rerun()

# KPI Row
c1, c2, c3, c4 = st.columns(4)
with c1:
    component_kpi_card("Experiments", f"{len(exp_df)}", "All-time runs")
with c2:
    component_kpi_card("Models", f"{len(mdl_df)}", "In Registry")
# Guard against missing 'stage' column when registry is empty/uninitialized.
_production_count = 0
if not mdl_df.empty and "stage" in mdl_df.columns:
    _production_count = len(mdl_df[mdl_df["stage"] == "production"])

with c3:
    component_kpi_card("Serving", str(_production_count), "Production", tone="success")
with c4:
    component_kpi_card("Accuracy", f"{best_acc:.3f}", "Best Result", tone="success")

render_spacer("md")

# Empty state handling
if exp_df.empty:
    component_alert_card("No experiments recorded yet. Run a pipeline to populate the dashboard.", tone="info")

# Executive Section
m_left, m_mid, m_right = st.columns([2, 1, 1], gap="medium")

with m_left:
    render_section_title("Production Throughput")
    if not exp_df.empty:
        exp_df["ts"] = pd.to_datetime(exp_df["created_at"], errors="coerce", format="mixed")
        fig = px.area(
            exp_df.sort_values("ts"), x="ts", y="duration_seconds", color_discrete_sequence=[CHART_SEQUENCE[0]]
        )
        fig.update_traces(line_width=1.5)
        fig.update_layout(
            height=260,
            margin=dict(l=8, r=8, t=8, b=8),
            xaxis_title=None,
            yaxis_title="Run duration (s)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        component_alert_card("No experiment data available for throughput chart.", tone="info")

with m_mid:
    render_section_title("Platform Score")
    component_health_score(health_score)

with m_right:
    render_section_title("AI Context")
    # Every line here must be derived from the data on screen; the panel used
    # to assert fixed model-comparison and latency claims that nothing measured.
    insights = []
    if not exp_df.empty:
        insights.append(f"Completion rate: {success_rate:.1f}% across {len(exp_df)} run(s).")
        if "model_type" in exp_df.columns and not exp_df["model_type"].empty:
            top_model = exp_df["model_type"].value_counts().idxmax()
            insights.append(f"Most-used estimator: {top_model}.")
        insights.append(f"Best recorded accuracy: {best_acc:.3f}.")
    else:
        insights.append("Run experiments to generate insights.")

    cpu = sys_snapshot.get("cpu_percent")
    mem = sys_snapshot.get("memory_percent")
    if cpu is not None and mem is not None:
        insights.append(f"Host load: {cpu:.0f}% CPU, {mem:.0f}% memory.")
    component_insight_panel(insights)

render_spacer("md")

# Activity & Distribution
b_left, b_right = st.columns([1.5, 1], gap="medium")

with b_left:
    render_section_title("Recent Activity Feed")
    if not exp_df.empty:
        events = []
        for _, r in exp_df.head(6).iterrows():
            events.append(
                {
                    "time": str(r["created_at"])[11:16],
                    "label": f"{r['model_type']} on {r['dataset']}",
                    "status": r["status"],
                }
            )
        component_timeline(events)
    else:
        component_alert_card("No recent activity.", tone="info")

with b_right:
    render_section_title("Registry Fleet")
    if not mdl_df.empty:
        fig_pie = px.pie(mdl_df, names="stage", hole=0.62, color_discrete_sequence=CHART_SEQUENCE)
        fig_pie.update_traces(textposition="outside", textinfo="label+value", sort=False)
        fig_pie.update_layout(height=260, margin=dict(l=8, r=8, t=8, b=8), showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        component_alert_card("No models in registry.", tone="info")

st.divider()
st.caption("ML Pipeline Monitor")
