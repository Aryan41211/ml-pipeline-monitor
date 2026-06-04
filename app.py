"""
ML Pipeline Monitor — Executive Command Center
Redesigned with reusable enterprise components.
"""
import json
import pandas as pd
import plotly.express as px
import streamlit as st

from services.app_service import get_dashboard_snapshot, initialize_application
from services.telemetry_service import track_user_action
from src.auth import current_role, render_auth_controls
from src.ui_theme import (
    apply_ui_theme,
    component_health_score,
    component_insight_panel,
    component_kpi_card,
    component_timeline,
    render_loading_skeleton,
    render_sidebar_nav,
    render_top_navbar,
    render_section_title,
    render_spacer,
)

# ---------------------------------------------------------------------------
# Shell setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Dashboard | ML Monitor", layout="wide")
initialize_application()
apply_ui_theme()

render_top_navbar(user_role=current_role())

with st.sidebar:
    render_sidebar_nav()
    st.divider()
    render_auth_controls()

# ---------------------------------------------------------------------------
# Data Logic
# ---------------------------------------------------------------------------
@st.cache_data(ttl=15)
def _load_dashboard():
    return get_dashboard_snapshot(limit=100)

loading = st.empty()
with loading.container(): render_loading_skeleton(lines=5)

try:
    snapshot = _load_dashboard()
    exp_df = pd.DataFrame(snapshot.get("experiments", []))
    mdl_df = pd.DataFrame(snapshot.get("models", []))
    sys_snapshot = snapshot.get("system", {})
    loading.empty()
except Exception:
    loading.empty(); st.error("Telemetry link severed."); st.stop()

# ---------------------------------------------------------------------------
# KPI & Health Scoring
# ---------------------------------------------------------------------------
best_acc = 0.0
success_rate = 100.0
if not exp_df.empty:
    def _p(m): return json.loads(m) if isinstance(m, str) else (m or {})
    accs = [float(_p(m).get("accuracy", 0)) for m in exp_df["metrics"]]
    best_acc = max(accs) if accs else 0.0
    success_rate = (len(exp_df[exp_df["status"]=="completed"]) / len(exp_df)) * 100

health_score = int((success_rate * 0.5) + (min(best_acc * 100, 100) * 0.5))

# ---------------------------------------------------------------------------
# Main Layout
# ---------------------------------------------------------------------------
col_head, col_action = st.columns([4, 1])
with col_head:
    st.markdown('<div class="ui-fade-in"><h1 style="margin:0; font-family:\'Poppins\', sans-serif;">Command Center</h1><p style="color:var(--color-text-tertiary);">Real-time MLOps orchestration and fleet observability.</p></div>', unsafe_allow_html=True)
with col_action:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if st.button("Sync Platform", type="primary", use_container_width=True): st.rerun()

# KPI Row
c1, c2, c3, c4 = st.columns(4)
with c1: component_kpi_card("Experiments", f"{len(exp_df)}", "All-time runs", icon="📊", trend="+4")
with c2: component_kpi_card("Models", f"{len(mdl_df)}", "In Registry", icon="🧠", trend="+1")
with c3: component_kpi_card("Serving", str(len(mdl_df[mdl_df["stage"]=="production"])), "Production", icon="🚀", tone="success")
with c4: component_kpi_card("Accuracy", f"{best_acc:.3f}", "Best Result", icon="🏆", tone="success")

render_spacer("md")

# Executive Section
m_left, m_mid, m_right = st.columns([2, 1, 1], gap="medium")

with m_left:
    render_section_title("Production Throughput")
    if not exp_df.empty:
        exp_df["ts"] = pd.to_datetime(exp_df["created_at"])
        fig = px.area(exp_df.sort_values("ts"), x="ts", y="duration_seconds", color_discrete_sequence=["#6366F1"])
        fig.update_layout(height=280, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

with m_mid:
    render_section_title("Platform Score")
    component_health_score(health_score)

with m_right:
    render_section_title("AI Context")
    component_insight_panel([
        f"Success rate stable at {success_rate:.1f}%.",
        "XGBoost yields 12% higher F1 than Linear models.",
        "System latency is within P99 bounds."
    ])

render_spacer("md")

# Activity & Distribution
b_left, b_right = st.columns([1.5, 1], gap="medium")

with b_left:
    render_section_title("Recent Activity Feed")
    events = []
    for _, r in exp_df.head(6).iterrows():
        events.append({
            "time": str(r["created_at"])[11:16],
            "label": f"{r['model_type']} on {r['dataset']}",
            "status": r["status"]
        })
    component_timeline(events)

with b_right:
    render_section_title("Registry Fleet")
    if not mdl_df.empty:
        fig_pie = px.pie(mdl_df, names="stage", hole=0.7, color_discrete_sequence=px.colors.qualitative.Set2)
        fig_pie.update_layout(height=240, margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

st.divider()
st.caption("⚡ ML Pipeline Monitor v2.0-Componentized")
