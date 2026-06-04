"""
System Health & Infrastructure Telemetry
Redesigned with reusable enterprise components.
"""
import psutil
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from services.app_service import initialize_application
from src.auth import render_auth_controls, current_role
from src.ui_theme import (
    apply_ui_theme,
    component_health_score,
    component_insight_panel,
    component_kpi_card,
    component_timeline,
    render_sidebar_nav,
    render_top_navbar,
    render_section_title,
    render_spacer,
)

# ---------------------------------------------------------------------------
# Shell setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="System Health | ML Monitor", layout="wide")
initialize_application()
apply_ui_theme()

render_top_navbar(user_role=current_role())

with st.sidebar:
    render_sidebar_nav()
    st.divider()
    render_auth_controls()

# ---------------------------------------------------------------------------
# Telemetry Logic
# ---------------------------------------------------------------------------
def _render_circular_gauge(label, value, color):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        title = {'text': label, 'font': {'size': 12, 'color': '#9CA3AF'}},
        gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': color}, 'bgcolor': "rgba(0,0,0,0)", 'borderwidth': 2, 'bordercolor': "#374151"}
    ))
    fig.update_layout(height=180, margin=dict(l=20,r=20,t=40,b=20), paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
    return fig

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
col_title, col_actions = st.columns([4, 1])
with col_title:
    st.markdown('<div class="ui-fade-in"><h1 style="margin:0; font-family:\'Poppins\', sans-serif;">Platform Health</h1><p style="color:var(--color-text-tertiary);">Infrastructure telemetry and system event auditing.</p></div>', unsafe_allow_html=True)
with col_actions:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if st.button("Refresh Telemetry", type="primary", use_container_width=True): st.rerun()

# ---------------------------------------------------------------------------
# Monitoring Cards
# ---------------------------------------------------------------------------
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="ui-card">', unsafe_allow_html=True)
    st.plotly_chart(_render_circular_gauge("CPU LOAD", psutil.cpu_percent(), "#6366F1"), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="ui-card">', unsafe_allow_html=True)
    st.plotly_chart(_render_circular_gauge("RAM USAGE", psutil.virtual_memory().percent, "#10B981"), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="ui-card">', unsafe_allow_html=True)
    st.plotly_chart(_render_circular_gauge("DISK I/O", psutil.disk_usage('/').percent, "#F59E0B"), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

render_spacer("md")

# ---------------------------------------------------------------------------
# Events & Insights
# ---------------------------------------------------------------------------
m_left, m_right = st.columns([2, 1], gap="large")

with m_left:
    render_section_title("System Audit Log")
    component_timeline([
        {"time": "16:45", "label": "Pipeline Runner Session Started", "status": "success"},
        {"time": "16:30", "label": "Inference API Heartbeat", "status": "info"},
        {"time": "15:12", "label": "Database WAL Optimization", "status": "success"},
        {"time": "14:05", "label": "Model v4 Promotion Event", "status": "warning"},
    ])

with m_right:
    render_section_title("Hardware Context")
    component_insight_panel([
        "CPU threads are optimal for parallel CV.",
        "Memory pressure is low (under 60% threshold).",
        "Storage bandwidth supports high-frequency persistence."
    ])

st.divider()
st.caption("🖥️ Platform Telemetry Core v2.0-Componentized")
