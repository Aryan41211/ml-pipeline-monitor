"""
Data Drift Observability
Redesigned with reusable enterprise components.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from services.app_service import initialize_application
from services.drift_service import (
    get_dataset_options,
    get_drift_preview_dataset,
    get_monitoring_defaults,
    list_drift_reports,
    run_drift_and_persist,
)
from src.auth import can_run_pipeline, render_auth_controls, current_role
from src.ui_theme import (
    apply_ui_theme,
    component_alert_card,
    component_health_score,
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
st.set_page_config(page_title="Data Drift | ML Monitor", layout="wide")
initialize_application()
apply_ui_theme()

render_top_navbar(user_role=current_role())

with st.sidebar:
    render_sidebar_nav()
    st.divider()
    render_auth_controls()

# ---------------------------------------------------------------------------
# Logic & Execution
# ---------------------------------------------------------------------------
DATASET_OPTIONS = get_dataset_options()

col_title, col_actions = st.columns([4, 1])
with col_title:
    st.markdown('<div class="ui-fade-in"><h1 style="margin:0; font-family:\'Poppins\', sans-serif;">Data Observability</h1><p style="color:var(--color-text-tertiary);">Distribution shift analysis using Kolmogorov-Smirnov and PSI.</p></div>', unsafe_allow_html=True)
with col_actions:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    dataset_label = st.selectbox("Target Dataset", list(DATASET_OPTIONS.keys()))
    dataset_key = DATASET_OPTIONS[dataset_label]

c1, c2 = st.columns([1, 2], gap="large")

with c1:
    render_section_title("Perturbation Settings")
    st.markdown('<div class="ui-card">', unsafe_allow_html=True)
    noise = st.slider("Signal Noise", 0.0, 2.0, 0.5)
    shift = st.slider("Mean Offset", 0.0, 1.0, 0.0)
    alpha = st.select_slider("Confidence (α)", [0.01, 0.05, 0.10], 0.05)
    
    if st.button("Run Drift Scan", type="primary", use_container_width=True, disabled=not can_run_pipeline()):
        with st.spinner("Analyzing distributions..."):
            payload = run_drift_and_persist(dataset_label, dataset_key, noise, shift, alpha)
            st.session_state["active_drift"] = payload["report"]
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    report = st.session_state.get("active_drift")
    if report:
        render_section_title("Overall Distribution Health")
        score = int(100 - (min(report["average_psi"] * 2, 1) * 100))
        component_health_score(score, label="STABLE" if score > 75 else "DRIFTED")
    else:
        component_alert_card("No active report. Configure and run a drift scan to begin monitoring.", tone="info")

# ---------------------------------------------------------------------------
# Statistical Deep Dive
# ---------------------------------------------------------------------------
if report:
    render_spacer("md")
    # KPI Row
    k1, k2, k3, k4 = st.columns(4)
    with k1: component_kpi_card("Analyzed", str(report["features_analyzed"]), "Dimensions", icon="🔢")
    with k2: component_kpi_card("Drifted", str(report["features_drifted"]), "Flagged", icon="⚠️", tone="danger" if report["features_drifted"] > 0 else "success")
    with k3: component_kpi_card("Avg PSI", f"{report['average_psi']:.4f}", "Stability Index", icon="📊", tone="warning" if report["average_psi"] > 0.1 else "success")
    with k4: component_kpi_card("Severity", report["overall_severity"].upper(), "Platform Risk", icon="⚖️", tone="danger" if report["overall_severity"]=="critical" else "success")
    
    render_spacer("md")
    t_feat, t_hist = st.tabs(["🧬 Feature Stability", "📜 Analysis History"])
    
    with t_feat:
        col_list, col_insights = st.columns([2, 1], gap="medium")
        with col_list:
            feat_df = pd.DataFrame(report["feature_results"])
            feat_df.columns = [c.replace('_',' ').title() for c in feat_df.columns]
            render_summary_table(feat_df, columns=["Feature", "Severity", "Psi", "P Value"], sort_by="Psi")
        with col_insights:
            component_insight_panel([
                f"PSI > 0.25 detected in {len(feat_df[feat_df['Severity']=='significant'])} features.",
                "Retraining is recommended to align with distribution shift.",
                "KS-tests confirm shape-level divergence."
            ])

    with t_hist:
        history = list_drift_reports(limit=15)
        if history:
            h_df = pd.DataFrame(history)
            h_df.columns = [c.replace('_',' ').title() for c in h_df.columns]
            render_summary_table(h_df, columns=["Dataset", "Features Drifted", "Drift Score", "Created At"])

st.divider()
st.caption("📈 Statistical Observability Core v2.0-Componentized")
