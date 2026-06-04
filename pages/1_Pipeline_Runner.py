"""
Pipeline Runner — Visual Workflow Orchestration
Redesigned with reusable enterprise components.
"""
import time
import pandas as pd
import plotly.express as px
import streamlit as st

from services.app_service import initialize_application
from services.pipeline_service import (
    get_dataset_options,
    get_dataset_preview,
    get_task_and_model_options,
    run_pipeline_and_persist,
)
from src.auth import can_run_pipeline, render_auth_controls, current_role
from src.ui_theme import (
    apply_ui_theme,
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
st.set_page_config(page_title="Pipeline Runner | ML Monitor", layout="wide")
initialize_application()
apply_ui_theme()

render_top_navbar(user_role=current_role())

with st.sidebar:
    render_sidebar_nav()
    st.divider()
    render_auth_controls()

DATASET_OPTIONS = get_dataset_options()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
col_title, col_actions = st.columns([4, 1])
with col_title:
    st.markdown('<div class="ui-fade-in"><h1 style="margin:0; font-family:\'Poppins\', sans-serif;">Workflow Orchestrator</h1><p style="color:var(--color-text-tertiary);">Live stage tracking and hyperparameter optimization.</p></div>', unsafe_allow_html=True)
with col_actions:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    run_btn = st.button("Execute Pipeline", type="primary", use_container_width=True, disabled=not can_run_pipeline())

# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------
tab_exec, tab_config = st.tabs(["⚡ Live Execution", "⚙️ Architecture Config"])

with tab_config:
    c1, c2 = st.columns([2, 1], gap="large")
    with c1:
        render_section_title("Hardware & Estimator")
        st.markdown('<div class="ui-card">', unsafe_allow_html=True)
        ds_label = st.selectbox("Target Dataset", list(DATASET_OPTIONS.keys()))
        ds_key = DATASET_OPTIONS[ds_label]
        
        task_meta = get_task_and_model_options(ds_key)
        task = task_meta["task"]
        model = st.selectbox("Algorithm", task_meta["model_options"])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with c2:
        render_section_title("Execution Strategy")
        st.markdown('<div class="ui-card">', unsafe_allow_html=True)
        test_size = st.slider("Validation Split", 0.1, 0.4, 0.2)
        cv_folds = st.slider("Cross-Validation Folds", 2, 10, 5)
        st.markdown('</div>', unsafe_allow_html=True)
        
        component_insight_panel([
            f"Orchestrating {task} pipeline.",
            f"Using {cv_folds}-fold Stratified CV.",
            "StandardScaler applied automatically."
        ])

with tab_exec:
    if run_btn:
        prog = st.progress(0.0)
        status_box = st.empty()
        timeline_box = st.empty()
        
        stages = []
        def _cb(stage, progress, msg):
            prog.progress(progress)
            status_box.markdown(f"**Current:** {stage} — {msg}")
            # Update virtual timeline
            stages.append({"time": time.strftime("%H:%M:%S"), "label": stage, "status": "success"})
            with timeline_box.container(): component_timeline(stages[-10:])

        try:
            payload = run_pipeline_and_persist(
                dataset_label=ds_label, dataset_key=ds_key, model_type=model, task=task,
                params={}, test_size=test_size, cv_folds=cv_folds, random_state=42,
                progress_callback=_cb
            )
            res = payload["result"]
            st.success(f"Run {res.run_id} finished in {res.duration:.2f}s")
            st.session_state["last_res"] = res
        except Exception as e:
            st.error(f"Execution Failed: {e}")

    # Results
    last_res = st.session_state.get("last_res")
    if last_res:
        render_spacer("md")
        render_section_title(f"Analysis: {last_res.run_id}")
        r1, r2, r3, r4 = st.columns(4)
        for i, (k, v) in enumerate(list(last_res.metrics.items())[:4]):
            with [r1, r2, r3, r4][i]:
                component_kpi_card(k.title(), f"{v:.4f}", "Primary metric", tone="success")

st.divider()
st.caption("⚡ Workflow Engine v2.0-Componentized")
