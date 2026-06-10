"""
Model Registry & Governance
Redesigned with reusable enterprise components.
"""
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from services.app_service import initialize_application
from src.auth import has_role, render_auth_controls, require_role, current_role
from services.model_service import (
    get_rollback_hint,
    list_lineage,
    list_models,
    revert_to_previous_production,
    set_model_stage,
)
from src.ui_theme import (
    apply_ui_theme,
    component_empty_state,
    component_insight_panel,
    component_kpi_card,
    component_registry_card,
    render_loading_skeleton,
    render_sidebar_nav,
    render_top_navbar,
    render_section_title,
    render_spacer,
    safe_render,
)

# ---------------------------------------------------------------------------
# Shell setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Model Registry | ML Monitor", layout="wide")
initialize_application()
apply_ui_theme()

render_top_navbar(user_role=current_role())

with st.sidebar:
    render_sidebar_nav()
    st.divider()
    render_auth_controls()

def _render_page():
    # ---------------------------------------------------------------------------
    # Data Loading
    # ---------------------------------------------------------------------------
    @st.cache_data(ttl=20)
    def _load_registry_data():
        raw = list_models(limit=100)
        rows = []
        for m in raw:
            metrics = m.get("metrics") or {}
            if isinstance(metrics, str): metrics = json.loads(metrics)
            rows.append({
                "model_id": m["model_id"],
                "name": m["name"],
                "dataset": m["dataset"],
                "version": m.get("version", 1),
                "stage": m.get("stage", "development"),
                "accuracy": metrics.get("accuracy", 0),
                "f1_score": metrics.get("f1_score", 0),
                "registered_at": m.get("registered_at", "")
            })
        return pd.DataFrame(rows)

    loading = st.empty()
    with loading.container(): render_loading_skeleton(lines=5)
    df = _load_registry_data()
    loading.empty()

    if df.empty:
        component_empty_state("Registry Empty", "No models have been registered yet.", "Train First Model", "pages/1_Pipeline_Runner.py")
        st.stop()

    # ---------------------------------------------------------------------------
    # Header
    # ---------------------------------------------------------------------------
    col_title, col_actions = st.columns([4, 1])
    with col_title:
        st.markdown('<div class="ui-fade-in"><h1 style="margin:0; font-family:\'Poppins\', sans-serif;">Model Inventory</h1><p style="color:var(--color-text-tertiary);">Enterprise model governance and versioned lineage.</p></div>', unsafe_allow_html=True)
    with col_actions:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("Sync Registry", type="primary", use_container_width=True): st.rerun()

    # KPI Row
    c1, c2, c3, c4 = st.columns(4)
    with c1: component_kpi_card("Total Models", str(len(df)), "Versions tracked", icon="📦")
    with c2: component_kpi_card("Production", str(len(df[df["stage"]=="production"])), "Active serving", icon="🚀", tone="success")
    with c3: component_kpi_card("Staging", str(len(df[df["stage"]=="staging"])), "Release testing", icon="🧪", tone="warning")
    with c4: component_kpi_card("Promotion Rate", "4.2d", "Avg cycle time", icon="⚡", tone="info")

    render_spacer("md")

    # ---------------------------------------------------------------------------
    # Workspace
    # ---------------------------------------------------------------------------
    tab_grid, tab_lineage, tab_governance = st.tabs(["📋 Inventory Grid", "⛓️ Production Lineage", "🛡️ Compliance & Management"])

    with tab_grid:
        render_section_title("Model Cards")
        grid_cols = st.columns(3)
        for i, (_, row) in enumerate(df.sort_values("registered_at", ascending=False).iterrows()):
            with grid_cols[i % 3]:
                clicked = component_registry_card(
                    name=row["name"], version=str(row["version"]), stage=row["stage"],
                    dataset=row["dataset"], metrics={"accuracy": row["accuracy"], "f1_score": row["f1_score"]},
                    model_id=row["model_id"]
                )
                if clicked:
                    st.session_state["active_model_id"] = row["model_id"]
                    st.rerun()

    with tab_lineage:
        render_section_title("Version Evolution")
        lineage_data = list_lineage(limit=50)
        if lineage_data:
            lin_df = pd.DataFrame(lineage_data)
            fig = px.scatter(lin_df, x="version", y="dataset", color="stage", size=[15]*len(lin_df),
                             color_discrete_map={"production": "#10B981", "staging": "#F59E0B", "development": "#6366F1"})
            fig.update_layout(height=400, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Lineage data pending.")

    with tab_governance:
        selected_id = st.session_state.get("active_model_id") or (df.iloc[0]["model_id"] if not df.empty else None)
        if selected_id:
            m_row = df[df["model_id"] == selected_id].iloc[0]
            st.markdown(f'<div class="ui-card">', unsafe_allow_html=True)
            st.markdown(f"**Target:** {m_row['name']} v{m_row['version']} on {m_row['dataset']}")
            new_stage = st.selectbox("Update Stage", ["development", "staging", "production", "archived"], index=["development", "staging", "production", "archived"].index(m_row["stage"]))
            if st.button("Commit Transition", type="primary"):
                if require_role("admin", "Update Stage"):
                    set_model_stage(selected_id, new_stage)
                    st.success(f"Model promoted to {new_stage}")
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        render_section_title("Rollback Protection")
        for ds in df["dataset"].unique():
            hint = get_rollback_hint(ds)
            if hint.get("previous_production"):
                st.warning(f"Rollback candidate found for {ds}")
                if st.button(f"Revert {ds} to Previous Prod", key=f"rev_{ds}"):
                    if require_role("admin", "Revert"):
                        revert_to_previous_production(ds)
                        st.rerun()

    st.divider()
    st.caption("🛡️ Model Governance Core v2.0-Componentized")

safe_render("Model Registry", _render_page)
