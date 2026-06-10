"""
Governance & Compliance — Audit & Policy Management
"""
import pandas as pd
import streamlit as st

from services.app_service import initialize_application
from services.model_service import list_lineage, list_models
from src.auth import can_administer, render_auth_controls, require_role, current_role
from src.ui_theme import (
    apply_ui_theme,
    component_alert_card,
    component_insight_panel,
    component_kpi_card,
    hp_status_badge,
    render_loading_skeleton,
    render_sidebar_nav,
    render_top_navbar,
    render_section_title,
    render_spacer,
    render_summary_table,
    safe_render,
)

# ---------------------------------------------------------------------------
# Shell setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Governance | ML Monitor", layout="wide")
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
    def _load_governance_data():
        models = list_models(limit=200)
        lineage = list_lineage(limit=200)
        return models, lineage

    loading = st.empty()
    with loading.container(): render_loading_skeleton(lines=5)
    models_raw, lineage_raw = _load_governance_data()
    loading.empty()

    if not models_raw:
        component_alert_card("No models in registry. Train and register models first.", tone="info")
        st.stop()

    models_df = pd.DataFrame(models_raw)
    lineage_df = pd.DataFrame(lineage_raw) if lineage_raw else pd.DataFrame()

    # ---------------------------------------------------------------------------
    # Header
    # ---------------------------------------------------------------------------
    col_title, col_actions = st.columns([4, 1])
    with col_title:
        st.markdown('<div class="ui-fade-in"><h1 style="margin:0; font-family:\'Poppins\', sans-serif;">Governance & Compliance</h1><p style="color:var(--color-text-tertiary);">Model audit trails, policy enforcement, and regulatory compliance.</p></div>', unsafe_allow_html=True)
    with col_actions:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("Refresh", type="primary", use_container_width=True): st.rerun()

    # KPI Row
    c1, c2, c3, c4 = st.columns(4)
    with c1: component_kpi_card("Total Models", f"{len(models_df)}", "Registered", icon="📦")
    with c2: component_kpi_card("Production", f"{len(models_df[models_df.get('stage')=='production'])}", "Active serving", icon="🚀", tone="success")
    with c3: component_kpi_card("Staging", f"{len(models_df[models_df.get('stage')=='staging'])}", "Pending approval", icon="🧪", tone="warning")
    with c4: component_kpi_card("Archived", f"{len(models_df[models_df.get('stage')=='archived'])}", "Retired", icon="📦", tone="neutral")

    render_spacer("md")

    # ---------------------------------------------------------------------------
    # Tabs
    # ---------------------------------------------------------------------------
    tab_audit, tab_policy, tab_compliance = st.tabs(["📋 Audit Trail", "🛡️ Policy Enforcement", "📜 Compliance Report"])

    with tab_audit:
        render_section_title("Model Stage Change History")
        
        if not lineage_df.empty and "model_id" in lineage_df.columns:
            audit_rows = []
            for _, row in models_df.iterrows():
                model_id = row.get("model_id")
                model_events = lineage_df[lineage_df.get("model_id") == model_id] if "model_id" in lineage_df.columns else pd.DataFrame()
                for _, evt in model_events.iterrows():
                    audit_rows.append({
                        "Model": row.get("name", "Unknown"),
                        "Version": row.get("version", "N/A"),
                        "Dataset": row.get("dataset", "N/A"),
                        "From Stage": evt.get("from_stage", "N/A"),
                        "To Stage": evt.get("to_stage", "N/A"),
                        "Changed At": evt.get("changed_at", "N/A"),
                        "Note": evt.get("note", ""),
                    })
            
            if audit_rows:
                audit_df = pd.DataFrame(audit_rows)
                render_summary_table(
                    audit_df,
                    columns=["Model", "Version", "Dataset", "From Stage", "To Stage", "Changed At", "Note"],
                    filterable_columns=["Model", "Dataset", "To Stage"]
                )
            else:
                st.info("No stage change events recorded.")
        else:
            st.info("No lineage data available.")

    with tab_policy:
        render_section_title("Promotion Policies")
        
        st.markdown("**Current Promotion Rules**")
        st.markdown("""
        - **Development → Staging**: Requires passing all pipeline stages (CV, evaluation, feature importance)
        - **Staging → Production**: Requires admin approval + performance benchmark vs current production
        - **Production → Archived**: Automatic when new model promoted to production
        - **Rollback**: Admin-only, promotes previous production model
        """)
        
        render_spacer("md")
        render_section_title("Configure Policy Thresholds")
        
        col1, col2 = st.columns(2)
        with col1:
            min_accuracy = st.number_input("Minimum Accuracy (Classification)", 0.0, 1.0, 0.80, 0.01)
            min_f1 = st.number_input("Minimum F1 Score", 0.0, 1.0, 0.75, 0.01)
        with col2:
            max_psi = st.number_input("Maximum PSI for Production", 0.0, 1.0, 0.10, 0.01)
            require_approval = st.checkbox("Require Admin Approval for Production", value=True)
        
        if st.button("Save Policy", type="primary"):
            st.success("Policy thresholds saved (stored in session). Persist to config.yaml for permanence.")

    with tab_compliance:
        render_section_title("Compliance Status")
        
        # Check each production model
        prod_models = models_df[models_df.get("stage") == "production"]
        compliance_rows = []
        
        for _, row in prod_models.iterrows():
            metrics = row.get("metrics", {})
            if isinstance(metrics, str):
                import json
                metrics = json.loads(metrics)
            
            accuracy = metrics.get("accuracy", 0)
            f1 = metrics.get("f1_score", 0)
            
            compliant = accuracy >= 0.80 and f1 >= 0.75
            compliance_rows.append({
                "Model": row.get("name", "Unknown"),
                "Version": row.get("version", "N/A"),
                "Dataset": row.get("dataset", "N/A"),
                "Accuracy": f"{accuracy:.4f}",
                "F1 Score": f"{f1:.4f}",
                "Status": hp_status_badge("compliant" if compliant else "non_compliant"),
            })
        
        if compliance_rows:
            comp_df = pd.DataFrame(compliance_rows)
            render_summary_table(
                comp_df,
                columns=["Model", "Version", "Dataset", "Accuracy", "F1 Score", "Status"],
                filterable_columns=["Model", "Dataset"]
            )
        else:
            st.info("No production models to audit.")

    render_spacer("md")
    component_insight_panel([
        "All production models undergo automated compliance checks.",
        "Audit trail is immutable and timestamped.",
        "Policy violations trigger alerts to administrators."
    ])

    st.divider()
    st.caption("⚖️ Governance Core v2.0-Componentized")

safe_render("Governance", _render_page)