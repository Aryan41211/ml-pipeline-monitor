"""
Governance & Compliance — Audit & Policy Management
"""

import json

import pandas as pd
import streamlit as st

from ml_pipeline_monitor.core.auth import current_role, render_auth_controls
from ml_pipeline_monitor.services.app_service import initialize_application
from ml_pipeline_monitor.services.governance_service import (
    alert_summary,
    list_alerts,
    list_dataset_versions,
    list_lineage_edges,
)
from ml_pipeline_monitor.services.model_service import get_stage_timeline, list_models
from ml_pipeline_monitor.utils.ui_theme import (
    apply_ui_theme,
    component_alert_card,
    component_insight_panel,
    component_kpi_card,
    hp_status_badge,
    page_header,
    render_loading_skeleton,
    render_section_title,
    render_sidebar_nav,
    render_spacer,
    render_summary_table,
    render_top_navbar,
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
        # The audit trail needs real stage transitions, which live in
        # model_stage_events -- not in the model rows themselves. Reading
        # from_stage/to_stage off a model row yielded "N/A" for every entry.
        events = []
        for model in models:
            model_id = model.get("model_id")
            if not model_id:
                continue
            for event in get_stage_timeline(str(model_id), limit=50):
                events.append(
                    {
                        "Model": model.get("name", "Unknown"),
                        "Version": model.get("version", "N/A"),
                        "Dataset": event.get("dataset") or model.get("dataset", "N/A"),
                        "From Stage": event.get("from_stage") or "-",
                        "To Stage": event.get("to_stage", "N/A"),
                        "Changed At": event.get("changed_at", "N/A"),
                        "Note": event.get("note", ""),
                    }
                )
        events.sort(key=lambda row: str(row["Changed At"]), reverse=True)
        return models, events

    loading = st.empty()
    with loading.container():
        render_loading_skeleton(lines=5)
    models_raw, audit_rows = _load_governance_data()
    loading.empty()

    # Deliberately no st.stop() here: Alert History and Data Lineage are
    # independent of the model registry, and stopping made them unreachable
    # until a model happened to be registered.
    has_models = bool(models_raw)
    models_df = pd.DataFrame(models_raw) if has_models else pd.DataFrame(columns=["stage", "name", "metrics"])

    # ---------------------------------------------------------------------------
    # Header
    # ---------------------------------------------------------------------------
    col_title, col_actions = st.columns([4, 1])
    with col_title:
        page_header("Governance & Compliance", "Model audit trails, policy enforcement, and regulatory compliance.")
    with col_actions:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("Refresh", type="primary", use_container_width=True):
            st.rerun()

    # KPI Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        component_kpi_card("Total Models", f"{len(models_df)}", "Registered")
    with c2:
        component_kpi_card(
            "Production",
            str(int((models_df["stage"] == "production").sum()) if has_models else 0),
            "Active serving",
            tone="success",
        )
    with c3:
        component_kpi_card(
            "Staging",
            str(int((models_df["stage"] == "staging").sum()) if has_models else 0),
            "Pending approval",
            tone="warning",
        )
    with c4:
        component_kpi_card(
            "Archived",
            str(int((models_df["stage"] == "archived").sum()) if has_models else 0),
            "Retired",
            tone="neutral",
        )

    render_spacer("md")

    # ---------------------------------------------------------------------------
    # Tabs
    # ---------------------------------------------------------------------------
    tab_audit, tab_alerts, tab_lineage, tab_policy, tab_compliance = st.tabs(
        ["Audit Trail", "Alert History", "Data Lineage", "Policy Enforcement", "Compliance Report"]
    )

    with tab_audit:
        render_section_title("Model Stage Change History")

        if not has_models:
            component_alert_card(
                "No models in registry yet. Train and register a model to build an audit trail.",
                tone="info",
            )
        elif audit_rows:
            render_summary_table(
                pd.DataFrame(audit_rows),
                columns=["Model", "Version", "Dataset", "From Stage", "To Stage", "Changed At", "Note"],
                filterable_columns=["Model", "Dataset", "To Stage"],
            )
        else:
            st.info("No stage change events recorded yet.")

    with tab_alerts:
        render_section_title("Alert History")
        alerts = list_alerts(limit=200)
        if not alerts:
            st.info(
                "No alerts recorded yet. Alerts are written when a pipeline run fails "
                "or a drift scan crosses its threshold."
            )
        else:
            counts = alert_summary(alerts)
            a1, a2, a3 = st.columns(3)
            with a1:
                component_kpi_card("Critical", str(counts.get("critical", 0)), "Highest severity", tone="danger")
            with a2:
                component_kpi_card("Warning", str(counts.get("warning", 0)), "Needs attention", tone="warning")
            with a3:
                component_kpi_card("Informational", str(counts.get("info", 0)), "Recorded", tone="neutral")
            render_spacer("sm")
            render_summary_table(
                pd.DataFrame(
                    [
                        {
                            "Created At": a.get("created_at", ""),
                            "Severity": a.get("severity", "info"),
                            "Type": a.get("alert_type", ""),
                            "Message": a.get("message", ""),
                        }
                        for a in alerts
                    ]
                ),
                columns=["Created At", "Severity", "Type", "Message"],
                filterable_columns=["Severity", "Type"],
            )

    with tab_lineage:
        render_section_title("Dataset to Model Lineage")
        edges = list_lineage_edges(limit=200)
        if not edges:
            st.info("No lineage recorded yet. It is written on each pipeline run.")
        else:
            render_summary_table(
                pd.DataFrame(
                    [
                        {
                            "Created At": e.get("created_at", ""),
                            "Dataset": e.get("from_dataset_id", "—"),
                            "Version": e.get("from_version", "—"),
                            "Edge": e.get("edge_type", ""),
                            "Model": e.get("to_model_id", "—"),
                            "Note": e.get("note", ""),
                        }
                        for e in edges
                    ]
                ),
                columns=["Created At", "Dataset", "Version", "Edge", "Model", "Note"],
                filterable_columns=["Dataset", "Edge"],
            )

            datasets = sorted({str(e.get("from_dataset_id")) for e in edges if e.get("from_dataset_id")})
            if datasets:
                render_spacer("sm")
                render_section_title("Version History")
                chosen = st.selectbox("Dataset", datasets)
                versions = list_dataset_versions(chosen, limit=50)
                if versions:
                    render_summary_table(
                        pd.DataFrame(
                            [
                                {
                                    "Version": v.get("version"),
                                    "Rows": v.get("row_count"),
                                    "Columns": v.get("column_count"),
                                    "Checksum": str(v.get("hash", ""))[:16],
                                    "Created At": v.get("created_at", ""),
                                }
                                for v in versions
                            ]
                        ),
                        columns=["Version", "Rows", "Columns", "Checksum", "Created At"],
                    )
                else:
                    st.caption("No versions recorded for this dataset.")

    with tab_policy:
        render_section_title("Promotion Policies")

        st.markdown("**Current Promotion Rules**")
        st.markdown(
            """- **Development  Staging**: Requires passing all pipeline stages (CV, evaluation, feature importance)
        - **Staging  Production**: Requires admin approval + performance benchmark vs current production
        - **Production  Archived**: Automatic when new model promoted to production
        - **Rollback**: Admin-only, promotes previous production model"""
        )

        render_spacer("md")
        render_section_title("Configure Policy Thresholds")

        col1, col2 = st.columns(2)
        with col1:
            min_accuracy = st.number_input(
                "Minimum Accuracy (Classification)",
                0.0,
                1.0,
                float(st.session_state.get("policy_min_accuracy", 0.80)),
                0.01,
            )
            min_f1 = st.number_input(
                "Minimum F1 Score",
                0.0,
                1.0,
                float(st.session_state.get("policy_min_f1", 0.75)),
                0.01,
            )
        with col2:
            st.number_input("Maximum PSI for Production", 0.0, 1.0, 0.10, 0.01, key="policy_max_psi")
            st.checkbox("Require Admin Approval for Production", value=True, key="policy_require_approval")

        if st.button("Apply Thresholds", type="primary"):
            st.session_state["policy_min_accuracy"] = min_accuracy
            st.session_state["policy_min_f1"] = min_f1
            st.success(
                "Thresholds applied to the compliance report below for this session. "
                "Edit config/config.prod.yaml to make them permanent."
            )

    with tab_compliance:
        render_section_title("Compliance Status")

        # Check each production model
        prod_models = (
            models_df[models_df["stage"] == "production"] if has_models else pd.DataFrame(columns=models_df.columns)
        )
        compliance_rows = []

        for _, row in prod_models.iterrows():
            metrics = row.get("metrics", {})
            if isinstance(metrics, str):
                metrics = json.loads(metrics)

            accuracy = metrics.get("accuracy", 0)
            f1 = metrics.get("f1_score", 0)

            compliant = accuracy >= min_accuracy and f1 >= min_f1
            compliance_rows.append(
                {
                    "Model": row.get("name", "Unknown"),
                    "Version": row.get("version", "N/A"),
                    "Dataset": row.get("dataset", "N/A"),
                    "Accuracy": f"{accuracy:.4f}",
                    "F1 Score": f"{f1:.4f}",
                    "Status": hp_status_badge("compliant" if compliant else "non_compliant"),
                }
            )

        if compliance_rows:
            comp_df = pd.DataFrame(compliance_rows)
            render_summary_table(
                comp_df,
                columns=["Model", "Version", "Dataset", "Accuracy", "F1 Score", "Status"],
                filterable_columns=["Model", "Dataset"],
            )
        else:
            st.info("No production models to audit.")

    render_spacer("md")
    non_compliant = sum(1 for row in compliance_rows if "non_compliant" in str(row["Status"]))
    component_insight_panel(
        [
            f"{len(prod_models)} production model(s) checked against the current thresholds.",
            (
                f"{non_compliant} model(s) below the accuracy/F1 policy."
                if non_compliant
                else "All production models meet the current accuracy/F1 policy."
            ),
            f"{len(audit_rows)} stage transition(s) recorded in the audit trail.",
        ]
    )

    st.divider()


safe_render("Governance", _render_page)
