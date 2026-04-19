"""
Model Registry page.

Lists all trained models stored in the local registry, supports promoting
models through development â†’ staging â†’ production lifecycle stages, and
renders side-by-side performance comparisons.
"""
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from services.app_service import initialize_application
from src.auth import has_role, render_auth_controls, require_role
from services.model_service import (
    get_rollback_hint,
    get_stage_timeline,
    list_lineage,
    list_models,
    revert_to_previous_production,
    set_model_stage,
)
from services.telemetry_service import track_user_action
from src.ui_theme import (
    apply_plotly_layout,
    apply_ui_theme,
    render_empty_data_explainer,
    render_expandable_rows,
    render_kpi_card,
    render_loading_skeleton,
    render_page_header,
    render_page_header_with_action,
    render_section_header,
    render_spacer,
    render_status_badge,
    render_section_title,
    render_sidebar_brand,
    render_sidebar_nav,
    render_summary_table,
    stage_badge_html,
)

st.set_page_config(page_title="Model Registry | ML Monitor", layout="wide")
initialize_application()
apply_ui_theme()
refresh_clicked = render_page_header_with_action(
    "Model registry",
    "Lifecycle stage management: development, staging, production.",
    "Refresh",
    action_key="registry_refresh",
)
if refresh_clicked:
    track_user_action("model_registry", "refresh")
    st.rerun()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
sk = st.empty()
with sk.container():
    render_loading_skeleton(lines=2, key="registry_load")


@st.cache_data(ttl=20, show_spinner=False)
def _cached_models(limit: int):
    return list_models(limit=limit)


try:
    raw = _cached_models(limit=100)
except Exception as exc:
    sk.empty()
    st.error("Unable to load model registry right now.")
    st.caption(f"Details: {exc}")
    if st.button("Retry", type="primary", key="registry_retry"):
        st.rerun()
    st.stop()
sk.empty()


def _parse(val):
    if isinstance(val, dict):
        return val
    try:
        return json.loads(val) if val else {}
    except Exception:
        return {}


rows = []
for m in raw:
    metrics = _parse(m.get("metrics", {}))
    rows.append(
        {
            "model_id":   m["model_id"],
            "run_id":     m["run_id"],
            "name":       m["name"],
            "dataset":    m["dataset"],
            "task":       m.get("task", "classification"),
            "stage":      m.get("stage", "development"),
            "version":    m.get("version", 1),
            "registered_at": m.get("registered_at", ""),
            # classification
            "accuracy":   metrics.get("accuracy"),
            "f1_score":   metrics.get("f1_score"),
            "roc_auc":    metrics.get("roc_auc"),
            # regression
            "rmse":       metrics.get("rmse"),
            "r2":         metrics.get("r2"),
            "cv_mean":    metrics.get("cv_mean"),
        }
    )

df = pd.DataFrame(rows)


@st.cache_data(ttl=20, show_spinner=False)
def _cached_timeline(model_id: str):
    return get_stage_timeline(model_id=model_id, limit=20)


def _apply_stage_change(model_id: str, stage: str, *, context: str = "manual_update") -> None:
    if not require_role("admin", "Update Model Stage"):
        raise PermissionError("Admin role required for model lifecycle changes.")
    track_user_action("model_registry", context, {"model_id": model_id, "new_stage": stage})
    set_model_stage(model_id, stage)

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
with st.sidebar:
    render_sidebar_brand()
    st.markdown("### Navigation")
    render_sidebar_nav()
    st.divider()

    st.markdown("### Configuration")
    st.divider()

    stage_opts = ["All"] + (sorted(df["stage"].unique().tolist()) if not df.empty else [])
    task_opts = ["All"] + (sorted(df["task"].unique().tolist()) if not df.empty else [])
    ds_opts = ["All"] + (sorted(df["dataset"].unique().tolist()) if not df.empty else [])

    sel_stage = st.selectbox("Stage",   stage_opts)
    sel_task  = st.selectbox("Task",    task_opts)
    sel_ds    = st.selectbox("Dataset", ds_opts)

    st.divider()
    st.markdown("### Access")
    render_auth_controls()

filtered = df.copy()
if sel_stage != "All":
    filtered = filtered[filtered["stage"] == sel_stage]
if sel_task != "All":
    filtered = filtered[filtered["task"] == sel_task]
if sel_ds != "All":
    filtered = filtered[filtered["dataset"] == sel_ds]

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
clf_f = filtered[filtered["task"] == "classification"]
reg_f = filtered[filtered["task"] == "regression"]

k1, k2, k3, k4 = st.columns(4)
with k1:
    render_kpi_card("Total Models", str(len(filtered)), subtitle="models in current filter", tone="info", icon="&#128421;")
with k2:
    render_kpi_card(
        "In Production",
        str(int((filtered["stage"] == "production").sum())),
        subtitle="active stage",
        tone="success",
        icon="&#128994;",
    )
with k3:
    render_kpi_card(
        "Best Accuracy",
        f"{clf_f['accuracy'].max():.4f}" if not clf_f.empty and clf_f["accuracy"].notna().any() else "â€”",
        subtitle="classification",
        tone="neutral",
        icon="&#127942;",
    )
with k4:
    render_kpi_card(
        "Best RÂ²",
        f"{reg_f['r2'].max():.4f}" if not reg_f.empty and reg_f["r2"].notna().any() else "â€”",
        subtitle="regression",
        tone="neutral",
        icon="&#128200;",
    )

render_spacer("lg")

if filtered.empty:
    render_empty_data_explainer(
        "No models are available in this registry view yet.",
        "Train and persist at least one pipeline run, or loosen active filters.",
        "The model cards, stage controls, and lineage graph will appear automatically.",
    )
    cta_left, cta_right = st.columns(2)
    with cta_left:
        st.page_link("pages/1_Pipeline_Runner.py", label="Run first pipeline")
    with cta_right:
        st.page_link("pages/2_Experiment_Tracking.py", label="Open experiment tracking")
    st.stop()

if "registry_pending_promotion" not in st.session_state:
    st.session_state["registry_pending_promotion"] = None

pending = st.session_state.get("registry_pending_promotion")
if pending:
    def _confirm_panel() -> None:
        st.warning(
            f"You are promoting model {pending['model_id']} to production for dataset {pending['dataset']}."
        )
        if pending.get("replace_model_id"):
            st.error(
                "This will replace the current production model "
                f"{pending['replace_model_id']} (v{pending['replace_version']})."
            )
        left, right = st.columns(2)
        with left:
            if st.button("Confirm Promotion", type="primary", key="confirm_stage_promotion"):
                try:
                    _apply_stage_change(pending["model_id"], "production", context="promote_confirmed")
                    st.session_state["registry_pending_promotion"] = None
                    st.success("Model promoted to production.")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
        with right:
            if st.button("Cancel", key="cancel_stage_promotion"):
                st.session_state["registry_pending_promotion"] = None
                st.rerun()

    if hasattr(st, "dialog"):
        @st.dialog("Confirm Production Promotion")
        def _promotion_dialog() -> None:
            _confirm_panel()

        _promotion_dialog()
    else:
        _confirm_panel()

st.markdown(
        """
        <style>
            .mr-grid-card {
                border: 1px solid #e6e9ef;
                border-radius: 14px;
                background: linear-gradient(180deg, #ffffff 0%, #fbfcfe 100%);
                box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
                padding: 14px;
                min-height: 182px;
                transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
            }
            .mr-grid-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 24px rgba(15, 23, 42, 0.12);
                border-color: #d8dfeb;
            }
            .mr-grid-card.selected {
                border-color: #2b6de9;
                box-shadow: 0 0 0 2px rgba(43, 109, 233, 0.22), 0 12px 24px rgba(43, 109, 233, 0.18);
            }
            .mr-card-head { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
            .mr-card-title { font-size: 1rem; font-weight: 700; color: #0f172a; line-height: 1.25; }
            .mr-stage {
                padding: 3px 10px;
                border-radius: 999px;
                font-size: 0.72rem;
                font-weight: 700;
                text-transform: capitalize;
            }
            .mr-stage.production { background: #e7f7ee; color: #166534; border: 1px solid #86efac; }
            .mr-stage.staging { background: #e8f0ff; color: #1d4ed8; border: 1px solid #93c5fd; }
            .mr-stage.development { background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; }
            .mr-stage.archived { background: #fff7ed; color: #9a3412; border: 1px solid #fdba74; }
            .mr-meta { margin-top: 8px; font-size: 0.78rem; color: #475569; }
            .mr-metrics { margin-top: 12px; display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
            .mr-metric { border: 1px solid #e8edf5; border-radius: 10px; padding: 8px; background: #ffffff; }
            .mr-metric-label { font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.03em; }
            .mr-metric-value { font-size: 0.92rem; font-weight: 700; color: #0f172a; }
            .mr-runid { margin-top: 10px; font-size: 0.72rem; color: #64748b; }
        </style>
        """,
        unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
tab_cards, tab_compare, tab_manage, tab_lineage = st.tabs(
    ["Model Cards", "Performance Comparison", "Stage Management", "Model Lineage"]
)

with tab_cards:
    if filtered.empty:
        st.info("No models match the current filters.")
    else:
        if "registry_selected_model" not in st.session_state:
            st.session_state["registry_selected_model"] = ""

        # Sort: production first, then staging, then development
        order = {"production": 0, "staging": 1, "development": 2, "archived": 3}
        sorted_df = filtered.copy()
        sorted_df["_order"] = sorted_df["stage"].map(order).fillna(9)
        sorted_df = sorted_df.sort_values(["_order", "registered_at"], ascending=[True, False]).drop(columns="_order")

        stage_counts = sorted_df["stage"].value_counts().reindex(["production", "staging", "development", "archived"], fill_value=0)
        chart_col, info_col = st.columns([1.3, 1], gap="large")

        with chart_col:
            render_section_title("Stage Distribution")
            stage_df = stage_counts.reset_index()
            stage_df.columns = ["stage", "count"]
            fig_stage = px.pie(
                stage_df[stage_df["count"] > 0],
                names="stage",
                values="count",
                hole=0.62,
                color="stage",
                color_discrete_map={
                    "production": "#16a34a",
                    "staging": "#2563eb",
                    "development": "#64748b",
                    "archived": "#c2410c",
                },
            )
            fig_stage.update_traces(
                textinfo="percent",
                textfont_size=11,
                marker=dict(line=dict(width=0.8, color="#ffffff")),
                sort=False,
            )
            fig_stage.update_layout(
                height=260,
                margin=dict(l=0, r=0, t=8, b=0),
                paper_bgcolor="white",
                plot_bgcolor="white",
                legend=dict(title=None, orientation="h", yanchor="bottom", y=-0.15, x=0),
            )
            st.plotly_chart(fig_stage, width="stretch")

        with info_col:
            render_section_title("Lifecycle Snapshot")
            st.caption(
                f"Production: {int(stage_counts.get('production', 0))}   |   "
                f"Staging: {int(stage_counts.get('staging', 0))}   |   "
                f"Development: {int(stage_counts.get('development', 0))}"
            )
            st.caption("Use Select to focus a card and Promote to push a model to production.")
            if st.session_state["registry_selected_model"]:
                st.info(f"Selected model: {st.session_state['registry_selected_model']}")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        by_dataset_production = (
            df[df["stage"] == "production"].sort_values("registered_at", ascending=False).drop_duplicates("dataset")
        )
        production_lookup = {
            r["dataset"]: {"model_id": r["model_id"], "version": r.get("version", "â€”")}
            for _, r in by_dataset_production.iterrows()
        }

        grid_cols = 3
        rows = [sorted_df.iloc[i:i + grid_cols] for i in range(0, len(sorted_df), grid_cols)]

        for block in rows:
            cols = st.columns(grid_cols, gap="large")
            for col_idx in range(grid_cols):
                with cols[col_idx]:
                    if col_idx >= len(block):
                        continue

                    row = block.iloc[col_idx]
                    selected = st.session_state["registry_selected_model"] == row["model_id"]
                    card_classes = "mr-grid-card selected" if selected else "mr-grid-card"
                    stage = str(row.get("stage", "development"))

                    acc_txt = f"{row['accuracy']:.4f}" if pd.notna(row.get("accuracy")) else "â€”"
                    roc_txt = f"{row['roc_auc']:.4f}" if pd.notna(row.get("roc_auc")) else "â€”"

                    st.markdown(
                        f"""
                        <div class='{card_classes}'>
                          <div class='mr-card-head'>
                            <div class='mr-card-title'>{row['name']}</div>
                            <span class='mr-stage {stage}'>{stage}</span>
                          </div>
                          <div class='mr-meta'>Dataset: <b>{row['dataset']}</b>  |  Version: <b>v{row['version']}</b></div>
                          <div class='mr-metrics'>
                            <div class='mr-metric'>
                              <div class='mr-metric-label'>Accuracy</div>
                              <div class='mr-metric-value'>{acc_txt}</div>
                            </div>
                            <div class='mr-metric'>
                              <div class='mr-metric-label'>ROC AUC</div>
                              <div class='mr-metric-value'>{roc_txt}</div>
                            </div>
                          </div>
                          <div class='mr-runid'>Run ID: {row['run_id']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    a, b = st.columns(2)
                    with a:
                        if st.button("Select", key=f"card_select_{row['model_id']}", width="stretch"):
                            st.session_state["registry_selected_model"] = row["model_id"]
                            st.rerun()
                    with b:
                        can_promote = bool(has_role("admin") and row.get("stage") != "production")
                        if st.button(
                            "Promote",
                            key=f"card_promote_{row['model_id']}",
                            type="primary",
                            disabled=not can_promote,
                            width="stretch",
                        ):
                            replacing = production_lookup.get(row["dataset"])
                            replacing_model_id = None
                            replacing_version = None
                            if replacing and replacing["model_id"] != row["model_id"]:
                                replacing_model_id = replacing["model_id"]
                                replacing_version = replacing["version"]

                            st.session_state["registry_pending_promotion"] = {
                                "model_id": row["model_id"],
                                "dataset": row["dataset"],
                                "replace_model_id": replacing_model_id,
                                "replace_version": replacing_version,
                            }
                            st.rerun()

                    if row.get("stage") == "production":
                        st.caption("Already in production")
                    elif not has_role("admin"):
                        render_status_badge("admin required", "warning")

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)


with tab_compare:
    render_section_header("Performance by Algorithm and Dataset", "Consistent visual hierarchy from shared theme helpers")

    clf_plot = filtered[filtered["task"] == "classification"].dropna(subset=["accuracy"])
    reg_plot = filtered[filtered["task"] == "regression"].dropna(subset=["r2"])

    if not clf_plot.empty:
        fig = px.bar(
            clf_plot.sort_values("accuracy", ascending=False),
            x="model_id",
            y="accuracy",
            color="name",
            text="accuracy",
            barmode="group",
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={"model_id": "Run ID", "accuracy": "Accuracy", "name": "Algorithm"},
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_layout(
            height=380,
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(showgrid=False, title="Run ID"),
            yaxis=dict(showgrid=True, gridcolor="#f1f5f9", range=[0, 1.1], title="Accuracy"),
            showlegend=True,
            legend=dict(title="Algorithm"),
        )
        st.plotly_chart(fig, width="stretch")

    if not reg_plot.empty:
        fig2 = px.bar(
            reg_plot.sort_values("r2", ascending=False),
            x="model_id",
            y="r2",
            color="name",
            text="r2",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            labels={"model_id": "Run ID", "r2": "RÂ²", "name": "Algorithm"},
        )
        fig2.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig2.update_layout(
            height=360,
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#f1f5f9", title="RÂ²"),
            showlegend=True,
        )
        st.plotly_chart(fig2, width="stretch")

    if clf_plot.empty and reg_plot.empty:
        st.info("Not enough data for comparison with current filters.")


with tab_manage:
    render_section_title("Promote or Archive Models")
    st.caption(
        "Changing a model's stage updates the registry record. "
        "Only one model per dataset should be promoted to Production."
    )

    with st.expander("Rollback hints", expanded=True):
        if not has_role("admin"):
            st.info("Read-only mode: admin role is required for rollback or stage updates.")
        for dataset_name in sorted(filtered["dataset"].dropna().unique().tolist()):
            hint = get_rollback_hint(dataset=dataset_name)
            current_prod = hint.get("current_production")
            previous_prod = hint.get("previous_production")

            h_left, h_mid, h_right = st.columns([3, 3, 2])
            with h_left:
                if current_prod:
                    st.markdown(
                        f"**{dataset_name}** current production: "
                        f"`{current_prod.get('model_id')}` (v{current_prod.get('version')})"
                    )
                else:
                    st.markdown(f"**{dataset_name}** current production: none")

            with h_mid:
                if previous_prod:
                    st.caption(
                        "Previous production: "
                        f"{previous_prod.get('model_id')} (v{previous_prod.get('version')})"
                    )
                else:
                    st.caption("Previous production: unavailable")

            with h_right:
                revert_disabled = previous_prod is None
                if st.button(
                    "Revert",
                    key=f"revert_prod_{dataset_name}",
                    disabled=(revert_disabled or not has_role("admin")),
                    help="Promote the previous production model back to production.",
                ):
                    try:
                        if not require_role("admin", "Revert Production Model"):
                            st.stop()
                        reverted = revert_to_previous_production(dataset=dataset_name)
                        track_user_action(
                            "model_registry",
                            "revert_production",
                            {"dataset": dataset_name, "model_id": reverted.get("model_id")},
                        )
                        st.success(f"Reverted production to {reverted.get('model_id')}")
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))

        st.divider()

    if filtered.empty:
        st.info("No models to manage.")
    else:
        by_dataset_production = (
            df[df["stage"] == "production"].sort_values("registered_at", ascending=False).drop_duplicates("dataset")
        )
        production_lookup = {
            r["dataset"]: {"model_id": r["model_id"], "version": r.get("version", "â€”")}
            for _, r in by_dataset_production.iterrows()
        }

        for _, row in filtered.iterrows():
            col_label, col_stage, col_btn = st.columns([3, 2, 1])

            with col_label:
                st.markdown(
                    f"**{row['name']}** &nbsp; `{row['model_id']}`  "
                    f"<br><span style='color:#64748b;font-size:0.78rem'>"
                    f"{row['dataset']} / {row['task']}</span>",
                    unsafe_allow_html=True,
                )

            with col_stage:
                new_stage = st.selectbox(
                    "Stage",
                    ["development", "staging", "production", "archived"],
                    index=["development", "staging", "production", "archived"].index(
                        row.get("stage", "development")
                    ),
                    key=f"stage_{row['model_id']}",
                    label_visibility="collapsed",
                )

            with col_btn:
                if st.button("Save", key=f"save_{row['model_id']}"):
                    try:
                        current_stage = row.get("stage", "development")
                        if new_stage == current_stage:
                            st.info("No stage change detected.")
                        elif not require_role("admin", "Promote Model"):
                            st.stop()
                        elif new_stage == "production":
                            replacing = production_lookup.get(row["dataset"])
                            replacing_model_id = None
                            replacing_version = None
                            if replacing and replacing["model_id"] != row["model_id"]:
                                replacing_model_id = replacing["model_id"]
                                replacing_version = replacing["version"]

                            st.session_state["registry_pending_promotion"] = {
                                "model_id": row["model_id"],
                                "dataset": row["dataset"],
                                "replace_model_id": replacing_model_id,
                                "replace_version": replacing_version,
                            }
                            st.rerun()
                        else:
                            _apply_stage_change(row["model_id"], new_stage)
                            st.success("Saved")
                            st.rerun()
                    except Exception as exc:
                        st.error(str(exc))

            timeline_events = _cached_timeline(str(row["model_id"]))
            if timeline_events:
                ordered = list(reversed(timeline_events))
                stage_steps = [evt.get("to_stage", "unknown") for evt in ordered]
                deduped_steps = []
                for step in stage_steps:
                    if not deduped_steps or deduped_steps[-1] != step:
                        deduped_steps.append(step)
                st.caption(f"Stage timeline: {' -> '.join(deduped_steps)}")
            else:
                st.caption("Stage timeline: development")

            st.divider()
with tab_lineage:
    render_section_title("Model Lineage")
    st.caption("Track dataset, parameters, experiment ID, and parent model relationship.")

    dataset_filter = st.selectbox(
        "Lineage dataset filter",
        options=["All"] + sorted(df["dataset"].dropna().unique().tolist()),
        key="lineage_ds_filter",
    )

    lineage_dataset = None if dataset_filter == "All" else dataset_filter
    try:
        lineage_rows = list_lineage(limit=300, dataset=lineage_dataset)
    except Exception as exc:
        st.error("Failed to load lineage data.")
        st.caption(f"Details: {exc}")
        if st.button("Retry", type="primary", key="lineage_retry"):
            st.rerun()
        st.stop()

    if not lineage_rows:
        st.info("No lineage records available yet.")
    else:
        lineage_df = pd.DataFrame(lineage_rows)
        lineage_df["params"] = lineage_df["params"].apply(
            lambda x: json.loads(x) if isinstance(x, str) and x else {}
        )

        display = lineage_df[
            ["model_id", "experiment_id", "dataset", "version", "stage", "parent_model_id", "created_at", "params"]
        ].copy()
        display.columns = [c.replace("_", " ").title() for c in display.columns]
        display = render_summary_table(
            display,
            key_prefix="registry_lineage",
            columns=display.columns.tolist(),
            sort_by="Created At" if "Created At" in display.columns else None,
            filterable_columns=[c for c in ["Dataset", "Stage"] if c in display.columns],
            max_rows=25,
        )
        render_expandable_rows(
            display,
            title_col="Model Id",
            detail_cols=[
                c for c in [
                    "Stage",
                    "Dataset",
                    "Version",
                    "Experiment Id",
                    "Parent Model Id",
                    "Created At",
                ] if c in display.columns
            ],
            badge_col="Stage" if "Stage" in display.columns else None,
            badge_mode="stage",
        )

        if "version" in lineage_df.columns and lineage_df["version"].notna().any():
            lineage_df_plot = lineage_df.dropna(subset=["version"]).copy()
            lineage_df_plot["version"] = lineage_df_plot["version"].astype(int)
            fig = px.line(
                lineage_df_plot.sort_values("version"),
                x="version",
                y="dataset",
                color="stage",
                markers=True,
                hover_data=["model_id", "experiment_id", "parent_model_id"],
                labels={"version": "Model Version", "dataset": "Dataset"},
            )
            fig.update_layout(
                legend=dict(title=None),
            )
            apply_plotly_layout(fig, height=320, x_title="Model Version", y_title="Dataset")
            fig.update_yaxes(showgrid=False)
            st.plotly_chart(fig, width="stretch")

