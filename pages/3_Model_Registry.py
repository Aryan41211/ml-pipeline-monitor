"""
Model Registry page.

Lists all trained models stored in the local registry, supports promoting
models through development → staging → production lifecycle stages, and
renders side-by-side performance comparisons.
"""
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from services.model_service import list_lineage, list_models, set_model_stage
from src.database import initialize_db

st.set_page_config(page_title="Model Registry | ML Monitor", layout="wide")
initialize_db()

st.markdown(
    """
    <style>
        html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
        [data-testid="metric-container"] {
            background: #f8fafc; border: 1px solid #e2e8f0;
            border-radius: 10px; padding: 18px 20px;
        }
        [data-testid="stMetricValue"] { font-size: 1.9rem !important; font-weight: 700; color: #0f172a; }
        .page-header { padding: 8px 0 24px 0; border-bottom: 2px solid #e2e8f0; margin-bottom: 28px; }
        .page-header h1 { font-size: 1.75rem; font-weight: 700; color: #0f172a; margin: 0; }
        .page-header p  { color: #64748b; margin: 4px 0 0 0; font-size: 0.9rem; }
        .section-title  { font-size: 1rem; font-weight: 600; color: #1e293b;
                          margin-bottom: 14px; padding-bottom: 6px; border-bottom: 1px solid #f1f5f9; }
        .model-card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 16px;
            margin-bottom: 12px;
        }
        .model-card h4 { margin: 0 0 4px 0; color: #0f172a; font-size: 0.95rem; }
        .model-card .meta { color: #64748b; font-size: 0.78rem; margin-bottom: 10px; }
        .stage-badge {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 600;
        }
        [data-testid="stSidebar"] { background: #f8fafc; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="page-header">
      <h1>Model Registry</h1>
      <p>Manage, compare, and promote trained models across lifecycle stages.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Stage badge helper
# ---------------------------------------------------------------------------
STAGE_COLORS = {
    "development": ("bg:#dbeafe;color:#1d4ed8", "Development"),
    "staging":     ("bg:#fef9c3;color:#854d0e", "Staging"),
    "production":  ("bg:#dcfce7;color:#15803d", "Production"),
    "archived":    ("bg:#f1f5f9;color:#475569",  "Archived"),
}


def _badge(stage: str) -> str:
    style, label = STAGE_COLORS.get(stage, ("bg:#f1f5f9;color:#475569", stage.title()))
    return (
        f'<span class="stage-badge" '
        f'style="{style.replace("bg:", "background:")}">{label}</span>'
    )


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
raw = list_models(limit=100)

if not raw:
    st.info(
        "No models registered yet.  "
        "Execute a pipeline run on the **Pipeline Runner** page to register your first model."
    )
    st.stop()


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

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Navigation")
    st.page_link("app.py", label="Overview")
    st.page_link("pages/1_Pipeline_Runner.py", label="Pipeline Runner")
    st.page_link("pages/2_Experiment_Tracking.py", label="Experiment Tracking")
    st.page_link("pages/3_Model_Registry.py", label="Model Registry")
    st.page_link("pages/4_Data_Drift.py", label="Data Drift")
    st.divider()

    st.markdown("### Filters")
    st.divider()

    stage_opts = ["All"] + sorted(df["stage"].unique().tolist())
    task_opts  = ["All"] + sorted(df["task"].unique().tolist())
    ds_opts    = ["All"] + sorted(df["dataset"].unique().tolist())

    sel_stage = st.selectbox("Stage",   stage_opts)
    sel_task  = st.selectbox("Task",    task_opts)
    sel_ds    = st.selectbox("Dataset", ds_opts)

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
k1.metric("Total Models",   len(filtered))
k2.metric("In Production",  (filtered["stage"] == "production").sum())
k3.metric(
    "Best Accuracy",
    f"{clf_f['accuracy'].max():.4f}" if not clf_f.empty and clf_f["accuracy"].notna().any() else "—",
)
k4.metric(
    "Best R²",
    f"{reg_f['r2'].max():.4f}" if not reg_f.empty and reg_f["r2"].notna().any() else "—",
)

st.markdown("<br>", unsafe_allow_html=True)

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
        # Sort: production first, then staging, then development
        order = {"production": 0, "staging": 1, "development": 2, "archived": 3}
        sorted_df = filtered.copy()
        sorted_df["_order"] = sorted_df["stage"].map(order).fillna(9)
        sorted_df = sorted_df.sort_values("_order").drop(columns="_order")

        for _, row in sorted_df.iterrows():
            with st.container():
                left, right = st.columns([3, 1])
                with left:
                    st.markdown(
                        f"""
                        <div class="model-card">
                          <h4>{row['name']}  &nbsp; {_badge(row['stage'])}</h4>
                          <div class="meta">
                            Run ID: {row['model_id']} &nbsp;|&nbsp;
                                                        Version: v{row['version']} &nbsp;|&nbsp;
                            Dataset: {row['dataset']} &nbsp;|&nbsp;
                            Task: {row['task'].title()} &nbsp;|&nbsp;
                            Registered: {row['registered_at'][:10] if row['registered_at'] else '—'}
                          </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Metric chips
                    chip_html = []
                    if pd.notna(row.get("accuracy")):
                        chip_html.append(f"<b>Accuracy</b> {row['accuracy']:.4f}")
                    if pd.notna(row.get("f1_score")):
                        chip_html.append(f"<b>F1</b> {row['f1_score']:.4f}")
                    if pd.notna(row.get("roc_auc")):
                        chip_html.append(f"<b>ROC-AUC</b> {row['roc_auc']:.4f}")
                    if pd.notna(row.get("rmse")):
                        chip_html.append(f"<b>RMSE</b> {row['rmse']:.4f}")
                    if pd.notna(row.get("r2")):
                        chip_html.append(f"<b>R²</b> {row['r2']:.4f}")
                    if pd.notna(row.get("cv_mean")):
                        chip_html.append(f"<b>CV</b> {row['cv_mean']:.4f}")

                    st.markdown(
                        "  &nbsp;&nbsp;".join(chip_html) + "</div>",
                        unsafe_allow_html=True,
                    )


with tab_compare:
    st.markdown('<div class="section-title">Performance by Algorithm and Dataset</div>', unsafe_allow_html=True)

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
        st.plotly_chart(fig, use_container_width=True)

    if not reg_plot.empty:
        fig2 = px.bar(
            reg_plot.sort_values("r2", ascending=False),
            x="model_id",
            y="r2",
            color="name",
            text="r2",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            labels={"model_id": "Run ID", "r2": "R²", "name": "Algorithm"},
        )
        fig2.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig2.update_layout(
            height=360,
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#f1f5f9", title="R²"),
            showlegend=True,
        )
        st.plotly_chart(fig2, use_container_width=True)

    if clf_plot.empty and reg_plot.empty:
        st.info("Not enough data for comparison with current filters.")


with tab_manage:
    st.markdown('<div class="section-title">Promote or Archive Models</div>', unsafe_allow_html=True)
    st.caption(
        "Changing a model's stage updates the registry record. "
        "Only one model per dataset should be promoted to Production."
    )

    if filtered.empty:
        st.info("No models to manage.")
    else:
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
                        set_model_stage(row["model_id"], new_stage)
                        st.success("Saved")
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))

            st.divider()
with tab_lineage:
    st.markdown('<div class="section-title">Model Lineage</div>', unsafe_allow_html=True)
    st.caption("Track dataset, parameters, experiment ID, and parent model relationship.")

    dataset_filter = st.selectbox(
        "Lineage dataset filter",
        options=["All"] + sorted(df["dataset"].dropna().unique().tolist()),
        key="lineage_ds_filter",
    )

    lineage_dataset = None if dataset_filter == "All" else dataset_filter
    lineage_rows = list_lineage(limit=300, dataset=lineage_dataset)

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
        st.dataframe(display, use_container_width=True, hide_index=True)

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
                height=320,
                margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor="white",
                paper_bgcolor="white",
                xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
                yaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig, use_container_width=True)
