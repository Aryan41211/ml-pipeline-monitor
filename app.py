"""
ML Pipeline Monitor â€” Home Dashboard

Provides an at-a-glance view of experiment history, model registry health,
and current host resource utilisation.  Use the sidebar navigation to access
the Pipeline Runner, Experiment Tracking, Model Registry, and Data Drift pages.
"""
from datetime import datetime
from html import escape
import json

import pandas as pd
import plotly.express as px
import streamlit as st

from services.app_service import get_dashboard_snapshot, get_ui_settings, initialize_application
from services.pipeline_service import get_dataset_options, run_pipeline_and_persist
from services.telemetry_service import track_user_action
from src.auth import can_run_pipeline, current_role, is_auth_enabled, render_auth_controls
from src.ui_theme import (
    apply_ui_theme,
    render_empty_data_explainer,
    render_loading_skeleton,
    render_kpi_row,
    render_page_header_with_action,
    render_section_title,
    render_sidebar_brand,
    render_sidebar_nav,
    render_spacer,
)

# ---------------------------------------------------------------------------
# Page config â€” must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ML Pipeline Monitor",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "ML Pipeline Monitor â€” production MLOps observability platform."},
)

initialize_application()
UI_SETTINGS = get_ui_settings()
MAX_EXPERIMENTS = int(UI_SETTINGS.get("max_experiments_displayed", 200))
DATASET_OPTIONS = get_dataset_options()
apply_ui_theme()
if "overview_limit" not in st.session_state:
    st.session_state["overview_limit"] = MAX_EXPERIMENTS

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    render_sidebar_brand()
    st.markdown("### Navigation")
    render_sidebar_nav()
    st.divider()

    st.markdown("### Configuration")
    st.session_state["overview_limit"] = int(
        st.number_input(
            "Experiments to display",
            min_value=20,
            max_value=500,
            value=int(st.session_state["overview_limit"]),
            step=20,
        )
    )
    st.divider()

    st.markdown("### User / Access")
    auth_ok = render_auth_controls()
    if auth_ok:
        st.caption(f"Role: {current_role()}")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
@st.cache_data(ttl=20, show_spinner=False)
def _load_dashboard(limit: int) -> dict:
    return get_dashboard_snapshot(limit=limit)


loading = st.empty()
with loading.container():
    render_loading_skeleton(lines=4, key="overview_load")


try:
    snapshot = _load_dashboard(int(st.session_state["overview_limit"]))
    experiments = snapshot.get("experiments", [])
    models = snapshot.get("models", [])
    sys_snapshot = snapshot.get("system", {})
    loading.empty()
except Exception as exc:
    loading.empty()
    st.error("Dashboard data could not be loaded right now.")
    st.caption(f"Details: {exc}")
    if st.button("Retry", type="primary", key="overview_retry"):
        st.rerun()
    st.stop()

exp_df = pd.DataFrame(experiments) if experiments else pd.DataFrame()
mdl_df = pd.DataFrame(models) if models else pd.DataFrame()

if not exp_df.empty:
    def _parse_metrics(row):
        try:
            return json.loads(row) if isinstance(row, str) else row
        except Exception:
            return {}

    exp_df["_metrics"] = exp_df["metrics"].apply(_parse_metrics)
# ---------------------------------------------------------------------------
# Premium overview helpers
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
      .ov-hero { margin: 2px 0 14px 0; }
      .ov-title { font-size: 2rem; font-weight: 800; letter-spacing: -0.02em; color: #0b1220; line-height: 1.1; }
      .ov-sub { margin-top: 5px; color: #5b6472; font-size: 0.93rem; }

      .ov-kpi {
        border-radius: 12px;
        padding: 14px 14px 12px 14px;
        background: linear-gradient(160deg, #ffffff 0%, #f7f9fc 100%);
        border: 1px solid #e5e9f1;
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.05);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
        min-height: 120px;
      }
      .ov-kpi:hover {
        transform: translateY(-2px);
        box-shadow: 0 14px 24px rgba(15, 23, 42, 0.11);
      }
      .ov-kpi-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; }
      .ov-kpi-icon {
        width: 28px; height: 28px; border-radius: 8px; display: inline-flex;
        align-items: center; justify-content: center; font-size: 14px; font-weight: 700;
        background: #e7eef9; color: #1f4f9a;
      }
      .ov-kpi-label { font-size: 0.78rem; color: #637083; }
      .ov-kpi-value { font-size: 1.45rem; font-weight: 800; color: #0b1220; line-height: 1.1; }
      .ov-kpi-sub { margin-top: 4px; font-size: 0.78rem; color: #5f6a7a; }

      .ov-panel {
        border: 1px solid #e5e9f1;
        border-radius: 12px;
        padding: 12px;
        background: #ffffff;
      }
      .ov-table-wrap {
        border: 1px solid #e6ebf3;
        border-radius: 10px;
        overflow: hidden;
      }
      table.ov-table { width: 100%; border-collapse: collapse; font-size: 0.81rem; }
      table.ov-table thead tr { background: #f4f7fb; }
      table.ov-table th {
        text-align: left; padding: 9px 10px; color: #4f5f75; font-weight: 700; border-bottom: 1px solid #e6ebf3;
      }
      table.ov-table td {
        padding: 9px 10px; border-bottom: 1px solid #edf1f7; color: #1e2a3b;
      }
      table.ov-table tbody tr:hover { background: #f8fbff; }
      .ov-badge {
        display: inline-flex; align-items: center; gap: 5px;
        padding: 2px 8px; border-radius: 999px; font-size: 0.72rem; font-weight: 600;
      }
      .ov-badge.model { background: #eaf2ff; color: #1e4fa3; }
      .ov-badge.dataset { background: #eef9f4; color: #1d7a54; }
      .ov-acc.high { color: #15803d; font-weight: 700; }
      .ov-acc.mid { color: #b45309; font-weight: 700; }
      .ov-acc.low { color: #b91c1c; font-weight: 700; }
      .ov-acc.na { color: #677489; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _apply_dark_plotly(fig, *, height: int = 280, x_title: str | None = None, y_title: str | None = None):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        transition={"duration": 420, "easing": "cubic-in-out"},
        margin={"l": 8, "r": 8, "t": 16, "b": 12},
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220",
        font={"family": "Inter, Segoe UI, sans-serif", "color": "#dbe5f4"},
    )
    fig.update_xaxes(showgrid=False, title=x_title, zeroline=False)
    fig.update_yaxes(showgrid=False, title=y_title, zeroline=False)
    return fig


def _accuracy_class(value) -> str:
    if value is None or pd.isna(value):
        return "na"
    val = float(value)
    if val >= 0.90:
        return "high"
    if val >= 0.75:
        return "mid"
    return "low"


def _render_kpi_card(icon_html: str, label: str, value: str, subtext: str) -> str:
    return (
        "<div class='ov-kpi'>"
        "<div class='ov-kpi-head'>"
        f"<div class='ov-kpi-label'>{escape(label)}</div>"
        f"<span class='ov-kpi-icon'>{icon_html}</span>"
        "</div>"
        f"<div class='ov-kpi-value'>{escape(value)}</div>"
        f"<div class='ov-kpi-sub'>{escape(subtext)}</div>"
        "</div>"
    )


def _render_recent_runs_table(source_df: pd.DataFrame, limit: int = 12) -> str:
    recent_df = source_df.head(limit).copy()
    rows_html = []
    for row in recent_df.itertuples(index=False):
        metrics = row._metrics if hasattr(row, "_metrics") else {}
        if isinstance(metrics, str):
            try:
                metrics = json.loads(metrics)
            except Exception:
                metrics = {}
        metrics = metrics or {}

        accuracy = metrics.get("accuracy")
        accuracy_str = f"{float(accuracy):.3f}" if accuracy is not None and pd.notna(accuracy) else "--"
        acc_class = _accuracy_class(accuracy)
        duration = f"{float(row.duration_seconds):.2f}s" if pd.notna(getattr(row, "duration_seconds", None)) else "--"
        created = str(getattr(row, "created_at", ""))[:19].replace("T", " ")

        rows_html.append(
            "<tr>"
            f"<td>{escape(str(row.run_id)[:8])}</td>"
            f"<td><span class='ov-badge dataset'>{escape(str(row.dataset))}</span></td>"
            f"<td><span class='ov-badge model'>{escape(str(row.model_type))}</span></td>"
            f"<td>{escape(str(row.task).title())}</td>"
            f"<td class='ov-acc {acc_class}'>{accuracy_str}</td>"
            f"<td>{duration}</td>"
            f"<td>{escape(created)}</td>"
            "</tr>"
        )

    if not rows_html:
        rows_html.append("<tr><td colspan='7' style='text-align:center;color:#677489'>No runs available</td></tr>")

    return (
        "<div class='ov-table-wrap'>"
        "<table class='ov-table'>"
        "<thead><tr>"
        "<th>Run</th><th>Dataset</th><th>Model</th><th>Task</th><th>Accuracy</th><th>Duration</th><th>Created</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        "</table>"
        "</div>"
    )


# ---------------------------------------------------------------------------
# Header + KPI summary
# ---------------------------------------------------------------------------
refresh_clicked = render_page_header_with_action(
    "Overview",
    "Monitor experiments, model inventory, and system health at a glance.",
    "Refresh",
    action_key="overview_refresh",
)
if refresh_clicked:
    track_user_action("overview", "refresh_dashboard")
    st.rerun()

total_runs = len(exp_df)
registered_models = len(mdl_df)
production_models = int((mdl_df["stage"] == "production").sum()) if not mdl_df.empty and "stage" in mdl_df.columns else 0
best_accuracy = None
best_accuracy_model = "No classification runs yet"
avg_duration = None
fastest_duration = None
runs_today = 0

if not exp_df.empty:
    acc_pairs = []
    for _, row in exp_df.iterrows():
        m = row.get("_metrics", {}) or {}
        if isinstance(m, dict) and m.get("accuracy") is not None:
            acc_pairs.append((float(m.get("accuracy")), row.get("model_type", "unknown")))
    if acc_pairs:
        best_accuracy, best_model = max(acc_pairs, key=lambda x: x[0])
        best_accuracy_model = f"best model: {best_model}"

    if "duration_seconds" in exp_df.columns and exp_df["duration_seconds"].notna().any():
        avg_duration = float(exp_df["duration_seconds"].mean())
        fastest_duration = float(exp_df["duration_seconds"].min())

    if "created_at" in exp_df.columns and exp_df["created_at"].notna().any():
        created_series = pd.to_datetime(exp_df["created_at"], errors="coerce")
        today = datetime.utcnow().date()
        runs_today = int((created_series.dt.date == today).sum())

render_kpi_row(
    [
        {
            "title": "Total Experiments",
            "value": f"{total_runs:,}",
            "subtitle": f"+{runs_today} today",
            "tone": "info",
            "icon": "&#128202;",
        },
        {
            "title": "Best Accuracy",
            "value": f"{best_accuracy:.3f}" if best_accuracy is not None else "--",
            "subtitle": best_accuracy_model,
            "tone": "success",
            "icon": "&#127942;",
        },
        {
            "title": "Pipeline Speed",
            "value": f"{avg_duration:.2f}s" if avg_duration is not None else "--",
            "subtitle": f"fastest: {fastest_duration:.2f}s" if fastest_duration is not None else "no timings yet",
            "tone": "neutral",
            "icon": "&#9201;",
        },
        {
            "title": "Model Inventory",
            "value": f"{registered_models:,}",
            "subtitle": f"production: {production_models}",
            "tone": "neutral",
            "icon": "&#129504;",
        },
    ]
)
render_spacer("sm")

# ---------------------------------------------------------------------------
# Main content layout
# ---------------------------------------------------------------------------
if exp_df.empty:
    render_section_title("First Run")
    render_empty_data_explainer(
        "No experiments yet.",
        "Run your first pipeline to unlock dashboard analytics.",
        "After the first run, KPI cards, charts, and tables will populate automatically.",
    )
    st.page_link("pages/1_Pipeline_Runner.py", label="Run First Pipeline")
else:
    top_left, top_right = st.columns([1.6, 1], gap="large")

    with top_left:
        render_section_title("Recent Runs")
        st.markdown("<div class='ov-panel'>", unsafe_allow_html=True)
        st.markdown(_render_recent_runs_table(exp_df), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with top_right:
        render_section_title("Experiment Activity")
        st.markdown("<div class='ov-panel'>", unsafe_allow_html=True)
        exp_df["date"] = pd.to_datetime(exp_df["created_at"]).dt.date
        activity = exp_df.groupby("date").size().reset_index(name="count")
        fig = px.bar(
            activity,
            x="date",
            y="count",
            labels={"date": "", "count": "Runs"},
            color_discrete_sequence=["#5ba3ff"],
        )
        fig.update_traces(marker_line_width=0, opacity=0.88)
        _apply_dark_plotly(fig, height=285, x_title="", y_title="Runs")
        st.plotly_chart(fig, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    lower_left, lower_right = st.columns([1, 1], gap="large")
    with lower_left:
        render_section_title("System Health")
        st.markdown("<div class='ov-panel'>", unsafe_allow_html=True)
        try:
            sys_m = sys_snapshot if isinstance(sys_snapshot, dict) and sys_snapshot else {}
            if not sys_m:
                raise RuntimeError("System metrics unavailable")

            def _bar_color(pct: float) -> str:
                if pct < 60:
                    return "#22c55e"
                if pct < 85:
                    return "#f59e0b"
                return "#ef4444"

            for label, pct in [
                ("CPU", sys_m["cpu_percent"]),
                ("Memory", sys_m["memory_percent"]),
                ("Disk", sys_m["disk_percent"]),
            ]:
                color = _bar_color(pct)
                st.markdown(
                    f"""
                    <div style="margin-bottom:14px">
                      <div style="display:flex;justify-content:space-between;font-size:0.82rem;color:#374151;margin-bottom:4px">
                        <span>{label}</span>
                        <span style="font-weight:700">{pct:.1f}%</span>
                      </div>
                      <div class="resource-bar"><div class="resource-bar-fill" style="width:{pct}%;background:{color}"></div></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.caption(
                f"CPU cores: {sys_m['cpu_logical_cores']} logical / {sys_m['cpu_physical_cores']} physical"
            )
            st.caption(
                f"Memory: {sys_m['memory_used_gb']:.1f} GB used / {sys_m['memory_total_gb']:.1f} GB total"
            )
            st.caption(
                f"Disk: {sys_m['disk_free_gb']:.1f} GB free / {sys_m['disk_total_gb']:.1f} GB total"
            )
        except Exception as exc:
            st.warning(f"System metrics unavailable: {exc}")
        st.markdown("</div>", unsafe_allow_html=True)

    with lower_right:
        render_section_title("Algorithm Mix")
        st.markdown("<div class='ov-panel'>", unsafe_allow_html=True)
        if "model_type" in exp_df.columns and exp_df["model_type"].notna().any():
            algo_counts = exp_df["model_type"].value_counts().reset_index()
            algo_counts.columns = ["Algorithm", "Count"]
            fig2 = px.pie(
                algo_counts,
                names="Algorithm",
                values="Count",
                color_discrete_sequence=["#7cb7ff", "#50d4ad", "#ff9f7b", "#ffd166", "#c4b5fd", "#9ae6b4"],
                hole=0.58,
            )
            fig2.update_traces(textposition="inside", textinfo="percent", pull=[0.02] + [0] * (len(algo_counts) - 1))
            _apply_dark_plotly(fig2, height=285)
            fig2.update_layout(showlegend=True, legend={"orientation": "v", "font": {"size": 10}})
            st.plotly_chart(fig2, width="stretch")
        else:
            st.info("No algorithm distribution available yet.")
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Footer note
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "ML Pipeline Monitor  â€”  built with Streamlit, scikit-learn, XGBoost, Plotly, and SQLite."
)

