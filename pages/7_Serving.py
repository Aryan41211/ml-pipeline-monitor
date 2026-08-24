"""
Serving — Inference request history and latency.

Shows what the FastAPI inference endpoint has actually served: which model
answered, whether it succeeded, and how long it took.
"""

import pandas as pd
import streamlit as st

from ml_pipeline_monitor.core.auth import current_role, render_auth_controls
from ml_pipeline_monitor.services.app_service import initialize_application
from ml_pipeline_monitor.services.model_service import (
    get_prediction_detail,
    list_prediction_history,
    prediction_stats,
)
from ml_pipeline_monitor.utils.ui_theme import (
    apply_ui_theme,
    component_alert_card,
    component_insight_panel,
    component_kpi_card,
    page_header,
    render_loading_skeleton,
    render_section_title,
    render_sidebar_nav,
    render_spacer,
    render_summary_table,
    render_top_navbar,
    safe_render,
)

st.set_page_config(page_title="Serving | ML Monitor", layout="wide")
initialize_application()
apply_ui_theme()

render_top_navbar(user_role=current_role())

with st.sidebar:
    render_sidebar_nav()
    st.divider()
    render_auth_controls()


def _render_page():
    page_header(
        "Serving",
        "Inference requests handled by the API, with the model that answered each one.",
    )

    @st.cache_data(ttl=15)
    def _load(limit: int):
        rows = list_prediction_history(limit=limit)
        return rows, prediction_stats(rows)

    loading = st.empty()
    with loading.container():
        render_loading_skeleton(lines=5)
    history, stats = _load(200)
    loading.empty()

    if not history:
        component_alert_card(
            "No inference requests recorded yet. Promote a model to production and "
            "call POST /v1/predict to populate this view.",
            tone="info",
        )
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        component_kpi_card("Requests", str(stats["total"]), "Recent history")
    with c2:
        component_kpi_card(
            "Success Rate",
            f"{stats['success_rate']:.1f}%",
            f"{stats['failed']} failed",
            tone="success" if stats["failed"] == 0 else "warning",
        )
    with c3:
        component_kpi_card("Median Latency", f"{stats['p50_ms']:.1f} ms", "p50")
    with c4:
        component_kpi_card(
            "p95 Latency",
            f"{stats['p95_ms']:.1f} ms",
            "95th percentile",
            tone="warning" if stats["p95_ms"] > 1000 else "neutral",
        )

    render_spacer("md")

    tab_requests, tab_detail = st.tabs(["Request History", "Request Detail"])

    with tab_requests:
        render_section_title("Recent Requests")
        frame = pd.DataFrame(history)
        display = pd.DataFrame(
            {
                "Request": frame["request_id"].astype(str).str[:12],
                "Model": frame["model_id"].astype(str),
                "Dataset": frame["dataset"].fillna("—").astype(str),
                "Rows": frame["num_predictions"].astype(int),
                "Duration (ms)": frame["duration_ms"].fillna(0).astype(float).round(2),
                "Status": frame["status"].astype(str),
                "Created At": frame["created_at"].astype(str),
            }
        )
        render_summary_table(
            display,
            columns=["Request", "Model", "Dataset", "Rows", "Duration (ms)", "Status", "Created At"],
            filterable_columns=["Model", "Dataset", "Status"],
        )

        failures = [row for row in history if str(row.get("status")) != "success"]
        if failures:
            render_spacer("sm")
            render_section_title("Recent Failures")
            for row in failures[:5]:
                component_alert_card(
                    f"{row.get('created_at', '')} — {row.get('error') or 'no error recorded'}",
                    tone="danger",
                    title=str(row.get("request_id", ""))[:12],
                )

    with tab_detail:
        render_section_title("Inspect a Request")
        options = [str(row["request_id"]) for row in history]
        selected = st.selectbox(
            "Request ID",
            options,
            format_func=lambda value: f"{value[:12]}  ({dict((r['request_id'], r['status']) for r in history)[value]})",
        )

        detail = get_prediction_detail(selected) if selected else None
        if not detail:
            component_alert_card("Request not found.", tone="warning")
        else:
            left, right = st.columns([1, 1], gap="medium")
            with left:
                st.markdown("**Request**")
                st.json(
                    {
                        "request_id": detail.get("request_id"),
                        "correlation_id": detail.get("correlation_id"),
                        "model_id": detail.get("model_id"),
                        "dataset": detail.get("dataset"),
                        "input_type": detail.get("input_type"),
                        "input_hash": detail.get("input_hash"),
                        "status": detail.get("status"),
                        "duration_ms": detail.get("duration_ms"),
                        "error": detail.get("error"),
                    }
                )
            with right:
                st.markdown("**Predictions**")
                rows = detail.get("predictions") or []
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    st.caption("No per-row predictions stored for this request.")

    render_spacer("md")
    models_used = len({str(row.get("model_id")) for row in history if row.get("model_id")})
    component_insight_panel(
        [
            f"{stats['total']} request(s) in view, served by {models_used} distinct model(s).",
            f"Median {stats['p50_ms']:.1f} ms, p95 {stats['p95_ms']:.1f} ms.",
            (f"{stats['failed']} request(s) failed." if stats["failed"] else "No failed requests in this window."),
        ]
    )


safe_render("Serving", _render_page)
