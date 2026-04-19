"""Data Health page: missingness, class balance, schema drift, and outlier checks."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from services.app_service import initialize_application
from services.data_health_service import (
    basic_statistics_report,
    class_imbalance_report,
    compare_schema,
    data_health_defaults,
    load_health_input,
    load_schema_baseline,
    missing_value_report,
    outlier_report,
    save_schema_baseline,
)
from services.telemetry_service import track_user_action
from src.auth import can_run_pipeline, render_auth_controls
from src.ui_theme import (
    apply_plotly_layout,
    apply_ui_theme,
    render_page_header_with_action,
    render_section_title,
    render_sidebar_brand,
    render_sidebar_nav,
    render_summary_table,
    status_badge_html,
)


st.set_page_config(page_title="Data Health | ML Monitor", layout="wide")
initialize_application()
apply_ui_theme()

DEFAULTS = data_health_defaults()
DATASET_OPTIONS = DEFAULTS["dataset_options"]

with st.sidebar:
    render_sidebar_brand()
    st.markdown("### Navigation")
    render_sidebar_nav()
    st.divider()

    st.markdown("### Configuration")
    st.divider()

    dataset_label = st.selectbox("Dataset", list(DATASET_OPTIONS.keys()))
    dataset_key = DATASET_OPTIONS[dataset_label]

    test_size = st.slider("Test split", 0.10, 0.40, float(DEFAULTS["test_size"]), 0.05)
    random_seed = st.number_input("Random seed", min_value=0, max_value=9999, value=int(DEFAULTS["random_seed"]))

    outlier_method = st.selectbox("Outlier method", ["iqr", "zscore"], index=0)
    z_threshold = st.slider("Z-score threshold", 2.0, 5.0, 3.0, 0.1)

    st.divider()
    st.markdown("### Access")
    render_auth_controls()

run_scan = render_page_header_with_action(
    "Data health",
    "Inspect dataset quality issues: missingness, imbalance, schema drift, and outliers.",
    "Run Health Scan",
    action_key="data_health_run_scan",
)
if run_scan:
    track_user_action("data_health", "run_scan", {"dataset": dataset_label, "outlier_method": outlier_method})


@st.cache_data(ttl=120, show_spinner=False)
def _cached_input(ds_key: str, split: float, seed: int):
    return load_health_input(ds_key, test_size=split, random_state=seed)


try:
    with st.spinner("Loading dataset for health checks..."):
        payload = _cached_input(dataset_key, float(test_size), int(random_seed))
except Exception as exc:
    st.error("Failed to load dataset for health analysis.")
    st.caption(f"Details: {exc}")
    if st.button("Retry", type="primary", key="data_health_retry_load"):
        st.rerun()
    st.stop()

feature_df: pd.DataFrame = payload["feature_frame"]
target = payload["target"]
task = payload["task"]

# Compute reports
missing = missing_value_report(feature_df)
imbalance = class_imbalance_report(target, task)
basic_stats = basic_statistics_report(feature_df)
outliers = outlier_report(feature_df, method=outlier_method, z_threshold=float(z_threshold))

baseline = load_schema_baseline(dataset_key)
schema_cmp = compare_schema(list(feature_df.columns), baseline)

# KPI strip
k1, k2, k3, k4 = st.columns(4)
k1.metric("Features", feature_df.shape[1])
k2.metric("Rows", feature_df.shape[0])
k3.metric("Total Missing %", f"{missing['total_missing_pct']:.2f}%")
k4.metric("Outlier Features", int((outliers["outlier_count"] > 0).sum()) if not outliers.empty else 0)

st.markdown("<br>", unsafe_allow_html=True)

# Schema drift
render_section_title("Schema Drift (Baseline Comparison)")
if not schema_cmp["has_baseline"]:
    st.info("No previous schema baseline exists for this dataset.")
    if st.button(
        "Save current schema as baseline",
        type="primary",
        key="save_schema_baseline",
        disabled=not can_run_pipeline(),
    ):
        save_schema_baseline(dataset_key, list(feature_df.columns))
        track_user_action("data_health", "save_schema_baseline", {"dataset": dataset_label})
        st.success("Schema baseline saved.")
        st.rerun()
    if not can_run_pipeline():
        st.caption("Read-only role: operator or admin is required to save baselines.")
else:
    new_cols = schema_cmp["new_columns"]
    missing_cols = schema_cmp["missing_columns"]

    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown(f"**New columns**: {len(new_cols)}")
        if new_cols:
            st.markdown("  ".join([status_badge_html("info") + f" {c}" for c in new_cols]), unsafe_allow_html=True)
        else:
            st.caption("No new columns detected.")

    with c_right:
        st.markdown(f"**Missing columns**: {len(missing_cols)}")
        if missing_cols:
            st.markdown("  ".join([status_badge_html("warning") + f" {c}" for c in missing_cols]), unsafe_allow_html=True)
        else:
            st.caption("No missing columns detected.")

    if st.button(
        "Update baseline to current schema",
        key="update_schema_baseline",
        disabled=not can_run_pipeline(),
    ):
        save_schema_baseline(dataset_key, list(feature_df.columns))
        track_user_action("data_health", "update_schema_baseline", {"dataset": dataset_label})
        st.success("Schema baseline updated.")
        st.rerun()
    if not can_run_pipeline():
        st.caption("Read-only role: operator or admin is required to update baselines.")

st.markdown("<br>", unsafe_allow_html=True)

# Missing values
render_section_title("Missing Values")
missing_df = missing["per_column"].copy()
missing_df.columns = ["Column", "Missing Count", "Missing %"]
render_summary_table(
    missing_df,
    key_prefix="health_missing",
    columns=["Column", "Missing Count", "Missing %"],
    sort_by="Missing Count",
    filterable_columns=[],
    max_rows=25,
)

if not missing_df.empty:
    top_missing = missing_df.head(12)
    fig_missing = px.bar(
        top_missing,
        x="Column",
        y="Missing %",
        color="Missing %",
        color_continuous_scale="Blues",
    )
    apply_plotly_layout(fig_missing, height=280, x_title="Feature", y_title="Missing %")
    st.plotly_chart(fig_missing, width="stretch")

st.markdown("<br>", unsafe_allow_html=True)

# Class imbalance
render_section_title("Class Imbalance")
if not imbalance["enabled"]:
    st.info("Class imbalance is only relevant for classification datasets.")
else:
    dist = imbalance["distribution"].copy()
    dist.columns = ["Class", "Count", "Percent"]

    c1, c2 = st.columns([2, 1])
    with c1:
        render_summary_table(
            dist,
            key_prefix="health_imbalance",
            columns=["Class", "Count", "Percent"],
            sort_by="Count",
            filterable_columns=[],
            max_rows=10,
        )
    with c2:
        ratio = imbalance["imbalance_ratio"]
        st.metric("Imbalance Ratio", f"{ratio:.2f}x" if ratio is not None else "â€”")
        if ratio is not None and ratio > 3:
            st.warning("Potential class imbalance risk detected.")

    fig_cls = px.pie(dist, names="Class", values="Count")
    apply_plotly_layout(fig_cls, height=280)
    st.plotly_chart(fig_cls, width="stretch")

st.markdown("<br>", unsafe_allow_html=True)

# Basic statistics
render_section_title("Basic Statistics")
stats_df = basic_stats.copy()
stats_df.columns = ["Feature", "Mean", "Std", "Min", "Max"]
render_summary_table(
    stats_df,
    key_prefix="health_stats",
    columns=["Feature", "Mean", "Std", "Min", "Max"],
    sort_by="Feature",
    filterable_columns=[],
    max_rows=25,
)

st.markdown("<br>", unsafe_allow_html=True)

# Outliers
render_section_title("Outlier Detection")
out_df = outliers.copy()
out_df.columns = ["Feature", "Outlier Count", "Outlier %"]
render_summary_table(
    out_df,
    key_prefix="health_outliers",
    columns=["Feature", "Outlier Count", "Outlier %"],
    sort_by="Outlier Count",
    filterable_columns=[],
    max_rows=25,
)

if not out_df.empty:
    top_out = out_df.head(12)
    fig_out = px.bar(top_out, x="Feature", y="Outlier Count", color="Outlier Count", color_continuous_scale="Oranges")
    apply_plotly_layout(fig_out, height=280, x_title="Feature", y_title="Outlier Count")
    st.plotly_chart(fig_out, width="stretch")

