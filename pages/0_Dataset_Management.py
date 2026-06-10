"""
Dataset Management — Data Hub
"""
import pandas as pd
import streamlit as st

from services.app_service import initialize_application
from services.pipeline_service import get_dataset_options, get_dataset_preview
from src.auth import render_auth_controls, current_role
from src.ui_theme import (
    apply_ui_theme,
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
st.set_page_config(page_title="Dataset Hub | ML Monitor", layout="wide")
initialize_application()
apply_ui_theme()

render_top_navbar(user_role=current_role())

with st.sidebar:
    render_sidebar_nav()
    st.divider()
    render_auth_controls()

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
DATASET_OPTIONS = get_dataset_options()

@st.cache_data(ttl=30)
def _load_dataset_preview(dataset_key: str):
    return get_dataset_preview(dataset_key, test_size=0.2, random_state=42)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
col_title, col_actions = st.columns([4, 1])
with col_title:
    st.markdown('<div class="ui-fade-in"><h1 style="margin:0; font-family:\'Poppins\', sans-serif;">Dataset Hub</h1><p style="color:var(--color-text-tertiary);">Explore, validate, and manage training datasets.</p></div>', unsafe_allow_html=True)
with col_actions:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if st.button("Refresh", type="primary", use_container_width=True): st.rerun()

# ---------------------------------------------------------------------------
# Dataset Selector
# ---------------------------------------------------------------------------
selected_label = st.selectbox("Select Dataset", list(DATASET_OPTIONS.keys()))
selected_key = DATASET_OPTIONS[selected_label]

# ---------------------------------------------------------------------------
# Dataset Preview
# ---------------------------------------------------------------------------
loading = st.empty()
with loading.container(): render_loading_skeleton(lines=5)

try:
    preview = _load_dataset_preview(selected_key)
    ds = preview["dataset"]
    feat_stats = preview["feature_stats"]
    loading.empty()
except Exception as e:
    loading.empty()
    st.error(f"Failed to load dataset: {e}")
    st.stop()

# KPI Row
c1, c2, c3, c4 = st.columns(4)
stats = ds.get("stats", {})
with c1: component_kpi_card("Samples", f"{stats.get('n_samples', 0):,}", "Total rows", icon="📊")
with c2: component_kpi_card("Features", f"{stats.get('n_features', 0)}", "Columns", icon="🔢")
with c3: component_kpi_card("Task", ds.get("task", "classification").title(), "Problem type", icon="🎯")
with c4: component_kpi_card("Missing", f"{stats.get('missing_values', 0)}", "Null values", icon="⚠️", tone="danger" if stats.get('missing_values', 0) > 0 else "success")

render_spacer("md")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_overview, tab_features, tab_split = st.tabs(["📋 Overview", "🔬 Feature Statistics", "✂️ Train/Test Split"])

with tab_overview:
    render_section_title(f"Dataset: {selected_label}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Basic Info**")
        st.write(f"**Task:** {ds.get('task', 'N/A').title()}")
        st.write(f"**Samples:** {stats.get('n_samples', 0):,}")
        st.write(f"**Features:** {stats.get('n_features', 0)}")
        st.write(f"**Missing Values:** {stats.get('missing_values', 0)}")
    
    with col2:
        if ds.get("task") == "classification":
            st.markdown("**Class Distribution**")
            dist = stats.get("class_distribution", {})
            if dist:
                dist_df = pd.DataFrame(list(dist.items()), columns=["Class", "Proportion"])
                dist_df["Count"] = (dist_df["Proportion"] * stats.get("n_samples", 0)).astype(int)
                st.dataframe(dist_df, use_container_width=True, hide_index=True)
        else:
            st.markdown("**Target Statistics**")
            st.write(f"Mean: {stats.get('target_mean', 0):.4f}")
            st.write(f"Std: {stats.get('target_std', 0):.4f}")
            st.write(f"Min: {stats.get('target_min', 0):.4f}")
            st.write(f"Max: {stats.get('target_max', 0):.4f}")

with tab_features:
    render_section_title("Feature Statistics")
    if not feat_stats.empty:
        display_stats = feat_stats.reset_index().rename(columns={"index": "feature"})
        render_summary_table(
            display_stats,
            columns=["feature", "count", "mean", "std", "min", "max", "missing", "missing_pct", "skewness", "kurtosis"],
            filterable_columns=["feature"]
        )
    else:
        st.info("No feature statistics available.")

with tab_split:
    render_section_title("Train/Test Split")
    train_size = stats.get('train_size', 0)
    test_size = stats.get('test_size', 0)
    
    c1, c2 = st.columns(2)
    with c1:
        component_kpi_card("Train Set", f"{train_size:,}", f"{train_size/stats.get('n_samples', 1)*100:.1f}%", icon="📚")
    with c2:
        component_kpi_card("Test Set", f"{test_size:,}", f"{test_size/stats.get('n_samples', 1)*100:.1f}%", icon="🧪")
    
    st.markdown("**Sample Data (Train)**")
    if len(ds["X_train"]) > 0:
        st.dataframe(ds["X_train"].head(10), use_container_width=True)
    
    st.markdown("**Sample Data (Test)**")
    if len(ds["X_test"]) > 0:
        st.dataframe(ds["X_test"].head(10), use_container_width=True)

# Insights
render_spacer("md")
component_insight_panel([
    f"Dataset '{selected_label}' contains {stats.get('n_samples', 0):,} samples with {stats.get('n_features', 0)} features.",
    f"Task type: {ds.get('task', 'classification').title()}.",
    "Use Pipeline Runner to train models on this dataset."
])

st.divider()
st.caption("🗂️ Dataset Hub v2.0-Componentized")