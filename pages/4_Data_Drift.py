"""
Data Drift Detection page.

Compares a reference dataset split against a perturbed (simulated current)
distribution using the Kolmogorov-Smirnov test and Population Stability Index.
Operators can control the perturbation intensity to explore drift thresholds.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from services.drift_service import run_drift_and_persist
from src.config_loader import load_config
from src.data_loader import DATASET_OPTIONS, load_dataset
from src.database import get_drift_reports, initialize_db

st.set_page_config(page_title="Data Drift | ML Monitor", layout="wide")
initialize_db()
APP_CONFIG = load_config()
MONITORING_CFG = APP_CONFIG.get("monitoring", {})

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
        [data-testid="stSidebar"] { background: #f8fafc; }
        .alert-box {
            padding: 12px 16px;
            border-radius: 8px;
            margin-bottom: 16px;
            font-size: 0.88rem;
        }
        .alert-red    { background: #fee2e2; border-left: 4px solid #dc2626; color: #7f1d1d; }
        .alert-yellow { background: #fef9c3; border-left: 4px solid #ca8a04; color: #713f12; }
        .alert-green  { background: #dcfce7; border-left: 4px solid #16a34a; color: #14532d; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="page-header">
      <h1>Data Drift Detection</h1>
      <p>
        Compare reference and current data distributions using the
        Kolmogorov-Smirnov test and Population Stability Index (PSI).
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Navigation")
    st.page_link("app.py", label="Overview")
    st.page_link("pages/1_Pipeline_Runner.py", label="Pipeline Runner")
    st.page_link("pages/2_Experiment_Tracking.py", label="Experiment Tracking")
    st.page_link("pages/3_Model_Registry.py", label="Model Registry")
    st.page_link("pages/4_Data_Drift.py", label="Data Drift")
    st.divider()

    st.markdown("### Configuration")
    st.divider()

    dataset_label = st.selectbox("Dataset", list(DATASET_OPTIONS.keys()))
    dataset_key   = DATASET_OPTIONS[dataset_label]

    st.markdown("**Perturbation settings**")
    st.caption(
        "Simulates real-world distribution shift by adding noise and/or "
        "applying a mean offset to the current split."
    )
    noise_level = st.slider(
        "Noise level (std)",
        min_value=0.0,
        max_value=3.0,
        value=0.5,
        step=0.1,
        help="Gaussian noise added to features in the current window.",
    )
    mean_shift = st.slider(
        "Mean shift",
        min_value=0.0,
        max_value=2.0,
        value=0.0,
        step=0.1,
        help="Constant offset added to all features in the current window.",
    )
    alpha = st.select_slider(
        "Significance level (alpha)",
        options=[0.01, 0.05, 0.10],
        value=float(MONITORING_CFG.get("drift_significance_level", 0.05)),
    )

    st.divider()
    run_btn = st.button("Run Drift Analysis", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Load baseline preview dataset
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_preview(key: str):
    return load_dataset(key)


with st.spinner("Loading reference dataset..."):
    try:
        ds = _load_preview(dataset_key)
    except Exception as exc:
        st.error(f"Failed to load dataset: {exc}")
        st.stop()

reference = ds["X_train"].copy()
current = ds["X_test"].copy()

# ---------------------------------------------------------------------------
# Split preview
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Reference samples", f"{len(reference):,}")
c2.metric("Current samples",   f"{len(current):,}")
c3.metric("Features",          reference.shape[1])
c4.metric("Significance level", f"{alpha}")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Run analysis on button press or show cached result
# ---------------------------------------------------------------------------
if "drift_result" not in st.session_state or run_btn:
    if run_btn or "drift_result" not in st.session_state:
        with st.spinner("Running statistical tests..."):
            payload = run_drift_and_persist(
                dataset_label=dataset_label,
                dataset_key=dataset_key,
                noise_level=float(noise_level),
                mean_shift=float(mean_shift),
                alpha=float(alpha),
            )

        report = payload["report"]
        st.session_state["drift_result"]   = report
        st.session_state["drift_reference"] = payload["reference"].copy()
        st.session_state["drift_current"]   = payload["current"].copy()
        st.session_state["drift_features"]  = payload["feature_names"]

report     = st.session_state.get("drift_result")
ref_arr    = st.session_state.get("drift_reference", reference)
cur_arr    = st.session_state.get("drift_current",   current)
feat_names = st.session_state.get("drift_features",  ds["feature_names"])

if report is None:
    st.info("Click **Run Drift Analysis** to start.")
    st.stop()

# ---------------------------------------------------------------------------
# Summary alert
# ---------------------------------------------------------------------------
overall_severity = report.get("overall_severity", "stable")
if overall_severity == "critical":
    severity_msg = (
        f"Critical drift detected — {report['features_drifted']} of "
        f"{report['features_analyzed']} features shifted (avg PSI = {report['average_psi']:.4f})."
    )
    st.markdown(f'<div class="alert-box alert-red">{severity_msg}</div>', unsafe_allow_html=True)
elif overall_severity == "warning":
    severity_msg = (
        f"Warning: moderate drift observed — drift ratio {report['drift_ratio']:.1%} "
        f"(avg PSI = {report['average_psi']:.4f})."
    )
    st.markdown(f'<div class="alert-box alert-yellow">{severity_msg}</div>', unsafe_allow_html=True)
else:
    st.markdown(
        f'<div class="alert-box alert-green">No significant drift detected. '
        f'Avg PSI = {report["average_psi"]:.4f}.</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Features Analyzed", report["features_analyzed"])
k2.metric("Features Drifted",  report["features_drifted"])
k3.metric("Drift Ratio",       f"{report['drift_ratio']:.1%}")
k4.metric("Avg PSI",           f"{report['average_psi']:.4f}")

sev_counts = pd.DataFrame(report["feature_results"])["severity"].value_counts() if report["feature_results"] else pd.Series(dtype=int)
sc1, sc2, sc3 = st.columns(3)
sc1.metric("Critical Features", int(sev_counts.get("significant", 0)))
sc2.metric("Warning Features", int(sev_counts.get("moderate", 0)))
sc3.metric("Stable Features", int(sev_counts.get("none", 0)))

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Per-feature results
# ---------------------------------------------------------------------------
tab_table, tab_dist, tab_psi, tab_history = st.tabs(
    ["Feature Results", "Distribution Comparison", "PSI Breakdown", "Report History"]
)

with tab_table:
    st.markdown('<div class="section-title">Feature-Level Drift Results</div>', unsafe_allow_html=True)

    feat_df = pd.DataFrame(report["feature_results"])
    if feat_df.empty:
        st.info("No feature results available.")
    else:
        styled = feat_df.copy()
        styled["drift_detected"] = styled["drift_detected"].map({True: "Yes", False: "No"})
        styled.columns = [c.replace("_", " ").title() for c in styled.columns]
        st.dataframe(
            styled.style.apply(
                lambda row: [
                    "background-color: #fee2e2" if row["Drift Detected"] == "Yes" else ""
                    for _ in row
                ],
                axis=1,
            ),
            use_container_width=True,
            hide_index=True,
        )

        affected = feat_df[feat_df["drift_detected"]].sort_values("psi", ascending=False)
        if not affected.empty:
            st.markdown('<div class="section-title" style="margin-top:20px">Most Affected Features</div>', unsafe_allow_html=True)
            st.dataframe(
                affected[["feature", "severity", "psi", "p_value"]].head(10),
                use_container_width=True,
                hide_index=True,
            )

        # PSI bar chart
        st.markdown('<div class="section-title" style="margin-top:20px">PSI by Feature</div>', unsafe_allow_html=True)
        psi_sorted = feat_df.sort_values("psi", ascending=False).head(20)
        bar_colors = psi_sorted["severity"].map(
            {"significant": "#ef4444", "moderate": "#f59e0b", "none": "#22c55e"}
        )
        fig = go.Figure(
            go.Bar(
                x=psi_sorted["psi"],
                y=psi_sorted["feature"],
                orientation="h",
                marker_color=bar_colors.tolist(),
                marker_line_width=0,
            )
        )
        fig.add_vline(x=0.10, line_dash="dot", line_color="#f59e0b",
                      annotation_text="Moderate (0.10)", annotation_position="top right")
        fig.add_vline(x=0.25, line_dash="dot", line_color="#ef4444",
                      annotation_text="Significant (0.25)", annotation_position="top right")
        fig.update_layout(
            height=max(300, len(psi_sorted) * 22),
            margin=dict(l=0, r=80, t=10, b=0),
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(title="PSI", showgrid=True, gridcolor="#f1f5f9"),
            yaxis=dict(title=None, showgrid=False, tickfont=dict(size=11)),
        )
        st.plotly_chart(fig, use_container_width=True)


with tab_dist:
    st.markdown('<div class="section-title">Distribution Overlay — Reference vs Current</div>', unsafe_allow_html=True)
    st.caption("Select features to compare their reference (blue) and current (orange) distributions.")

    ref_df = pd.DataFrame(ref_arr, columns=feat_names)
    cur_df = pd.DataFrame(cur_arr, columns=feat_names)

    selected = st.multiselect(
        "Features",
        options=feat_names,
        default=feat_names[:4],
        max_selections=9,
    )

    if not selected:
        st.info("Select at least one feature.")
    else:
        n_cols = min(3, len(selected))
        rows_list = [selected[i:i + n_cols] for i in range(0, len(selected), n_cols)]

        for row_feats in rows_list:
            cols = st.columns(n_cols)
            for col_widget, feat in zip(cols, row_feats):
                with col_widget:
                    fig = go.Figure()
                    fig.add_trace(
                        go.Histogram(
                            x=ref_df[feat],
                            name="Reference",
                            opacity=0.65,
                            nbinsx=30,
                            marker_color="#2563eb",
                        )
                    )
                    fig.add_trace(
                        go.Histogram(
                            x=cur_df[feat],
                            name="Current",
                            opacity=0.65,
                            nbinsx=30,
                            marker_color="#f97316",
                        )
                    )
                    fig.update_layout(
                        barmode="overlay",
                        height=220,
                        margin=dict(l=0, r=0, t=28, b=0),
                        paper_bgcolor="white",
                        plot_bgcolor="white",
                        title=dict(text=feat, font=dict(size=11)),
                        xaxis=dict(showgrid=False, title=None),
                        yaxis=dict(showgrid=False, title=None, showticklabels=False),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.0,
                            xanchor="right",
                            x=1,
                            font=dict(size=9),
                        ),
                        showlegend=(feat == selected[0]),
                    )
                    st.plotly_chart(fig, use_container_width=True)


with tab_psi:
    st.markdown('<div class="section-title">PSI Interpretation Guide</div>', unsafe_allow_html=True)

    guide_data = {
        "PSI Range": ["< 0.10", "0.10 – 0.25", "> 0.25"],
        "Interpretation": [
            "No significant change — distribution is stable",
            "Moderate shift — review and monitor closely",
            "Significant change — model retraining recommended",
        ],
        "Recommended Action": [
            "Continue monitoring; no immediate action required",
            "Investigate root cause; consider retraining soon",
            "Retrain on recent data; audit upstream data pipeline",
        ],
    }
    st.dataframe(pd.DataFrame(guide_data), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title" style="margin-top:24px">KS Test p-value Heatmap</div>', unsafe_allow_html=True)
    feat_df_psi = pd.DataFrame(report["feature_results"]).sort_values("psi", ascending=False)
    if not feat_df_psi.empty:
        top20 = feat_df_psi.head(20)
        fig = go.Figure(
            go.Bar(
                x=top20["feature"],
                y=top20["p_value"],
                marker_color=[
                    "#ef4444" if p < alpha else "#22c55e" for p in top20["p_value"]
                ],
                text=[f"{p:.4f}" for p in top20["p_value"]],
                textposition="outside",
            )
        )
        fig.add_hline(y=alpha, line_dash="dot", line_color="#374151",
                      annotation_text=f"alpha = {alpha}")
        fig.update_layout(
            height=340,
            margin=dict(l=0, r=0, t=10, b=60),
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(showgrid=False, tickangle=-30, title=None),
            yaxis=dict(showgrid=True, gridcolor="#f1f5f9", title="p-value"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)


with tab_history:
    st.markdown('<div class="section-title">Past Drift Reports</div>', unsafe_allow_html=True)

    history = get_drift_reports(limit=30)
    if not history:
        st.info("No historical drift reports found.")
    else:
        hist_df = pd.DataFrame(history)[
            ["report_id", "dataset", "reference_size", "current_size",
             "drift_detected", "features_drifted", "drift_score", "created_at"]
        ]
        hist_df["drift_detected"] = hist_df["drift_detected"].map({1: "Yes", 0: "No"})
        hist_df["drift_score"]    = hist_df["drift_score"].apply(
            lambda x: f"{x:.4f}" if x is not None else "—"
        )
        hist_df.columns = [c.replace("_", " ").title() for c in hist_df.columns]
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
