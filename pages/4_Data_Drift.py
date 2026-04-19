"""
Data Drift Detection page.

Compares a reference dataset split against a perturbed (simulated current)
distribution using the Kolmogorov-Smirnov test and Population Stability Index.
Operators can control the perturbation intensity to explore drift thresholds.
"""
from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from services.app_service import initialize_application
from services.drift_service import (
    get_dataset_options,
    get_drift_preview_dataset,
    get_monitoring_defaults,
    list_drift_reports,
    run_drift_and_persist,
)
from services.telemetry_service import track_user_action
from src.auth import can_run_pipeline, render_auth_controls
from src.ui_theme import (
    apply_plotly_layout,
    apply_ui_theme,
    render_alert,
    render_expandable_rows,
    render_loading_skeleton,
    render_page_header_with_action,
    render_section_title,
    render_sidebar_brand,
    render_sidebar_nav,
    render_summary_table,
    status_badge_html,
)

st.set_page_config(page_title="Data Drift | ML Monitor", layout="wide")
initialize_application()
apply_ui_theme()
MONITORING_CFG = get_monitoring_defaults()
DATASET_OPTIONS = get_dataset_options()

# ---------------------------------------------------------------------------
# Sidebar â€” configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    render_sidebar_brand()
    st.markdown("### Navigation")
    render_sidebar_nav()
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
    can_execute_drift = can_run_pipeline()
    run_btn = st.button(
        "Run Drift Analysis",
        type="primary",
        width="stretch",
        disabled=not can_execute_drift,
    )
    if not can_execute_drift:
        st.caption("Read-only role: operator or admin is required to run drift analysis.")

    st.divider()
    st.markdown("### Access")
    render_auth_controls()

run_btn_top = render_page_header_with_action(
    "Data drift detection",
    "Compare reference and current data distributions using KS test and PSI.",
    "Run Drift Analysis",
    action_key="drift_run_top",
    disabled=not can_execute_drift,
)
run_btn = bool(run_btn or run_btn_top)
if run_btn_top:
    track_user_action("data_drift", "run_drift_clicked_top", {"dataset": dataset_label})

# ---------------------------------------------------------------------------
# Load baseline preview dataset
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_preview(key: str):
    return get_drift_preview_dataset(key)


sk = st.empty()
with sk.container():
    render_loading_skeleton(lines=3, key="drift_load")
with st.spinner("Loading reference dataset..."):
    try:
        ds = _load_preview(dataset_key)
    except Exception as exc:
        st.error("Failed to load dataset preview.")
        st.caption(f"Details: {exc}")
        if st.button("Retry", type="primary", key="drift_preview_retry"):
            st.rerun()
        st.stop()
sk.empty()

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
        if not can_run_pipeline():
            st.error("Permission denied: operator or admin role required.")
            st.stop()
        track_user_action("data_drift", "run_drift_execute", {"dataset": dataset_label})
        prog = st.progress(0.0)
        with st.spinner("Running statistical tests..."):
            try:
                prog.progress(0.25)
                payload = run_drift_and_persist(
                    dataset_label=dataset_label,
                    dataset_key=dataset_key,
                    noise_level=float(noise_level),
                    mean_shift=float(mean_shift),
                    alpha=float(alpha),
                )
                prog.progress(1.0)
            except Exception as exc:
                prog.empty()
                st.error("Drift analysis failed. Please retry.")
                st.caption(f"Details: {exc}")
                if st.button("Retry drift analysis", type="primary", key="drift_retry_run"):
                    st.rerun()
                st.stop()
        prog.empty()

        report = payload["report"]
        st.session_state["drift_result"]   = report
        st.session_state["drift_reference"] = payload["reference"].copy()
        st.session_state["drift_current"]   = payload["current"].copy()
        st.session_state["drift_features"]  = payload["feature_names"]
        st.success("Drift analysis completed successfully.")

report     = st.session_state.get("drift_result")
ref_arr    = st.session_state.get("drift_reference", reference)
cur_arr    = st.session_state.get("drift_current",   current)
feat_names = st.session_state.get("drift_features",  ds["feature_names"])

if report is None:
    st.info("Click **Run Drift Analysis** to start.")
    st.stop()

st.markdown(
        """
        <style>
            .drift-kpi-row { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin-top: 8px; }
            .drift-kpi {
                border-radius: 12px;
                padding: 12px;
                border: 1px solid #e7eaf0;
                background: #ffffff;
                box-shadow: 0 3px 10px rgba(15, 23, 42, 0.04);
                transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
            }
            .drift-kpi:hover { transform: translateY(-1px); box-shadow: 0 10px 18px rgba(15, 23, 42, 0.10); }
            .drift-kpi-label { font-size: .74rem; color: #64748b; text-transform: uppercase; letter-spacing: .03em; }
            .drift-kpi-value { margin-top: 6px; font-size: 1.25rem; font-weight: 800; line-height: 1.1; color: #0f172a; }
            .drift-kpi-sub { margin-top: 4px; font-size: .75rem; color: #64748b; }

            .drift-kpi.stable { border-color: #86efac; background: linear-gradient(180deg, #ffffff 0%, #f0fdf4 100%); }
            .drift-kpi.warning { border-color: #fdba74; background: linear-gradient(180deg, #ffffff 0%, #fff7ed 100%); }
            .drift-kpi.critical { border-color: #fca5a5; background: linear-gradient(180deg, #ffffff 0%, #fef2f2 100%); }

            .feat-stack { display: grid; gap: 10px; margin-top: 8px; }
            .feat-item {
                border: 1px solid #e8ecf2;
                border-radius: 11px;
                padding: 10px;
                background: #fff;
                transition: transform .18s ease, box-shadow .18s ease;
            }
            .feat-item:hover { transform: translateY(-1px); box-shadow: 0 8px 16px rgba(15,23,42,.08); }
            .feat-top { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 7px; }
            .feat-name { font-size: .84rem; font-weight: 700; color: #0f172a; }
            .feat-metas { font-size: .75rem; color: #64748b; text-align: right; }
            .feat-bar-wrap { height: 10px; border-radius: 999px; background: #f1f5f9; overflow: hidden; }
            .feat-bar { height: 100%; border-radius: 999px; transition: width .35s ease; }
            .feat-bar.stable { background: linear-gradient(90deg, #34d399 0%, #22c55e 100%); }
            .feat-bar.monitor { background: linear-gradient(90deg, #fbbf24 0%, #f59e0b 100%); }
            .feat-bar.retrain { background: linear-gradient(90deg, #fb7185 0%, #ef4444 100%); }
            .feat-badge {
                display: inline-flex; align-items: center; gap: 4px; padding: 2px 8px; border-radius: 999px;
                font-size: .72rem; font-weight: 700;
            }
            .feat-badge.stable { background: #e8fbe9; color: #166534; }
            .feat-badge.monitor { background: #fff3e0; color: #9a3412; }
            .feat-badge.retrain { background: #fee2e2; color: #991b1b; }

            .psi-guide-wrap { display: grid; gap: 10px; margin-top: 6px; }
            .psi-guide {
                border: 1px solid #e7ebf3; border-radius: 11px; padding: 10px;
                background: #fff; transition: box-shadow .18s ease, transform .18s ease;
            }
            .psi-guide:hover { transform: translateY(-1px); box-shadow: 0 8px 16px rgba(15,23,42,.08); }
            .psi-chip {
                display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: .72rem; font-weight: 800;
                text-transform: uppercase; letter-spacing: .03em;
            }
            .psi-chip.stable { background: #e8fbe9; color: #166534; }
            .psi-chip.monitor { background: #fff3e0; color: #9a3412; }
            .psi-chip.retrain { background: #fee2e2; color: #991b1b; }
            .psi-desc { margin-top: 6px; font-size: .8rem; color: #475569; }
        </style>
        """,
        unsafe_allow_html=True,
)

feat_df_all = pd.DataFrame(report.get("feature_results", []))
if not feat_df_all.empty:
    severity_weight = feat_df_all["severity"].map({"none": 0.0, "moderate": 0.5, "significant": 1.0}).fillna(0.0)
    psi_norm = (feat_df_all["psi"] / 0.25).clip(0, 2)
    ks_norm = (feat_df_all["ks_statistic"] / 0.20).clip(0, 2)
    feat_df_all["drift_priority_score"] = (0.45 * psi_norm + 0.35 * ks_norm + 0.20 * severity_weight).round(4)
else:
    feat_df_all = pd.DataFrame(
        columns=["feature", "severity", "drift_detected", "psi", "p_value", "ks_statistic", "drift_priority_score"]
    )

critical_count = int((feat_df_all["severity"] == "significant").sum()) if not feat_df_all.empty else 0
moderate_count = int((feat_df_all["severity"] == "moderate").sum()) if not feat_df_all.empty else 0
critical_ratio = (critical_count / report["features_analyzed"]) if report.get("features_analyzed") else 0.0
retrain_score = min(
    1.0,
    0.45 * float(report.get("drift_ratio", 0.0))
    + 0.35 * min(float(report.get("average_psi", 0.0)) / 0.25, 1.0)
    + 0.20 * critical_ratio,
)
retrain_suggested = (
    report.get("overall_severity") == "critical"
    or critical_count > 0
    or retrain_score >= 0.60
)
confidence_pct = retrain_score * 100.0
confidence_label = "High" if retrain_score >= 0.75 else "Medium" if retrain_score >= 0.50 else "Low"

recommended_actions = []
if retrain_suggested:
    recommended_actions.append("Retrain model with fresh data window and validate against baseline metrics.")
if critical_count > 0:
    recommended_actions.append("Review top significant-drift features for upstream data pipeline and feature engineering changes.")
if moderate_count > 0:
    recommended_actions.append("Increase monitoring cadence for moderate-drift features and set alert thresholds.")
if not recommended_actions:
    recommended_actions.append("No immediate intervention required; continue routine monitoring.")

# ---------------------------------------------------------------------------
# Summary alert
# ---------------------------------------------------------------------------
overall_severity = report.get("overall_severity", "stable")
if overall_severity == "critical":
    severity_msg = (
        f"Critical drift detected â€” {report['features_drifted']} of "
        f"{report['features_analyzed']} features shifted (avg PSI = {report['average_psi']:.4f})."
    )
    render_alert(severity_msg, tone="danger")
elif overall_severity == "warning":
    severity_msg = (
        f"Warning: moderate drift observed â€” drift ratio {report['drift_ratio']:.1%} "
        f"(avg PSI = {report['average_psi']:.4f})."
    )
    render_alert(severity_msg, tone="warning")
else:
    render_alert(
        f'No significant drift detected. Avg PSI = {report["average_psi"]:.4f}.',
        tone="success",
    )

render_section_title("Retrain Indicator", margin_top_px=10)
badge_tone = "danger" if retrain_suggested else "success"
badge_text = "Retrain Suggested" if retrain_suggested else "Retrain Not Required"
st.markdown(f"<span class='ui-badge {badge_tone}'>{badge_text}</span>", unsafe_allow_html=True)

r_left, r_mid, r_right = st.columns([1, 1, 2])
r_left.metric("Confidence", f"{confidence_pct:.1f}%")
r_mid.metric("Confidence Level", confidence_label)
with r_right:
    st.markdown("**Recommended actions**")
    for action in recommended_actions:
        st.markdown(f"- {action}")

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
verdict_class = "critical" if overall_severity == "critical" else "warning" if overall_severity == "warning" else "stable"
verdict_text = "High Drift" if overall_severity == "critical" else "Moderate Drift" if overall_severity == "warning" else "Stable"
st.markdown(
        f"""
        <div class="drift-kpi-row">
            <div class="drift-kpi {verdict_class}" title="Total numeric features evaluated for drift.">
                <div class="drift-kpi-label">Features Analyzed</div>
                <div class="drift-kpi-value">{report['features_analyzed']}</div>
                <div class="drift-kpi-sub">reference vs current</div>
            </div>
            <div class="drift-kpi {verdict_class}" title="Number of features flagged by KS or PSI thresholds.">
                <div class="drift-kpi-label">Features Drifted</div>
                <div class="drift-kpi-value">{report['features_drifted']}</div>
                <div class="drift-kpi-sub">ratio: {report['drift_ratio']:.1%}</div>
            </div>
            <div class="drift-kpi {verdict_class}" title="Average PSI across all analyzed features.">
                <div class="drift-kpi-label">Avg PSI</div>
                <div class="drift-kpi-value">{report['average_psi']:.4f}</div>
                <div class="drift-kpi-sub">stable < 0.10, retrain > 0.25</div>
            </div>
            <div class="drift-kpi {verdict_class}" title="Overall drift verdict from feature ratio and PSI severity.">
                <div class="drift-kpi-label">Drift Verdict</div>
                <div class="drift-kpi-value">{verdict_text}</div>
                <div class="drift-kpi-sub">confidence {confidence_pct:.1f}%</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Per-feature results
# ---------------------------------------------------------------------------
load_drift_charts = st.toggle(
    "Load heavy drift visualizations",
    value=False,
    help="Enable advanced distribution and PSI charts when needed.",
)

tab_table, tab_dist, tab_psi, tab_history = st.tabs(
    ["Feature Results", "Distribution Comparison", "PSI Breakdown", "Report History"]
)

with tab_table:
    render_section_title("Feature-Level Drift Results")

    feat_df = feat_df_all.copy()
    if feat_df.empty:
        st.info("No feature results available.")
    else:
        rank_metric = st.selectbox(
            "Feature importance metric",
            options=["PSI", "KS Statistic", "Composite Severity"],
            index=0,
            help="Ranks feature drift severity using PSI, KS statistic, or a blended score.",
        )

        if rank_metric == "KS Statistic":
            ranked = feat_df.sort_values("ks_statistic", ascending=False).copy()
        elif rank_metric == "Composite Severity":
            ranked = feat_df.sort_values("drift_priority_score", ascending=False).copy()
        else:
            ranked = feat_df.sort_values("psi", ascending=False).copy()

        render_section_title("Feature Importance Ranking", margin_top_px=12)
        ranked_view = ranked[["feature", "severity", "psi", "ks_statistic", "p_value", "drift_priority_score"]].copy()
        ranked_view.insert(0, "rank", range(1, len(ranked_view) + 1))
        ranked_view.columns = ["Rank", "Feature", "Severity", "PSI", "KS Statistic", "P Value", "Priority Score"]
        render_summary_table(
            ranked_view,
            key_prefix="drift_feature_rank",
            columns=ranked_view.columns.tolist(),
            sort_by="Rank",
            filterable_columns=["Severity"],
            max_rows=25,
        )

        render_section_title("Feature Drift Health Bars", margin_top_px=12)
        max_psi = max(float(ranked["psi"].max()), 0.25)
        bar_rows = []
        for _, r in ranked.head(18).iterrows():
            sev = str(r.get("severity", "none"))
            sev_key = "retrain" if sev == "significant" else "monitor" if sev == "moderate" else "stable"
            width_pct = max(2.0, min(100.0, float(r.get("psi", 0.0)) / max_psi * 100.0))
            drift_badge = "drifted" if bool(r.get("drift_detected", False)) else "stable"
            tooltip = (
                f"Feature: {r.get('feature')} | PSI: {float(r.get('psi', 0.0)):.4f} | "
                f"KS: {float(r.get('ks_statistic', 0.0)):.4f} | p-value: {float(r.get('p_value', 0.0)):.4f}"
            )
            bar_rows.append(
                "<div class='feat-item' title='" + escape(tooltip) + "'>"
                "<div class='feat-top'>"
                "<div class='feat-name'>" + escape(str(r.get("feature", ""))) + "</div>"
                "<div class='feat-metas'>"
                f"PSI {float(r.get('psi', 0.0)):.4f} | KS {float(r.get('ks_statistic', 0.0)):.4f} | "
                f"<span class='feat-badge {sev_key}'>{drift_badge}</span>"
                "</div></div>"
                "<div class='feat-bar-wrap'>"
                f"<div class='feat-bar {sev_key}' style='width:{width_pct:.1f}%'></div>"
                "</div></div>"
            )

        st.markdown("<div class='feat-stack'>" + "".join(bar_rows) + "</div>", unsafe_allow_html=True)

        affected = feat_df[feat_df["drift_detected"]].sort_values("psi", ascending=False)
        if not affected.empty:
            render_section_title("Most Affected Features", margin_top_px=20)
            preview = affected[["feature", "severity", "psi", "p_value"]].head(10).copy()
            preview.columns = ["Feature", "Severity", "PSI", "P Value"]
            preview["Severity Badge"] = preview["Severity"].apply(status_badge_html)
            st.dataframe(preview[["Feature", "Severity", "PSI", "P Value"]], width="stretch", hide_index=True)

        # PSI bar chart
        render_section_title("PSI by Feature", margin_top_px=20)
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
            margin=dict(r=80),
        )
        apply_plotly_layout(fig, height=max(300, len(psi_sorted) * 22), x_title="PSI")
        fig.update_yaxes(showgrid=False, tickfont=dict(size=11))
        st.plotly_chart(fig, width="stretch")


with tab_dist:
    if not load_drift_charts:
        st.info("Enable 'Load heavy drift visualizations' above to render distribution overlays.")
    else:
        render_section_title("Distribution Overlay - Reference vs Current")
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
                        st.plotly_chart(fig, width="stretch")


with tab_psi:
    if not load_drift_charts:
        st.info("Enable 'Load heavy drift visualizations' above to render PSI charts.")
    else:
                st.markdown('<div class="section-title">PSI Interpretation Guide</div>', unsafe_allow_html=True)
                st.markdown(
                        """
                        <div class='psi-guide-wrap'>
                            <div class='psi-guide' title='PSI below 0.10 usually indicates healthy stability.'>
                                <span class='psi-chip stable'>Stable</span>
                                <div class='psi-desc'><b>PSI &lt; 0.10</b> â€” distribution remains healthy. Continue baseline monitoring.</div>
                            </div>
                            <div class='psi-guide' title='PSI between 0.10 and 0.25 requires closer observation and root-cause checks.'>
                                <span class='psi-chip monitor'>Monitor</span>
                                <div class='psi-desc'><b>0.10 â‰¤ PSI â‰¤ 0.25</b> â€” moderate shift. Review data quality and feature behavior.</div>
                            </div>
                            <div class='psi-guide' title='PSI above 0.25 often suggests meaningful distribution change and retraining need.'>
                                <span class='psi-chip retrain'>Retrain</span>
                                <div class='psi-desc'><b>PSI &gt; 0.25</b> â€” significant drift. Plan retraining and upstream data audit.</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                )

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
            st.plotly_chart(fig, width="stretch")


with tab_history:
    st.markdown('<div class="section-title">Past Drift Reports</div>', unsafe_allow_html=True)

    history = list_drift_reports(limit=30)
    if not history:
        st.info("No historical drift reports found.")
    else:
        hist_df = pd.DataFrame(history)[
            ["report_id", "dataset", "reference_size", "current_size",
             "drift_detected", "features_drifted", "drift_score", "created_at"]
        ]
        hist_df["drift_detected"] = hist_df["drift_detected"].map({1: "Yes", 0: "No"})
        hist_df["drift_score"]    = hist_df["drift_score"].apply(
            lambda x: f"{x:.4f}" if x is not None else "â€”"
        )
        hist_df.columns = [c.replace("_", " ").title() for c in hist_df.columns]
        st.dataframe(hist_df, width="stretch", hide_index=True)

