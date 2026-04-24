"""Shared UI theme helpers that mirror the provided frontend demo."""

from __future__ import annotations

from html import escape
from typing import Iterable, Sequence

import plotly.io as pio
import plotly.graph_objects as go
import pandas as pd
import streamlit as st


DESIGN_TOKENS = {
    "spacing": {
        "xs": "8px",
        "sm": "16px",
        "md": "24px",
        "lg": "32px",
        "xl": "40px",
    },
    "font_sizes": {
        "h1": "32px",
        "h2": "20px",
        "body": "14px",
        "caption": "12px",
    },
    "colors": {
        "background_primary": "#f6f7fb",
        "background_secondary": "#ffffff",
        "border_primary": "#e6e8ef",
        "border_secondary": "#d8dce7",
        "text_primary": "#151923",
        "text_secondary": "#3a4357",
        "text_tertiary": "#697387",
        "accent": "#4f46e5",
        "accent_soft": "#eef0ff",
    },
    "radius": {
        "sm": "10px",
        "md": "12px",
        "lg": "16px",
        "pill": "999px",
    },
    "shadows": {
        "none": "none",
        "sm": "0 1px 2px rgba(17, 24, 39, 0.04)",
    },
    "semantic": {
        "success": {"bg": "#edf8f1", "fg": "#256f45", "border": "#c9e8d2"},
        "warning": {"bg": "#fff6e8", "fg": "#8a5a12", "border": "#f0ddbc"},
        "danger": {"bg": "#fff0f1", "fg": "#9b3a46", "border": "#f3cdd3"},
        "info": {"bg": "#eef2ff", "fg": "#3f46a8", "border": "#d9ddff"},
        "neutral": {"bg": "#f3f5f9", "fg": "#4d5970", "border": "#dfe4ee"},
    },
}

STAGE_TO_TONE = {
    "production": "success",
    "staging": "info",
    "development": "neutral",
    "archived": "neutral",
}

STATUS_TO_TONE = {
    "completed": "success",
    "success": "success",
    "done": "success",
    "warning": "warning",
    "moderate": "warning",
    "failed": "danger",
    "error": "danger",
    "critical": "danger",
    "pending": "info",
    "running": "info",
    "queued": "neutral",
    "skipped": "neutral",
    "none": "neutral",
    "stable": "success",
    "significant": "danger",
}

PLOTLY_COLORWAY = [
    "#4f46e5",
    "#0ea5a4",
    "#64748b",
    "#22c55e",
    "#f59e0b",
    "#ef4444",
]


def apply_plotly_defaults() -> None:
    """Register and activate a project-wide Plotly template."""
    if "mlmonitor" not in pio.templates:
        base = pio.templates["plotly_white"]
        template_dict = base.to_plotly_json() if hasattr(base, "to_plotly_json") else {}
        layout_config = template_dict.get("layout", {}) if isinstance(template_dict, dict) else {}
        layout_config.update(
            {
                "colorway": PLOTLY_COLORWAY,
                "font": {"family": "Inter, Segoe UI, sans-serif", "size": 12, "color": "#1f1f1e"},
                "plot_bgcolor": "#ffffff",
                "paper_bgcolor": "#ffffff",
                "legend": {"title": None, "orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0},
                "margin": {"l": 0, "r": 0, "t": 12, "b": 0},
                "hoverlabel": {"bgcolor": "#ffffff", "bordercolor": "#d8dce7", "font": {"color": "#151923", "size": 12}},
                "xaxis": {
                    "showgrid": False,
                    "zeroline": False,
                    "title": None,
                    "linecolor": "#e6e8ef",
                    "tickfont": {"color": "#697387", "size": 11},
                },
                "yaxis": {
                    "showgrid": True,
                    "gridcolor": "#f1f3f8",
                    "gridwidth": 1,
                    "zeroline": False,
                    "title": None,
                    "linecolor": "#e6e8ef",
                    "tickfont": {"color": "#697387", "size": 11},
                },
            }
        )
        pio.templates["mlmonitor"] = go.layout.Template(layout=layout_config)
    pio.templates.default = "mlmonitor"


def apply_plotly_layout(fig, *, height: int = 340, x_title: str | None = None, y_title: str | None = None):
    """Apply shared layout polish to a Plotly figure."""
    fig.update_layout(
        height=height,
        transition={"duration": 350, "easing": "cubic-in-out"},
        hovermode="x unified",
    )
    if x_title is not None:
        fig.update_xaxes(title=x_title)
    if y_title is not None:
        fig.update_yaxes(title=y_title)
    return fig


def apply_ui_theme() -> None:
    """Apply a compact demo-like visual language across all Streamlit pages."""
    st.markdown(
        """
        <style>
            :root {
                --space-xs: 8px;
                --space-sm: 16px;
                --space-md: 24px;
                --space-lg: 32px;
                --space-xl: 40px;

                --font-h1: 32px;
                --font-h2: 20px;
                --font-body: 14px;
                --font-caption: 12px;

                --color-background-primary: #f6f7fb;
                --color-background-secondary: #ffffff;
                --color-border-tertiary: #e6e8ef;
                --color-border-secondary: #d8dce7;
                --color-text-primary: #151923;
                --color-text-secondary: #3a4357;
                --color-text-tertiary: #697387;
                --color-accent: #4f46e5;
                --color-accent-soft: #eef0ff;

                --radius-sm: 10px;
                --radius-md: 12px;
                --radius-lg: 16px;
                --radius-pill: 999px;

                --shadow-sm: 0 1px 2px rgba(17, 24, 39, 0.04);

                --semantic-success-bg: #edf8f1;
                --semantic-success-fg: #256f45;
                --semantic-success-border: #c9e8d2;

                --semantic-warning-bg: #fff6e8;
                --semantic-warning-fg: #8a5a12;
                --semantic-warning-border: #f0ddbc;

                --semantic-danger-bg: #fff0f1;
                --semantic-danger-fg: #9b3a46;
                --semantic-danger-border: #f3cdd3;

                --semantic-info-bg: #eef2ff;
                --semantic-info-fg: #3f46a8;
                --semantic-info-border: #d9ddff;

                --semantic-neutral-bg: #f3f5f9;
                --semantic-neutral-fg: #4d5970;
                --semantic-neutral-border: #dfe4ee;
            }

            html, body, [class*="css"] {
                font-family: "Inter", "Manrope", "Segoe UI", sans-serif;
                color: var(--color-text-primary);
            }

            .stApp {
                background: var(--color-background-primary);
            }

            .block-container {
                padding-top: 1.5rem;
                padding-bottom: 1.25rem;
                max-width: none;
            }

            [data-testid="stSidebar"] {
                background: var(--color-background-secondary);
                border-right: 1px solid var(--color-border-tertiary);
                min-width: 220px;
                width: 220px;
            }

            [data-testid="stSidebar"] .block-container {
                padding-top: 1rem;
                padding-left: 0.8rem;
                padding-right: 0.8rem;
            }

            /* Hide Streamlit's default multipage list to avoid duplicate navigation. */
            [data-testid="stSidebarNav"] {
                display: none;
            }

            [data-testid="stSidebar"] .stPageLink a,
            [data-testid="stSidebar"] .stPageLink span {
                font-size: 12px;
                color: var(--color-text-secondary);
                text-decoration: none;
            }

            [data-testid="stSidebar"] .stPageLink > div {
                border-radius: 0;
                padding: 2px 2px;
            }

            [data-testid="stSidebar"] .stPageLink > div > a {
                display: flex;
                align-items: center;
                gap: 8px;
                padding: 7px 10px;
                border-radius: 0;
                transition: background-color 0.15s ease;
            }

            [data-testid="stSidebar"] .stPageLink > div > a::before {
                content: "";
                width: 6px;
                height: 6px;
                border-radius: 50%;
                background: currentColor;
                flex-shrink: 0;
            }

            [data-testid="stSidebar"] .stPageLink > div > a:hover {
                background: var(--color-background-secondary);
            }

            [data-testid="stSidebar"] .stPageLink a[aria-current="page"] {
                color: var(--color-accent);
                background: var(--color-accent-soft);
                font-weight: 500;
            }

            [data-testid="metric-container"] {
                background: var(--color-background-secondary);
                border: 1px solid var(--color-border-tertiary);
                border-radius: var(--radius-md);
                padding: 14px;
                box-shadow: var(--shadow-sm);
            }

            [data-testid="stMetricLabel"] {
                color: var(--color-text-tertiary);
                font-size: 11px;
                font-weight: 400;
            }

            [data-testid="stMetricValue"] {
                color: var(--color-text-primary);
                font-size: 22px;
                font-weight: 500;
                line-height: 1.1;
            }

            [data-testid="stMetricDelta"] {
                font-size: 11px;
            }

            [data-testid="stDataFrame"] {
                border: 0.5px solid var(--color-border-tertiary);
                border-radius: var(--radius-md);
                overflow: hidden;
            }

            .page-header {
                margin-bottom: 20px;
            }

            .page-title {
                font-size: var(--font-h1);
                font-weight: 650;
                line-height: 1.1;
                color: var(--color-text-primary);
            }

            .page-sub {
                font-size: var(--font-body);
                color: var(--color-text-tertiary);
                margin-top: 8px;
            }

            .section-title {
                font-size: var(--font-h2);
                font-weight: 600;
                color: var(--color-text-secondary);
                margin-bottom: 12px;
                text-transform: none;
                letter-spacing: 0;
            }

            .sidebar-brand {
                padding: 0 4px 12px 4px;
                border-bottom: 0.5px solid var(--color-border-tertiary);
                margin-bottom: 8px;
            }

            .brand-title {
                font-size: 13px;
                color: var(--color-text-primary);
                font-weight: 500;
            }

            .brand-sub {
                font-size: 10px;
                color: var(--color-text-tertiary);
                margin-top: 2px;
            }

            .stage-row {
                display: flex;
                align-items: center;
                gap: 10px;
                padding: 8px 12px;
                border: 0.5px solid var(--color-border-tertiary);
                border-radius: var(--radius-md);
                margin-bottom: 6px;
                background: var(--color-background-primary);
            }

            .stage-icon {
                width: 20px;
                height: 20px;
                border-radius: 50%;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                font-size: 9px;
                font-weight: 600;
                background: #eaf3de;
                color: #27500a;
                flex-shrink: 0;
            }

            .stage-name {
                color: var(--color-text-primary);
                font-size: 12px;
                font-weight: 500;
                flex: 1;
            }

            .stage-time {
                color: var(--color-text-tertiary);
                font-size: 11px;
            }

            .resource-bar {
                width: 100%;
                height: 6px;
                background: var(--color-background-secondary);
                border-radius: var(--radius-pill);
                overflow: hidden;
            }

            .resource-bar-fill {
                height: 100%;
                border-radius: var(--radius-pill);
            }

            .stButton > button[kind="primary"] {
                background: var(--color-accent);
                color: var(--color-accent-soft);
                border: none;
                border-radius: var(--radius-md);
                font-size: 12px;
                font-weight: 500;
            }

            .stButton > button[kind="primary"]:hover {
                background: #0c447c;
            }

            .stSelectbox label,
            .stSlider label,
            .stNumberInput label {
                font-size: 12px;
                color: var(--color-text-secondary);
            }

            .stTabs [data-baseweb="tab-list"] {
                gap: 6px;
            }

            .stTabs [data-baseweb="tab"] {
                font-size: 12px;
                padding-top: 7px;
                padding-bottom: 7px;
            }

            .stTabs [aria-selected="true"] {
                color: var(--color-accent);
                font-weight: 500;
            }

            .ui-badge {
                display: inline-flex;
                align-items: center;
                gap: 4px;
                font-size: 11px;
                font-weight: 500;
                line-height: 1;
                padding: 5px 10px;
                border-radius: var(--radius-pill);
                border: 1px solid transparent;
                text-transform: capitalize;
                letter-spacing: 0.01em;
            }

            .ui-badge.success {
                background: var(--semantic-success-bg);
                color: var(--semantic-success-fg);
                border-color: var(--semantic-success-border);
            }

            .ui-badge.warning {
                background: var(--semantic-warning-bg);
                color: var(--semantic-warning-fg);
                border-color: var(--semantic-warning-border);
            }

            .ui-badge.danger {
                background: var(--semantic-danger-bg);
                color: var(--semantic-danger-fg);
                border-color: var(--semantic-danger-border);
            }

            .ui-badge.info {
                background: var(--semantic-info-bg);
                color: var(--semantic-info-fg);
                border-color: var(--semantic-info-border);
            }

            .ui-badge.neutral {
                background: var(--semantic-neutral-bg);
                color: var(--semantic-neutral-fg);
                border-color: var(--semantic-neutral-border);
            }

            .ui-alert {
                padding: 12px 14px;
                border-radius: var(--radius-md);
                margin-bottom: 16px;
                font-size: 13px;
                border: 1px solid transparent;
            }

            .ui-alert.success {
                background: var(--semantic-success-bg);
                color: var(--semantic-success-fg);
                border-color: var(--semantic-success-border);
            }

            .ui-alert.warning {
                background: var(--semantic-warning-bg);
                color: var(--semantic-warning-fg);
                border-color: var(--semantic-warning-border);
            }

            .ui-alert.danger {
                background: var(--semantic-danger-bg);
                color: var(--semantic-danger-fg);
                border-color: var(--semantic-danger-border);
            }

            .ui-alert.info {
                background: var(--semantic-info-bg);
                color: var(--semantic-info-fg);
                border-color: var(--semantic-info-border);
            }

            .ui-alert.neutral {
                background: var(--semantic-neutral-bg);
                color: var(--semantic-neutral-fg);
                border-color: var(--semantic-neutral-border);
            }

            .ui-card {
                border: 1px solid var(--color-border-tertiary);
                border-radius: var(--radius-lg);
                background: var(--color-background-secondary);
                box-shadow: none;
                padding: 16px;
                transition: border-color .18s ease, transform .18s ease;
            }
            .ui-card:hover { border-color: var(--color-border-secondary); transform: translateY(-1px); }

            .ui-kpi-card {
                border: 1px solid var(--color-border-tertiary);
                border-radius: var(--radius-lg);
                background: var(--color-background-secondary);
                box-shadow: none;
                padding: 16px;
                transition: border-color .18s ease, transform .18s ease;
            }
            .ui-kpi-card:hover { border-color: var(--color-border-secondary); transform: translateY(-1px); }

            .ui-kpi-head {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 8px;
            }

            .ui-kpi-title {
                font-size: 11px;
                color: var(--color-text-tertiary);
                text-transform: uppercase;
                letter-spacing: 0.03em;
            }

            .ui-kpi-icon {
                font-size: 12px;
                color: var(--color-text-secondary);
            }

            .ui-kpi-value {
                font-size: 24px;
                font-weight: 600;
                color: var(--color-text-primary);
                line-height: 1.1;
            }

            .ui-kpi-sub {
                margin-top: 4px;
                font-size: 11px;
                color: var(--color-text-tertiary);
            }

            .ui-kpi-card.success { border-top: 2px solid var(--semantic-success-border); }
            .ui-kpi-card.warning { border-top: 2px solid var(--semantic-warning-border); }
            .ui-kpi-card.danger { border-top: 2px solid var(--semantic-danger-border); }
            .ui-kpi-card.info { border-top: 2px solid var(--semantic-info-border); }
            .ui-kpi-card.neutral { border-top: 2px solid var(--semantic-neutral-border); }

            .ui-table-wrap {
                border: 1px solid var(--color-border-tertiary);
                border-radius: var(--radius-md);
                overflow: hidden;
                background: var(--color-background-secondary);
            }
            table.ui-table {
                width: 100%;
                border-collapse: collapse;
                font-size: 12px;
            }
            table.ui-table thead tr {
                background: #f8f9fc;
                border-bottom: 1px solid var(--color-border-tertiary);
            }
            table.ui-table th {
                text-align: left;
                color: var(--color-text-tertiary);
                font-weight: 600;
                padding: 10px 12px;
                letter-spacing: 0.01em;
            }
            table.ui-table td {
                padding: 9px 12px;
                color: var(--color-text-secondary);
                border-bottom: 1px solid #f2f4f8;
                vertical-align: middle;
            }
            table.ui-table tbody tr:hover {
                background: #f8faff;
            }
            table.ui-table tbody tr:last-child td {
                border-bottom: none;
            }
            .ui-fade-in {
                animation: uiFadeIn .22s ease-out;
            }
            @keyframes uiFadeIn {
                from { opacity: 0; transform: translateY(2px); }
                to { opacity: 1; transform: translateY(0); }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    apply_plotly_defaults()


def _resolve_tone(value: str, default: str = "neutral") -> str:
    key = str(value or "").strip().lower()
    return STATUS_TO_TONE.get(key, default)


def section_title_html(title: str, margin_top_px: int = 0) -> str:
    return f'<div class="section-title" style="margin-top:{margin_top_px}px">{title}</div>'


def render_section_title(title: str, margin_top_px: int = 0) -> None:
    st.markdown(section_title_html(title, margin_top_px=margin_top_px), unsafe_allow_html=True)


def render_section_header(title: str, subtitle: str | None = None, *, margin_top_px: int = 0) -> None:
    """Render a standardized section header with optional subtitle."""
    render_section_title(title, margin_top_px=margin_top_px)
    if subtitle:
        st.caption(subtitle)


def render_spacer(size: str = "md") -> None:
    """Render vertical spacing using design token sizes."""
    px = DESIGN_TOKENS.get("spacing", {}).get(size, DESIGN_TOKENS["spacing"]["md"])
    st.markdown(f"<div style='height:{px}'></div>", unsafe_allow_html=True)


def badge_html(label: str, tone: str = "neutral") -> str:
    tone_key = _resolve_tone(tone, default=tone)
    return f'<span class="ui-badge {tone_key}">{label}</span>'


def stage_badge_html(stage: str) -> str:
    stage_key = str(stage or "").strip().lower()
    tone = STAGE_TO_TONE.get(stage_key, "neutral")
    return badge_html(stage_key or "unknown", tone=tone)


def status_badge_html(status: str) -> str:
    status_key = str(status or "").strip().lower()
    return badge_html(status_key or "unknown", tone=_resolve_tone(status_key))


def drift_severity_badge_html(severity: str) -> str:
    sev = str(severity or "").strip().lower()
    tone_map = {"high": "danger", "significant": "danger", "medium": "warning", "moderate": "warning", "low": "success", "stable": "success"}
    return badge_html(sev or "unknown", tone=tone_map.get(sev, "neutral"))


def render_status_badge(label: str, status: str | None = None) -> None:
    """Render an inline semantic badge."""
    if status is None:
        st.markdown(status_badge_html(label), unsafe_allow_html=True)
    else:
        st.markdown(badge_html(label, tone=_resolve_tone(status)), unsafe_allow_html=True)


def render_kpi_card(
    title: str,
    value: str,
    *,
    subtitle: str | None = None,
    tone: str = "neutral",
    icon: str | None = None,
) -> None:
    """Render a reusable KPI card for dashboard-style pages."""
    tone_key = _resolve_tone(tone, default=tone)
    safe_icon = icon or ""
    safe_sub = subtitle or ""
    st.markdown(
        f"""
        <div class="ui-kpi-card {tone_key}">
            <div class="ui-kpi-head">
                <div class="ui-kpi-title">{title}</div>
                <div class="ui-kpi-icon">{safe_icon}</div>
            </div>
            <div class="ui-kpi-value">{value}</div>
            <div class="ui-kpi-sub">{safe_sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_row(items: Sequence[dict]) -> None:
    """Render a consistent KPI row from simple config dictionaries."""
    if not items:
        return

    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            render_kpi_card(
                title=str(item.get("title", "")),
                value=str(item.get("value", "—")),
                subtitle=str(item.get("subtitle", "")),
                tone=str(item.get("tone", "neutral")),
                icon=item.get("icon"),
            )



def alert_html(message: str, tone: str = "info") -> str:
    tone_key = _resolve_tone(tone, default=tone)
    return f'<div class="ui-alert {tone_key}">{message}</div>'


def render_alert(message: str, tone: str = "info") -> None:
    st.markdown(alert_html(message, tone=tone), unsafe_allow_html=True)


def render_sidebar_brand() -> None:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-title">ML Pipeline Monitor</div>
            <div class="brand-sub">MLOps Observability</div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def render_sidebar_nav() -> None:
    """Render demo-aligned app navigation in the sidebar."""
    st.page_link("app.py", label="Overview")
    st.page_link("pages/1_Pipeline_Runner.py", label="Pipeline runner")
    st.page_link("pages/2_Experiment_Tracking.py", label="Experiment tracking")
    st.page_link("pages/3_Model_Registry.py", label="Model registry")
    st.page_link("pages/4_Data_Drift.py", label="Data drift")
    st.page_link("pages/5_Data_Health.py", label="Data health")



def render_page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="page-header">
            <div class="page-title">{title}</div>
            <div class="page-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )




def render_page_header_with_action(
    title: str,
    subtitle: str,
    action_label: str,
    *,
    action_key: str,
    disabled: bool = False,
    help_text: str | None = None,
) -> bool:
    """Render a standard title/subtitle block with a right-aligned primary action."""
    col_left, col_right = st.columns([5, 1])
    with col_left:
        render_page_header(title, subtitle)
    with col_right:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        clicked = st.button(
            action_label,
            key=action_key,
            type="primary",
            width="stretch",
            disabled=disabled,
            help=help_text,
        )
    return clicked


def render_empty_data_explainer(why: str, action: str, next_step: str) -> None:
    """Render a consistent empty-data explanation card."""
    st.markdown(
        f"""
        <div class="ui-alert info">
            <b>Why this is empty</b><br>{why}<br><br>
            <b>What to do</b><br>{action}<br><br>
            <b>What happens next</b><br>{next_step}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_loading_skeleton(lines: int = 4, key: str = "skeleton") -> None:
    """Render a lightweight skeleton placeholder while data is loading."""
    blocks = "".join([f'<div class="sk-line" style="width:{90 - (i % 3) * 10}%"></div>' for i in range(lines)])
    st.markdown(
        f"""
        <style>
          .sk-wrap {{ padding: 12px; border: 0.5px solid #e9e7e4; border-radius: 8px; margin-bottom: 10px; }}
          .sk-line {{ height: 12px; border-radius: 8px; margin-bottom: 8px;
                      background: linear-gradient(90deg, #f3f3f1 25%, #ecece9 37%, #f3f3f1 63%);
                      background-size: 400% 100%; animation: skPulse 1.2s ease-in-out infinite; }}
          @keyframes skPulse {{ 0% {{ background-position: 100% 50%; }} 100% {{ background-position: 0 50%; }} }}
        </style>
        <div class="sk-wrap" id="{key}">{blocks}</div>
        """,
        unsafe_allow_html=True,
    )


def render_summary_table(
    df: pd.DataFrame,
    *,
    key_prefix: str,
    columns: Sequence[str],
    sort_by: str | None = None,
    filterable_columns: Iterable[str] | None = None,
    max_rows: int = 25,
) -> pd.DataFrame:
    """Render a compact sortable/filterable summary table and return the shown frame."""
    data = df.copy()
    if data.empty:
        st.info("No rows to display.")
        return data

    if filterable_columns:
        with st.expander("Filter and sort", expanded=False):
            for col in filterable_columns:
                if col in data.columns:
                    opts = sorted(data[col].dropna().astype(str).unique().tolist())
                    selected = st.multiselect(
                        f"{col}",
                        options=opts,
                        default=opts,
                        key=f"{key_prefix}_filter_{col}",
                    )
                    if selected:
                        data = data[data[col].astype(str).isin(selected)]

            sort_field = st.selectbox(
                "Sort by",
                options=[c for c in columns if c in data.columns],
                index=(
                    [c for c in columns if c in data.columns].index(sort_by)
                    if sort_by in [c for c in columns if c in data.columns]
                    else 0
                ),
                key=f"{key_prefix}_sort_field",
            )
            desc = st.toggle("Descending", value=True, key=f"{key_prefix}_sort_desc")
            data = data.sort_values(sort_field, ascending=not desc)
    elif sort_by and sort_by in data.columns:
        data = data.sort_values(sort_by, ascending=False)

    show_cols = [c for c in columns if c in data.columns]
    page_size = int(
        st.selectbox(
            "Rows per page",
            options=[10, 25, 50, 100],
            index=[10, 25, 50, 100].index(max_rows) if max_rows in [10, 25, 50, 100] else 1,
            key=f"{key_prefix}_page_size",
        )
    )
    total_rows = len(data)
    total_pages = max(1, (total_rows + page_size - 1) // page_size)
    page_no = int(
        st.number_input(
            "Page",
            min_value=1,
            max_value=total_pages,
            value=1,
            step=1,
            key=f"{key_prefix}_page_no",
        )
    )
    start = (page_no - 1) * page_size
    end = start + page_size
    shown = data.iloc[start:end][show_cols].copy()
    st.caption(f"Showing rows {start + 1}-{min(end, total_rows)} of {total_rows}")

    badge_cols_status = {"status", "drift detected"}
    badge_cols_stage = {"stage"}
    badge_cols_severity = {"severity"}

    header_html = "".join(f"<th>{escape(str(col))}</th>" for col in shown.columns)
    body_rows = []
    for _, row in shown.iterrows():
        cell_html = []
        for col in shown.columns:
            col_key = str(col).strip().lower()
            raw_val = row[col]
            if col_key in badge_cols_stage:
                rendered = stage_badge_html(str(raw_val))
            elif col_key in badge_cols_severity:
                rendered = drift_severity_badge_html(str(raw_val))
            elif col_key in badge_cols_status:
                rendered = status_badge_html(str(raw_val))
            else:
                rendered = escape(str(raw_val))
            cell_html.append(f"<td>{rendered}</td>")
        body_rows.append("<tr>" + "".join(cell_html) + "</tr>")

    st.markdown(
        """
        <div class="ui-table-wrap ui-fade-in">
          <table class="ui-table">
            <thead><tr>"""
        + header_html
        + """</tr></thead>
            <tbody>"""
        + "".join(body_rows)
        + """</tbody>
          </table>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return data


def render_expandable_rows(
    df: pd.DataFrame,
    *,
    title_col: str,
    detail_cols: Sequence[str],
    badge_col: str | None = None,
    badge_mode: str = "status",
    max_rows: int = 12,
) -> None:
    """Render compact row cards as expanders with optional semantic badge."""
    if df.empty:
        return

    for _, row in df.head(max_rows).iterrows():
        title = str(row.get(title_col, "row"))
        badge = ""
        if badge_col and badge_col in row.index:
            value = str(row.get(badge_col, ""))
            if badge_mode == "stage":
                badge = stage_badge_html(value)
            else:
                badge = status_badge_html(value)
        header = f"{title} {badge}".strip()
        with st.expander(header):
            for col in detail_cols:
                if col in row.index:
                    st.markdown(f"**{col}**: {row[col]}")
