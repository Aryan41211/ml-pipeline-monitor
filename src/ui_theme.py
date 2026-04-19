"""Shared UI theme helpers that mirror the provided frontend demo."""

from __future__ import annotations

from typing import Iterable, Sequence

import plotly.io as pio
import plotly.graph_objects as go
import pandas as pd
import streamlit as st


DESIGN_TOKENS = {
    "spacing": {
        "xs": "4px",
        "sm": "8px",
        "md": "12px",
        "lg": "16px",
        "xl": "24px",
    },
    "font_sizes": {
        "h1": "22px",
        "h2": "16px",
        "body": "12px",
        "caption": "10px",
    },
    "colors": {
        "background_primary": "#ffffff",
        "background_secondary": "#f6f6f4",
        "border_primary": "#e9e7e4",
        "border_secondary": "#d4d2cf",
        "text_primary": "#1f1f1e",
        "text_secondary": "#4d4c48",
        "text_tertiary": "#888780",
        "accent": "#185fa5",
        "accent_soft": "#e6f1fb",
    },
    "radius": {
        "sm": "6px",
        "md": "8px",
        "lg": "12px",
        "pill": "999px",
    },
    "shadows": {
        "none": "none",
        "sm": "0 1px 2px rgba(15, 23, 42, 0.06)",
    },
    "semantic": {
        "success": {"bg": "#eaf3de", "fg": "#27500a", "border": "#639922"},
        "warning": {"bg": "#faeeda", "fg": "#633806", "border": "#ba7517"},
        "danger": {"bg": "#fcebeb", "fg": "#791f1f", "border": "#e24b4a"},
        "info": {"bg": "#e6f1fb", "fg": "#0c447c", "border": "#378add"},
        "neutral": {"bg": "#f1f5f9", "fg": "#475569", "border": "#94a3b8"},
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
    "#378add",
    "#1d9e75",
    "#ba7517",
    "#e24b4a",
    "#888780",
    "#185fa5",
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
                "xaxis": {"showgrid": False, "title": None, "linecolor": "#e9e7e4"},
                "yaxis": {"showgrid": True, "gridcolor": "#f1f5f9", "title": None, "linecolor": "#e9e7e4"},
            }
        )
        pio.templates["mlmonitor"] = go.layout.Template(layout=layout_config)
    pio.templates.default = "mlmonitor"


def apply_plotly_layout(fig, *, height: int = 340, x_title: str | None = None, y_title: str | None = None):
    """Apply shared layout polish to a Plotly figure."""
    fig.update_layout(height=height)
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
                --space-xs: 4px;
                --space-sm: 8px;
                --space-md: 12px;
                --space-lg: 16px;
                --space-xl: 24px;

                --font-h1: 22px;
                --font-h2: 16px;
                --font-body: 12px;
                --font-caption: 10px;

                --color-background-primary: #ffffff;
                --color-background-secondary: #f6f6f4;
                --color-border-tertiary: #e9e7e4;
                --color-border-secondary: #d4d2cf;
                --color-text-primary: #1f1f1e;
                --color-text-secondary: #4d4c48;
                --color-text-tertiary: #888780;
                --color-accent: #185fa5;
                --color-accent-soft: #e6f1fb;

                --radius-sm: 6px;
                --radius-md: 8px;
                --radius-lg: 12px;
                --radius-pill: 999px;

                --shadow-sm: 0 1px 2px rgba(15, 23, 42, 0.06);

                --semantic-success-bg: #eaf3de;
                --semantic-success-fg: #27500a;
                --semantic-success-border: #639922;

                --semantic-warning-bg: #faeeda;
                --semantic-warning-fg: #633806;
                --semantic-warning-border: #ba7517;

                --semantic-danger-bg: #fcebeb;
                --semantic-danger-fg: #791f1f;
                --semantic-danger-border: #e24b4a;

                --semantic-info-bg: #e6f1fb;
                --semantic-info-fg: #0c447c;
                --semantic-info-border: #378add;

                --semantic-neutral-bg: #f1f5f9;
                --semantic-neutral-fg: #475569;
                --semantic-neutral-border: #94a3b8;
            }

            html, body, [class*="css"] {
                font-family: "Inter", "Segoe UI", sans-serif;
                color: var(--color-text-primary);
            }

            .stApp {
                background: var(--color-background-primary);
            }

            .block-container {
                padding-top: 1.25rem;
                padding-bottom: 1.1rem;
                max-width: none;
            }

            [data-testid="stSidebar"] {
                background: var(--color-background-primary);
                border-right: 0.5px solid var(--color-border-tertiary);
                min-width: 180px;
                width: 180px;
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
                border: none;
                border-radius: var(--radius-md);
                padding: 12px;
                box-shadow: none;
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
                margin-bottom: 16px;
            }

            .page-title {
                font-size: 16px;
                font-weight: 500;
                color: var(--color-text-primary);
            }

            .page-sub {
                font-size: 12px;
                color: var(--color-text-tertiary);
                margin-top: 3px;
            }

            .section-title {
                font-size: 12px;
                font-weight: 500;
                color: var(--color-text-secondary);
                margin-bottom: 10px;
                text-transform: uppercase;
                letter-spacing: 0.05em;
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
                font-size: 10px;
                font-weight: 600;
                line-height: 1;
                padding: 4px 8px;
                border-radius: var(--radius-pill);
                border: 1px solid transparent;
                text-transform: lowercase;
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
                padding: 12px 16px;
                border-radius: var(--radius-md);
                margin-bottom: 16px;
                font-size: 12px;
                border-left: 4px solid transparent;
            }

            .ui-alert.success {
                background: var(--semantic-success-bg);
                color: var(--semantic-success-fg);
                border-left-color: var(--semantic-success-border);
            }

            .ui-alert.warning {
                background: var(--semantic-warning-bg);
                color: var(--semantic-warning-fg);
                border-left-color: var(--semantic-warning-border);
            }

            .ui-alert.danger {
                background: var(--semantic-danger-bg);
                color: var(--semantic-danger-fg);
                border-left-color: var(--semantic-danger-border);
            }

            .ui-alert.info {
                background: var(--semantic-info-bg);
                color: var(--semantic-info-fg);
                border-left-color: var(--semantic-info-border);
            }

            .ui-alert.neutral {
                background: var(--semantic-neutral-bg);
                color: var(--semantic-neutral-fg);
                border-left-color: var(--semantic-neutral-border);
            }

            .ui-card {
                border: 1px solid var(--color-border-tertiary);
                border-radius: var(--radius-md);
                background: var(--color-background-primary);
                box-shadow: var(--shadow-sm);
                padding: 12px;
            }

            .ui-kpi-card {
                border: 1px solid var(--color-border-tertiary);
                border-radius: var(--radius-md);
                background: var(--color-background-primary);
                box-shadow: var(--shadow-sm);
                padding: 12px;
            }

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
                font-size: 22px;
                font-weight: 600;
                color: var(--color-text-primary);
                line-height: 1.1;
            }

            .ui-kpi-sub {
                margin-top: 4px;
                font-size: 11px;
                color: var(--color-text-tertiary);
            }

            .ui-kpi-card.success { border-color: var(--semantic-success-border); }
            .ui-kpi-card.warning { border-color: var(--semantic-warning-border); }
            .ui-kpi-card.danger { border-color: var(--semantic-danger-border); }
            .ui-kpi-card.info { border-color: var(--semantic-info-border); }
            .ui-kpi-card.neutral { border-color: var(--semantic-neutral-border); }
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
    st.dataframe(shown, width="stretch", hide_index=True)
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
