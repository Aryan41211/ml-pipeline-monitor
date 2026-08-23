"""
Design system for the ML Pipeline Monitor UI.

A single light theme built from neutral greys and one accent. Every component
here draws its colours from the token tables below rather than hardcoding hex
values, so the palette can be changed in one place.

The palette is mirrored in `.streamlit/config.toml`, which styles Streamlit's
own widgets (buttons, inputs, tabs). Keep the two in sync.
"""

from __future__ import annotations

import contextlib
import html as _html
from collections.abc import Sequence
from typing import Any

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------

LIGHT: dict[str, str] = {
    # Surfaces
    "background": "#FFFFFF",
    "surface": "#FAFAFA",
    "surface_alt": "#F4F4F5",
    "card": "#FFFFFF",
    "card_hover": "#FAFAFA",
    # Lines
    "border": "#E4E4E7",
    "border_strong": "#D4D4D8",
    # Text
    "text_primary": "#18181B",
    "text_secondary": "#52525B",
    "text_tertiary": "#A1A1AA",
    # Accent
    "accent": "#4F46E5",
    "accent_hover": "#4338CA",
    "accent_soft": "#EEF2FF",
    # Semantic
    "success": "#059669",
    "success_soft": "#ECFDF5",
    "warning": "#B45309",
    "warning_soft": "#FFFBEB",
    "error": "#DC2626",
    "error_soft": "#FEF2F2",
    "neutral": "#52525B",
    "neutral_soft": "#F4F4F5",
}

# The app ships a single light theme. DARK is retained so that
# get_color("dark", ...) keeps working for any caller that asks for it.
DARK = dict(LIGHT)

COLORS: dict[str, dict[str, str]] = {"light": LIGHT, "dark": DARK}

TYPOGRAPHY: dict[str, str] = {
    "font_family": (
        "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
    ),
    "font_mono": "'JetBrains Mono', 'SF Mono', Menlo, Consolas, monospace",
}

# 4px base grid.
SPACING_PX: dict[str, int] = {"xs": 4, "sm": 8, "md": 16, "lg": 24, "xl": 32, "2xl": 48}
SPACING: dict[str, str] = {k: f"{v}px" for k, v in SPACING_PX.items()}

BORDER_RADIUS: dict[str, str] = {
    "sm": "4px",
    "md": "6px",
    "lg": "8px",
    "xl": "12px",
    "pill": "999px",
}

# Deliberately restrained: one hairline shadow for raised surfaces, nothing more.
SHADOWS: dict[str, str] = {
    "none": "none",
    "soft": "0 1px 2px rgba(24, 24, 27, 0.04)",
    "raised": "0 1px 3px rgba(24, 24, 27, 0.08), 0 1px 2px rgba(24, 24, 27, 0.04)",
}

ANIMATION: dict[str, str] = {
    "fast": "120ms cubic-bezier(0.4, 0, 0.2, 1)",
    "base": "180ms cubic-bezier(0.4, 0, 0.2, 1)",
}

GRADIENTS: dict[str, str] = {"accent": LIGHT["accent"], "surface": LIGHT["surface"]}

# Colour-blind-safe categorical sequence, accent first.
CHART_SEQUENCE: list[str] = [
    "#4F46E5",
    "#0891B2",
    "#059669",
    "#B45309",
    "#BE185D",
    "#7C3AED",
    "#0F766E",
    "#A16207",
]

CHART_COLORS: dict[str, Any] = {
    "sequence": CHART_SEQUENCE,
    "primary": LIGHT["accent"],
    "success": LIGHT["success"],
    "warning": LIGHT["warning"],
    "error": LIGHT["error"],
    "grid": LIGHT["border"],
    "text": LIGHT["text_secondary"],
}

# Semantic mapping for statuses and lifecycle stages.
_TONE_BY_STATUS: dict[str, str] = {
    "success": "success",
    "completed": "success",
    "healthy": "success",
    "passed": "success",
    "pass": "success",
    "stable": "success",
    "compliant": "success",
    "production": "success",
    "ready": "success",
    "active": "success",
    "warning": "warning",
    "warn": "warning",
    "moderate": "warning",
    "degraded": "warning",
    "pending": "warning",
    "running": "warning",
    "staging": "warning",
    "queued": "warning",
    "error": "error",
    "failed": "error",
    "fail": "error",
    "critical": "error",
    "significant": "error",
    "non_compliant": "error",
    "unhealthy": "error",
    "development": "info",
    "info": "info",
    "none": "neutral",
    "archived": "neutral",
    "skipped": "neutral",
    "unknown": "neutral",
}


def _tone_for(status: str) -> str:
    return _TONE_BY_STATUS.get(str(status).strip().lower().replace(" ", "_"), "neutral")


def _tone_colors(tone: str) -> tuple[str, str]:
    """Return (foreground, background) for a semantic tone."""
    mapping = {
        "success": (LIGHT["success"], LIGHT["success_soft"]),
        "warning": (LIGHT["warning"], LIGHT["warning_soft"]),
        "error": (LIGHT["error"], LIGHT["error_soft"]),
        "danger": (LIGHT["error"], LIGHT["error_soft"]),
        "info": (LIGHT["accent"], LIGHT["accent_soft"]),
        "accent": (LIGHT["accent"], LIGHT["accent_soft"]),
        "neutral": (LIGHT["text_secondary"], LIGHT["neutral_soft"]),
    }
    return mapping.get(str(tone).lower(), mapping["neutral"])


def get_color(theme: str = "light", key: str = "background") -> str:
    """Return a palette colour by key."""
    return COLORS.get(theme, LIGHT).get(key, LIGHT.get(key, "#000000"))


def esc(value: Any) -> str:
    """Escape a value for safe interpolation into component markup."""
    return _html.escape(str(value), quote=True)


def _render(*parts: str) -> None:
    """Write component markup to the page as a single unbroken line.

    Streamlit parses st.markdown as Markdown first. A blank line inside an HTML
    block closes it and the following indented lines become a code block, so
    markup is joined with no newlines and empty parts are dropped.
    """
    st.markdown("".join(part for part in parts if part), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Plotly
# ---------------------------------------------------------------------------


def apply_plotly_defaults(theme: str = "light") -> None:
    """Register a Plotly template matching the design system."""
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError:  # pragma: no cover - plotly is a hard dependency
        return

    axis = {
        "gridcolor": LIGHT["border"],
        "linecolor": LIGHT["border"],
        "zerolinecolor": LIGHT["border"],
        "tickfont": {"size": 11, "color": LIGHT["text_tertiary"]},
        "title": {"font": {"size": 12, "color": LIGHT["text_secondary"]}},
        "automargin": True,
    }

    pio.templates["mlmonitor"] = go.layout.Template(
        layout={
            "colorway": CHART_SEQUENCE,
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "font": {"family": TYPOGRAPHY["font_family"], "size": 12, "color": LIGHT["text_secondary"]},
            "xaxis": axis,
            "yaxis": axis,
            "hoverlabel": {
                "bgcolor": LIGHT["text_primary"],
                "font": {"color": "#FFFFFF", "size": 12, "family": TYPOGRAPHY["font_family"]},
                "bordercolor": LIGHT["text_primary"],
            },
            "legend": {
                "bgcolor": "rgba(0,0,0,0)",
                "font": {"size": 11, "color": LIGHT["text_secondary"]},
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "x": 0,
            },
            "margin": {"l": 8, "r": 8, "t": 8, "b": 8},
        }
    )
    pio.templates.default = "mlmonitor"


# ---------------------------------------------------------------------------
# Global stylesheet
# ---------------------------------------------------------------------------


def apply_ui_theme() -> None:
    """Inject the global stylesheet. Call once per page, after set_page_config."""
    apply_plotly_defaults()

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

        :root {{
            --c-bg: {LIGHT["background"]};
            --c-surface: {LIGHT["surface"]};
            --c-surface-alt: {LIGHT["surface_alt"]};
            --c-border: {LIGHT["border"]};
            --c-border-strong: {LIGHT["border_strong"]};
            --c-text: {LIGHT["text_primary"]};
            --c-text-2: {LIGHT["text_secondary"]};
            --c-text-3: {LIGHT["text_tertiary"]};
            --c-accent: {LIGHT["accent"]};
            --c-accent-hover: {LIGHT["accent_hover"]};
            --c-accent-soft: {LIGHT["accent_soft"]};
            --c-success: {LIGHT["success"]};
            --c-warning: {LIGHT["warning"]};
            --c-error: {LIGHT["error"]};
            --r-sm: {BORDER_RADIUS["sm"]};
            --r-md: {BORDER_RADIUS["md"]};
            --r-lg: {BORDER_RADIUS["lg"]};
            --shadow: {SHADOWS["soft"]};
            --font: {TYPOGRAPHY["font_family"]};
            --font-mono: {TYPOGRAPHY["font_mono"]};
        }}

        /* ---------- Base ---------- */
        html, body, .stApp, [class*="css"] {{
            font-family: var(--font);
            color: var(--c-text);
            -webkit-font-smoothing: antialiased;
        }}
        .stApp {{ background: var(--c-bg); }}

        .block-container {{
            max-width: 1280px;
            padding: 2.5rem 2rem 4rem;
        }}

        /* Streamlit's floating toolbar otherwise overlaps page content. */
        [data-testid="stHeader"] {{ background: transparent; height: 0; }}
        [data-testid="stDecoration"] {{ display: none; }}
        #MainMenu, footer {{ visibility: hidden; }}

        /* ---------- Typography ---------- */
        h1, h2, h3, h4, h5, h6 {{
            font-family: var(--font);
            color: var(--c-text);
            font-weight: 600;
            letter-spacing: -0.011em;
        }}
        .stApp h1, .ui-page-head h1 {{
            font-size: 1.75rem !important; line-height: 1.25; margin: 0 0 0.25rem;
            font-weight: 600; padding: 0;
        }}
        .stApp h2 {{
            font-size: 1.0625rem !important; line-height: 1.4; margin: 0 0 0.75rem;
            font-weight: 600; padding: 0;
        }}
        .stApp h3 {{ font-size: 0.9375rem !important; line-height: 1.4; margin: 0 0 0.5rem; padding: 0; }}
        /* Streamlit renders anchor links beside markdown headings. */
        .stApp h1 a, .stApp h2 a, .stApp h3 a {{ display: none !important; }}
        p, li, label, span {{ font-size: 0.875rem; line-height: 1.55; }}
        p {{ color: var(--c-text-2); }}
        a {{ color: var(--c-accent); text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        code {{ font-family: var(--font-mono); font-size: 0.8125rem; }}
        hr {{ border: none; border-top: 1px solid var(--c-border); margin: 2rem 0; }}

        /* ---------- Page header ---------- */
        .ui-page-head {{
            display: flex; align-items: flex-start; justify-content: space-between;
            gap: 1.5rem; padding-bottom: 1.25rem; margin-bottom: 1.5rem;
            border-bottom: 1px solid var(--c-border);
        }}
        .ui-page-head .eyebrow {{
            font-size: 0.6875rem; font-weight: 600; letter-spacing: 0.06em;
            text-transform: uppercase; color: var(--c-text-3); margin-bottom: 0.375rem;
        }}
        .ui-page-head .lede {{ color: var(--c-text-2); font-size: 0.875rem; margin: 0.25rem 0 0; }}

        /* ---------- Identity strip ---------- */
        .ui-idbar {{
            display: flex; align-items: center; justify-content: space-between;
            padding: 0 0 1rem; margin-bottom: 0.5rem;
        }}
        .ui-idbar .brand {{
            display: flex; align-items: center; gap: 0.5rem;
            font-size: 0.8125rem; font-weight: 600; color: var(--c-text);
            letter-spacing: -0.005em;
        }}
        .ui-idbar .mark {{
            width: 20px; height: 20px; border-radius: var(--r-sm);
            background: var(--c-accent); display: inline-flex;
            align-items: center; justify-content: center;
            color: #fff; font-size: 0.625rem; font-weight: 600;
        }}
        .ui-idbar .who {{
            font-size: 0.75rem; color: var(--c-text-3);
            display: flex; align-items: center; gap: 0.5rem;
        }}

        /* ---------- Cards ---------- */
        .ui-card {{
            background: var(--c-bg);
            border: 1px solid var(--c-border);
            border-radius: var(--r-lg);
            padding: 1.25rem;
        }}
        .ui-card--flush {{ padding: 0; overflow: hidden; }}

        /* ---------- Metric ---------- */
        .ui-metric .label {{
            font-size: 0.6875rem; font-weight: 600; letter-spacing: 0.06em;
            text-transform: uppercase; color: var(--c-text-3);
            margin-bottom: 0.5rem;
        }}
        .ui-metric .value {{
            font-size: 1.75rem; font-weight: 600; line-height: 1.1;
            color: var(--c-text); font-variant-numeric: tabular-nums;
            letter-spacing: -0.02em;
        }}
        .ui-metric .value.is-success {{ color: var(--c-success); }}
        .ui-metric .value.is-warning {{ color: var(--c-warning); }}
        .ui-metric .value.is-error {{ color: var(--c-error); }}
        .ui-metric .sub {{ font-size: 0.75rem; color: var(--c-text-3); margin-top: 0.25rem; }}

        /* ---------- Badge ---------- */
        .ui-badge {{
            display: inline-flex; align-items: center;
            padding: 0.125rem 0.5rem; border-radius: var(--r-sm);
            font-size: 0.6875rem; font-weight: 600; letter-spacing: 0.02em;
            white-space: nowrap; line-height: 1.5;
        }}

        /* ---------- Notice ---------- */
        .ui-notice {{
            display: flex; gap: 0.625rem; padding: 0.75rem 0.875rem;
            border: 1px solid var(--c-border); border-left-width: 3px;
            border-radius: var(--r-md); margin-bottom: 1rem;
            font-size: 0.8125rem; line-height: 1.5;
        }}
        .ui-notice .title {{ font-weight: 600; display: block; margin-bottom: 0.125rem; }}

        /* ---------- Rows / timeline ---------- */
        .ui-row {{
            display: flex; align-items: center; gap: 0.75rem;
            padding: 0.6875rem 1rem;
            border-bottom: 1px solid var(--c-border);
            font-size: 0.8125rem;
        }}
        .ui-row:last-child {{ border-bottom: none; }}
        .ui-row .dot {{ width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }}
        .ui-row .time {{
            color: var(--c-text-3); font-variant-numeric: tabular-nums;
            min-width: 42px; font-size: 0.75rem;
        }}
        .ui-row .main {{ flex: 1; color: var(--c-text); }}

        /* ---------- List ---------- */
        .ui-list {{ margin: 0; padding: 0; list-style: none; }}
        .ui-list li {{
            position: relative; padding: 0.375rem 0 0.375rem 0.875rem;
            font-size: 0.8125rem; color: var(--c-text-2); line-height: 1.5;
        }}
        .ui-list li::before {{
            content: ""; position: absolute; left: 0; top: 0.8125rem;
            width: 4px; height: 4px; border-radius: 50%; background: var(--c-border-strong);
        }}

        /* ---------- Table ---------- */
        .ui-table {{
            width: 100%; border-collapse: collapse;
            border: 1px solid var(--c-border); border-radius: var(--r-lg);
            overflow: hidden; font-size: 0.8125rem;
        }}
        .ui-table th {{
            text-align: left; padding: 0.625rem 0.875rem;
            background: var(--c-surface);
            border-bottom: 1px solid var(--c-border);
            font-size: 0.6875rem; font-weight: 600; letter-spacing: 0.04em;
            text-transform: uppercase; color: var(--c-text-3); white-space: nowrap;
        }}
        .ui-table td {{
            padding: 0.625rem 0.875rem;
            border-bottom: 1px solid var(--c-border);
            color: var(--c-text); vertical-align: middle;
        }}
        .ui-table tbody tr:last-child td {{ border-bottom: none; }}
        .ui-table tbody tr:hover td {{ background: var(--c-surface); }}
        .ui-table-wrap {{ overflow-x: auto; }}

        /* ---------- Registry tile ---------- */
        .ui-tile {{
            border: 1px solid var(--c-border); border-radius: var(--r-lg);
            padding: 1rem; margin-bottom: 0.5rem; background: var(--c-bg);
            transition: border-color {ANIMATION["fast"]};
        }}
        .ui-tile:hover {{ border-color: var(--c-border-strong); }}
        .ui-tile .name {{ font-size: 0.875rem; font-weight: 600; color: var(--c-text); }}
        .ui-tile .meta {{ font-size: 0.75rem; color: var(--c-text-3); margin-top: 0.125rem; }}
        .ui-tile .stats {{
            display: flex; gap: 1.25rem; margin-top: 0.75rem;
            padding-top: 0.75rem; border-top: 1px solid var(--c-border);
            font-size: 0.75rem; color: var(--c-text-2);
            font-variant-numeric: tabular-nums;
        }}
        .ui-tile .stats b {{ color: var(--c-text); font-weight: 600; }}

        /* ---------- Score ---------- */
        .ui-score {{ text-align: center; padding: 0.5rem 0; }}
        .ui-score .num {{
            font-size: 2.5rem; font-weight: 600; line-height: 1;
            font-variant-numeric: tabular-nums; letter-spacing: -0.03em;
        }}
        .ui-score .cap {{
            font-size: 0.6875rem; font-weight: 600; letter-spacing: 0.06em;
            text-transform: uppercase; color: var(--c-text-3); margin-top: 0.5rem;
        }}
        .ui-score .track {{
            height: 4px; border-radius: 999px; background: var(--c-surface-alt);
            margin-top: 0.875rem; overflow: hidden;
        }}
        .ui-score .fill {{ height: 100%; border-radius: 999px; }}

        /* ---------- Empty state ---------- */
        .ui-empty {{
            text-align: center; padding: 3rem 1.5rem;
            border: 1px dashed var(--c-border); border-radius: var(--r-lg);
        }}
        .ui-empty .t {{ font-size: 0.9375rem; font-weight: 600; color: var(--c-text); }}
        .ui-empty .m {{ font-size: 0.8125rem; color: var(--c-text-2); margin-top: 0.375rem; }}

        /* ---------- Skeleton ---------- */
        @keyframes ui-pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.45; }} }}
        .ui-skel {{
            height: 10px; border-radius: var(--r-sm); background: var(--c-surface-alt);
            margin-bottom: 0.625rem; animation: ui-pulse 1.4s ease-in-out infinite;
        }}

        /* ---------- Sidebar ---------- */
        [data-testid="stSidebar"] {{
            background: var(--c-surface);
            border-right: 1px solid var(--c-border);
        }}
        [data-testid="stSidebar"] > div:first-child {{ padding-top: 1.75rem; }}
        [data-testid="stSidebarNav"] {{ display: none; }}
        [data-testid="stSidebar"] h3 {{
            font-size: 0.6875rem; font-weight: 600; letter-spacing: 0.06em;
            text-transform: uppercase; color: var(--c-text-3);
            margin: 1.25rem 0 0.5rem;
        }}
        .ui-navgroup {{
            font-size: 0.6875rem; font-weight: 600; letter-spacing: 0.06em;
            text-transform: uppercase; color: var(--c-text-3);
            padding: 0 0.5rem; margin: 1.25rem 0 0.375rem;
        }}
        .ui-navgroup:first-child {{ margin-top: 0; }}
        [data-testid="stSidebar"] [data-testid="stPageLink"] a {{
            display: flex; align-items: center;
            padding: 0.4375rem 0.5rem; border-radius: var(--r-md);
            font-size: 0.8125rem; font-weight: 500;
            color: var(--c-text-2); text-decoration: none;
            transition: background {ANIMATION["fast"]}, color {ANIMATION["fast"]};
        }}
        [data-testid="stSidebar"] [data-testid="stPageLink"] a:hover {{
            background: var(--c-surface-alt); color: var(--c-text);
        }}
        [data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"] {{
            background: var(--c-accent-soft); color: var(--c-accent); font-weight: 600;
        }}

        /* ---------- Streamlit widget overrides ---------- */
        .stButton > button {{
            border-radius: var(--r-md); font-size: 0.8125rem; font-weight: 500;
            padding: 0.4375rem 0.875rem; border: 1px solid var(--c-border);
            background: var(--c-bg); color: var(--c-text);
            box-shadow: none; transition: all {ANIMATION["fast"]};
        }}
        .stButton > button:hover {{
            border-color: var(--c-border-strong); background: var(--c-surface); color: var(--c-text);
        }}
        .stButton > button[kind="primary"] {{
            background: var(--c-accent); border-color: var(--c-accent); color: #fff;
        }}
        .stButton > button[kind="primary"]:hover {{
            background: var(--c-accent-hover); border-color: var(--c-accent-hover); color: #fff;
        }}
        .stButton > button:focus:not(:active) {{
            border-color: var(--c-accent); box-shadow: 0 0 0 3px var(--c-accent-soft); color: var(--c-text);
        }}
        .stButton > button[kind="primary"]:focus:not(:active) {{ color: #fff; }}
        .stButton > button:disabled {{ opacity: 0.5; }}
        /* Streamlit wraps button text in <p>, which the global `p` colour rule
           would otherwise win against, leaving dark text on the accent fill. */
        .stButton > button p, .stButton > button div, .stButton > button span {{
            color: inherit !important; font-size: 0.8125rem; font-weight: 500; margin: 0;
        }}
        [data-testid="stPageLink"] p {{ color: inherit !important; margin: 0; }}

        .stTextInput input, .stNumberInput input, .stTextArea textarea,
        .stSelectbox div[data-baseweb="select"] > div {{
            border-radius: var(--r-md) !important;
            border-color: var(--c-border) !important;
            font-size: 0.8125rem !important;
            background: var(--c-bg) !important;
        }}
        .stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {{
            border-color: var(--c-accent) !important;
            box-shadow: 0 0 0 3px var(--c-accent-soft) !important;
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap: 1.5rem; border-bottom: 1px solid var(--c-border);
            background: transparent; padding: 0; margin-bottom: 1.25rem;
        }}
        .stTabs [data-baseweb="tab"] {{
            height: auto; padding: 0.625rem 0; background: transparent;
            font-size: 0.8125rem; font-weight: 500; color: var(--c-text-2);
            border-bottom: 2px solid transparent; border-radius: 0;
        }}
        .stTabs [aria-selected="true"] {{
            color: var(--c-accent) !important; border-bottom-color: var(--c-accent) !important;
        }}
        .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display: none; }}

        [data-testid="stExpander"] {{
            border: 1px solid var(--c-border); border-radius: var(--r-lg); background: var(--c-bg);
        }}
        [data-testid="stExpander"] summary {{ font-size: 0.8125rem; font-weight: 500; }}

        [data-testid="stMetricValue"] {{ font-size: 1.5rem; font-weight: 600; }}
        [data-testid="stMetricLabel"] {{ font-size: 0.75rem; color: var(--c-text-3); }}

        [data-testid="stAlert"] {{ border-radius: var(--r-md); font-size: 0.8125rem; }}
        [data-testid="stDataFrame"] {{ border: 1px solid var(--c-border); border-radius: var(--r-lg); }}
        [data-testid="stCaptionContainer"] p {{ font-size: 0.75rem; color: var(--c-text-3); }}

        .stSlider [data-baseweb="slider"] div[role="slider"] {{ background: var(--c-accent); }}
        .stProgress > div > div > div {{ background: var(--c-accent); }}

        /* Tighten Streamlit's default vertical rhythm. */
        [data-testid="stVerticalBlock"] {{ gap: 0.875rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Layout primitives
# ---------------------------------------------------------------------------


def render_top_navbar(user_role: str = "viewer") -> None:
    """Render the compact identity strip at the top of every page."""
    _render(
        '<div class="ui-idbar">',
        '<div class="brand"><span class="mark">M</span> ML Pipeline Monitor</div>',
        f'<div class="who">Signed in as {esc(str(user_role).lower())}</div>',
        "</div>",
    )


_NAV_SECTIONS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "Platform",
        [
            ("app.py", "Dashboard"),
            ("pages/0_Dataset_Management.py", "Datasets"),
            ("pages/1_Pipeline_Runner.py", "Pipeline Runner"),
            ("pages/2_Experiment_Tracking.py", "Experiments"),
            ("pages/3_Model_Registry.py", "Model Registry"),
        ],
    ),
    (
        "Observability",
        [
            ("pages/4_Data_Drift.py", "Data Drift"),
            ("pages/5_Data_Health.py", "Data Health"),
            ("pages/6_Governance.py", "Governance"),
        ],
    ),
]


def render_sidebar_nav() -> None:
    """Render grouped sidebar navigation."""
    for group, links in _NAV_SECTIONS:
        st.markdown(f'<div class="ui-navgroup">{group}</div>', unsafe_allow_html=True)
        for path, label in links:
            try:
                st.page_link(path, label=label)
            except Exception:
                # page_link raises if the target is not a registered page.
                st.markdown(
                    f'<div style="padding:0.4375rem 0.5rem;font-size:0.8125rem;">{esc(label)}</div>',
                    unsafe_allow_html=True,
                )


def page_header(title: str, subtitle: str = "", eyebrow: str = "") -> None:
    """Render a page title block with an underline rule."""
    _render(
        '<div class="ui-page-head"><div>',
        f'<div class="eyebrow">{esc(eyebrow)}</div>' if eyebrow else "",
        f"<h1>{esc(title)}</h1>",
        f'<p class="lede">{esc(subtitle)}</p>' if subtitle else "",
        "</div></div>",
    )


def render_section_title(title: str, margin_top_px: int = 0) -> None:
    """Render a section heading."""
    st.markdown(f'<h2 style="margin-top:{int(margin_top_px)}px;">{esc(title)}</h2>', unsafe_allow_html=True)


def section_header(title: str, subtitle: str = "", icon: str = "", theme: str = "light") -> None:
    """Render a section heading with an optional subtitle."""
    render_section_title(title)
    if subtitle:
        st.markdown(
            f'<p style="margin:-0.5rem 0 0.75rem;color:var(--c-text-3);font-size:0.8125rem;">{esc(subtitle)}</p>',
            unsafe_allow_html=True,
        )


def render_spacer(size: str = "md") -> None:
    """Insert vertical space from the spacing scale."""
    st.markdown(f"<div style='height:{SPACING_PX.get(size, 16)}px'></div>", unsafe_allow_html=True)


def render_loading_skeleton(lines: int = 4, key: str = "skeleton") -> None:
    """Render a placeholder block while data loads."""
    widths = [100, 82, 91, 68, 76]
    bars = "".join(f'<div class="ui-skel" style="width:{widths[i % 5]}%"></div>' for i in range(max(1, lines)))
    st.markdown(f'<div class="ui-card">{bars}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Display components
# ---------------------------------------------------------------------------


def status_badge(status: str, label: str | None = None) -> str:
    """Return badge markup for a status value."""
    fg, bg = _tone_colors(_tone_for(status))
    text = label if label is not None else str(status).replace("_", " ")
    return f'<span class="ui-badge" style="color:{fg};background:{bg};">{esc(text.upper())}</span>'


def hp_status_badge(status: str) -> str:
    """Return badge markup for a status value."""
    return status_badge(status)


def status_badge_html(status: str) -> str:
    """Return badge markup for a status value."""
    return status_badge(status)


def stage_badge_html(stage: str) -> str:
    """Return badge markup for a model lifecycle stage."""
    return status_badge(stage)


def component_metric_badge(label: str, value: str, tone: str = "info") -> str:
    """Return badge markup for an inline label/value pair."""
    fg, bg = _tone_colors(tone)
    return f'<span class="ui-badge" style="color:{fg};background:{bg};">{esc(label)}: {esc(value)}</span>'


def metric_card(
    label: str,
    value: str,
    subtitle: str = "",
    tone: str = "neutral",
    icon: str = "",
    theme: str = "light",
) -> None:
    """Render a single metric."""
    tone_class = {
        "success": "is-success",
        "warning": "is-warning",
        "error": "is-error",
        "danger": "is-error",
    }.get(str(tone).lower(), "")
    _render(
        '<div class="ui-metric">',
        f'<div class="label">{esc(label)}</div>',
        f'<div class="value {tone_class}">{esc(value)}</div>',
        f'<div class="sub">{esc(subtitle)}</div>' if subtitle else "",
        "</div>",
    )


def kpi_card(
    label: str,
    value: str,
    subtitle: str = "",
    tone: str = "neutral",
    icon: str = "",
    theme: str = "light",
) -> None:
    """Render a single metric."""
    metric_card(label, value, subtitle, tone, icon, theme)


def hp_kpi_card(title: str, value: str, subtitle: str = "", tone: str = "info", icon: str = "") -> None:
    """Render a single metric."""
    metric_card(title, value, subtitle, tone)


def component_kpi_card(
    title: str,
    value: str,
    subtitle: str | None = None,
    tone: str = "neutral",
    icon: str | None = None,
    trend: str | None = None,
) -> None:
    """Render a single metric.

    ``icon`` and ``trend`` are accepted for backward compatibility and ignored:
    the design system uses no decorative icons, and a trend has to be computed
    from data rather than passed in as a literal.
    """
    metric_card(title, value, subtitle or "", tone)


def render_kpi_row(items: Sequence[dict]) -> None:
    """Render a row of metrics, one column each."""
    if not items:
        return
    for col, item in zip(st.columns(len(items)), items, strict=False):
        with col:
            metric_card(
                label=item.get("title", item.get("label", "")),
                value=str(item.get("value", "—")),
                subtitle=item.get("subtitle", ""),
                tone=item.get("tone", "neutral"),
            )


def hp_alert_card(message: str, tone: str = "info", title: str | None = None) -> None:
    """Render an inline notice."""
    fg, bg = _tone_colors(tone)
    _render(
        f'<div class="ui-notice" style="border-left-color:{fg};background:{bg};"><div>',
        f'<span class="title" style="color:{fg};">{esc(title)}</span>' if title else "",
        f'<span style="color:var(--c-text-2);">{esc(message)}</span>',
        "</div></div>",
    )


def hp_timeline(events: list[dict[str, str]]) -> None:
    """Render a compact activity list."""
    if not events:
        st.markdown('<div class="ui-card"><p style="margin:0;">No recent activity.</p></div>', unsafe_allow_html=True)
        return

    rows = []
    for event in events:
        fg, _ = _tone_colors(_tone_for(event.get("status", "neutral")))
        rows.append(
            f'<div class="ui-row">'
            f'<span class="dot" style="background:{fg};"></span>'
            f'<span class="time">{esc(event.get("time", "--:--"))}</span>'
            f'<span class="main">{esc(event.get("label", ""))}</span>'
            f'{status_badge(event.get("status", "unknown"))}'
            f"</div>"
        )
    _render(f'<div class="ui-card ui-card--flush">{"".join(rows)}</div>')


def hp_health_score(score: int, label: str = "HEALTHY") -> None:
    """Render a 0-100 score with a progress track."""
    value = max(0, min(100, int(score)))
    tone = "success" if value >= 75 else "warning" if value >= 50 else "error"
    fg, _ = _tone_colors(tone)
    _render(
        '<div class="ui-card ui-score">',
        f'<div class="num" style="color:{fg};">{value}</div>',
        f'<div class="cap">{esc(label)}</div>',
        f'<div class="track"><div class="fill" style="width:{value}%;background:{fg};"></div></div>',
        "</div>",
    )


def hp_empty_state(
    title: str,
    message: str,
    action_label: str | None = None,
    page_link: str | None = None,
) -> None:
    """Render an empty-state panel with an optional call to action."""
    _render(
        '<div class="ui-empty">',
        f'<div class="t">{esc(title)}</div>',
        f'<div class="m">{esc(message)}</div>',
        "</div>",
    )
    if action_label and page_link:
        render_spacer("sm")
        left, _ = st.columns([1, 3])
        # suppress: page_link raises if the target is not a registered page.
        with left, contextlib.suppress(Exception):
            st.page_link(page_link, label=action_label)


def hp_insight_panel(insights: list[str]) -> None:
    """Render a bulleted list of observations."""
    if not insights:
        return
    items = "".join(f"<li>{esc(text)}</li>" for text in insights)
    _render(
        '<div class="ui-card">',
        '<div class="ui-metric"><div class="label">Summary</div></div>',
        f'<ul class="ui-list">{items}</ul>',
        "</div>",
    )


def hp_registry_card(
    name: str,
    version: str,
    stage: str,
    dataset: str,
    metrics: dict[str, float],
    model_id: str,
) -> bool:
    """Render a model tile. Returns True when its select button is pressed."""
    accuracy = float(metrics.get("accuracy", 0) or 0)
    f1 = float(metrics.get("f1_score", 0) or 0)
    _render(
        '<div class="ui-tile">',
        '<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:0.5rem;"><div>',
        f'<div class="name">{esc(name)}</div>',
        f'<div class="meta">v{esc(version)} &middot; {esc(dataset)}</div>',
        f"</div>{status_badge(stage)}</div>",
        f'<div class="stats"><span>Accuracy <b>{accuracy:.4f}</b></span>' f"<span>F1 <b>{f1:.4f}</b></span></div>",
        f'<div class="meta" style="margin-top:0.5rem;font-family:var(--font-mono);">{esc(model_id)}</div>',
        "</div>",
    )
    return st.button("Select", key=f"select_{model_id}", use_container_width=True)


def hp_chevron_header(title: str, subtitle: str = "") -> None:
    """Render a page title block."""
    page_header(title, subtitle)


def glass_container(content_html: str, title: str | None = None, theme: str = "light") -> None:
    """Render arbitrary markup inside a card."""
    heading = f"<h2>{esc(title)}</h2>" if title else ""
    st.markdown(f'<div class="ui-card">{heading}{content_html}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

_BADGE_COLUMNS = {"status", "stage", "severity", "drift", "to stage", "from stage", "result"}


def render_summary_table(df: pd.DataFrame, *, columns: Sequence[str], **kwargs: Any) -> pd.DataFrame:
    """Render a DataFrame as a styled table, badging status-like columns."""
    if df is None or df.empty:
        st.markdown('<div class="ui-empty"><div class="m">No data available.</div></div>', unsafe_allow_html=True)
        return df

    show = [c for c in columns if c in df.columns] or list(df.columns)

    head = "".join(f"<th>{esc(c)}</th>" for c in show)
    body = []
    for _, row in df.iterrows():
        cells = []
        for column in show:
            raw = row[column]
            if str(column).strip().lower() in _BADGE_COLUMNS:
                text = str(raw)
                # Some callers pass pre-rendered badge markup.
                cells.append(f"<td>{text if text.startswith('<span') else status_badge(text)}</td>")
            elif isinstance(raw, float):
                cells.append(f"<td>{raw:.4f}</td>")
            else:
                cells.append(f"<td>{esc(raw)}</td>")
        body.append(f"<tr>{''.join(cells)}</tr>")

    _render(
        '<div class="ui-table-wrap"><table class="ui-table">',
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody>",
        "</table></div>",
    )
    return df


def render_expandable_rows(
    df: pd.DataFrame,
    *,
    title_col: str,
    detail_cols: Sequence[str],
    badge_col: str | None = None,
    **kwargs: Any,
) -> None:
    """Render each row as an expander with its detail fields."""
    for _, row in df.iterrows():
        suffix = f" — {row[badge_col]}" if badge_col and badge_col in df.columns else ""
        with st.expander(f"{row[title_col]}{suffix}"):
            for column in detail_cols:
                if column in df.columns:
                    st.markdown(f"**{column}:** {row[column]}")


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def render_error_boundary(error: Exception, page_name: str = "page") -> None:
    """Render a user-facing error panel for a page-level exception."""
    import traceback

    st.error(f"Something went wrong while loading {page_name}.")
    with st.expander("Technical details"):
        st.code("".join(traceback.format_exception(type(error), error, error.__traceback__)))
    st.caption("Refresh the page, or check the application logs if this persists.")


def safe_render(page_name: str, render_fn, *args: Any, **kwargs: Any):
    """Run a page renderer, converting an exception into an error panel."""
    try:
        return render_fn(*args, **kwargs)
    except Exception as exc:
        render_error_boundary(exc, page_name)
        st.stop()


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------

component_alert_card = hp_alert_card
component_timeline = hp_timeline
component_status_badge = hp_status_badge
component_health_score = hp_health_score
component_empty_state = hp_empty_state
component_insight_panel = hp_insight_panel
component_registry_card = hp_registry_card
