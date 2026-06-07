"""
Enterprise UI/UX Design System for ML Pipeline Monitor.
HP-inspired white/ink/blue design language — premium, clean, enterprise SaaS.
Forma DJR Micro via Inter, HP Electric Blue (#024ad8) as lone signal CTA.
"""

from __future__ import annotations

from html import escape
import json
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ===========================================================================
# HP Design System Tokens
# ===========================================================================

class HP:
    """HP Design Language tokens — white/ink/blue system."""
    
    # Brand & Accent
    primary = "#024ad8"
    primary_bright = "#296ef9"
    primary_deep = "#0e3191"
    primary_soft = "#c9e0fc"
    on_primary = "#ffffff"
    
    # Ink
    ink = "#1a1a1a"
    ink_deep = "#000000"
    ink_soft = "#292929"
    on_ink = "#ffffff"
    
    # Surfaces
    canvas = "#ffffff"
    paper = "#ffffff"
    cloud = "#f7f7f7"
    fog = "#e8e8e8"
    steel = "#c2c2c2"
    graphite = "#636363"
    charcoal = "#3d3d3d"
    
    # Semantic
    bloom_coral = "#ff5050"
    bloom_rose = "#f9d4d2"
    bloom_deep = "#b3262b"
    storm_mist = "#8ebdce"
    storm_sea = "#7fadbe"
    storm_deep = "#356373"
    
    # Hairlines
    hairline = "#e8e8e8"
    hairline_strong = "#c2c2c2"
    
    # Typography
    font_family = "Inter, 'Segoe UI', Arial, sans-serif"
    font_display = "Inter, 'Segoe UI', Arial, sans-serif"
    
    # Radius
    radius_none = "0px"
    radius_sm = "3px"
    radius_md = "4px"
    radius_lg = "8px"
    radius_xl = "16px"
    radius_pill = "9999px"
    
    # Shadows
    shadow_soft = "0 2px 8px rgba(26, 26, 26, 0.08)"
    shadow_modal = "0 8px 24px rgba(26, 26, 26, 0.12)"
    
    # Chevron decoration
    chevron_color = "#024ad8"


STAGE_TO_TONE = {
    "production": "success",
    "staging": "warning",
    "development": "info",
    "archived": "neutral",
}

STATUS_TO_TONE = {
    "completed": "success", "success": "success", "done": "success",
    "warning": "warning", "moderate": "warning",
    "failed": "danger", "error": "danger", "critical": "danger",
    "pending": "info", "running": "info",
    "queued": "neutral", "skipped": "neutral", "none": "neutral",
    "stable": "success", "significant": "danger",
}

PLOTLY_COLORWAY = ["#024ad8", "#1a1a1a", "#296ef9", "#636363", "#0e3191", "#ff5050"]

# ===========================================================================
# Plotly Theming — HP white canvas
# ===========================================================================

def apply_plotly_defaults() -> None:
    if "hp_white" not in pio.templates:
        pio.templates["hp_white"] = go.layout.Template(
            layout={
                "colorway": PLOTLY_COLORWAY,
                "font": {"family": "Inter, Arial, sans-serif", "size": 13, "color": "#1a1a1a"},
                "plot_bgcolor": "#ffffff",
                "paper_bgcolor": "#ffffff",
                "legend": {"title": None, "orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
                "margin": {"l": 10, "r": 10, "t": 40, "b": 10},
                "xaxis": {"showgrid": False, "linecolor": "#e8e8e8", "tickfont": {"color": "#636363", "size": 11}},
                "yaxis": {"showgrid": True, "gridcolor": "rgba(232, 232, 232, 0.6)", "linecolor": "#e8e8e8", "tickfont": {"color": "#636363", "size": 11}},
                "title": {"font": {"family": "Inter, Arial, sans-serif", "size": 16, "color": "#1a1a1a", "weight": 500}},
            }
        )
    pio.templates.default = "hp_white"

# ===========================================================================
# HP CSS Injection
# ===========================================================================

def apply_ui_theme() -> None:
    st.markdown(f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
            
            :root {{
                --hp-primary: {HP.primary};
                --hp-primary-bright: {HP.primary_bright};
                --hp-primary-deep: {HP.primary_deep};
                --hp-primary-soft: {HP.primary_soft};
                --hp-ink: {HP.ink};
                --hp-ink-deep: {HP.ink_deep};
                --hp-ink-soft: {HP.ink_soft};
                --hp-canvas: {HP.canvas};
                --hp-cloud: {HP.cloud};
                --hp-fog: {HP.fog};
                --hp-steel: {HP.steel};
                --hp-graphite: {HP.graphite};
                --hp-charcoal: {HP.charcoal};
                --hp-hairline: {HP.hairline};
                --hp-radius-sm: {HP.radius_sm};
                --hp-radius-md: {HP.radius_md};
                --hp-radius-lg: {HP.radius_lg};
                --hp-radius-xl: {HP.radius_xl};
                --hp-radius-pill: {HP.radius_pill};
                --hp-shadow-soft: {HP.shadow_soft};
                --font-family: {HP.font_family};
            }}
            
            /* === Base Reset === */
            html, body, [class*="css"] {{ 
                font-family: var(--font-family); 
                color: var(--hp-ink); 
                font-size: 15px;
                font-weight: 400;
                line-height: 1.38;
            }}
            
            .stApp {{ 
                background: var(--hp-canvas); 
            }}
            
            .block-container {{ 
                padding-top: 0.5rem; 
                max-width: 1366px;
                margin: 0 auto;
            }}
            
            /* === Sidebar === */
            [data-testid="stSidebar"] {{ 
                background: var(--hp-canvas); 
                border-right: 1px solid var(--hp-hairline); 
                min-width: 220px;
            }}
            
            [data-testid="stSidebarNav"] {{ display: none; }}
            
            /* === Typography === */
            h1, h2, h3, h4, h5, h6 {{
                font-family: var(--font-family);
                font-weight: 500;
                color: var(--hp-ink);
                letter-spacing: 0;
            }}
            
            h1 {{ font-size: 32px; line-height: 1.0; margin-bottom: 8px; }}
            h2 {{ font-size: 24px; line-height: 1.17; margin-bottom: 6px; }}
            h3 {{ font-size: 20px; line-height: 1.0; margin-bottom: 4px; }}
            
            p, li {{ color: var(--hp-charcoal); font-size: 15px; line-height: 1.38; }}
            
            /* === HP Cards === */
            .hp-card {{
                background: var(--hp-canvas);
                border-radius: var(--hp-radius-xl);
                padding: 24px;
                margin-bottom: 20px;
                box-shadow: var(--hp-shadow-soft);
                border: none;
            }}
            
            .hp-card:hover {{
                box-shadow: 0 4px 12px rgba(26, 26, 26, 0.1);
            }}
            
            .hp-card-cloud {{
                background: var(--hp-cloud);
                border-radius: var(--hp-radius-xl);
                padding: 24px;
                margin-bottom: 20px;
                border: none;
            }}
            
            /* === HP Badges === */
            .hp-badge {{
                display: inline-flex;
                align-items: center;
                gap: 6px;
                padding: 4px 12px;
                border-radius: var(--hp-radius-lg);
                font-size: 13px;
                font-weight: 400;
                border: 1px solid transparent;
            }}
            .hp-badge.success {{ background: transparent; color: {HP.primary}; border-color: {HP.primary_soft}; }}
            .hp-badge.warning {{ background: transparent; color: #b8860b; border-color: #f0e68c; }}
            .hp-badge.danger {{ background: transparent; color: {HP.bloom_deep}; border-color: {HP.bloom_rose}; }}
            .hp-badge.info {{ background: transparent; color: {HP.primary}; border-color: {HP.primary_soft}; }}
            .hp-badge.neutral {{ background: transparent; color: {HP.graphite}; border-color: var(--hp-hairline); }}
            
            /* === HP Buttons === */
            .stButton > button {{
                font-family: var(--font-family);
                font-size: 13px;
                font-weight: 600;
                letter-spacing: 0.5px;
                text-transform: uppercase;
                border-radius: var(--hp-radius-md) !important;
                height: 40px;
                padding: 0 20px;
                transition: all 0.15s ease;
            }}
            
            .stButton > button[data-testid="baseButton-primary"] {{
                background: {HP.primary} !important;
                color: white !important;
                border: none !important;
            }}
            
            .stButton > button[data-testid="baseButton-primary"]:hover {{
                background: {HP.primary_deep} !important;
                box-shadow: none;
            }}
            
            .stButton > button[kind="secondary"] {{
                background: white !important;
                color: {HP.ink} !important;
                border: 1px solid var(--hp-hairline) !important;
            }}
            
            /* === HP Section Bands === */
            .hp-section-cloud {{
                background: var(--hp-cloud);
                padding: 40px 24px;
                margin: 32px -24px;
            }}
            
            .hp-section-ink {{
                background: var(--hp-ink);
                color: white;
                padding: 40px 24px;
                margin: 32px -24px;
                border-radius: var(--hp-radius-xl);
            }}
            
            .hp-section-ink h2, 
            .hp-section-ink h3, 
            .hp-section-ink p {{ color: white; }}
            
            /* === HP Chevron Decoration === */
            .hp-chevron {{
                display: inline-block;
                width: 8px;
                height: 32px;
                background: {HP.primary};
                transform: skewX(-20deg);
                margin: 0 2px;
                border-radius: 0;
            }}
            
            .hp-chevron-pair {{
                display: flex;
                gap: 4px;
                margin-bottom: 12px;
            }}
            
            /* === HP KPI Metric === */
            .hp-metric {{
                padding: 16px 0;
            }}
            .hp-metric-label {{
                font-size: 12px;
                font-weight: 500;
                color: var(--hp-graphite);
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 4px;
            }}
            .hp-metric-value {{
                font-size: 28px;
                font-weight: 500;
                color: var(--hp-ink);
                line-height: 1.0;
            }}
            .hp-metric-sub {{
                font-size: 13px;
                color: var(--hp-charcoal);
                margin-top: 2px;
            }}
            
            /* === HP Timeline === */
            .hp-timeline-item {{
                display: flex;
                align-items: center;
                padding: 10px 0;
                border-bottom: 1px solid var(--hp-hairline);
            }}
            .hp-timeline-item:last-child {{ border-bottom: none; }}
            
            /* === HP Section Divider === */
            .hp-divider {{
                height: 1px;
                background: var(--hp-hairline);
                margin: 24px 0;
                border: none;
            }}
            
            /* === HP Table === */
            .hp-table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 13px;
            }}
            .hp-table th {{
                text-align: left;
                padding: 12px 16px;
                font-weight: 600;
                color: var(--hp-graphite);
                text-transform: uppercase;
                font-size: 11px;
                letter-spacing: 0.5px;
                border-bottom: 2px solid var(--hp-hairline);
                background: var(--hp-cloud);
            }}
            .hp-table td {{
                padding: 10px 16px;
                color: var(--hp-ink);
                border-bottom: 1px solid var(--hp-hairline);
            }}
            
            /* === HP Tags === */
            .hp-tag {{
                display: inline-block;
                padding: 2px 10px;
                border-radius: var(--hp-radius-lg);
                font-size: 12px;
                font-weight: 500;
            }}
            .hp-tag-blue {{ background: {HP.primary_soft}; color: {HP.primary}; }}
            .hp-tag-ink {{ background: var(--hp-ink); color: white; }}
            
            /* === HP Empty State === */
            .hp-empty {{
                text-align: center;
                padding: 48px 24px;
            }}
            .hp-empty-icon {{ font-size: 40px; margin-bottom: 12px; opacity: 0.4; }}
            .hp-empty-title {{ font-size: 20px; font-weight: 500; color: var(--hp-ink); margin-bottom: 8px; }}
            .hp-empty-desc {{ font-size: 14px; color: var(--hp-charcoal); max-width: 360px; margin: 0 auto 24px; }}
            
            /* === Animations === */
            .hp-fade-in {{ animation: hpFadeIn 0.35s ease-out; }}
            @keyframes hpFadeIn {{ from {{ opacity: 0; transform: translateY(4px); }} to {{ opacity: 1; transform: translateY(0); }} }}
            
            /* === Streamlit overrides === */
            .stTabs [data-baseweb="tab-list"] {{
                gap: 0;
                border-bottom: 2px solid var(--hp-hairline);
            }}
            .stTabs [data-baseweb="tab"] {{
                font-size: 13px;
                font-weight: 500;
                padding: 8px 20px;
                text-transform: uppercase;
                letter-spacing: 0.3px;
                color: var(--hp-charcoal);
            }}
            .stTabs [aria-selected="true"] {{
                color: {HP.primary} !important;
                border-bottom: 2px solid {HP.primary} !important;
            }}
            
            /* Metrics */
            [data-testid="stMetricValue"] {{
                font-size: 28px;
                font-weight: 500;
                color: var(--hp-ink);
            }}
            [data-testid="stMetricLabel"] {{
                font-size: 12px;
                font-weight: 500;
                color: var(--hp-graphite);
                text-transform: uppercase;
                letter-spacing: 0.3px;
            }}
            
            /* Expander */
            .streamlit-expanderHeader {{
                font-weight: 500;
                font-size: 14px;
                background: transparent;
                border: 1px solid var(--hp-hairline);
                border-radius: var(--hp-radius-lg);
            }}
        </style>
    """, unsafe_allow_html=True)
    apply_plotly_defaults()

# ===========================================================================
# HP UI Components
# ===========================================================================

def hp_chevron_header(title: str, subtitle: str = "") -> None:
    """Render an HP-style section header with chevron decoration."""
    html = f"""
        <div class="hp-fade-in" style="margin-bottom: 16px;">
            <div class="hp-chevron-pair">
                <div class="hp-chevron"></div>
                <div class="hp-chevron" style="opacity:0.6;"></div>
            </div>
            <h1 style="margin:0; font-size:32px; font-weight:500; color:#1a1a1a;">{title}</h1>
            {f'<p style="color:#636363; margin-top:4px; font-size:15px;">{subtitle}</p>' if subtitle else ''}
        </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def hp_kpi_card(title: str, value: str, subtitle: str = "", tone: str = "info", icon: str = "") -> None:
    """HP-style KPI metric card — clean, minimal, no borders."""
    st.markdown(f"""
        <div class="hp-fade-in" style="padding: 8px 0;">
            <div style="font-size:12px; font-weight:500; color:#636363; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px;">
                {icon} {title}
            </div>
            <div style="font-size:28px; font-weight:500; color:#1a1a1a; line-height:1.0;">
                {value}
            </div>
            {f'<div style="font-size:13px; color:#3d3d3d; margin-top:2px;">{subtitle}</div>' if subtitle else ''}
        </div>
    """, unsafe_allow_html=True)


def hp_alert_card(message: str, tone: str = "info", title: Optional[str] = None) -> None:
    """HP-style alert — minimal, blue-tinted."""
    border = "#024ad8" if tone in ("info", "success") else "#b8860b" if tone == "warning" else "#b3262b"
    bg = "#c9e0fc" if tone == "info" else "#d4edda" if tone == "success" else "#fff3cd" if tone == "warning" else "#f9d4d2"
    text = "#0e3191" if tone in ("info", "success") else "#856404" if tone == "warning" else "#b3262b"
    
    st.markdown(f"""
        <div style="padding:12px 16px; border-left:3px solid {border}; background:{bg}; border-radius:4px; margin-bottom:16px;">
            {f'<div style="font-weight:600; color:{text}; margin-bottom:2px;">{title}</div>' if title else ''}
            <div style="font-size:14px; color:{text};">{message}</div>
        </div>
    """, unsafe_allow_html=True)


def hp_timeline(events: List[Dict[str, str]]) -> None:
    """HP-style activity timeline — clean rows with left accent."""
    html = ['<div class="hp-card" style="padding:0;">']
    for ev in events:
        tone = ev.get("status", "neutral")
        dot = "#10b981" if tone == "success" else "#024ad8" if tone == "info" else "#ff5050" if tone == "danger" else "#c2c2c2"
        label = tone.upper()
        html.append(f"""
            <div class="hp-timeline-item" style="padding:12px 16px;">
                <div style="width:8px; height:8px; border-radius:50%; background:{dot}; margin-right:12px; flex-shrink:0;"></div>
                <div style="width:50px; font-size:12px; color:#636363;">{ev.get('time', '--:--')}</div>
                <div style="flex:1; font-size:14px; color:#1a1a1a;">{ev.get('label', '')}</div>
                <span style="font-size:10px; font-weight:600; color:{dot}; background:{dot}15; padding:2px 8px; border-radius:8px;">{label}</span>
            </div>
        """)
    html.append('</div>')
    st.markdown("".join(html), unsafe_allow_html=True)


def hp_status_badge(status: str) -> str:
    """HP-style status badge HTML."""
    tone_map = {
        "completed": "success", "success": "success", "done": "success",
        "warning": "warning", "failed": "danger", "error": "danger", "critical": "danger",
        "running": "info", "pending": "info", "queued": "neutral",
    }
    tone = tone_map.get(status.lower(), "neutral")
    return f'<span class="hp-badge {tone}">{status.upper()}</span>'


def hp_health_score(score: int, label: str = "HEALTHY") -> None:
    """HP-style health score — large, clean, emoji-free."""
    color = "#10b981" if score > 80 else "#b8860b" if score > 60 else "#b3262b"
    st.markdown(f"""
        <div class="hp-fade-in" style="text-align:center; padding:24px 16px;">
            <div style="font-size:11px; color:#636363; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:8px;">Composite Health</div>
            <div style="font-size:56px; font-weight:500; color:{color}; line-height:1;">{score}</div>
            <div style="font-size:14px; font-weight:500; color:{color}; margin-top:4px;">{label}</div>
        </div>
    """, unsafe_allow_html=True)


def hp_empty_state(title: str, message: str, action_label: Optional[str] = None, page_link: Optional[str] = None) -> None:
    """HP-style empty state — centered, clean, with CTA."""
    st.markdown(f"""
        <div class="hp-empty hp-fade-in">
            <div class="hp-empty-icon">&#9744;</div>
            <div class="hp-empty-title">{title}</div>
            <div class="hp-empty-desc">{message}</div>
        </div>
    """, unsafe_allow_html=True)
    if action_label and page_link:
        st.page_link(page_link, label=action_label)


def hp_insight_panel(insights: List[str]) -> None:
    """HP-style insight panel — blue left border accent."""
    items = "".join([f'<div style="font-size:14px; color:#1a1a1a; margin-bottom:8px; display:flex; gap:8px;"><span style="color:#024ad8;">&rarr;</span><span>{ins}</span></div>' for ins in insights])
    st.markdown(f"""
        <div class="hp-fade-in" style="padding:16px; border-left:3px solid #024ad8; background:#f7f7f7; border-radius:4px; margin-top:8px;">
            <div style="font-size:11px; font-weight:600; color:#636363; text-transform:uppercase; letter-spacing:0.3px; margin-bottom:8px;">Insights</div>
            {items}
        </div>
    """, unsafe_allow_html=True)


def hp_registry_card(name: str, version: str, stage: str, dataset: str, metrics: Dict[str, float], model_id: str) -> bool:
    """HP-style registry card — product tile layout."""
    tone = STAGE_TO_TONE.get(stage.lower(), "info")
    stage_color = "#10b981" if tone == "success" else "#b8860b" if tone == "warning" else "#024ad8"
    
    st.markdown(f"""
        <div class="hp-card hp-fade-in" style="border-top: none;">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:8px;">
                <div>
                    <div style="font-size:16px; font-weight:500; color:#1a1a1a;">{name}</div>
                    <div style="font-size:12px; color:#636363;">v{version} &middot; {dataset}</div>
                </div>
                <span class="hp-tag hp-tag-blue">{stage.upper()}</span>
            </div>
            <div style="display:flex; gap:16px; margin:12px 0; font-size:13px; color:#3d3d3d;">
                <span>Acc: {metrics.get('accuracy', 0):.4f}</span>
                <span>F1: {metrics.get('f1_score', 0):.4f}</span>
            </div>
            <div style="font-size:11px; color:#636363;">ID: {model_id[:12]}...</div>
        </div>
    """, unsafe_allow_html=True)
    return st.button("Configure Lifecycle", key=f"btn_{model_id}", use_container_width=True)


# ===========================================================================
# Global Shell — HP-style Top Nav & Sidebar
# ===========================================================================

def render_top_navbar(user_role: str = "Admin") -> None:
    st.markdown(f"""
        <style>
            .hp-topnav {{
                display:flex; align-items:center; justify-content:space-between;
                height:56px; padding:0 24px;
                background:white; border-bottom:1px solid #e8e8e8;
                margin:-0.5rem -1rem 0 -1rem;
            }}
            .hp-logo {{
                font-weight:600; font-size:16px; color:#1a1a1a;
                display:flex; align-items:center; gap:10px;
            }}
            .hp-logo-mark {{
                display:inline-block;
                width:20px; height:20px;
                background:#024ad8;
                border-radius:4px;
                position:relative;
            }}
            .hp-logo-mark::after {{
                content:'';
                position:absolute;
                top:2px; left:2px; right:2px; bottom:2px;
                background:white;
                border-radius:2px;
            }}
            .hp-nav-search {{
                display:flex; align-items:center;
                background:#f7f7f7; border:1px solid #e8e8e8;
                border-radius:4px; padding:4px 12px;
                width:260px; height:34px;
            }}
            .hp-nav-search input {{
                background:transparent; border:none; color:#1a1a1a;
                font-size:13px; width:100%; outline:none;
                font-family:Inter, sans-serif;
            }}
        </style>
    """, unsafe_allow_html=True)
    
    col_nav, col_actions = st.columns([2, 1])
    with col_nav:
        st.markdown("""
            <div class="hp-topnav" style="justify-content:flex-start; gap:24px;">
                <div class="hp-logo">
                    <span class="hp-logo-mark"></span>
                    ML MONITOR
                    <span style="font-size:10px; font-weight:600; color:#024ad8; background:#c9e0fc; padding:2px 8px; border-radius:4px; letter-spacing:0.3px;">PROD</span>
                </div>
                <div class="hp-nav-search">
                    <span style="color:#636363; margin-right:6px;">&#128269;</span>
                    <input type="text" placeholder="Search experiments..." />
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col_actions:
        st.markdown('<div class="hp-topnav" style="justify-content:flex-end; gap:12px;">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1: 
            with st.popover("&#128276;"): st.markdown("**Notifications**"); st.divider(); st.caption("System Healthy")
        with c2: 
            with st.popover("&#9889;"): st.markdown("**Quick Actions**"); st.page_link("pages/1_Pipeline_Runner.py", label="Run Pipeline"); st.page_link("pages/0_Dataset_Management.py", label="Import Dataset")
        with c3:
            st.markdown(f'<div style="display:flex; align-items:center; gap:8px; height:38px;"><div style="width:28px; height:28px; border-radius:50%; background:#024ad8; display:flex; align-items:center; justify-content:center; font-size:12px; font-weight:600; color:white;">{user_role[0]}</div><span style="font-size:13px; font-weight:500; color:#1a1a1a;">{user_role}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)


def render_sidebar_nav() -> None:
    st.markdown(f"""
        <style>
            .hp-nav-section {{
                font-size:10px; font-weight:600; color:#636363;
                text-transform:uppercase; letter-spacing:0.5px;
                padding:12px 16px 4px;
            }}
            .hp-nav-item {{
                display:flex; align-items:center; gap:10px;
                padding:8px 16px;
                border-radius:4px;
                color:#1a1a1a;
                text-decoration:none;
                font-size:13px; font-weight:500;
                margin-bottom:2px;
                transition:all 0.1s ease;
            }}
            .hp-nav-item:hover {{
                background:#f7f7f7;
            }}
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="hp-nav-section">Platform</div>', unsafe_allow_html=True)
    
    nav_items = [
        ("🏠", "Dashboard", "app.py"),
        ("🗂️", "Dataset Hub", "pages/0_Dataset_Management.py"),
        ("⚙️", "Pipeline Runner", "pages/1_Pipeline_Runner.py"),
        ("📊", "Experiment Tracking", "pages/2_Experiment_Tracking.py"),
        ("📦", "Model Registry", "pages/3_Model_Registry.py"),
    ]
    
    for icon, label, page in nav_items:
        st.page_link(page, label=f"{icon} {label}")
    
    st.markdown('<div class="hp-nav-section" style="margin-top:16px;">Observability</div>', unsafe_allow_html=True)
    
    obs_items = [
        ("📈", "Data Drift", "pages/4_Data_Drift.py"),
        ("🖥️", "System Health", "pages/5_System_Health.py"),
        ("⚖️", "Governance", "pages/6_Governance.py"),
    ]
    
    for icon, label, page in obs_items:
        st.page_link(page, label=f"{icon} {label}")


# ===========================================================================
# Compatibility stubs — map old names to HP versions
# ===========================================================================

# Legacy aliases
def component_kpi_card(title: str, value: str, subtitle: Optional[str] = None, tone: str = "info", icon: Optional[str] = None, trend: Optional[str] = None) -> None:
    hp_kpi_card(title, value, subtitle or "", tone, icon or "")
    
component_alert_card = hp_alert_card
component_timeline = hp_timeline
component_status_badge = hp_status_badge
component_health_score = hp_health_score
component_empty_state = hp_empty_state
component_insight_panel = hp_insight_panel
component_registry_card = hp_registry_card

def component_metric_badge(label: str, value: str, tone: str = "info") -> str:
    return f'<span class="hp-tag hp-tag-blue">{label}: {value}</span>'

def render_kpi_row(items: Sequence[dict]) -> None:
    if not items: return
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col: hp_kpi_card(title=item.get("title", ""), value=item.get("value", "—"), subtitle=item.get("subtitle", ""), tone=item.get("tone", "info"), icon=item.get("icon", ""))

def render_section_title(title: str, margin_top_px: int = 0) -> None:
    st.markdown(f'<h2 style="margin-top:{margin_top_px}px;">{title}</h2>', unsafe_allow_html=True)

def render_spacer(size: str = "md") -> None:
    px_map = {"xs": 8, "sm": 16, "md": 24, "lg": 32, "xl": 40}
    st.markdown(f"<div style='height:{px_map.get(size, 24)}px'></div>", unsafe_allow_html=True)

def render_loading_skeleton(lines: int = 4, key: str = "skeleton") -> None:
    blocks = "".join([f'<div style="height:10px; border-radius:4px; background:#f0f0f0; margin-bottom:8px; width:{70-(i%3)*10}%"></div>' for i in range(lines)])
    st.markdown(f'<div class="hp-card">{blocks}</div>', unsafe_allow_html=True)

def status_badge_html(status: str) -> str: return hp_status_badge(status)
def stage_badge_html(stage: str) -> str: return f'<span class="hp-tag hp-tag-blue">{stage.upper()}</span>'

def render_summary_table(df: pd.DataFrame, *, columns: Sequence[str], **kwargs) -> pd.DataFrame:
    if df.empty: st.info("No data available."); return df
    show_cols = [c for c in columns if c in df.columns]
    header_html = "".join([f'<th>{c}</th>' for c in show_cols])
    rows_html = []
    for _, row in df.iterrows():
        cells = []
        for c in show_cols:
            val = str(row[c])
            if c.lower() in ["status", "stage", "severity", "drift"]:
                val = hp_status_badge(val) if c.lower() != "stage" else stage_badge_html(val)
            cells.append(f'<td>{val}</td>')
        rows_html.append(f"<tr>{''.join(cells)}</tr>")
    st.markdown(f'<table class="hp-table"><thead><tr>{header_html}</tr></thead><tbody>{"".join(rows_html)}</tbody></table>', unsafe_allow_html=True)
    return df

def render_expandable_rows(df: pd.DataFrame, *, title_col: str, detail_cols: Sequence[str], badge_col: Optional[str] = None, **kwargs) -> None:
    for _, row in df.iterrows():
        badge = hp_status_badge(str(row[badge_col])) if badge_col else ""
        with st.expander(f"{row[title_col]} {badge}"):
            for c in detail_cols: st.markdown(f"**{c}:** {row[c]}")