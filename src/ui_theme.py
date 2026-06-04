"""
Enterprise UI/UX Design System for ML Pipeline Monitor.
Implements a reusable, component-based architecture for premium Dark SaaS interfaces.
"""

from __future__ import annotations

from html import escape
import json
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ---------------------------------------------------------------------------
# Design System Tokens
# ---------------------------------------------------------------------------

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

PLOTLY_COLORWAY = ["#6366F1", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899"]

# ---------------------------------------------------------------------------
# Plotly Theming
# ---------------------------------------------------------------------------

def apply_plotly_defaults() -> None:
    if "mlmonitor_dark" not in pio.templates:
        base = pio.templates["plotly_dark"]
        template_dict = base.to_plotly_json() if hasattr(base, "to_plotly_json") else {}
        layout_config = template_dict.get("layout", {}) if isinstance(template_dict, dict) else {}
        layout_config.update({
            "colorway": PLOTLY_COLORWAY,
            "font": {"family": "Inter, sans-serif", "size": 12, "color": "#F9FAFB"},
            "plot_bgcolor": "rgba(0,0,0,0)",
            "paper_bgcolor": "rgba(0,0,0,0)",
            "legend": {"title": None, "orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
            "margin": {"l": 10, "r": 10, "t": 40, "b": 10},
            "xaxis": {"showgrid": False, "linecolor": "#374151", "tickfont": {"color": "#9CA3AF", "size": 10}},
            "yaxis": {"showgrid": True, "gridcolor": "rgba(55, 65, 81, 0.4)", "linecolor": "#374151", "tickfont": {"color": "#9CA3AF", "size": 10}},
        })
        pio.templates["mlmonitor_dark"] = go.layout.Template(layout=layout_config)
    pio.templates.default = "mlmonitor_dark"

# ---------------------------------------------------------------------------
# Global Style Injection
# ---------------------------------------------------------------------------

def apply_ui_theme() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Poppins:wght@600&display=swap');
            :root {
                --color-background-primary: #0F172A;
                --color-background-secondary: #111827;
                --color-background-panel: #1F2937;
                --color-border-primary: #374151;
                --color-border-secondary: #4B5563;
                --color-text-primary: #F9FAFB;
                --color-text-secondary: #D1D5DB;
                --color-text-tertiary: #9CA3AF;
                --color-accent: #6366F1;
                --radius-md: 8px;
                --radius-lg: 12px;
                --radius-pill: 999px;
            }
            html, body, [class*="css"] { font-family: "Inter", sans-serif; color: var(--color-text-primary); }
            .stApp { background: var(--color-background-primary); }
            .block-container { padding-top: 1rem; max-width: none; }
            [data-testid="stSidebar"] { background: var(--color-background-secondary); border-right: 1px solid var(--color-border-primary); min-width: 240px; }
            [data-testid="stSidebarNav"] { display: none; }
            
            /* Component Specific Styles */
            .ui-card { background: var(--color-background-panel); border: 1px solid var(--color-border-primary); border-radius: var(--radius-lg); padding: 20px; margin-bottom: 20px; transition: border-color 0.2s ease; }
            .ui-card:hover { border-color: var(--color-border-secondary); }
            
            .ui-badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; border-radius: var(--radius-pill); font-size: 11px; font-weight: 600; border: 1px solid transparent; }
            .ui-badge.success { background: rgba(16, 185, 129, 0.1); color: #10B981; border-color: rgba(16, 185, 129, 0.2); }
            .ui-badge.warning { background: rgba(245, 158, 11, 0.1); color: #F59E0B; border-color: rgba(245, 158, 11, 0.2); }
            .ui-badge.danger { background: rgba(239, 68, 68, 0.1); color: #EF4444; border-color: rgba(239, 68, 68, 0.2); }
            .ui-badge.info { background: rgba(99, 102, 241, 0.1); color: #6366F1; border-color: rgba(99, 102, 241, 0.2); }
            
            .ui-fade-in { animation: fadeIn 0.4s ease-out; }
            @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
            
            .nav-link { display: flex; align-items: center; gap: 12px; padding: 10px 14px; border-radius: var(--radius-md); color: var(--color-text-secondary); text-decoration: none; font-size: 13px; margin-bottom: 4px; transition: all 0.2s ease; }
            .nav-link:hover { background: rgba(255, 255, 255, 0.05); color: var(--color-text-primary); }
            .nav-link.active { background: rgba(99, 102, 241, 0.12); color: var(--color-accent); font-weight: 600; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    apply_plotly_defaults()

# ---------------------------------------------------------------------------
# Reusable UI Components
# ---------------------------------------------------------------------------

def component_kpi_card(title: str, value: str, subtitle: Optional[str] = None, tone: str = "info", icon: Optional[str] = None, trend: Optional[str] = None) -> None:
    """Standardized KPI Card component."""
    tone_key = STATUS_TO_TONE.get(tone.lower(), "neutral")
    trend_html = ""
    if trend:
        color = "#10B981" if trend.startswith("+") else "#EF4444"
        trend_html = f'<span style="color: {color}; font-size: 11px; font-weight: 700; margin-left: 8px;">{trend}</span>'
    
    st.markdown(
        f"""
        <div class="ui-card ui-fade-in" style="padding: 16px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <div style="font-size:11px; font-weight:600; color:var(--color-text-tertiary); text-transform:uppercase; letter-spacing:0.05em;">{title}</div>
                <div style="font-size:14px;">{icon or ""}</div>
            </div>
            <div style="display:flex; align-items:baseline;">
                <div style="font-size:26px; font-weight:700; color:white;">{value}</div>
                {trend_html}
            </div>
            <div style="font-size:11px; color:var(--color-text-tertiary); margin-top:4px;">{subtitle or ""}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def component_alert_card(message: str, tone: str = "warning", title: Optional[str] = None) -> None:
    """Standardized Alert/Notification Card."""
    tone_key = STATUS_TO_TONE.get(tone.lower(), "warning")
    st.markdown(
        f"""
        <div class="ui-badge {tone_key} ui-fade-in" style="width:100%; border-radius:8px; padding:16px; margin-bottom:16px; display:block;">
            {f'<div style="font-weight:700; margin-bottom:4px;">{title}</div>' if title else ''}
            <div style="font-size:13px; opacity:0.9;">{message}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def component_timeline(events: List[Dict[str, str]]) -> None:
    """Standardized Activity Timeline component."""
    html = ['<div class="ui-card ui-fade-in" style="padding: 0;">']
    for ev in events:
        tone = ev.get("status", "neutral")
        badge = f'<span class="ui-badge {STATUS_TO_TONE.get(tone, "neutral")}" style="padding: 2px 8px; font-size: 9px;">{tone.upper()}</span>'
        html.append(f"""
            <div style="display: flex; align-items: center; padding: 12px 16px; border-bottom: 1px solid rgba(255,255,255,0.03);">
                <div style="width: 50px; font-size: 11px; color: var(--color-text-tertiary);">{ev.get('time', '--:--')}</div>
                <div style="flex: 1; font-size: 13px; color: white;">{ev.get('label', '')}</div>
                <div>{badge}</div>
            </div>
        """)
    html.append('</div>')
    st.markdown("".join(html), unsafe_allow_html=True)

def component_metric_badge(label: str, value: str, tone: str = "info") -> str:
    """Returns HTML for a small metric badge."""
    return f"""
        <div style="background:rgba(255,255,255,0.02); padding:8px 12px; border-radius:6px; border:1px solid rgba(255,255,255,0.05); display:inline-block; min-width:80px;">
            <div style="font-size:9px; color:var(--color-text-tertiary); text-transform:uppercase;">{label}</div>
            <div style="font-size:14px; font-weight:700; color:white;">{value}</div>
        </div>
    """

def component_status_badge(status: str) -> str:
    """Returns HTML for a status badge."""
    tone = STATUS_TO_TONE.get(status.lower(), "neutral")
    return f'<span class="ui-badge {tone}">{status.upper()}</span>'

def component_health_score(score: int, label: str = "HEALTHY") -> None:
    """Large Composite Health Score component."""
    color = "#10B981" if score > 80 else "#F59E0B" if score > 60 else "#EF4444"
    st.markdown(
        f"""
        <div class="ui-card ui-fade-in" style="text-align: center; padding: 32px 16px;">
            <div style="font-size: 11px; color: var(--color-text-tertiary); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 12px;">Composite Health</div>
            <div style="font-size: 64px; font-weight: 800; color: {color}; line-height: 1;">{score}</div>
            <div style="font-size: 14px; font-weight: 600; color: {color}; margin-top: 8px;">{label}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def component_registry_card(name: str, version: str, stage: str, dataset: str, metrics: Dict[str, float], model_id: str) -> bool:
    """Standardized Registry Model Card with action button."""
    tone = STAGE_TO_TONE.get(stage.lower(), "info")
    stage_color = "#10B981" if tone == "success" else "#F59E0B" if tone == "warning" else "#6366F1"
    
    st.markdown(f"""
        <div class="ui-card ui-fade-in" style="border-top: 4px solid {stage_color};">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:12px;">
                <div>
                    <div style="font-size:16px; font-weight:700; color:white;">{name}</div>
                    <div style="font-size:11px; color:var(--color-text-tertiary);">v{version} • {dataset}</div>
                </div>
                <span class="ui-badge {tone}">{stage.upper()}</span>
            </div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; margin:16px 0;">
                {component_metric_badge("Accuracy", f"{metrics.get('accuracy', 0):.4f}")}
                {component_metric_badge("F1 Score", f"{metrics.get('f1_score', 0):.4f}")}
            </div>
            <div style="font-size:10px; color:var(--color-text-tertiary); margin-bottom:12px;">ID: {model_id[:16]}...</div>
        </div>
    """, unsafe_allow_html=True)
    return st.button("Configure Lifecycle", key=f"btn_{model_id}", use_container_width=True)

def component_empty_state(title: str, message: str, action_label: Optional[str] = None, page_link: Optional[str] = None) -> None:
    """Standardized Empty State component."""
    st.markdown(
        f"""
        <div class="ui-card ui-fade-in" style="text-align:center; padding:60px 40px; border-style:dashed;">
            <div style="font-size:48px; margin-bottom:20px;">📂</div>
            <div style="font-size:20px; font-weight:700; color:white; margin-bottom:12px;">{title}</div>
            <div style="font-size:14px; color:var(--color-text-tertiary); margin-bottom:24px; max-width:400px; margin-left:auto; margin-right:auto;">{message}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    if action_label and page_link:
        st.page_link(page_link, label=action_label, icon="🚀")

def component_insight_panel(insights: List[str]) -> None:
    """Standardized AI Insight Panel component."""
    insights_html = "".join([f'<div style="font-size: 13px; color: var(--color-text-secondary); margin-bottom: 10px; display:flex; gap:8px;"><span>•</span><span>{ins}</span></div>' for ins in insights])
    st.markdown(
        f"""
        <div class="ui-fade-in" style="padding: 16px; background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(139, 92, 246, 0.08) 100%); border: 1px solid rgba(99, 102, 241, 0.2); border-radius: var(--radius-lg); margin-top: 10px;">
            <div style="display: flex; align-items: center; gap: 8px; color: var(--color-accent); font-weight: 700; font-size: 12px; margin-bottom: 12px; text-transform: uppercase;">
                <span>✨</span> AI SYSTEM INSIGHTS
            </div>
            {insights_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Global Shell Components
# ---------------------------------------------------------------------------

def render_top_navbar(user_role: str = "Admin") -> None:
    st.markdown(
        """
        <style>
            .navbar-search { background: rgba(255, 255, 255, 0.05); border: 1px solid var(--color-border-primary); border-radius: 6px; padding: 4px 12px; display: flex; align-items: center; gap: 8px; width: 300px; }
            .navbar-search input { background: transparent; border: none; color: white; font-size: 13px; width: 100%; outline: none; }
        </style>
        """,
        unsafe_allow_html=True
    )
    col_nav, col_actions = st.columns([2, 1])
    with col_nav:
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:24px; padding:12px 24px; background:var(--color-background-secondary); border-bottom:1px solid var(--color-border-primary); margin:-1rem -1rem 0 -1rem;">
                <div style="display:flex; align-items:center; gap:12px; font-family:'Poppins', sans-serif; font-weight:600; font-size:16px; color:white;">
                    <span style="font-size:20px;">⚡</span><span>ML MONITOR</span><span style="padding:2px 8px; border-radius:4px; font-size:10px; font-weight:700; text-transform:uppercase; background:rgba(99,102,241,0.2); color:var(--color-accent); border:1px solid var(--color-accent);">PROD</span>
                </div>
                <div class="navbar-search"><span style="color:var(--color-text-tertiary);">🔍</span><input type="text" placeholder="Search experiments..." /></div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_actions:
        st.markdown('<div style="display:flex; align-items:center; justify-content:flex-end; gap:20px; padding:12px 24px; background:var(--color-background-secondary); border-bottom:1px solid var(--color-border-primary); margin:-1rem -1rem 0 -1rem;">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1: 
            with st.popover("🔔"): st.markdown("**Notifications**"); st.divider(); st.caption("System Healthy")
        with c2: 
            with st.popover("⚡"): st.markdown("**Quick Actions**"); st.page_link("pages/1_Pipeline_Runner.py", label="Run Pipeline")
        with c3:
            st.markdown(f'<div style="display:flex; align-items:center; gap:10px; height:38px;"><div style="width:28px; height:28px; border-radius:50%; background:var(--color-accent); display:flex; align-items:center; justify-content:center; font-size:12px; font-weight:700;">{user_role[0]}</div><span style="font-size:13px; font-weight:500; color:var(--color-text-secondary);">{user_role}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div style="height: 1.5rem;"></div>', unsafe_allow_html=True)

def render_sidebar_nav() -> None:
    nav_items = [
        {"label": "Dashboard", "icon": "🏠", "page": "app.py"},
        {"label": "Pipeline Runner", "icon": "⚙️", "page": "pages/1_Pipeline_Runner.py"},
        {"label": "Experiment Tracking", "icon": "📊", "page": "pages/2_Experiment_Tracking.py"},
        {"label": "Model Registry", "icon": "📦", "page": "pages/3_Model_Registry.py"},
        {"label": "Data Drift", "icon": "📈", "page": "pages/4_Data_Drift.py"},
        {"label": "System Health", "icon": "🖥️", "page": "pages/5_Data_Health.py"},
    ]
    for item in nav_items:
        st.page_link(item["page"], label=item["label"], icon=item["icon"])

# ---------------------------------------------------------------------------
# Backward Compatibility Stubs (Internal redirects)
# ---------------------------------------------------------------------------
def render_kpi_row(items: Sequence[dict]) -> None:
    if not items: return
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col: component_kpi_card(title=item.get("title", ""), value=item.get("value", "—"), subtitle=item.get("subtitle"), tone=item.get("tone", "neutral"), icon=item.get("icon"), trend=item.get("trend"))

def render_section_title(title: str, margin_top_px: int = 0) -> None:
    st.markdown(f'<div style="font-family:\'Poppins\', sans-serif; font-size:18px; font-weight:600; color:white; margin-bottom:16px; margin-top:{margin_top_px}px;">{title}</div>', unsafe_allow_html=True)

def render_spacer(size: str = "md") -> None:
    px_map = {"xs": 8, "sm": 16, "md": 24, "lg": 32, "xl": 40}
    st.markdown(f"<div style='height:{px_map.get(size, 24)}px'></div>", unsafe_allow_html=True)

def render_loading_skeleton(lines: int = 4, key: str = "skeleton") -> None:
    blocks = "".join([f'<div style="height:12px; border-radius:6px; background:rgba(255,255,255,0.05); margin-bottom:10px; width:{90-(i%3)*10}%"></div>' for i in range(lines)])
    st.markdown(f'<div class="ui-card ui-fade-in">{blocks}</div>', unsafe_allow_html=True)

def status_badge_html(status: str) -> str: return component_status_badge(status)
def stage_badge_html(stage: str) -> str: return f'<span class="ui-badge {STAGE_TO_TONE.get(stage.lower(), "info")}">{stage.upper()}</span>'

def render_summary_table(df: pd.DataFrame, *, columns: Sequence[str], **kwargs) -> pd.DataFrame:
    if df.empty: st.info("No data available."); return df
    show_cols = [c for c in columns if c in df.columns]
    header_html = "".join([f'<th style="text-align:left; padding:14px 16px; background:rgba(255,255,255,0.03); color:var(--color-text-tertiary); font-size:11px; font-weight:700; text-transform:uppercase; border-bottom:1px solid var(--color-border-primary);">{c}</th>' for c in show_cols])
    rows_html = []
    for _, row in df.iterrows():
        cells = []
        for c in show_cols:
            val = str(row[c])
            if c.lower() in ["status", "stage", "severity", "drift"]:
                val = component_status_badge(val) if c.lower() != "stage" else stage_badge_html(val)
            cells.append(f'<td style="padding:12px 16px; color:var(--color-text-secondary); font-size:13px; border-bottom:1px solid rgba(255,255,255,0.02);">{val}</td>')
        rows_html.append(f"<tr>{''.join(cells)}</tr>")
    st.markdown(f'<div style="border-radius:12px; border:1px solid var(--color-border-primary); overflow:hidden; margin-bottom:20px;"><table style="width:100%; border-collapse:collapse; background:var(--color-background-panel);"><thead><tr>{header_html}</tr></thead><tbody>{"".join(rows_html)}</tbody></table></div>', unsafe_allow_html=True)
    return df

def render_expandable_rows(df: pd.DataFrame, *, title_col: str, detail_cols: Sequence[str], badge_col: Optional[str] = None, **kwargs) -> None:
    for _, row in df.iterrows():
        badge = component_status_badge(str(row[badge_col])) if badge_col else ""
        with st.expander(f"{row[title_col]} {badge}"):
            for c in detail_cols: st.markdown(f"**{c}:** {row[c]}")
