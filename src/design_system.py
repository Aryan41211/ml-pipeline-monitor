"""Premium Enterprise Design System for ML Pipeline Monitor
Inspired by Vercel, Linear, Stripe, and modern SaaS interfaces.
Dark mode first with elegant light mode support.
"""

from typing import Optional, Dict

# ===========================================================================
# Color Palette — Dual Theme (Dark + Light)
# ===========================================================================

# Dark Mode (Default) — ui_theme HP palette (applied across all pages)
DARK = {
    "background": "#0B0F19",
    "surface": "#111827",
    "card": "#1F2937",
    "card_hover": "#263244",
    "text_primary": "#F9FAFB",
    "text_secondary": "#9CA3AF",
    "text_tertiary": "#6B7280",
    "accent": "#3B82F6",
    "accent_bright": "#60A5FA",
    "accent_deep": "#2563EB",
    "accent_soft": "#DBEAFE",
    "success": "#10B981",
    "success_soft": "#10B98115",
    "warning": "#FBBF24",
    "warning_soft": "#FBBF2415",
    "error": "#EF4444",
    "error_soft": "#EF444415",
    "border": "#374151",
    "border_strong": "#4B5563",
    "glass": "rgba(26, 26, 26, 0.6)",
    "glass_border": "rgba(255, 255, 255, 0.08)",
    "ink": "#F9FAFB",
    "ink_deep": "#E5E7EB",
    "ink_soft": "#9CA3AF",
    "canvas": "#0B0F19",
    "cloud": "#111827",
    "fog": "#1F2937",
    "steel": "#4B5563",
    "graphite": "#9CA3AF",
    "charcoal": "#6B7280",
    "hairline": "#374151",
    "bloom_deep": "#EF4444",
    "bloom_rose": "#FCA5A5",
}

# Light Mode — Clean minimal
LIGHT = {
    "background": "#FFFFFF",
    "surface": "#FAFAFA",
    "card": "#FFFFFF",
    "card_hover": "#F5F5F5",
    "text_primary": "#111111",
    "text_secondary": "#555555",
    "text_tertiary": "#777777",
    "accent": "#2563EB",
    "accent_bright": "#3B82F6",
    "accent_deep": "#1D4ED8",
    "accent_soft": "#DBEAFE",
    "success": "#10B981",
    "success_soft": "#10B98115",
    "warning": "#FBBF24",
    "warning_soft": "#FBBF2415",
    "error": "#EF4444",
    "error_soft": "#EF444415",
    "border": "#E5E5E5",
    "border_strong": "#CCCCCC",
    "glass": "rgba(255, 255, 255, 0.7)",
    "glass_border": "rgba(0, 0, 0, 0.08)",
    "ink": "#111111",
    "ink_deep": "#333333",
    "ink_soft": "#555555",
    "canvas": "#FFFFFF",
    "cloud": "#FAFAFA",
    "fog": "#FFFFFF",
    "steel": "#CCCCCC",
    "graphite": "#555555",
    "charcoal": "#777777",
    "hairline": "#E5E5E5",
    "bloom_deep": "#EF4444",
    "bloom_rose": "#FECACA",
}

# Original COLORS dict for backward compatibility with components.py
COLORS = {
    "dark": {
        "background": "#0A0A0A",
        "surface": "#111111",
        "card": "#1A1A1A",
        "card_hover": "#222222",
        "glass": "rgba(26, 26, 26, 0.6)",
        "glass_border": "rgba(255, 255, 255, 0.08)",
        "text_primary": "#FFFFFF",
        "text_secondary": "#A0A0A0",
        "text_tertiary": "#707070",
        "accent": "#0070F3",
        "accent_bright": "#3399FF",
        "accent_soft": "#1A3FFF",
        "success": "#00C951",
        "success_soft": "#10B98115",
        "warning": "#F59E0B",
        "warning_soft": "#F59E0B15",
        "error": "#EF4444",
        "error_soft": "#EF444415",
        "border": "#2A2A2A",
        "border_strong": "#404040",
    },
    "light": {
        "background": "#FFFFFF",
        "surface": "#FAFAFA",
        "card": "#FFFFFF",
        "card_hover": "#F5F5F5",
        "glass": "rgba(255, 255, 255, 0.7)",
        "glass_border": "rgba(0, 0, 0, 0.08)",
        "text_primary": "#111111",
        "text_secondary": "#555555",
        "text_tertiary": "#777777",
        "accent": "#0070F3",
        "accent_bright": "#0052CC",
        "accent_soft": "#4DA6FF",
        "success": "#00C951",
        "success_soft": "#00C95115",
        "warning": "#F59E0B",
        "warning_soft": "#F59E0B15",
        "error": "#EF4444",
        "error_soft": "#EF444415",
        "border": "#E5E5E5",
        "border_strong": "#CCCCCC",
    }
}

def get_color(theme: str = "dark", key: str = "background") -> str:
    return COLORS.get(theme, DARK).get(key, "")

# ===========================================================================
# Premium Typography
# ===========================================================================

TYPOGRAPHY = {
    "font_family": "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif",
    "font_mono": "'JetBrains Mono', 'Fira Code', monospace",
    "h1": {"size": "32px", "weight": "700", "line_height": "1.2", "letter_spacing": "-0.02em"},
    "h2": {"size": "24px", "weight": "600", "line_height": "1.3", "letter_spacing": "-0.01em"},
    "h3": {"size": "18px", "weight": "600", "line_height": "1.4", "letter_spacing": "0"},
    "h4": {"size": "16px", "weight": "600", "line_height": "1.4", "letter_spacing": "0.01em"},
    "body": {"size": "15px", "weight": "400", "line_height": "1.5", "letter_spacing": "0"},
    "caption": {"size": "13px", "weight": "500", "line_height": "1.4", "letter_spacing": "0.02em"},
    "small": {"size": "12px", "weight": "500", "line_height": "1.3", "letter_spacing": "0.03em"},
}

# ===========================================================================
# Spacing Scale
# ===========================================================================

SPACING = {
    "xs": "4px",
    "sm": "8px",
    "md": "16px",
    "lg": "24px",
    "xl": "32px",
    "xxl": "48px",
}

SPACING_PX = {
    "xs": 4,
    "sm": 8,
    "md": 16,
    "lg": 24,
    "xl": 32,
    "xxl": 48,
}

# ===========================================================================
# Premium Shadows
# ===========================================================================

SHADOWS = {
    "card": "0 4px 12px rgba(0, 0, 0, 0.15)",
    "card_hover": "0 8px 24px rgba(0, 0, 0, 0.25)",
    "modal": "0 24px 60px rgba(0, 0, 0, 0.4)",
    "glass": "0 8px 32px rgba(0, 0, 0, 0.12)",
    "glow": "0 0 20px rgba(0, 112, 243, 0.3)",
    "soft": "0 4px 12px rgba(0, 0, 0, 0.25)",
    "hover": "0 8px 24px rgba(0, 0, 0, 0.35)",
    "light": "0 2px 8px rgba(0, 0, 0, 0.08)",
}

# ===========================================================================
# Border Radius
# ===========================================================================

BORDER_RADIUS = {
    "sm": "6px",
    "md": "8px",
    "lg": "12px",
    "xl": "16px",
    "pill": "9999px",
    "full": "50%",
}

# ===========================================================================
# Animation Presets
# ===========================================================================

ANIMATION = {
    "fast": "120ms",
    "normal": "200ms",
    "slow": "300ms",
    "easing": "cubic-bezier(0.4, 0, 0.2, 1)",
    "spring": "cubic-bezier(0.34, 1.56, 0.64, 1)",
}

# ===========================================================================
# Gradients
# ===========================================================================

GRADIENTS = {
    "accent": "linear-gradient(135deg, #0070F3 0%, #1A3FFF 100%)",
    "success": "linear-gradient(135deg, #00C951 0%, #10B981 100%)",
    "warning": "linear-gradient(135deg, #F59E0B 0%, #FFA500 100%)",
    "error": "linear-gradient(135deg, #EF4444 0%, #F87171 100%)",
    "glass": "linear-gradient(135deg, rgba(26,26,26,0.8) 0%, rgba(36,36,36,0.9) 100%)",
    "glass_light": "linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(245,245,245,0.95) 100%)",
}

# ===========================================================================
# Chart Colors — Premium palette
# ===========================================================================

CHART_COLORS = {
    "primary": ["#0070F3", "#1A3FFF", "#3399FF"],
    "success": ["#00C951", "#10B981", "#34D399"],
    "warning": ["#F59E0B", "#FFA500", "#FCD34D"],
    "error": ["#EF4444", "#F87171", "#FCA5A5"],
    "neutral": ["#A0A0A0", "#808080", "#606060"],
    "palette": ["#0070F3", "#00C951", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899", "#14B8A6", "#F97316"],
}

# ===========================================================================
# Plotly Color Theme
# ===========================================================================

PLOTLY_COLORWAY = ["#3B82F6", "#06B6D4", "#8B5CF6", "#10B981", "#F59E0B", "#EF4444", "#EC4899"]

# ===========================================================================
# Status Mappings
# ===========================================================================

STATUS_TO_TONE: Dict[str, str] = {
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

# ===========================================================================
# Plotly Theming — Premium Dark / Light Templates
# ===========================================================================

def apply_plotly_defaults(theme: str = "dark") -> None:
    import plotly.graph_objects as go
    import plotly.io as pio

    tokens = DARK if theme == "dark" else LIGHT
    template_name = "hp_dark" if theme == "dark" else "hp_light"

    if template_name not in pio.templates:
        pio.templates[template_name] = go.layout.Template(
            layout={
                "colorway": PLOTLY_COLORWAY,
                "font": {
                    "family": TYPOGRAPHY["font_family"],
                    "size": 13,
                    "color": tokens["text_primary"],
                },
                "paper_bgcolor": "rgba(0,0,0,0)",
                "plot_bgcolor": tokens["card"],
                "legend": {
                    "title": None,
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.02,
                    "x": 0,
                    "font": {"color": tokens["text_secondary"]},
                },
                "margin": {"l": 10, "r": 10, "t": 40, "b": 10},
                "xaxis": {
                    "showgrid": True,
                    "gridcolor": tokens["border"],
                    "zerolinecolor": tokens["border"],
                    "linecolor": tokens["border"],
                    "tickfont": {"color": tokens["text_secondary"], "size": 11},
                },
                "yaxis": {
                    "showgrid": True,
                    "gridcolor": tokens["border"],
                    "zerolinecolor": tokens["border"],
                    "linecolor": tokens["border"],
                    "tickfont": {"color": tokens["text_secondary"], "size": 11},
                },
                "title": {
                    "font": {
                        "family": TYPOGRAPHY["font_family"],
                        "size": 16,
                        "color": tokens["text_primary"],
                        "weight": 500,
                    }
                },
                "hoverlabel": {
                    "bgcolor": tokens["card"],
                    "font": {"color": tokens["text_primary"]},
                },
            }
        )
    pio.templates.default = template_name

# ===========================================================================
# Reusable UI Components
# ===========================================================================

def section_header(title: str, subtitle: str = "", icon: str = "", theme: str = "dark") -> None:
    import streamlit as st

    tokens = DARK if theme == "dark" else LIGHT
    html = f"""
        <div style="margin-bottom: 16px;">
            <h2 style="margin:0; font-size:24px; font-weight:600; color:{tokens['text_primary']}; line-height:1.3;">
                {icon + " " if icon else ""}{title}
            </h2>
            {f'<p style="color:{tokens["text_secondary"]}; margin-top:4px; font-size:15px;">{subtitle}</p>' if subtitle else ''}
        </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def status_badge(status: str, label: Optional[str] = None) -> str:
    tone = STATUS_TO_TONE.get(status.lower(), "neutral")
    tokens = DARK
    color_map = {
        "success": tokens["success"],
        "warning": tokens["warning"],
        "danger": tokens["error"],
        "info": tokens["accent"],
        "neutral": tokens["text_secondary"],
    }
    bg_map = {
        "success": tokens["success_soft"],
        "warning": tokens["warning_soft"],
        "danger": tokens["error_soft"],
        "info": tokens["accent_soft"],
        "neutral": tokens["border"],
    }
    text_color = color_map.get(tone, tokens["text_secondary"])
    bg = bg_map.get(tone, tokens["border"])
    display = label.upper() if label else status.upper()
    return (
        f'<span style="display:inline-flex;align-items:center;gap:6px;'
        f'padding:4px 12px;border-radius:{BORDER_RADIUS["lg"]};'
        f'font-size:{TYPOGRAPHY["caption"]["size"]};font-weight:{TYPOGRAPHY["caption"]["weight"]};'
        f'background:{bg};color:{text_color};border:1px solid transparent;">'
        f'{display}</span>'
    )


def metric_card(title: str, value: str, subtitle: str = "", tone: str = "info", icon: str = "", theme: str = "dark") -> None:
    import streamlit as st

    tokens = DARK if theme == "dark" else LIGHT
    tone_colors = {
        "success": tokens["success"],
        "warning": tokens["warning"],
        "danger": tokens["error"],
        "info": tokens["accent"],
    }
    accent = tone_colors.get(tone, tokens["accent"])

    st.markdown(f"""
        <div style="background:{tokens['card']}; border-radius:{BORDER_RADIUS['xl']}; padding:24px;
                    box-shadow:{SHADOWS['soft']}; border:1px solid {tokens['border']}; margin-bottom:16px;">
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div>
                    <div style="font-size:12px; font-weight:500; color:{tokens['text_secondary']};
                                text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px;">
                        {icon + " " if icon else ""}{title}
                    </div>
                    <div style="font-size:28px; font-weight:500; color:{accent}; line-height:1;">
                        {value}
                    </div>
                    {f'<div style="font-size:13px; color:{tokens["text_tertiary"]}; margin-top:2px;">{subtitle}</div>' if subtitle else ''}
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def glass_container(content_html: str, title: Optional[str] = None, theme: str = "dark") -> None:
    import streamlit as st

    tokens = DARK if theme == "dark" else LIGHT
    header_html = f'<div style="font-size:14px; font-weight:600; color:{tokens["text_primary"]}; margin-bottom:12px;">{title}</div>' if title else ""
    st.markdown(f"""
        <div style="background:{tokens['glass']}; border-radius:{BORDER_RADIUS['xl']};
                    backdrop-filter: blur(12px); border:1px solid {tokens['glass_border']};
                    padding:24px; margin-bottom:16px;">
            {header_html}
            {content_html}
        </div>
    """, unsafe_allow_html=True)


def kpi_card(title: str, value: str, change: Optional[str] = None, icon: Optional[str] = None, is_positive: bool = True, theme: str = "dark") -> None:
    import streamlit as st

    tokens = DARK if theme == "dark" else LIGHT
    change_color = tokens["success"] if is_positive else tokens["error"]

    st.markdown(f"""
        <div style="background:{tokens['card']}; border-radius:{BORDER_RADIUS['md']};
                    padding:{SPACING['lg']}; box-shadow:{SHADOWS['card']};
                    margin-bottom:{SPACING['md']}; transition:all {ANIMATION['normal']};">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <h3 style="color:{tokens['text_primary']}; font-size:{TYPOGRAPHY['h3']['size']};
                                font-weight:{TYPOGRAPHY['h3']['weight']}; margin:0 0 {SPACING['xs']} 0;">
                        {title}
                    </h3>
                    <div style="font-size:{TYPOGRAPHY['h1']['size']}; font-weight:{TYPOGRAPHY['h1']['weight']};
                                color:{tokens['accent']}; margin:{SPACING['sm']} 0;">
                        {value}
                    </div>
                    {f'<p style="color:{change_color}; font-size:{TYPOGRAPHY["caption"]["size"]}; margin:0;">{change}</p>' if change else ''}
                </div>
                {f'<div style="font-size:24px;">{icon}</div>' if icon else ''}
            </div>
        </div>
    """, unsafe_allow_html=True)