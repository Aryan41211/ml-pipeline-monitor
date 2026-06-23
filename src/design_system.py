"""Premium Enterprise Design System for ML Pipeline Monitor
Inspired by Vercel, Linear, Stripe, and modern SaaS interfaces.
Dark mode first with elegant light mode support.
"""

# Color Palette - Premium SaaS Theme
COLORS = {
    # Dark Mode (Default) - Vercel-inspired
    "dark": {
        # Background hierarchy
        "background": "#0A0A0A",
        "surface": "#111111",
        "card": "#1A1A1A",
        "card_hover": "#222222",
        
        # Glassmorphism surfaces
        "glass": "rgba(26, 26, 26, 0.6)",
        "glass_border": "rgba(255, 255, 255, 0.08)",
        
        # Text hierarchy
        "text_primary": "#FFFFFF",
        "text_secondary": "#A0A0A0",
        "text_tertiary": "#707070",
        
        # Accent colors - Stripe blue palette
        "accent": "#0070F3",
        "accent_bright": "#3399FF",
        "accent_soft": "#1A3FFF",
        
        # Status colors
        "success": "#00C951",
        "success_soft": "#10B98115",
        "warning": "#F59E0B",
        "warning_soft": "#F59E0B15",
        "error": "#EF4444",
        "error_soft": "#EF444415",
        
        # Borders
        "border": "#2A2A2A",
        "border_strong": "#404040",
    },
    # Light Mode - Clean minimal
    "light": {
        "background": "#FFFFFF",
        "surface": "#FAFAFA",
        "card": "#FFFFFF",
        "card_hover": "#F5F5F5",
        
        # Glassmorphism surfaces
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

# Premium Typography
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

# Spacing Scale
SPACING = {
    "xs": "4px",
    "sm": "8px",
    "md": "16px",
    "lg": "24px",
    "xl": "32px",
    "xxl": "48px",
}

# Premium Shadows
SHADOWS = {
    "card": "0 4px 12px rgba(0, 0, 0, 0.15)",
    "card_hover": "0 8px 24px rgba(0, 0, 0, 0.25)",
    "modal": "0 24px 60px rgba(0, 0, 0, 0.4)",
    "glass": "0 8px 32px rgba(0, 0, 0, 0.12)",
    "glow": "0 0 20px rgba(0, 112, 243, 0.3)",
}

# Border Radius
BORDER_RADIUS = {
    "sm": "6px",
    "md": "8px",
    "lg": "12px",
    "xl": "16px",
    "pill": "9999px",
    "full": "50%",
}

# Animation Presets
ANIMATION = {
    "fast": "120ms",
    "normal": "200ms",
    "slow": "300ms",
    "easing": "cubic-bezier(0.4, 0, 0.2, 1)",
    "spring": "cubic-bezier(0.34, 1.56, 0.64, 1)",
}

# Gradients
GRADIENTS = {
    "accent": "linear-gradient(135deg, #0070F3 0%, #1A3FFF 100%)",
    "success": "linear-gradient(135deg, #00C951 0%, #10B981 100%)",
    "warning": "linear-gradient(135deg, #F59E0B 0%, #FFA500 100%)",
    "error": "linear-gradient(135deg, #EF4444 0%, #F87171 100%)",
    "glass": "linear-gradient(135deg, rgba(26,26,26,0.8) 0%, rgba(36,36,36,0.9) 100%)",
    "glass_light": "linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(245,245,245,0.95) 100%)",
}

# Chart Colors - Premium palette
CHART_COLORS = {
    "primary": ["#0070F3", "#1A3FFF", "#3399FF"],
    "success": ["#00C951", "#10B981", "#34D399"],
    "warning": ["#F59E0B", "#FFA500", "#FCD34D"],
    "error": ["#EF4444", "#F87171", "#FCA5A5"],
    "neutral": ["#A0A0A0", "#808080", "#606060"],
    "palette": ["#0070F3", "#00C951", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899", "#14B8A6", "#F97316"],
}