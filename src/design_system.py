"""Centralized UI Design System for ML Pipeline Monitor"""

# Color Palette
COLORS = {
    # Dark Mode
    "dark": {
        "background": "#121212",
        "surface": "#1E1E1E",
        "card": "#2C2C2C",
        "text_primary": "#FFFFFF",
        "text_secondary": "#AAAAAA",
        "accent": "#61DAFB",
        "success": "#4CAF50",
        "warning": "#FF9800",
        "error": "#F44336",
        "border": "#333333"
    },
    # Light Mode
    "light": {
        "background": "#FFFFFF",
        "surface": "#F5F5F5",
        "card": "#EEEEEE",
        "text_primary": "#121212",
        "text_secondary": "#666666",
        "accent": "#61DAFB",
        "success": "#4CAF50",
        "warning": "#FF9800",
        "error": "#F44336",
        "border": "#CCCCCC"
    }
}

# Typography
TYPOGRAPHY = {
    "font_family": "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif",
    "h1": {"size": "28px", "weight": "700", "line_height": "1.2"},
    "h2": {"size": "24px", "weight": "600", "line_height": "1.3"},
    "h3": {"size": "20px", "weight": "600", "line_height": "1.4"},
    "body": {"size": "16px", "weight": "400", "line_height": "1.5"},
    "caption": {"size": "14px", "weight": "400", "line_height": "1.4"},
    "small": {"size": "12px", "weight": "400", "line_height": "1.3"}
}

# Spacing
SPACING = {
    "xs": "4px",
    "sm": "8px",
    "md": "16px",
    "lg": "24px",
    "xl": "32px",
    "xxl": "40px"
}

# Shadows
SHADOWS = {
    "light": "0 2px 4px rgba(0, 0, 0, 0.1)",
    "medium": "0 4px 6px rgba(0, 0, 0, 0.15)",
    "heavy": "0 8px 16px rgba(0, 0, 0, 0.2)"
}

# Border Radius
BORDER_RADIUS = {
    "sm": "4px",
    "md": "8px",
    "lg": "12px"
}

# Animation
ANIMATION = {
    "duration": "0.3s",
    "timing_function": "ease-in-out"
}

# Responsive Breakpoints
BREAKPOINTS = {
    "mobile": "768px",
    "tablet": "1024px",
    "desktop": "1200px"
}
