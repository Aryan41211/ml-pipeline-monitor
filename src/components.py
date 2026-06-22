"""Reusable UI Components for ML Pipeline Monitor"""

import streamlit as st
from src.design_system import COLORS, TYPOGRAPHY, SPACING, SHADOWS, BORDER_RADIUS

def kpi_card(title, value, change=None, icon=None, is_positive=True):
    """Premium KPI Card Component"""
    st.markdown(
        f"""
        <div style="background: {COLORS['dark']['card']}; 
                    border-radius: {BORDER_RADIUS['md']};
                    padding: {SPACING['lg']}; 
                    box-shadow: {SHADOWS['light']};
                    margin-bottom: {SPACING['md']};
                    transition: all {COLORS['dark']['accent']};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h3 style="color: {COLORS['dark']['text_primary']}; 
                               font-size: {TYPOGRAPHY['h3']['size']};
                               font-weight: {TYPOGRAPHY['h3']['weight']};
                               margin: 0 0 {SPACING['xs']} 0;">
                        {title}
                    </h3>
                    <div style="font-size: {TYPOGRAPHY['h1']['size']};
                                font-weight: {TYPOGRAPHY['h1']['weight']};
                                color: {COLORS['dark']['accent']};
                                margin: {SPACING['sm']} 0;">
                        {value}
                    </div>
                    {f'<p style="color: {"#4CAF50" if is_positive else "#F44336"}; 
                               font-size: {TYPOGRAPHY['caption']['size']}; 
                               margin: 0;">{change}</p>' if change else ''}
                </div>
                {f'<div style="font-size: 24px;">{icon}</div>' if icon else ''}
            </div>
        </div>
        """, unsafe_allow_html=True
    )

def metric_counter(value, label, is_positive=True):
    """Animated Metric Counter"""
    st.markdown(
        f"""
        <div style="text-align: center; padding: {SPACING['md']};">
            <div style="font-size: 32px; 
                        font-weight: {TYPOGRAPHY['h1']['weight']};
                        color: {COLORS['dark']['accent']};">
                {value}
            </div>
            <div style="color: {COLORS['dark']['text_secondary']}; 
                        font-size: {TYPOGRAPHY['caption']['size']};">
                {label}
            </div>
        </div>
        """, unsafe_allow_html=True
    )

def card_header(title, subtitle=None):
    """Card Header Component"""
    st.markdown(
        f"""
        <div style="padding: {SPACING['md']} 0;">
            <h2 style="color: {COLORS['dark']['text_primary']}; 
                       font-size: {TYPOGRAPHY['h2']['size']};
                       font-weight: {TYPOGRAPHY['h2']['weight']};
                       margin: 0 0 {SPACING['xs']} 0;">
                {title}
            </h2>
            {f'<p style="color: {COLORS['dark']['text_secondary']}; 
                       font-size: {TYPOGRAPHY['body']['size']};
                       margin: 0;">{subtitle}</p>' if subtitle else ''}
        </div>
        """, unsafe_allow_html=True
    )

def status_badge(status):
    """Status Badge Component"""
    color_map = {
        "active": COLORS['dark']['success'],
        "inactive": COLORS['dark']['warning'],
        "error": COLORS['dark']['error'],
        "completed": COLORS['dark']['success']
    }
    
    st.markdown(
        f"""
        <span style="background: {color_map.get(status, COLORS['dark']['warning'])};
                     color: {COLORS['dark']['background']};
                     padding: {SPACING['xs']} {SPACING['sm']};
                     border-radius: {BORDER_RADIUS['sm']};
                     font-size: {TYPOGRAPHY['caption']['size']};
                     font-weight: {TYPOGRAPHY['caption']['weight']};
                     text-transform: uppercase;">
            {status}
        </span>
        """, unsafe_allow_html=True
    )

def sidebar_header(title):
    """Sidebar Header Component"""
    st.markdown(
        f"""
        <h3 style="color: {COLORS['dark']['text_primary']}; 
                   font-size: {TYPOGRAPHY['h3']['size']};
                   font-weight: {TYPOGRAPHY['h3']['weight']};
                   margin: {SPACING['lg']} 0 {SPACING['md']} 0;
                   padding-bottom: {SPACING['sm']};
                   border-bottom: 1px solid {COLORS['dark']['border']};">
            {title}
        </h3>
        """, unsafe_allow_html=True
    )

def navigation_item(title, icon=None, is_active=False):
    """Navigation Item Component"""
    bg_color = COLORS['dark']['accent'] if is_active else COLORS['dark']['surface']
    color = COLORS['dark']['text_primary'] if is_active else COLORS['dark']['text_secondary']
    
    st.markdown(
        f"""
        <div style="padding: {SPACING['sm']} {SPACING['md']};
                    margin-bottom: {SPACING['xs']};
                    border-radius: {BORDER_RADIUS['sm']};
                    background: {bg_color};
                    color: {color};
                    cursor: pointer;
                    transition: all {COLORS['dark']['accent']};">
            {f'<span style="margin-right: {SPACING['sm']}">{icon}</span>' if icon else ''}
            <span>{title}</span>
        </div>
        """, unsafe_allow_html=True
    )