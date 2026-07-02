"""
Dashboard Overview
"""
import streamlit as st
from src.components import kpi_card, metric_counter, card_header, status_badge, sidebar_header, navigation_item
from src.design_system import COLORS, TYPOGRAPHY, SPACING, SHADOWS, BORDER_RADIUS
import plotly.express as px
import pandas as pd

# Set page config
st.set_page_config(
    page_title="ML Pipeline Monitor",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for premium look
st.markdown(f"""
<style>
    .css-1d3915o {{
        background-color: {COLORS['dark']['background']};
    }}
    .stApp {{
        background-color: {COLORS['dark']['background']};
        color: {COLORS['dark']['text_primary']};
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: {COLORS['dark']['text_primary']} !important;
    }}
    .css-18e3th9 {{
        background-color: {COLORS['dark']['surface']};
        border-radius: {BORDER_RADIUS['md']};
        padding: {SPACING['md']};
    }}
    .css-1v0mbdj {{
        background-color: {COLORS['dark']['surface']};
        border-radius: {BORDER_RADIUS['md']};
        padding: {SPACING['md']};
    }}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown(f"<h1 style='color: {COLORS['dark']['accent']}; font-size: 24px;'>🤖 ML Pipeline Monitor</h1>", unsafe_allow_html=True)
    
    sidebar_header("Navigation")
    navigation_item("Dashboard", "📊", is_active=True)
    navigation_item("Pipelines", "⚙️")
    navigation_item("Models", "🧠")
    navigation_item("Data Quality", "🔍")
    navigation_item("Alerts", "🔔")
    navigation_item("Settings", "⚙️")
    
    st.markdown(f"<div style='margin-top: {SPACING['xl']}; padding: {SPACING['md']}; color: {COLORS['dark']['text_secondary']}; font-size: 12px;'>Version 1.0.0</div>", unsafe_allow_html=True)

# Main content
st.markdown(f"<h1 style='color: {COLORS['dark']['text_primary']};'>Dashboard Overview</h1>", unsafe_allow_html=True)

# KPI Cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    kpi_card("Active Pipelines", "12", "+2 from last week", "⚙️", True)
with col2:
    kpi_card("Models Deployed", "8", "+1 from last week", "🧠", True)
with col3:
    kpi_card("Data Quality Score", "94.2%", "-0.5% from last week", "🔍", False)
with col4:
    kpi_card("Alerts Active", "3", "+1 from last week", "🔔", True)

# Charts and Metrics
st.markdown("<h2 style='color: #ffffff;'>System Overview</h2>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    # Mock data for system performance
    performance_data = pd.DataFrame({
        'Metric': ['CPU Usage', 'Memory Usage', 'Disk I/O', 'Network'],
        'Value': [65, 42, 30, 75],
        'Max': [100, 100, 100, 100]
    })
    
    fig = px.bar(performance_data, x='Metric', y='Value', 
                title='System Performance Metrics',
                color='Value',
                color_continuous_scale='Blues')
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Mock data for pipeline status
    pipeline_data = pd.DataFrame({
        'Status': ['Completed', 'Running', 'Failed', 'Queued'],
        'Count': [45, 12, 3, 8]
    })
    
    fig = px.pie(pipeline_data, values='Count', names='Status',
                title='Pipeline Status Distribution',
                color_discrete_sequence=px.colors.qualitative.Set3)
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

# Recent Activity
st.markdown("<h2 style='color: #ffffff;'>Recent Activity</h2>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    kpi_card("New Model Deployed", "Model v2.1", icon="🧠")
    
with col2:
    kpi_card("Pipeline Completed", "Data Processing", icon="⚙️")
    
with col3:
    kpi_card("Alert Resolved", "High CPU Usage", icon="🔔")

# System Status
st.markdown("<h2 style='color: #ffffff;'>System Status</h2>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    status_badge("active")
    st.write("Data Pipeline Service")
    
with col2:
    status_badge("active")
    st.write("Model Training Service")
    
with col3:
    status_badge("error")
    st.write("Alerting Service")

st.divider()
st.caption("📊 Dashboard v1.0 - Modern UI Components")