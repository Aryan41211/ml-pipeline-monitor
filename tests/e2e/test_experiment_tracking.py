"""E2E tests for Experiment Tracking page."""

import pytest
from playwright.sync_api import Page, expect


@pytest.fixture(autouse=True)
def login(page: Page):
    """Login before each test."""
    import os
    os.environ["AUTH_USERNAME"] = "testadmin"
    os.environ["AUTH_PASSWORD"] = "testpass123"
    os.environ["AUTH_ROLE"] = "admin"
    
    page.goto("/")
    page.wait_for_load_state("networkidle")
    page.fill("input[type='text']", "testadmin")
    page.fill("input[type='password']", "testpass123")
    page.click("button:has-text('Login')")
    page.wait_for_load_state("networkidle")


def test_experiment_tracking_loads(page: Page):
    """Test Experiment Tracking page loads."""
    page.goto("/pages/2_Experiment_Tracking.py")
    page.wait_for_load_state("networkidle")
    
    expect(page.locator("text=Experiment Workspace")).to_be_visible()
    expect(page.locator("text=Data Grid")).to_be_visible()
    expect(page.locator("text=Interactive Visuals")).to_be_visible()


def test_experiment_grid_display(page: Page):
    """Test experiment grid displays data."""
    page.goto("/pages/2_Experiment_Tracking.py")
    page.wait_for_load_state("networkidle")
    
    # Check KPI cards
    expect(page.locator("text=Total Runs")).to_be_visible()
    expect(page.locator("text=Best Accuracy")).to_be_visible()
    expect(page.locator("text=Avg Latency")).to_be_visible()


def test_experiment_analytics_tab(page: Page):
    """Test analytics tab loads."""
    page.goto("/pages/2_Experiment_Tracking.py")
    page.wait_for_load_state("networkidle")
    
    page.click("text=Interactive Visuals")
    page.wait_for_load_state("networkidle")
    
    expect(page.locator("text=Performance Correlation")).to_be_visible()
    # Check for plotly chart
    expect(page.locator(".js-plotly-plot")).to_be_visible()


def test_refresh_button(page: Page):
    """Test refresh button works."""
    page.goto("/pages/2_Experiment_Tracking.py")
    page.wait_for_load_state("networkidle")
    
    page.click("button:has-text('Refresh Grid')")
    page.wait_for_load_state("networkidle")
    
    expect(page.locator("text=Experiment Workspace")).to_be_visible()
