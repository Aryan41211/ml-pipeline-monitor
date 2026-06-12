"""E2E tests for Data Drift page."""

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


def test_data_drift_loads(page: Page):
    """Test Data Drift page loads."""
    page.goto("/pages/4_Data_Drift.py")
    page.wait_for_load_state("networkidle")
    
    expect(page.locator("text=Data Observability")).to_be_visible()
    expect(page.locator("text=Perturbation Settings")).to_be_visible()


def test_drift_settings(page: Page):
    """Test drift configuration settings."""
    page.goto("/pages/4_Data_Drift.py")
    page.wait_for_load_state("networkidle")
    
    # Check dataset selector
    expect(page.locator("select")).to_be_visible()
    
    # Check sliders
    expect(page.locator("input[type='range'] >> nth=0")).to_be_visible()  # Noise
    expect(page.locator("input[type='range'] >> nth=1")).to_be_visible()  # Shift
    
    # Check alpha selector
    expect(page.locator("text=Confidence")).to_be_visible()


def test_run_drift_scan(page: Page):
    """Test running drift scan."""
    page.goto("/pages/4_Data_Drift.py")
    page.wait_for_load_state("networkidle")
    
    page.select_option("select", "Breast Cancer Wisconsin")
    page.wait_for_timeout(500)
    
    page.click("button:has-text('Run Drift Scan')")
    
    # Wait for results
    page.wait_for_selector("text=Overall Distribution Health", timeout=30000)
    
    # Check KPI cards appear
    expect(page.locator("text=Analyzed")).to_be_visible()
    expect(page.locator("text=Drifted")).to_be_visible()
    expect(page.locator("text=Avg PSI")).to_be_visible()


def test_drift_history_tab(page: Page):
    """Test drift history tab."""
    page.goto("/pages/4_Data_Drift.py")
    page.wait_for_load_state("networkidle")
    
    page.select_option("select", "Breast Cancer Wisconsin")
    page.wait_for_timeout(500)
    
    page.click("button:has-text('Run Drift Scan')")
    page.wait_for_selector("text=Overall Distribution Health", timeout=30000)
    
    page.click("text=Analysis History")
    page.wait_for_load_state("networkidle")
    
    expect(page.locator("text=Dataset")).to_be_visible()
