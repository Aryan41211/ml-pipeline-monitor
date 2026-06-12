"""E2E tests for System Health page."""

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


def test_system_health_loads(page: Page):
    """Test System Health page loads."""
    page.goto("/pages/5_Data_Health.py")
    page.wait_for_load_state("networkidle")
    
    expect(page.locator("text=Platform Health")).to_be_visible()
    expect(page.locator("text=CPU LOAD")).to_be_visible()
    expect(page.locator("text=RAM USAGE")).to_be_visible()
    expect(page.locator("text=DISK I/O")).to_be_visible()


def test_gauge_charts(page: Page):
    """Test gauge charts render."""
    page.goto("/pages/5_Data_Health.py")
    page.wait_for_load_state("networkidle")
    
    # Check for plotly gauge charts
    expect(page.locator(".js-plotly-plot")).to_be_visible()


def test_refresh_telemetry(page: Page):
    """Test refresh telemetry button."""
    page.goto("/pages/5_Data_Health.py")
    page.wait_for_load_state("networkidle")
    
    page.click("button:has-text('Refresh Telemetry')")
    page.wait_for_load_state("networkidle")
    
    expect(page.locator("text=Platform Health")).to_be_visible()


def test_audit_log(page: Page):
    """Test system audit log displays."""
    page.goto("/pages/5_Data_Health.py")
    page.wait_for_load_state("networkidle")
    
    expect(page.locator("text=System Audit Log")).to_be_visible()
    # Check timeline items
    expect(page.locator(".hp-card")).to_be_visible()


def test_hardware_context(page: Page):
    """Test hardware context insights."""
    page.goto("/pages/5_Data_Health.py")
    page.wait_for_load_state("networkidle")
    
    expect(page.locator("text=Hardware Context")).to_be_visible()
