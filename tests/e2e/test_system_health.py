"""E2E tests for System Health page."""

from playwright.sync_api import Page, expect


def test_system_health_loads(page: Page):
    """Test System Health page loads."""
    page.goto("/Data_Health")
    page.wait_for_load_state("networkidle")

    expect(page.get_by_role("heading", name="Platform Health")).to_be_visible()
    expect(page.get_by_text("CPU LOAD", exact=True)).to_be_visible()
    expect(page.get_by_text("RAM USAGE", exact=True)).to_be_visible()
    expect(page.get_by_text("DISK I/O", exact=True)).to_be_visible()


def test_gauge_charts(page: Page):
    """Test gauge charts render."""
    page.goto("/Data_Health")
    page.wait_for_load_state("networkidle")

    # Three circular gauges render as plotly charts
    expect(page.locator(".js-plotly-plot").first).to_be_visible()
    expect(page.locator(".js-plotly-plot")).to_have_count(3)


def test_refresh_telemetry(page: Page):
    """Test refresh telemetry button."""
    page.goto("/Data_Health")
    page.wait_for_load_state("networkidle")

    page.get_by_role("button", name="Refresh Telemetry").click()
    page.wait_for_load_state("networkidle")

    expect(page.get_by_role("heading", name="Platform Health")).to_be_visible()


def test_audit_log(page: Page):
    """Test system audit log displays."""
    page.goto("/Data_Health")
    page.wait_for_load_state("networkidle")

    expect(page.get_by_role("heading", name="System Audit Log")).to_be_visible()
    expect(page.get_by_text("Pipeline Runner Session Started")).to_be_visible()


def test_hardware_context(page: Page):
    """Test hardware context insights."""
    page.goto("/Data_Health")
    page.wait_for_load_state("networkidle")

    expect(page.get_by_role("heading", name="Hardware Context")).to_be_visible()
