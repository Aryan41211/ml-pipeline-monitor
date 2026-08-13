"""E2E tests for Data Drift page."""

import re

from playwright.sync_api import Page, expect

# Auth is disabled for the whole e2e session (see conftest), so content renders
# and pipeline/drift actions are enabled without logging in.


def _run_scan(page: Page) -> None:
    """Trigger a drift scan on the default dataset and wait for results."""
    page.get_by_role("button", name="Run Drift Scan").click()
    page.wait_for_selector("text=Overall Distribution Health", timeout=60000)


def test_data_drift_loads(page: Page):
    """Test Data Drift page loads."""
    page.goto("/Data_Drift")
    page.wait_for_load_state("networkidle")

    expect(page.get_by_role("heading", name="Data Observability")).to_be_visible()
    expect(page.get_by_role("heading", name="Perturbation Settings")).to_be_visible()


def test_drift_settings(page: Page):
    """Test drift configuration settings."""
    page.goto("/Data_Drift")
    page.wait_for_load_state("networkidle")

    # Check dataset selector
    expect(page.get_by_label("Target Dataset")).to_be_visible()

    # Check sliders
    expect(page.get_by_label("Signal Noise")).to_be_visible()
    expect(page.get_by_label("Mean Offset")).to_be_visible()

    # Check alpha selector label
    expect(page.get_by_text("Confidence (α)")).to_be_visible()


def test_run_drift_scan(page: Page):
    """Test running drift scan."""
    page.goto("/Data_Drift")
    page.wait_for_load_state("networkidle")

    _run_scan(page)

    # Check KPI cards appear (icon-prefixed regex avoids KPI collisions)
    expect(page.get_by_text(re.compile(r"🔢\s*Analyzed", re.I))).to_be_visible()
    expect(page.get_by_text(re.compile(r"⚠️\s*Drifted", re.I))).to_be_visible()
    expect(page.get_by_text(re.compile(r"📊\s*Avg PSI", re.I))).to_be_visible()


def test_drift_history_tab(page: Page):
    """Test drift history tab."""
    page.goto("/Data_Drift")
    page.wait_for_load_state("networkidle")

    _run_scan(page)

    page.get_by_role("tab", name=re.compile("Analysis History")).click()
    page.wait_for_load_state("networkidle")

    expect(page.get_by_text("Dataset", exact=True)).to_be_visible()
    expect(page.get_by_text("Features Drifted", exact=True)).to_be_visible()
