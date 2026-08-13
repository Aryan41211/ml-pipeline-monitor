"""E2E tests for Pipeline Runner page."""

import re

from playwright.sync_api import Page, expect


def _open_config_tab(page: Page) -> None:
    """Switch to the Architecture Config tab."""
    page.get_by_role("tab", name=re.compile("Architecture Config")).click()
    page.wait_for_load_state("networkidle")


def _select_combobox(page: Page, label: str, value: str) -> None:
    """Choose an option from a Streamlit selectbox (combobox widget)."""
    page.get_by_label(label).click()
    page.get_by_role("option", name=value).click()
    page.wait_for_load_state("networkidle")


def test_pipeline_runner_loads(page: Page):
    """Test Pipeline Runner page loads."""
    page.goto("/Pipeline_Runner")
    page.wait_for_load_state("networkidle")

    expect(page.get_by_role("heading", name="Workflow Orchestrator")).to_be_visible()
    expect(page.get_by_role("tab", name=re.compile("Live Execution"))).to_be_visible()
    expect(page.get_by_role("tab", name=re.compile("Architecture Config"))).to_be_visible()


def test_dataset_selection(page: Page):
    """Test dataset selection works."""
    page.goto("/Pipeline_Runner")
    page.wait_for_load_state("networkidle")

    _open_config_tab(page)

    # Verify both selectboxes are present in the config tab
    expect(page.get_by_label("Target Dataset")).to_be_visible()
    expect(page.get_by_label("Algorithm")).to_be_visible()

    # Change dataset and verify the selection sticks
    _select_combobox(page, "Target Dataset", "Iris Species")
    expect(page.get_by_label("Target Dataset")).to_have_value("Iris Species")
    expect(page.get_by_label("Algorithm")).to_be_visible()


def test_pipeline_execution(page: Page):
    """Test pipeline execution flow."""
    page.goto("/Pipeline_Runner")
    page.wait_for_load_state("networkidle")

    _open_config_tab(page)

    _select_combobox(page, "Algorithm", "Random Forest")

    # Go to Live Execution tab
    page.get_by_role("tab", name=re.compile("Live Execution")).click()
    page.wait_for_load_state("networkidle")

    # Click Execute Pipeline
    page.get_by_role("button", name="Execute Pipeline").click()

    # Wait for progress bar to appear
    expect(page.locator("[role='progressbar']")).to_be_visible(timeout=30000)

    # Wait for completion (with timeout)
    page.wait_for_selector("text=finished", timeout=60000)

    # Verify results appear
    expect(page.get_by_text("Analysis:")).to_be_visible()


def test_hyperparameter_configuration(page: Page):
    """Test hyperparameter configuration."""
    page.goto("/Pipeline_Runner")
    page.wait_for_load_state("networkidle")

    _open_config_tab(page)

    _select_combobox(page, "Algorithm", "XGBoost")

    # Verify XGBoost hyperparameter inputs appear
    expect(page.locator("input[aria-label*='n_estimators']")).to_be_visible()
    expect(page.locator("input[aria-label*='learning_rate']")).to_be_visible()
    expect(page.locator("input[aria-label*='max_depth']")).to_be_visible()
