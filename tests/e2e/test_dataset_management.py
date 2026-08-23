"""E2E tests for Dataset Management page."""

import re

from playwright.sync_api import Page, expect


def _select_dataset(page: Page, label: str = "Breast Cancer Wisconsin") -> None:
    """Choose a dataset from the Select Dataset combobox."""
    page.get_by_label("Select Dataset").click()
    page.get_by_role("option", name=label).click()
    page.wait_for_load_state("networkidle")


def test_dataset_management_loads(page: Page):
    """Test Dataset Management page loads."""
    page.goto("/Dataset_Management")
    page.wait_for_load_state("networkidle")

    expect(page.get_by_role("heading", name="Dataset Hub")).to_be_visible()
    expect(page.get_by_label("Select Dataset")).to_be_visible()


def test_dataset_selection(page: Page):
    """Test dataset selection and preview."""
    page.goto("/Dataset_Management")
    page.wait_for_load_state("networkidle")

    _select_dataset(page)

    # Scope to the KPI label element: the design system carries no icons, and
    # bare text like "Samples"/"Missing" also appears in the Basic Info lines.
    labels = page.locator(".ui-metric .label")
    for name in ("Samples", "Features", "Task", "Missing"):
        expect(labels.filter(has_text=re.compile(rf"^{name}$", re.I)).first).to_be_visible()


def test_overview_tab(page: Page):
    """Test overview tab displays dataset info."""
    page.goto("/Dataset_Management")
    page.wait_for_load_state("networkidle")

    _select_dataset(page)

    expect(page.get_by_text("Basic Info")).to_be_visible()
    expect(page.get_by_text("Class Distribution")).to_be_visible()


def test_feature_statistics_tab(page: Page):
    """Test feature statistics tab."""
    page.goto("/Dataset_Management")
    page.wait_for_load_state("networkidle")

    _select_dataset(page)

    page.get_by_role("tab", name=re.compile("Feature Statistics")).click()
    page.wait_for_load_state("networkidle")

    expect(page.get_by_role("heading", name="Feature Statistics")).to_be_visible()
    # Check for table
    # .hp-table was renamed to .ui-table in the design system rewrite.
    expect(page.locator(".ui-table").first).to_be_visible()


def test_train_test_split_tab(page: Page):
    """Test train/test split tab."""
    page.goto("/Dataset_Management")
    page.wait_for_load_state("networkidle")

    _select_dataset(page)

    page.get_by_role("tab", name="Train/Test Split").click()
    page.wait_for_load_state("networkidle")

    expect(page.get_by_text("Sample Data (Train)")).to_be_visible()
    expect(page.get_by_text("Sample Data (Test)")).to_be_visible()
