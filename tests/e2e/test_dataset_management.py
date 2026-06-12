"""E2E tests for Dataset Management page."""

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


def test_dataset_management_loads(page: Page):
    """Test Dataset Management page loads."""
    page.goto("/pages/0_Dataset_Management.py")
    page.wait_for_load_state("networkidle")
    
    expect(page.locator("text=Dataset Hub")).to_be_visible()
    expect(page.locator("text=Select Dataset")).to_be_visible()


def test_dataset_selection(page: Page):
    """Test dataset selection and preview."""
    page.goto("/pages/0_Dataset_Management.py")
    page.wait_for_load_state("networkidle")
    
    page.select_option("select", "Breast Cancer Wisconsin")
    page.wait_for_load_state("networkidle")
    
    # Check KPI cards
    expect(page.locator("text=Samples")).to_be_visible()
    expect(page.locator("text=Features")).to_be_visible()
    expect(page.locator("text=Task")).to_be_visible()
    expect(page.locator("text=Missing")).to_be_visible()


def test_overview_tab(page: Page):
    """Test overview tab displays dataset info."""
    page.goto("/pages/0_Dataset_Management.py")
    page.wait_for_load_state("networkidle")
    
    page.select_option("select", "Breast Cancer Wisconsin")
    page.wait_for_load_state("networkidle")
    
    expect(page.locator("text=Basic Info")).to_be_visible()
    expect(page.locator("text=Class Distribution")).to_be_visible()


def test_feature_statistics_tab(page: Page):
    """Test feature statistics tab."""
    page.goto("/pages/0_Dataset_Management.py")
    page.wait_for_load_state("networkidle")
    
    page.select_option("select", "Breast Cancer Wisconsin")
    page.wait_for_load_state("networkidle")
    
    page.click("text=Feature Statistics")
    page.wait_for_load_state("networkidle")
    
    expect(page.locator("text=Feature Statistics")).to_be_visible()
    # Check for table
    expect(page.locator(".hp-table")).to_be_visible()


def test_train_test_split_tab(page: Page):
    """Test train/test split tab."""
    page.goto("/pages/0_Dataset_Management.py")
    page.wait_for_load_state("networkidle")
    
    page.select_option("select", "Breast Cancer Wisconsin")
    page.wait_for_load_state("networkidle")
    
    page.click("text=Train/Test Split")
    page.wait_for_load_state("networkidle")
    
    expect(page.locator("text=Train Set")).to_be_visible()
    expect(page.locator("text=Test Set")).to_be_visible()
    expect(page.locator("text=Sample Data (Train)")).to_be_visible()
