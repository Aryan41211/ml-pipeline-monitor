"""E2E tests for Pipeline Runner page."""

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


def test_pipeline_runner_loads(page: Page):
    """Test Pipeline Runner page loads."""
    page.goto("/pages/1_Pipeline_Runner.py")
    page.wait_for_load_state("networkidle")
    
    expect(page.locator("text=Workflow Orchestrator")).to_be_visible()
    expect(page.locator("text=Live Execution")).to_be_visible()
    expect(page.locator("text=Architecture Config")).to_be_visible()


def test_dataset_selection(page: Page):
    """Test dataset selection works."""
    page.goto("/pages/1_Pipeline_Runner.py")
    page.wait_for_load_state("networkidle")
    
    # Click on Architecture Config tab
    page.click("text=Architecture Config")
    page.wait_for_load_state("networkidle")
    
    # Select dataset
    page.select_option("select", "Breast Cancer Wisconsin")
    page.wait_for_timeout(500)
    
    # Verify model options appear
    expect(page.locator("select >> nth=1")).to_be_visible()


def test_pipeline_execution(page: Page):
    """Test pipeline execution flow."""
    page.goto("/pages/1_Pipeline_Runner.py")
    page.wait_for_load_state("networkidle")
    
    page.click("text=Architecture Config")
    page.wait_for_load_state("networkidle")
    
    page.select_option("select", "Breast Cancer Wisconsin")
    page.wait_for_timeout(500)
    
    # Select model
    page.select_option("select >> nth=1", "Random Forest")
    page.wait_for_timeout(500)
    
    # Go to Live Execution tab
    page.click("text=Live Execution")
    page.wait_for_load_state("networkidle")
    
    # Click Execute Pipeline
    page.click("button:has-text('Execute Pipeline')")
    
    # Wait for progress bar to appear
    expect(page.locator("[role='progressbar']")).to_be_visible()
    
    # Wait for completion (with timeout)
    page.wait_for_selector("text=finished", timeout=60000)
    
    # Verify results appear
    expect(page.locator("text=Analysis:")).to_be_visible()


def test_hyperparameter_configuration(page: Page):
    """Test hyperparameter configuration."""
    page.goto("/pages/1_Pipeline_Runner.py")
    page.wait_for_load_state("networkidle")
    
    page.click("text=Architecture Config")
    page.wait_for_load_state("networkidle")
    
    page.select_option("select", "Breast Cancer Wisconsin")
    page.wait_for_timeout(500)
    
    page.select_option("select >> nth=1", "XGBoost")
    page.wait_for_timeout(500)
    
    # Verify hyperparameter inputs appear
    expect(page.locator("input[aria-label*='n_estimators']")).to_be_visible()
    expect(page.locator("input[aria-label*='learning_rate']")).to_be_visible()
    expect(page.locator("input[aria-label*='max_depth']")).to_be_visible()
