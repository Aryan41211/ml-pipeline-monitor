"""E2E tests for Model Registry page."""

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


def test_model_registry_loads(page: Page):
    """Test Model Registry page loads."""
    page.goto("/pages/3_Model_Registry.py")
    page.wait_for_load_state("networkidle")
    
    expect(page.locator("text=Model Inventory")).to_be_visible()
    expect(page.locator("text=Inventory Grid")).to_be_visible()
    expect(page.locator("text=Production Lineage")).to_be_visible()
    expect(page.locator("text=Compliance & Management")).to_be_visible()


def test_model_registry_kpis(page: Page):
    """Test KPI cards display."""
    page.goto("/pages/3_Model_Registry.py")
    page.wait_for_load_state("networkidle")
    
    expect(page.locator("text=Total Models")).to_be_visible()
    expect(page.locator("text=Production")).to_be_visible()
    expect(page.locator("text=Staging")).to_be_visible()


def test_model_cards_display(page: Page):
    """Test model cards display in grid."""
    page.goto("/pages/3_Model_Registry.py")
    page.wait_for_load_state("networkidle")
    
    # Check for model cards (hp-card class)
    expect(page.locator(".hp-card")).to_be_visible()


def test_stage_promotion(page: Page):
    """Test model stage promotion flow."""
    page.goto("/pages/3_Model_Registry.py")
    page.wait_for_load_state("networkidle")
    
    page.click("text=Compliance & Management")
    page.wait_for_load_state("networkidle")
    
    # Check stage selector
    expect(page.locator("select")).to_be_visible()
    
    # Check commit button
    expect(page.locator("button:has-text('Commit Transition')")).to_be_visible()


def test_rollback_section(page: Page):
    """Test rollback protection section."""
    page.goto("/pages/3_Model_Registry.py")
    page.wait_for_load_state("networkidle")
    
    page.click("text=Compliance & Management")
    page.wait_for_load_state("networkidle")
    
    expect(page.locator("text=Rollback Protection")).to_be_visible()
