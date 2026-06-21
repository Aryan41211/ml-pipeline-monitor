"""E2E tests for authentication flows."""

import os

import pytest
from playwright.sync_api import Page, expect

# IMPORTANT: Streamlit is started once per test session in tests/e2e/conftest.py.
# Auth env vars must be set before Streamlit process starts (i.e., at import time).
os.environ.setdefault("AUTH_USERNAME", "testadmin")
os.environ.setdefault("AUTH_PASSWORD", "testpass123")
os.environ.setdefault("AUTH_ROLE", "admin")


def test_login_page_loads(page: Page):
    """Test that the login page loads correctly."""
    page.goto("/")
    page.wait_for_load_state("networkidle")

    # Check for login form (avoid strict-mode ambiguity for "Access"/"access")
    expect(page.get_by_role("heading", name="Access")).to_be_visible()
    # When auth is enabled but no credentials are configured, the UI can show
    # an access prompt without rendering password/text inputs.
    expect(page.get_by_text("Please log in to access the dashboard.")).to_be_visible()


def test_login_with_invalid_credentials(page: Page):
    """Test login with invalid credentials shows error."""
    page.goto("/")
    page.wait_for_load_state("networkidle")

    page.wait_for_selector("input[type='text']")
    page.wait_for_selector("input[type='password']")

    page.fill("input[type='text']", "invalid_user")
    page.fill("input[type='password']", "wrong_password")
    page.click("button:has-text('Login')")

    expect(page.locator("text=Invalid username or password")).to_be_visible()


def test_login_with_valid_credentials(page: Page):
    """Test login with valid credentials succeeds."""
    page.goto("/")
    page.wait_for_load_state("networkidle")

    page.wait_for_selector("input[type='text']")
    page.wait_for_selector("input[type='password']")

    page.fill("input[type='text']", "testadmin")
    page.fill("input[type='password']", "testpass123")
    page.click("button:has-text('Login')")

    expect(page.locator("text=Signed in as testadmin")).to_be_visible()


def test_logout(page: Page):
    """Test logout functionality."""
    import os

    os.environ["AUTH_USERNAME"] = "testadmin"
    os.environ["AUTH_PASSWORD"] = "testpass123"
    os.environ["AUTH_ROLE"] = "admin"

    page.goto("/")
    page.wait_for_load_state("networkidle")

    page.get_by_label("Username").fill("testadmin")
    page.get_by_label("Password").fill("testpass123")
    page.click("button:has-text('Login')")

    expect(page.locator("text=Signed in as testadmin")).to_be_visible()

    page.click("button:has-text('Logout')")
    page.wait_for_load_state("networkidle")

    expect(page.locator("text=Access")).to_be_visible()


def test_navigation_requires_auth(page: Page):
    """Test that navigation to protected pages requires authentication."""
    page.goto("/pages/1_Pipeline_Runner.py")
    page.wait_for_load_state("networkidle")

    expect(page.locator("text=Access")).to_be_visible()
    expect(page.locator("text=Please log in to access the dashboard")).to_be_visible()
