"""E2E tests for authentication flows."""

import os

from playwright.sync_api import Page, expect

# IMPORTANT: Streamlit is started once per test session in tests/e2e/conftest.py.
# Auth env vars must be set before Streamlit process starts (i.e., at import time).
os.environ.setdefault("AUTH_USERNAME", "testadmin")
os.environ.setdefault("AUTH_PASSWORD", "testpass123")
os.environ.setdefault("AUTH_ROLE", "admin")


def _login(page: Page, username: str = "testadmin", password: str = "testpass123") -> None:
    """Fill the sidebar login form and submit it."""
    page.wait_for_selector("input[aria-label='Username']")
    page.locator("input[aria-label='Username']").fill(username)
    page.locator("input[aria-label='Password'][type='password']").fill(password)
    page.click("button:has-text('Login')")


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

    _login(page, username="invalid_user", password="wrong_password")

    expect(page.get_by_text("Invalid username or password")).to_be_visible()


def test_login_with_valid_credentials(page: Page):
    """Test login with valid credentials succeeds."""
    page.goto("/")
    page.wait_for_load_state("networkidle")

    _login(page)

    expect(page.get_by_text("Signed in as testadmin")).to_be_visible()


def test_logout(page: Page):
    """Test logout functionality."""
    page.goto("/")
    page.wait_for_load_state("networkidle")

    _login(page)

    expect(page.get_by_text("Signed in as testadmin")).to_be_visible()

    page.click("button:has-text('Logout')")
    page.wait_for_load_state("networkidle")

    expect(page.get_by_role("heading", name="Access")).to_be_visible()


def test_navigation_requires_auth(page: Page):
    """Test that protected pages expose the access gate when not logged in."""
    page.goto("/Pipeline_Runner")
    page.wait_for_load_state("networkidle")

    expect(page.get_by_role("heading", name="Access")).to_be_visible()
    expect(page.get_by_text("Please log in to access the dashboard")).to_be_visible()
