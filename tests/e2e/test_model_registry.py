"""E2E tests for Model Registry page."""

from playwright.sync_api import Page, expect


def test_model_registry_loads(page: Page):
    """Test Model Registry page loads."""
    page.goto("/Model_Registry")
    page.wait_for_load_state("networkidle")

    # Fresh (isolated) database renders the deterministic empty state
    expect(page.get_by_text("Registry Empty", exact=True)).to_be_visible()
    expect(page.get_by_text("No models have been registered yet.", exact=True)).to_be_visible()


def test_empty_state_cta_navigates_to_runner(page: Page):
    """Test the 'Train First Model' empty-state action routes to the Pipeline Runner."""
    page.goto("/Model_Registry")
    page.wait_for_load_state("networkidle")

    expect(page.get_by_role("link", name="Train First Model")).to_be_visible()
    page.get_by_role("link", name="Train First Model").click()
    page.wait_for_load_state("networkidle")

    expect(page.get_by_role("heading", name="Workflow Orchestrator")).to_be_visible()
