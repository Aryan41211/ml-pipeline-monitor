"""E2E tests for Experiment Tracking page."""

from playwright.sync_api import Page, expect


def test_experiment_tracking_loads(page: Page):
    """Test Experiment Tracking page loads."""
    page.goto("/Experiment_Tracking")
    page.wait_for_load_state("networkidle")

    # Fresh (isolated) database renders the deterministic empty state
    expect(page.get_by_text("No Experiments", exact=True)).to_be_visible()
    expect(page.get_by_text("Start a pipeline run to populate this workspace.", exact=True)).to_be_visible()


def test_empty_state_cta_navigates_to_runner(page: Page):
    """Test the 'Run Pipeline' empty-state action routes to the Pipeline Runner."""
    page.goto("/Experiment_Tracking")
    page.wait_for_load_state("networkidle")

    expect(page.get_by_role("link", name="Run Pipeline")).to_be_visible()
    page.get_by_role("link", name="Run Pipeline").click()
    page.wait_for_load_state("networkidle")

    expect(page.get_by_role("heading", name="Workflow Orchestrator")).to_be_visible()
