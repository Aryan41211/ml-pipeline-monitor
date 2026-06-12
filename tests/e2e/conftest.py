"""Playwright E2E test configuration."""

import pytest
from playwright.sync_api import sync_playwright


@pytest.fixture(scope="session")
def browser():
    """Launch browser for test session."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        yield browser
        browser.close()


@pytest.fixture
def page(browser):
    """Create a new page for each test."""
    context = browser.new_context(
        viewport={"width": 1280, "height": 720},
        base_url="http://localhost:8501",
    )
    page = context.new_page()
    yield page
    context.close()


@pytest.fixture(scope="session", autouse=True)
def start_streamlit(request):
    """Start Streamlit app before tests and stop after."""
    # Only run for e2e tests
    if "e2e" not in str(request.config.rootpath):
        yield
        return
    
    import subprocess
    import time
    import requests

    proc = subprocess.Popen(
        ["python", "-m", "streamlit", "run", "app.py", "--server.headless", "true", "--server.port", "8501"],
        cwd="C:/projects/ML-pipeline-monitor",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for Streamlit to start
    for _ in range(30):
        try:
            response = requests.get("http://localhost:8501/_stcore/health")
            if response.status_code == 200:
                break
        except requests.ConnectionError:
            pass
        time.sleep(1)
    else:
        proc.terminate()
        raise RuntimeError("Streamlit failed to start")

    yield

    proc.terminate()
    proc.wait(timeout=5)
