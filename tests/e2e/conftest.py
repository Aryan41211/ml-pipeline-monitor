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
    """Start Streamlit app before e2e tests and stop after."""
    import subprocess
    import time
    import requests
    from pathlib import Path
    import sys

    # Run Streamlit from the repository root irrespective of OS/path.
    # This keeps e2e tests working in CI and local environments.
    repo_root = Path(__file__).resolve().parents[2]
    app_py = repo_root / "app.py"

    if not app_py.exists():
        raise RuntimeError(f"app.py not found at expected location: {app_py}")

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_py),
            "--server.headless",
            "true",
            "--server.port",
            "8501",
            "--server.address",
            "0.0.0.0",
        ],
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
    )

    # Wait for Streamlit to start
    health_url = "http://localhost:8501/_stcore/health"
    root_url = "http://localhost:8501/"
    last_exc: str = ""

    for _ in range(120):  # up to ~2 minutes
        # If the process already died, fail early with captured logs.
        if proc.poll() is not None:
            try:
                stdout, stderr = proc.communicate(timeout=1)
                out_txt = (stdout or b"").decode(errors="ignore")
                err_txt = (stderr or b"").decode(errors="ignore")
            except Exception:
                out_txt = "<failed to capture streamlit stdout>"
                err_txt = "<failed to capture streamlit stderr>"

            rc = proc.poll()
            raise RuntimeError(
                "Streamlit process exited early. "
                f"proc_return_code={rc}\n---stdout---\n{out_txt}\n---stderr---\n{err_txt}"
            )

        try:
            health_resp = requests.get(health_url, timeout=1)
            if health_resp.status_code != 200:
                last_exc = f"health status={health_resp.status_code}"
                time.sleep(1)
                continue

            # Extra confirmation: root page should be reachable too.
            root_resp = requests.get(root_url, timeout=1)
            if root_resp.status_code >= 500:
                last_exc = f"root status={root_resp.status_code}"
                time.sleep(1)
                continue

            # Give the server a moment to settle
            time.sleep(0.5)
            break
        except requests.RequestException as e:
            last_exc = str(e)
            time.sleep(1)
    else:
        # Capture logs to help debugging
        try:
            stdout, stderr = proc.communicate(timeout=2)
            out_txt = (stdout or b"").decode(errors="ignore")
            err_txt = (stderr or b"").decode(errors="ignore")
        except Exception:
            out_txt = "<failed to capture streamlit stdout>"
            err_txt = "<failed to capture streamlit stderr>"

        try:
            rc = proc.poll()
        except Exception:
            rc = None

        # Ensure process is stopped
        try:
            proc.terminate()
        except Exception:
            pass

        debug_msg = (
            f"Streamlit failed readiness checks.\nlast_exc={last_exc}\nproc_return_code={rc}\n"
            f"---stdout_last---\n{out_txt[-4000:] if out_txt else out_txt}\n"
            f"---stderr_last---\n{err_txt[-4000:] if err_txt else err_txt}\n"
        )
        raise RuntimeError(debug_msg)

    # Final verification: ensure root is actually serving content
    try:
        final_root = requests.get(root_url, timeout=2)
        if final_root.status_code >= 500:
            try:
                stdout, stderr = proc.communicate(timeout=2)
                out_txt = (stdout or b"").decode(errors="ignore")
                err_txt = (stderr or b"").decode(errors="ignore")
            except Exception:
                out_txt = "<failed to capture streamlit stdout>"
                err_txt = "<failed to capture streamlit stderr>"

            raise RuntimeError(
                "Streamlit root endpoint returned error "
                f"status={final_root.status_code}\n---stdout_last---\n{out_txt[-4000:]}\n---stderr_last---\n{err_txt[-4000:]}\n"
            )
    except Exception as e:
        raise RuntimeError(
            "Streamlit root endpoint not reachable after readiness.\n" + str(e)
        )

    yield

    proc.terminate()
    proc.wait(timeout=5)
