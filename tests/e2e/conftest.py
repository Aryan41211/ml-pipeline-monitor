"""Playwright E2E test configuration."""

import os

import pytest
from playwright.sync_api import expect, sync_playwright

# Cold page loads (first streamlit script run imports heavy ML modules) can take
# several seconds; give expect() assertions a generous default timeout.
expect.set_options(timeout=45000)

# Auth credentials must be set before the Streamlit server process starts so
# every e2e run (including individual test files) can log in. setdefault keeps
# any explicitly-provided values (e.g. CI overrides) intact.
os.environ.setdefault("AUTH_USERNAME", "testadmin")
os.environ.setdefault("AUTH_PASSWORD", "testpass123")
os.environ.setdefault("AUTH_ROLE", "admin")


@pytest.fixture(scope="session")
def browser():
    """Launch browser for test session."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        yield browser
        browser.close()


@pytest.fixture
def page(browser, start_streamlit):
    """Create a new page for each test."""
    context = browser.new_context(
        viewport={"width": 1280, "height": 720},
        base_url=f"http://localhost:{start_streamlit}",
    )
    page = context.new_page()
    page.set_default_timeout(60000)
    yield page
    context.close()


@pytest.fixture(scope="session", autouse=True)
def start_streamlit(request):
    """Start Streamlit app before e2e tests and stop after."""
    import subprocess
    import sys
    import time
    from pathlib import Path

    import requests

    # Stabilize E2E tests by disabling Streamlit UI auth.
    # Auth logic is covered by unit tests; E2E should validate page flows without login flakiness.
    os.environ["MLMONITOR_AUTH_ENABLED"] = "false"

    # Isolate e2e runs from any local database: point the Streamlit server at a
    # fresh temporary SQLite DB so experiment/model pages deterministically show
    # their empty states and pipeline runs never leak into developer data.
    import shutil
    import tempfile

    _e2e_db_dir = Path(tempfile.mkdtemp(prefix="mlmonitor-e2e-"))
    os.environ["PIPELINE_DB"] = str(_e2e_db_dir / "e2e.db")

    # Run Streamlit from the repository root irrespective of OS/path.
    # This keeps e2e tests working in CI and local environments.
    repo_root = Path(__file__).resolve().parents[2]
    app_py = repo_root / "app.py"

    if not app_py.exists():
        raise RuntimeError(f"app.py not found at expected location: {app_py}")

    # Use a dynamic free port instead of a fixed one so stale/orphaned listeners
    # (e.g. from interrupted local runs) never block the suite in CI.
    import socket

    _sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    _sock.bind(("127.0.0.1", 0))
    port = _sock.getsockname()[1]
    _sock.close()

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
            str(port),
            "--server.address",
            "0.0.0.0",
        ],
        cwd=str(repo_root),
        # Redirect to files instead of PIPE: Streamlit logs a lot (file-watcher
        # spam, per-run traces) and an undrained pipe buffer (~64KB) fills up,
        # blocking the server on write and freezing the app mid-suite.
        stdout=open(str(_e2e_db_dir / "streamlit.stdout.log"), "w", encoding="utf-8"),
        stderr=open(str(_e2e_db_dir / "streamlit.stderr.log"), "w", encoding="utf-8"),
        text=False,
    )

    # Wait for Streamlit to start
    health_url = f"http://localhost:{port}/_stcore/health"
    root_url = f"http://localhost:{port}/"
    last_exc: str = ""

    stdout_log = _e2e_db_dir / "streamlit.stdout.log"
    stderr_log = _e2e_db_dir / "streamlit.stderr.log"

    def _read_logs() -> tuple[str, str]:
        try:
            out_txt = stdout_log.read_text(encoding="utf-8", errors="ignore")
            err_txt = stderr_log.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            out_txt = "<failed to read streamlit stdout log>"
            err_txt = "<failed to read streamlit stderr log>"
        return out_txt, err_txt

    for _ in range(120):  # up to ~2 minutes
        # If the process already died, fail early with captured logs.
        if proc.poll() is not None:
            out_txt, err_txt = _read_logs()

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
        out_txt, err_txt = _read_logs()

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
            out_txt, err_txt = _read_logs()

            raise RuntimeError(
                "Streamlit root endpoint returned error "
                f"status={final_root.status_code}\n---stdout_last---\n{out_txt[-4000:]}\n---stderr_last---\n{err_txt[-4000:]}\n"
            )
    except Exception as e:
        raise RuntimeError(
            "Streamlit root endpoint not reachable after readiness.\n" + str(e)
        )

    yield port

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    shutil.rmtree(_e2e_db_dir, ignore_errors=True)
