"""Start Streamlit app on the first open port (starting at 8501)."""

from __future__ import annotations

import socket
import subprocess
import sys
from pathlib import Path


START_PORT = 8501
MAX_PORT_ATTEMPTS = 100
HOST = "127.0.0.1"


def _is_port_open(port: int, host: str = HOST) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex((host, port)) == 0


def _find_open_port(start_port: int = START_PORT) -> int:
    for port in range(start_port, start_port + MAX_PORT_ATTEMPTS):
        if not _is_port_open(port):
            return port
    raise RuntimeError(
        f"No open port found in range {start_port}-{start_port + MAX_PORT_ATTEMPTS - 1}."
    )


def main() -> int:
    root = Path(__file__).resolve().parent
    app_path = root / "app.py"
    if not app_path.exists():
        print(f"Error: app.py not found at {app_path}", file=sys.stderr)
        return 1

    port = _find_open_port()
    url = f"http://{HOST}:{port}"

    print(f"Launching Streamlit on available port: {port}")
    print(f"App URL: {url}")

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
        "--server.address",
        HOST,
    ]

    try:
        return subprocess.call(cmd, cwd=str(root))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
