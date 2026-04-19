"""CLI entrypoints for running ML Pipeline Monitor."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Launch Streamlit app via `mlmonitor` console script."""
    root = Path(__file__).resolve().parent.parent
    app_file = root / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_file)]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
