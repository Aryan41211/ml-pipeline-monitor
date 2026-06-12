"""CLI entrypoints for running ML Pipeline Monitor."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.auth import hash_password


def hash_password_cli() -> int:
    """Generate bcrypt hash for a password."""
    parser = argparse.ArgumentParser(description="Generate bcrypt password hash")
    parser.add_argument("password", help="Password to hash")
    parser.add_argument("--rounds", type=int, default=12, help="Bcrypt rounds (default: 12)")
    args = parser.parse_args(sys.argv[2:])

    import bcrypt
    salt = bcrypt.gensalt(rounds=args.rounds)
    hashed = bcrypt.hashpw(args.password.encode("utf-8"), salt).decode("utf-8")
    print(hashed)
    return 0


def main() -> int:
    """Launch Streamlit app via `mlmonitor` console script."""
    if len(sys.argv) > 1 and sys.argv[1] == "hash-password":
        return hash_password_cli()

    root = Path(__file__).resolve().parent.parent
    app_file = root / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_file)]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())