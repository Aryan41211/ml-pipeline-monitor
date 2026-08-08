"""One-time script to update imports after repository restructuring."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

REPLACEMENTS = [
    ("from ml_pipeline_monitor.database.connection", "from ml_pipeline_monitor.database.connection"),
    ("import ml_pipeline_monitor.database.connection", "import ml_pipeline_monitor.database.connection"),
    ("from ml_pipeline_monitor.database.interfaces", "from ml_pipeline_monitor.database.interfaces"),
    ("import ml_pipeline_monitor.database.interfaces", "import ml_pipeline_monitor.database.interfaces"),
    ("from ml_pipeline_monitor.database", "from ml_pipeline_monitor.database"),
    ("import ml_pipeline_monitor.database", "import ml_pipeline_monitor.database"),
    ("from ml_pipeline_monitor.core.config_loader", "from ml_pipeline_monitor.core.config_loader"),
    ("import ml_pipeline_monitor.core.config_loader", "import ml_pipeline_monitor.core.config_loader"),
    ("from ml_pipeline_monitor.core.logger", "from ml_pipeline_monitor.core.logger"),
    ("import ml_pipeline_monitor.core.logger", "import ml_pipeline_monitor.core.logger"),
    ("from ml_pipeline_monitor.core.secrets", "from ml_pipeline_monitor.core.secrets"),
    ("import ml_pipeline_monitor.core.secrets", "import ml_pipeline_monitor.core.secrets"),
    ("from ml_pipeline_monitor.core.auth", "from ml_pipeline_monitor.core.auth"),
    ("import ml_pipeline_monitor.core.auth", "import ml_pipeline_monitor.core.auth"),
    ("from ml_pipeline_monitor.core.jwt_auth", "from ml_pipeline_monitor.core.jwt_auth"),
    ("import ml_pipeline_monitor.core.jwt_auth", "import ml_pipeline_monitor.core.jwt_auth"),
    ("from ml_pipeline_monitor.core.alerts", "from ml_pipeline_monitor.core.alerts"),
    ("import ml_pipeline_monitor.core.alerts", "import ml_pipeline_monitor.core.alerts"),
    ("from ml_pipeline_monitor.core.metrics", "from ml_pipeline_monitor.core.metrics"),
    ("import ml_pipeline_monitor.core.metrics", "import ml_pipeline_monitor.core.metrics"),
    ("from ml_pipeline_monitor.core.system_monitor", "from ml_pipeline_monitor.core.system_monitor"),
    ("import ml_pipeline_monitor.core.system_monitor", "import ml_pipeline_monitor.core.system_monitor"),
    ("from ml_pipeline_monitor.utils.ui_theme", "from ml_pipeline_monitor.utils.ui_theme"),
    ("import ml_pipeline_monitor.utils.ui_theme", "import ml_pipeline_monitor.utils.ui_theme"),
    ("from ml_pipeline_monitor.ml.pipeline", "from ml_pipeline_monitor.ml.pipeline"),
    ("import ml_pipeline_monitor.ml.pipeline", "import ml_pipeline_monitor.ml.pipeline"),
    ("from ml_pipeline_monitor.ml.data_loader", "from ml_pipeline_monitor.ml.data_loader"),
    ("import ml_pipeline_monitor.ml.data_loader", "import ml_pipeline_monitor.ml.data_loader"),
    ("from ml_pipeline_monitor.ml.data_validation", "from ml_pipeline_monitor.ml.data_validation"),
    ("import ml_pipeline_monitor.ml.data_validation", "import ml_pipeline_monitor.ml.data_validation"),
    ("from ml_pipeline_monitor.ml.drift_detector", "from ml_pipeline_monitor.ml.drift_detector"),
    ("import ml_pipeline_monitor.ml.drift_detector", "import ml_pipeline_monitor.ml.drift_detector"),
    ("from ml_pipeline_monitor.ml.feature_store", "from ml_pipeline_monitor.ml.feature_store"),
    ("import ml_pipeline_monitor.ml.feature_store", "import ml_pipeline_monitor.ml.feature_store"),
    ("from ml_pipeline_monitor.ml.mlflow_tracker", "from ml_pipeline_monitor.ml.mlflow_tracker"),
    ("import ml_pipeline_monitor.ml.mlflow_tracker", "import ml_pipeline_monitor.ml.mlflow_tracker"),
    ("from ml_pipeline_monitor.ml.model_cache", "from ml_pipeline_monitor.ml.model_cache"),
    ("import ml_pipeline_monitor.ml.model_cache", "import ml_pipeline_monitor.ml.model_cache"),
    ("from ml_pipeline_monitor.services.", "from ml_pipeline_monitor.services."),
    ("import ml_pipeline_monitor.services.", "import ml_pipeline_monitor.services."),
    ("ml_pipeline_monitor.api.main", "ml_pipeline_monitor.api.main"),
    ("ml_pipeline_monitor.api.__main__", "ml_pipeline_monitor.api.__main__"),
    ("from ml_pipeline_monitor.", "from ml_pipeline_monitor."),
    ("import ml_pipeline_monitor.", "import ml_pipeline_monitor."),
]

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".pytest_cache", ".mypy_cache", ".ruff_cache", "htmlcov", "playwright-report", "artifacts", "data", "logs"}
SKIP_EXTS = {".pyc", ".pyo", ".db", ".db-shm", ".db-wal", ".joblib", ".pkl", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".woff", ".woff2", ".ttf", ".eot", ".pdf", ".sqlite3"}


def process_file(path: Path) -> bool:
    try:
        content = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, PermissionError):
        return False

    original = content
    for old, new in REPLACEMENTS:
        content = content.replace(old, new)

    if content != original:
        path.write_text(content, encoding="utf-8")
        return True
    return False


def main() -> None:
    changed = []
    for path in ROOT.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".ini", ".cfg", ".toml", ".yaml", ".yml", ".md", ".txt", ".sh", ".bat", ".ps1", ".ts", ".json"}:
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            if path.suffix in SKIP_EXTS:
                continue
            if process_file(path):
                changed.append(str(path.relative_to(ROOT)))

    print(f"Updated {len(changed)} files:")
    for f in sorted(changed):
        print(f"  {f}")


if __name__ == "__main__":
    main()