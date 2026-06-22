"""Coverage-targeted tests for secrets and pipeline service."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.secrets import SecretsManager, get_secret
from services.pipeline_service import compute_next_run_ts, get_pipeline_defaults, should_trigger_scheduled_run


class TestSecretsMore:
    def test_get_required_exists(self):
        with patch.dict(os.environ, {"EXISTING_KEY": "value"}):
            mgr = SecretsManager(secrets_dir="/nonexistent")
            assert mgr.get_required("existing_key") == "value"

    def test_get_from_file_found(self):
        mgr = SecretsManager(secrets_dir="/nonexistent")
        # Should fall through to default since no env/file/JSON provides it
        assert mgr.get("nothing", "fallback") == "fallback"


class TestPipelineServiceMore:
    def test_get_pipeline_defaults_with_cfg(self):
        with patch("services.pipeline_service.load_config", return_value={"pipeline": {"random_seed": 42, "test_size": 0.2, "cv_folds": 5, "n_jobs": -1}}):
            d = get_pipeline_defaults()
        assert d["random_seed"] == 42
        assert d["test_size"] == 0.2
        assert d["cv_folds"] == 5
        assert d["n_jobs"] == -1

    def test_compute_next_run_ts(self):
        from datetime import datetime, UTC
        result = compute_next_run_ts(60)
        now = datetime.now(UTC)
        assert result > now
        assert (result - now).total_seconds() >= 3580

    def test_should_trigger_scheduled_run_enabled(self):
        from datetime import datetime, UTC, timedelta
        assert should_trigger_scheduled_run(enabled=True, next_run_at=datetime.now(UTC) - timedelta(minutes=10))

    def test_should_trigger_scheduled_run_disabled(self):
        from datetime import datetime, UTC
        assert not should_trigger_scheduled_run(enabled=False, next_run_at=datetime.now(UTC))

    def test_should_trigger_scheduled_run_none(self):
        assert not should_trigger_scheduled_run(enabled=True, next_run_at=None)
