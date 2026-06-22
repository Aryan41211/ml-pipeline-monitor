"""Final coverage-targeted tests for pipeline_service and secrets."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from services.pipeline_service import get_dataset_preview, list_experiments
from src.secrets import SecretsManager


class TestPipelineServiceMore:
    def test_list_experiments_with_limit(self):
        with patch("services.pipeline_service.get_experiments", return_value=[{"run_id": i} for i in range(5)]) as m:
            result = list_experiments(limit=5)
        m.assert_called_once_with(limit=5)
        assert len(result) == 5

    def test_get_dataset_preview(self):
        with patch("services.pipeline_service.load_dataset", return_value={"X_train": "x", "X_test": "x", "y_train": "y", "y_test": "y", "feature_names": ["f1", "f2"]}):
            with patch("services.pipeline_service.get_feature_statistics", return_value={"mean": [1.0]}):
                result = get_dataset_preview("iris", test_size=0.2, random_state=42)
        assert "dataset" in result
        assert "feature_stats" in result


class TestSecretsMore:
    def test_get_secret_from_file(self, tmp_path):
        secret_file = tmp_path / "my_key"
        secret_file.write_text("file_value", encoding="utf-8")
        mgr = SecretsManager(secrets_dir=str(tmp_path))
        assert mgr.get("my_key") == "file_value"

    def test_secrets_manager_load_all_no_env(self):
        env = dict(os.environ)
        env.pop("TEST_PREFIX_X", None)
        with patch.dict(os.environ, env, clear=True):
            mgr = SecretsManager(secrets_dir="/nonexistent")
            result = mgr.load_all(prefix="test_prefix_x")
        assert result == {}
