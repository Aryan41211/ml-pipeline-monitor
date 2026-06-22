"""Tests for secrets management."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.secrets import SecretsManager, get_secret, get_secrets_manager


def test_secrets_manager_env_priority(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "env_value")
    mgr = SecretsManager()
    assert mgr.get("test_key") == "env_value"


def test_secrets_manager_file_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("TEST_KEY2", raising=False)
    secret_file = tmp_path / "test_key2"
    secret_file.write_text("file_value", encoding="utf-8")
    mgr = SecretsManager(secrets_dir=str(tmp_path))
    assert mgr.get("test_key2") == "file_value"


def test_secrets_manager_local_json_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("TEST_KEY3", raising=False)
    local = tmp_path / ".secrets.json"
    local.write_text('{"test_key3": "json_value"}', encoding="utf-8")
    mgr = SecretsManager(secrets_dir=str(tmp_path))
    with patch("src.secrets.Path", side_effect=lambda p: tmp_path / p if isinstance(p, str) else p):
        # patch cwd to tmp_path so .secrets.json is found
        with patch("pathlib.Path.cwd", return_value=tmp_path):
            assert mgr.get("test_key3") == "json_value"


def test_secrets_manager_default():
    mgr = SecretsManager(secrets_dir="/nonexistent")
    assert mgr.get("missing", "fallback") == "fallback"
    assert mgr.get("missing") is None


def test_get_required_secret_missing():
    mgr = SecretsManager(secrets_dir="/nonexistent")
    with pytest.raises(ValueError, match="not found"):
        mgr.get_required("missing_key")


def test_global_secrets_manager_singleton():
    get_secrets_manager.cache_clear() if hasattr(get_secrets_manager, "cache_clear") else None
    # Just verify the function returns a SecretsManager instance
    mgr = get_secrets_manager()
    assert isinstance(mgr, SecretsManager)


def test_get_secret_from_env():
    mgr = SecretsManager(secrets_dir="/nonexistent")
    with patch.dict(os.environ, {"MY_KEY": "env_val"}):
        assert mgr.get("my_key") == "env_val"


def test_get_required_secret_from_env():
    mgr = SecretsManager(secrets_dir="/nonexistent")
    with patch.dict(os.environ, {"REQUIRED_KEY": "required_val"}):
        assert mgr.get_required("required_key") == "required_val"


def test_load_all_with_prefix():
    mgr = SecretsManager(secrets_dir="/nonexistent")
    with patch.dict(os.environ, {"TEST_A": "1", "OTHER_B": "2"}):
        result = mgr.load_all(prefix="test")
    assert "test_a" in result
    assert result["test_a"] == "1"
    assert "test_b" not in result


def test_convenience_functions():
    with patch.dict(os.environ, {"CONV_KEY": "conv_val"}):
        assert get_secret("conv_key") == "conv_val"
