"""Tests for config_loader module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from src.config_loader import (
    DEFAULT_CONFIG,
    _deep_merge,
    get_artifact_dirs,
    load_config,
    ROOT_DIR,
)


def test_deep_merge():
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    override = {"b": {"c": 10}, "e": 5}
    result = _deep_merge(base, override)
    assert result["a"] == 1
    assert result["b"]["c"] == 10
    assert result["b"]["d"] == 3
    assert result["e"] == 5


def test_deep_merge_non_dict_override():
    base = {"a": 1}
    override = {"a": 2}
    result = _deep_merge(base, override)
    assert result["a"] == 2


def test_load_config_returns_default_when_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("CONFIG_PATH", str(tmp_path / "missing.yaml"))
    from src import config_loader
    config_loader.load_config.cache_clear()
    result = load_config()
    assert result["pipeline"]["random_seed"] == DEFAULT_CONFIG["pipeline"]["random_seed"]


def test_load_config_from_yaml(tmp_path, monkeypatch):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump({"pipeline": {"random_seed": 99}}), encoding="utf-8")
    monkeypatch.setenv("CONFIG_PATH", str(cfg_file))
    from src import config_loader
    config_loader.load_config.cache_clear()
    result = load_config()
    assert result["pipeline"]["random_seed"] == 99
    assert result["pipeline"]["test_size"] == DEFAULT_CONFIG["pipeline"]["test_size"]


def test_load_config_invalid_yaml(tmp_path, monkeypatch):
    cfg_file = tmp_path / "bad.yaml"
    cfg_file.write_text("not: valid: yaml: [", encoding="utf-8")
    monkeypatch.setenv("CONFIG_PATH", str(cfg_file))
    from src import config_loader
    config_loader.load_config.cache_clear()
    result = load_config()
    assert result["pipeline"]["random_seed"] == DEFAULT_CONFIG["pipeline"]["random_seed"]


def test_get_artifact_dirs(tmp_path, monkeypatch):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump({"storage": {"artifacts_root": str(tmp_path / "art")}}), encoding="utf-8")
    monkeypatch.setenv("CONFIG_PATH", str(cfg_file))
    from src import config_loader
    config_loader.load_config.cache_clear()
    dirs = get_artifact_dirs()
    assert dirs["models"].exists()
    assert dirs["scalers"].exists()
