"""Tests for config_loader module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import yaml

from ml_pipeline_monitor.core.config_loader import (
    DEFAULT_CONFIG,
    _deep_merge,
    _parse_env_line,
    get_artifact_dirs,
    load_config,
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


def test_load_config_returns_default_when_missing():
    with patch("ml_pipeline_monitor.core.config_loader.CONFIG_PATH", Path("/nonexistent/path/config.yaml")):
        load_config.cache_clear()
        result = load_config()
        load_config.cache_clear()
    assert result["pipeline"]["random_seed"] == DEFAULT_CONFIG["pipeline"]["random_seed"]


def test_load_config_from_yaml(tmp_path, monkeypatch):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump({"pipeline": {"random_seed": 99}}), encoding="utf-8")
    monkeypatch.setenv("CONFIG_PATH", str(cfg_file))
    with patch("ml_pipeline_monitor.core.config_loader.CONFIG_PATH", cfg_file):
        load_config.cache_clear()
        result = load_config()
        load_config.cache_clear()
    assert result["pipeline"]["random_seed"] == 99
    assert result["pipeline"]["test_size"] == DEFAULT_CONFIG["pipeline"]["test_size"]


def test_get_artifact_dirs(tmp_path, monkeypatch):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump({"storage": {"artifacts_root": str(tmp_path / "art")}}), encoding="utf-8")
    monkeypatch.setenv("CONFIG_PATH", str(cfg_file))
    with patch("ml_pipeline_monitor.core.config_loader.CONFIG_PATH", cfg_file):
        load_config.cache_clear()
        dirs = get_artifact_dirs()
        load_config.cache_clear()
    assert dirs["models"].exists()
    assert dirs["scalers"].exists()


class TestEnvLineParsing:
    """Regression tests for .env parsing.

    Copying .env.example verbatim used to assign the explanatory comment text
    as the value, so POSTGRES_PASSWORD literally became
    '# REQUIRED: Generate with: ...' instead of empty.
    """

    def test_inline_comment_after_empty_value_is_not_the_value(self):
        parsed = _parse_env_line("POSTGRES_PASSWORD=          # REQUIRED: generate one")
        assert parsed == "POSTGRES_PASSWORD="

    def test_inline_comment_is_stripped(self):
        assert _parse_env_line("SPACED=abc   # note") == "SPACED=abc"

    def test_double_quotes_are_removed(self):
        assert _parse_env_line('JWT_SECRET="quoted-value"') == "JWT_SECRET=quoted-value"

    def test_single_quotes_are_removed(self):
        assert _parse_env_line("AUTH_ROLE='admin'") == "AUTH_ROLE=admin"

    def test_hash_without_leading_space_is_kept(self):
        """A '#' inside a URL fragment or password is not a comment."""
        assert _parse_env_line("PW=has#hash") == "PW=has#hash"
        assert _parse_env_line("URL=postgresql://u:p@h:5432/db?x=1#frag") == "URL=postgresql://u:p@h:5432/db?x=1#frag"

    def test_export_prefix_is_accepted(self):
        assert _parse_env_line("export FOO=bar") == "FOO=bar"

    def test_comments_and_blanks_are_skipped(self):
        assert _parse_env_line("# just a comment") is None
        assert _parse_env_line("") is None
        assert _parse_env_line("no_equals_here") is None

    def test_quoted_value_keeps_inline_hash(self):
        assert _parse_env_line('PW="abc # notacomment"') == "PW=abc # notacomment"
