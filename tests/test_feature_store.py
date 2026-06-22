"""Tests for feature_store module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.feature_store import (
    _feature_store_root,
    load_cached_splits,
    make_feature_key,
    save_cached_splits,
)


def test_make_feature_key():
    key = make_feature_key("iris", 0.2, 42)
    assert isinstance(key, str)
    assert len(key) == 16


def test_make_feature_key_deterministic():
    key1 = make_feature_key("iris", 0.2, 42)
    key2 = make_feature_key("iris", 0.2, 42)
    assert key1 == key2


def test_make_feature_key_different_inputs():
    key1 = make_feature_key("iris", 0.2, 42)
    key2 = make_feature_key("iris", 0.3, 42)
    assert key1 != key2


def test_load_cached_splits_missing():
    with patch("src.feature_store._feature_store_root", return_value=Path("/nonexistent")):
        result = load_cached_splits("nonexistent_key")
    assert result is None


def test_save_and_load_cached_splits(tmp_path):
    payload = {"X_train": [[1, 2]], "y_train": [0]}
    with patch("src.feature_store._feature_store_root", return_value=tmp_path):
        path = save_cached_splits("test_key", payload)
        assert isinstance(path, str)
        loaded = load_cached_splits("test_key")
    assert loaded == payload
